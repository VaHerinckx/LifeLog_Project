"""
Music Topic Coordinator

This module coordinates music data from multiple sources:
1. Last.fm (real-time listening history via API)
2. Spotify legacy (historical 2013-2023 JSON exports)

Enriches with Spotify API metadata (artist/track info, genres, audio features).
Generates website-optimized files with listening statistics.
Uploads results to Google Drive.

Output: files/topic_processed_files/music/music_processed.csv
        files/website_files/music/music_page_data.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from src.utils.utils_functions import enforce_snake_case, record_successful_run
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.spotify_api_utils import spotify_authentication, get_artist_info, get_track_info
from src.topic_processing.music.genre_mapping import get_simplified_genre
from src.topic_processing.website_maintenance.website_maintenance_processing import full_website_maintenance_pipeline
from src.sources_processing.lastfm.lastfm_processing import full_lastfm_pipeline

load_dotenv()
from src.utils.logger import log


# ============================================================================
# DATA LOADING AND MERGING
# ============================================================================

def load_source_data():
    """
    Load data from source processors.

    Returns:
        pandas.DataFrame: Combined data from Last.fm and Spotify sources
    """
    log.progress("\n Loading source data...")

    lastfm_path = 'files/source_processed_files/lastfm/lastfm_processed.csv'
    spotify_path = 'files/source_processed_files/spotify/spotify_processed.csv'

    dfs_to_merge = []

    # Load Last.fm data
    if os.path.exists(lastfm_path):
        df_lastfm = pd.read_csv(lastfm_path, sep='|', encoding='utf-8', low_memory=False)
        df_lastfm['timestamp'] = pd.to_datetime(df_lastfm['timestamp'])
        log.success(f"Loaded {len(df_lastfm):,} tracks from Last.fm")
        dfs_to_merge.append(df_lastfm)
    else:
        log.warning(f"Last.fm source file not found: {lastfm_path}")

    # Load Spotify legacy data
    if os.path.exists(spotify_path):
        df_spotify = pd.read_csv(spotify_path, sep='|', encoding='utf-8', low_memory=False)
        df_spotify['timestamp'] = pd.to_datetime(df_spotify['timestamp'])
        log.success(f"Loaded {len(df_spotify):,} tracks from Spotify legacy")
        dfs_to_merge.append(df_spotify)
    else:
        log.warning(f"Spotify source file not found: {spotify_path}")

    if not dfs_to_merge:
        log.error("No source data found")
        return pd.DataFrame()

    # Merge all sources
    df_combined = pd.concat(dfs_to_merge, ignore_index=True)

    # Remove duplicates (same song_key + timestamp)
    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['song_key', 'timestamp'], keep='first')
    final_count = len(df_combined)

    duplicates_removed = initial_count - final_count
    if duplicates_removed > 0:
        log.progress(f"Removed {duplicates_removed:,} duplicate tracks")

    # Sort by timestamp (newest first)
    df_combined = df_combined.sort_values('timestamp', ascending=False)

    log.success(f"Combined dataset: {len(df_combined):,} unique tracks")
    log.info(f"📅 Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")

    return df_combined


# ============================================================================
# SPOTIFY ENRICHMENT
# ============================================================================

def enrich_with_spotify_metadata(df):
    """
    Enrich data with Spotify API metadata (artists, tracks, genres, audio features).

    Args:
        df (pandas.DataFrame): Combined source data

    Returns:
        pandas.DataFrame: Enriched data with Spotify metadata
    """
    log.progress("\n Enriching with Spotify API metadata...")

    # Get Spotify API credentials
    client_id = os.environ.get('Spotify_API_Client_ID')
    client_secret = os.environ.get('Spotify_API_Client_Secret')

    if not client_id or not client_secret:
        log.warning("Spotify API credentials not found in environment variables")
        log.info(" Proceeding without Spotify enrichment...")
        return df

    # Authenticate with Spotify API
    log.info("Authenticating with Spotify API...")
    token = spotify_authentication(client_id, client_secret)

    # Prepare unique artists and tracks
    unique_artists = list(df['artist_name'].astype(str).replace("nan", "nan_").unique())
    unique_tracks = list(df['song_key'].astype(str).unique())

    log.progress(f"Found {len(unique_artists):,} unique artists and {len(unique_tracks):,} unique tracks")

    # Define work file paths
    artists_work_file = 'files/work_files/lastfm_work_files/artists_infos.csv'
    tracks_work_file = 'files/work_files/lastfm_work_files/tracks_infos.csv'

    # Count how many are NEW (not in work files) - helps estimate API call time
    existing_artists = set()
    existing_tracks = set()
    if os.path.exists(artists_work_file):
        existing_df = pd.read_csv(artists_work_file, sep='|', low_memory=False)
        existing_artists = set(existing_df['artist_name'].str.lower())
    if os.path.exists(tracks_work_file):
        existing_df = pd.read_csv(tracks_work_file, sep='|', low_memory=False)
        existing_tracks = set(existing_df['song_key'].str.lower())

    new_artists_count = len([a for a in unique_artists if str(a).lower() not in existing_artists])
    new_tracks_count = len([t for t in unique_tracks if str(t).lower() not in existing_tracks])
    log.progress(f"New artists to fetch: {new_artists_count:,} (cached: {len(existing_artists):,})")
    log.progress(f"New tracks to fetch: {new_tracks_count:,} (cached: {len(existing_tracks):,})")
    log.normal(f"New artists to fetch: {new_artists_count:,} (cached: {len(existing_artists):,})")
    log.normal(f"New tracks to fetch: {new_tracks_count:,} (cached: {len(existing_tracks):,})")

    # Get artist information
    log.info("🎤 Gathering artist information from Spotify API...")
    artist_df = get_artist_info(token, unique_artists, artists_work_file)

    # Get track information
    log.info("🎶 Gathering track information from Spotify API...")
    track_df = get_track_info(token, unique_tracks, tracks_work_file)

    # Merge artist info (use lowercase for case-insensitive merge)
    log.progress("Merging artist metadata...")
    df['_artist_name_lower'] = df['artist_name'].astype(str).str.lower()
    artist_df['_artist_name_lower'] = artist_df['artist_name'].astype(str).str.lower()
    artist_df_no_name = artist_df.drop(columns=['artist_name'])  # Remove to avoid duplicate column

    df_merge_artist = pd.merge(df, artist_df_no_name, how='left', on='_artist_name_lower')
    df_merge_artist = df_merge_artist.drop(columns=['_artist_name_lower'])

    # Check if artist_artwork_url was merged
    if 'artist_artwork_url' in df_merge_artist.columns:
        non_null = df_merge_artist['artist_artwork_url'].notna().sum()
        log.success(f"artist_artwork_url merged ({non_null:,} non-null values)")
    else:
        log.warning(f"artist_artwork_url NOT in artist merge result")

    # Merge track info (only new columns)
    log.progress("Merging track metadata...")
    cols_to_use = list(track_df.columns.difference(df_merge_artist.columns))
    cols_to_use.append('song_key')

    # Ensure album_artwork_url is included if it exists in track_df
    if 'album_artwork_url' in track_df.columns and 'album_artwork_url' not in cols_to_use:
        cols_to_use.append('album_artwork_url')

    # Use lowercase song_key for case-insensitive merge (work file stores lowercase keys)
    df_merge_artist['_song_key_lower'] = df_merge_artist['song_key'].astype(str).str.lower()
    track_df_subset = track_df[cols_to_use].copy()
    track_df_subset['_song_key_lower'] = track_df_subset['song_key'].astype(str).str.lower()
    track_df_subset = track_df_subset.drop(columns=['song_key'])  # Remove original to avoid duplicate

    df_enriched = pd.merge(df_merge_artist, track_df_subset, how='left', on='_song_key_lower')
    df_enriched = df_enriched.drop(columns=['_song_key_lower'])  # Clean up temp column

    # Check if album_artwork_url was merged
    if 'album_artwork_url' in df_enriched.columns:
        non_null = df_enriched['album_artwork_url'].notna().sum()
        log.success(f"album_artwork_url merged ({non_null:,} non-null values)")
    else:
        log.warning(f"album_artwork_url NOT in track merge result")

    log.success(f"Enrichment complete: {len(df_enriched.columns)} total columns")

    return df_enriched


# ============================================================================
# LISTENING STATISTICS
# ============================================================================

def compute_completion(df):
    """
    Calculate listening completion percentage and skip detection.

    Completion is calculated by comparing consecutive plays:
    - If time to next track < track duration, track was likely skipped
    - Completion % = time_elapsed / track_duration

    Uses vectorized pandas operations for performance.

    Args:
        df (pandas.DataFrame): Data with timestamp and track_duration columns

    Returns:
        pandas.DataFrame: Data with 'completion' and 'is_skipped_track' columns
    """
    log.progress("\n Calculating listening statistics...")

    df = df.copy()

    # Sort chronologically for time-diff calculation
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # Convert track_duration to numeric seconds (handle non-numeric values)
    df['_duration_sec'] = pd.to_numeric(df['track_duration'], errors='coerce') / 1000

    # Calculate time diff to next track (vectorized)
    df['_time_to_next'] = df['timestamp'].shift(-1) - df['timestamp']
    df['_time_to_next_sec'] = df['_time_to_next'].dt.total_seconds()

    # Calculate completion (vectorized) - clip between 0 and 1
    df['completion'] = (df['_time_to_next_sec'] / df['_duration_sec']).clip(0, 1)

    # Handle missing/zero durations (assume full listen)
    df.loc[df['_duration_sec'].isna() | (df['_duration_sec'] == 0), 'completion'] = 1.0

    # Last track - assume completed (no next track to compare)
    df.loc[df.index[-1], 'completion'] = 1.0

    # Fill any remaining NaN completions with 1.0
    df['completion'] = df['completion'].fillna(1.0)

    # Skip detection: True if completion < 1.0 (boolean)
    df['is_skipped_track'] = df['completion'] < 1.0

    # Clean up temp columns
    df = df.drop(columns=['_duration_sec', '_time_to_next', '_time_to_next_sec'])

    # Sort back to descending timestamp
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    # Calculate and print statistics
    avg_completion = df['completion'].mean() * 100
    skip_rate = df['is_skipped_track'].mean() * 100
    log.success(f"Average completion: {avg_completion:.1f}%")
    log.success(f"Skip rate: {skip_rate:.1f}%")

    return df


def calculate_discovery_flags(df):
    """
    Calculate new artist/track discovery flags (boolean columns).

    Flags:
    - is_new_artist: True only for the first listen of an artist
    - is_new_track: True only for the first listen of a track
    - is_new_recurring_artist: True only at 10th listen of an artist (milestone)
    - is_new_recurring_track: True only at 5th listen of a track (milestone)
    - is_recurring_artist: True for ALL listens once artist has 10+ total listens
    - is_recurring_track: True for ALL listens once track has 5+ total listens

    Args:
        df (pandas.DataFrame): Data with timestamp column

    Returns:
        pandas.DataFrame: Data with discovery flag columns (boolean)
    """
    log.progress("Calculating discovery flags...")

    # Sort chronologically for cumulative counting
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # First listen flags (True only for the first occurrence)
    df['is_new_artist'] = df.groupby('artist_name').cumcount() == 0
    df['is_new_track'] = df.groupby('track_name').cumcount() == 0

    # Milestone flags (True only at the 10th/5th listen)
    df['is_new_recurring_artist'] = df.groupby('artist_name').cumcount() == 9  # 10th listen (0-indexed)
    df['is_new_recurring_track'] = df.groupby('track_name').cumcount() == 4  # 5th listen (0-indexed)

    # Recurring status (True for ALL listens once threshold reached)
    # Use transform to get total count per artist/track across all time
    artist_total_count = df.groupby('artist_name')['artist_name'].transform('count')
    track_total_count = df.groupby('track_name')['track_name'].transform('count')

    df['is_recurring_artist'] = artist_total_count >= 10
    df['is_recurring_track'] = track_total_count >= 5

    # Sort back to descending timestamp
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    # Print statistics
    new_artists = df['is_new_artist'].sum()
    new_tracks = df['is_new_track'].sum()
    recurring_artists = df['is_recurring_artist'].sum()
    recurring_tracks = df['is_recurring_track'].sum()
    log.success(f"Discovered {new_artists:,} new artists and {new_tracks:,} new tracks")
    log.success(f"Recurring: {recurring_artists:,} artist listens, {recurring_tracks:,} track listens")

    return df


# ============================================================================
# OUTPUT FILE GENERATION
# ============================================================================

def select_output_columns(df):
    """
    Select only columns needed for final output.

    Work files contain ~60+ columns (all Spotify API fields).
    Output files contain ~35 columns (only what's needed for website/analysis).

    Args:
        df (pandas.DataFrame): Full enriched data

    Returns:
        pandas.DataFrame: Data with only output columns
    """
    output_columns = [
        # Core identifiers
        'timestamp', 'song_key', 'artist_name', 'album_name', 'track_name',

        # Album metadata
        'album_release_date', 'album_type',

        # Artist metadata
        'followers', 'followers_total', 'artist_popularity', 'popularity',

        # Genres (first 14 for display, plus JSON for future use)
        'genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6',
        'genre_7', 'genre_8', 'genre_9', 'genre_10', 'genre_11', 'genre_12',
        'genre_13', 'genre_14', 'genres_json',

        # Track metadata
        'track_duration', 'track_popularity', 'track_number', 'explicit',

        # Audio features
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',

        # Artwork URLs
        'album_artwork_url', 'artist_artwork_url',

        # Listening behavior (calculated fields)
        'completion', 'is_skipped_track',

        # Discovery flags (boolean)
        'is_new_artist', 'is_new_track', 'is_new_recurring_artist', 'is_new_recurring_track',
        'is_recurring_artist', 'is_recurring_track',

        # Source tracking
        'source'
    ]

    # Keep only columns that exist in the dataframe
    existing_output_columns = [col for col in output_columns if col in df.columns]

    # Log how many columns are being filtered out
    removed_count = len(df.columns) - len(existing_output_columns)
    if removed_count > 0:
        log.debug(f"Filtered output: keeping {len(existing_output_columns)} columns, removing {removed_count} work-file-only columns")

    return df[existing_output_columns]


def power_bi_processing(df):
    """
    Data type conversions for better display in analysis tools.

    Args:
        df (pandas.DataFrame): Output data

    Returns:
        pandas.DataFrame: Cleaned data
    """
    df = df.copy()

    # Fill missing genre_1 with 'Unknown'
    if 'genre_1' in df.columns:
        df['genre_1'] = df['genre_1'].fillna('Unknown')

    # Convert track_duration to float (replace non-numeric strings with 0)
    if 'track_duration' in df.columns:
        df['track_duration'] = df['track_duration'].replace(['No API result', 'Unknown', 'Unknown Error'], '0').astype(float)

    return df


# ============================================================================
# WEBSITE FILE GENERATION
# ============================================================================

def generate_music_website_files(df):
    """
    Generate website-optimized files for the Music page.

    Args:
        df (pandas.DataFrame): Processed data (topic-level)

    Returns:
        bool: True if successful, False otherwise
    """
    log.progress("\n Generating website files for Music page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/music'
        os.makedirs(website_dir, exist_ok=True)

        # Work with copy to avoid modifying original
        df_web = df.copy()

        # Enforce snake_case before saving
        df_web = enforce_snake_case(df_web, "music_page_data")

        # Add toggle_id - unique identifier for each listening event
        df_web['toggle_id'] = range(1, len(df_web) + 1)
        log.success(f"Added toggle_id column ({len(df_web):,} toggles)")

        # Add listening_seconds - calculated from completion and track_duration
        # listening_seconds = (completion * track_duration_ms) / 1000
        df_web['listening_seconds'] = (df_web['completion'] * df_web['track_duration']) / 1000
        df_web['listening_seconds'] = df_web['listening_seconds'].fillna(0).astype(int)
        log.success(f"Added listening_seconds column")

        # Combine genres from available genre columns into single 'genres' column
        # Use comma separator (not pipe, which is the CSV delimiter)
        # Filter to only existing genre columns (may have genre_1 through genre_8 or more)
        genre_cols = [f'genre_{i}' for i in range(1, 15) if f'genre_{i}' in df_web.columns]
        if genre_cols:
            df_web['genres'] = df_web[genre_cols].apply(
                lambda row: ', '.join([str(g) for g in row if pd.notna(g) and str(g) != '' and str(g) != 'nan']),
                axis=1
            )
            log.success(f"Combined genres from {len(genre_cols)} genre columns into single column (comma-separated)")
        else:
            df_web['genres'] = ''
            log.warning(f"No genre columns found, genres will be empty")

        # Add simplified_genre column using genre_1
        df_web['simplified_genre'] = df_web['genre_1'].apply(get_simplified_genre)
        log.success(f"Created simplified_genre from genre_1 using genre mapping")

        # Select columns for website file
        website_columns = [
            'toggle_id',
            'song_key',
            'artist_name',
            'album_name',
            'track_name',
            'timestamp',
            'album_release_date',
            'followers',
            'artist_popularity',
            'genre_1',
            'simplified_genre',
            'genres',
            'track_duration',
            'track_popularity',
            'completion',
            'is_skipped_track',
            'listening_seconds'
        ]

        # Add discovery flag columns if they exist in the data
        if 'is_new_artist' in df_web.columns:
            website_columns.append('is_new_artist')
            log.success(f"Including is_new_artist column")

        if 'is_new_track' in df_web.columns:
            website_columns.append('is_new_track')
            log.success(f"Including is_new_track column")

        # Add is_recurring_artist if it exists in the data
        if 'is_recurring_artist' in df_web.columns:
            website_columns.append('is_recurring_artist')
            log.success(f"Including is_recurring_artist column")

        # Add is_recurring_track if it exists in the data
        if 'is_recurring_track' in df_web.columns:
            website_columns.append('is_recurring_track')
            log.success(f"Including is_recurring_track column")

        # Add is_new_recurring_artist if it exists in the data (milestone: 10th listen)
        if 'is_new_recurring_artist' in df_web.columns:
            website_columns.append('is_new_recurring_artist')
            log.success(f"Including is_new_recurring_artist column")

        # Add is_new_recurring_track if it exists in the data (milestone: 5th listen)
        if 'is_new_recurring_track' in df_web.columns:
            website_columns.append('is_new_recurring_track')
            log.success(f"Including is_new_recurring_track column")

        # Add album_artwork_url if it exists in the data
        if 'album_artwork_url' in df_web.columns:
            website_columns.append('album_artwork_url')
            non_null = df_web['album_artwork_url'].notna().sum()
            has_url = df_web['album_artwork_url'].str.startswith('http', na=False).sum()
            log.success(f"Including album_artwork_url column ({non_null:,} non-null, {has_url:,} valid URLs)")
        else:
            log.warning(f"album_artwork_url column NOT found in data")

        # Add artist_artwork_url if it exists in the data
        if 'artist_artwork_url' in df_web.columns:
            website_columns.append('artist_artwork_url')
            non_null = df_web['artist_artwork_url'].notna().sum()
            has_url = df_web['artist_artwork_url'].str.startswith('http', na=False).sum()
            log.success(f"Including artist_artwork_url column ({non_null:,} non-null, {has_url:,} valid URLs)")
        else:
            log.warning(f"artist_artwork_url column NOT found in data")

        # Filter to only website columns
        df_web = df_web[website_columns]
        log.success(f"Filtered to {len(website_columns)} website columns")

        # Convert boolean columns to integers for proper CSV storage
        # (Python "True"/"False" strings are truthy in JavaScript, 0/1 works correctly)
        bool_cols = ['is_skipped_track', 'is_new_artist', 'is_new_track',
                     'is_recurring_artist', 'is_recurring_track',
                     'is_new_recurring_artist', 'is_new_recurring_track']
        for col in bool_cols:
            if col in df_web.columns:
                df_web[col] = df_web[col].astype(int)
        log.success(f"Converted boolean columns to integers")

        # Add derived date columns
        log.info("📅 Adding derived date columns...")
        df_web['listening_year'] = pd.to_datetime(df_web['timestamp']).dt.year.astype('Int64').astype(str)
        df_web['listening_month'] = pd.to_datetime(df_web['timestamp']).dt.month.astype('Int64').astype(str)
        df_web['listening_quarter'] = pd.to_datetime(df_web['timestamp']).dt.quarter.astype('Int64').astype(str)

        # Reduce the dataset size by removing all tracks with less than 10% completion (half)
        df_web = df_web[df_web["completion"] >= 0.09]

        # Save website file
        website_path = f'{website_dir}/music_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        log.success(f"Website file: {len(df_web):,} records  {website_path}")

        return True

    except Exception as e:
        log.error(f"Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def create_music_file():
    """
    Main processing function for Music topic coordinator.

    Workflow:
    1. Load data from source processors (Last.fm + Spotify legacy)
    2. Merge and deduplicate
    3. Enrich with Spotify API metadata
    4. Calculate listening statistics (completion, skip detection)
    5. Calculate discovery flags
    6. Select output columns
    7. Generate website files
    8. Save processed data

    Returns:
        bool: True if successful, False otherwise
    """
    log.info("\n" + "="*70)
    log.progress("MUSIC TOPIC COORDINATOR PROCESSING")
    log.info("="*70)

    try:
        # Step 1: Load source data
        df = load_source_data()

        if df.empty:
            log.error("No source data available")
            return False

        # Step 2: Enrich with Spotify metadata
        df = enrich_with_spotify_metadata(df)

        # Step 3: Calculate listening statistics
        df = compute_completion(df)

        # Step 4: Calculate discovery flags
        df = calculate_discovery_flags(df)

        # Step 5: Select output columns (filter work columns)
        df = select_output_columns(df)

        # Step 6: Power BI processing (data type conversions)
        df = power_bi_processing(df)

        # Step 7: Enforce snake_case
        df = enforce_snake_case(df, "music_processed")

        # Step 8: Save topic-level processed file
        output_path = 'files/topic_processed_files/music/music_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        log.success(f"\n Saving topic processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        log.success(f"Successfully processed music data!")
        log.progress(f"Output: {len(df):,} tracks")
        log.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        log.info(f"🎤 Unique artists: {df['artist_name'].nunique():,}")
        log.progress(f"Unique tracks: {df['song_key'].nunique():,}")

        # Show sample
        log.debug(f"\n Sample records:")
        sample_df = df.head(3)[['timestamp', 'artist_name', 'track_name', 'completion', 'is_skipped_track']]
        for _, row in sample_df.iterrows():
            completion_pct = row['completion'] * 100
            skip_status = "⏭️ " if row['is_skipped_track'] else "✓"
            log.info(f"{skip_status} {str(row['timestamp'])[:16]} | {row['artist_name']} - {row['track_name']} ({completion_pct:.0f}%)")

        # Step 9: Generate website files
        website_success = generate_music_website_files(df)

        if not website_success:
            log.warning("Warning: Website file generation had issues")
            return False

        return True

    except Exception as e:
        log.error(f"Error in music topic processing: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# UPLOAD FUNCTIONS
# ============================================================================

def upload_music_results():
    """
    Upload music results to Google Drive.

    Returns:
        bool: True if successful, False otherwise
    """
    log.info("\n⬆️  Uploading music results to Google Drive...")

    files_to_upload = ['files/website_files/music/music_page_data.csv']

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        log.error("No files found to upload")
        return False

    log.info(f"📤 Uploading {len(existing_files)} file(s)...")
    success = upload_multiple_files(existing_files)

    if success:
        log.success("Music results uploaded successfully!")
    else:
        log.error("Some files failed to upload")

    return success


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_music_pipeline(auto_full=False, auto_process_only=False, merge_only=False):
    """
    Complete Music TOPIC coordinator pipeline.

    Options:
    1. Process source data, enrich with Spotify, generate website files, and upload to Drive
    2. Process existing topic data and upload to Drive
    3. Upload existing processed files to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input
        merge_only (bool): If True, runs create_music_file + upload (sources already processed)

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    log.normal("Processing Music...")
    log.info("\n" + "="*70)
    log.progress("MUSIC TOPIC COORDINATOR PIPELINE")
    log.info("="*70)

    if merge_only:
        # Sources already processed by main orchestrator — just merge + enrich + upload
        log.info("🔗 Merge mode: Processing source data and uploading...")
        process_success = create_music_file()
        if not process_success:
            log.error("Processing failed, skipping upload")
            return False
        upload_success = upload_music_results()
        success = upload_success
        if success:
            record_successful_run('topic_music', 'coordination')
        log.info("\n" + "="*70)
        if success:
            log.success("Music topic coordinator completed successfully!")
            log.normal_success("Music processing completed successfully")
            full_website_maintenance_pipeline(auto_mode=True, quiet=True)
        else:
            log.error("Music topic coordinator failed")
            log.normal("Music processing failed")
        log.info("="*70)
        return success

    if auto_process_only:
        log.info("🤖 Auto process mode: Processing existing data and uploading...")
        choice = "2"
    elif auto_full:
        log.info("🤖 Auto mode: Full processing from source data...")
        choice = "1"
    else:
        log.prompt("\nSelect an option:")
        log.prompt("1. Process source data, enrich with Spotify, and upload to Drive")
        log.prompt("2. Process existing topic data and upload to Drive")
        log.prompt("3. Upload existing processed files to Drive")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        log.milestone("\n Option 1: Full processing from source data...")

        # Step 1: Run Last.fm source pipeline (download + process)
        log.info("\n📥 Running Last.fm source pipeline...")
        full_lastfm_pipeline(auto_full=True)

        # Step 2: Process music topic data
        process_success = create_music_file()

        if not process_success:
            log.error("Processing failed, skipping upload")
            return False

        # Test drive connection before upload
        if not verify_drive_connection():
            log.warning("Warning: Google Drive connection issues detected")
            proceed = input("Continue with upload anyway? (Y/N): ").upper() == 'Y'
            if not proceed:
                log.success("Processing completed successfully (upload skipped)")
                return True

        # Upload results
        upload_success = upload_music_results()
        success = upload_success

        if success:
            record_successful_run('topic_music', 'coordination')

    elif choice == "2":
        log.info("\n⚙️  Option 2: Process existing topic data and upload...")

        # Check if topic processed file exists
        topic_file = 'files/topic_processed_files/music/music_processed.csv'

        if os.path.exists(topic_file):
            # Regenerate website files from existing topic data
            df = pd.read_csv(topic_file, sep='|', encoding='utf-8', low_memory=False)
            website_success = generate_music_website_files(df)

            if not website_success:
                log.error("Website file generation failed")
                return False
        else:
            log.error(f"Topic processed file not found: {topic_file}")
            log.info(" Run option 1 first to process source data")
            return False

        # Test drive connection before upload
        if not verify_drive_connection():
            log.warning("Warning: Google Drive connection issues detected")
            proceed = input("Continue with upload anyway? (Y/N): ").upper() == 'Y'
            if not proceed:
                log.success("Processing completed successfully (upload skipped)")
                return True

        # Upload results
        upload_success = upload_music_results()
        success = upload_success

        if success:
            record_successful_run('topic_music', 'coordination')

    elif choice == "3":
        log.info("\n📤 Option 3: Upload existing processed files...")

        # Test drive connection before upload
        if not verify_drive_connection():
            log.warning("Warning: Google Drive connection issues detected")
            proceed = input("Continue with upload anyway? (Y/N): ").upper() == 'Y'
            if not proceed:
                log.success("Upload skipped")
                return True

        # Upload results
        upload_success = upload_music_results()
        success = upload_success

        if success:
            record_successful_run('topic_music', 'coordination')

    else:
        log.error("Invalid choice. Please select 1-3.")
        return False

    # Final status
    log.info("\n" + "="*70)
    if success:
        log.success("Music topic coordinator completed successfully!")
        log.normal_success("Music processing completed successfully")
        log.progress("Topic output: files/topic_processed_files/music/music_processed.csv")
        log.progress("Website output: files/website_files/music/music_page_data.csv")
        # Update website tracking file
        full_website_maintenance_pipeline(auto_mode=True, quiet=True)
    else:
        log.error("Music topic coordinator failed")
        log.normal("Music processing failed")
    log.info("="*70)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    log.progress("Music Topic Coordinator")
    log.info("This tool processes music data from multiple sources and enriches with Spotify metadata.")
    log.info("Note: Run source processors (Last.fm, Spotify) first to generate source data\n")

    # Run the pipeline
    full_music_pipeline(auto_full=False)
