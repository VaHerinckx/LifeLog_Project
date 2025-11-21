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

load_dotenv()


# ============================================================================
# DATA LOADING AND MERGING
# ============================================================================

def load_source_data():
    """
    Load data from source processors.

    Returns:
        pandas.DataFrame: Combined data from Last.fm and Spotify sources
    """
    print("\n📊 Loading source data...")

    lastfm_path = 'files/source_processed_files/lastfm/lastfm_processed.csv'
    spotify_path = 'files/source_processed_files/spotify/spotify_processed.csv'

    dfs_to_merge = []

    # Load Last.fm data
    if os.path.exists(lastfm_path):
        df_lastfm = pd.read_csv(lastfm_path, sep='|', encoding='utf-8', low_memory=False)
        df_lastfm['timestamp'] = pd.to_datetime(df_lastfm['timestamp'])
        print(f"✅ Loaded {len(df_lastfm):,} tracks from Last.fm")
        dfs_to_merge.append(df_lastfm)
    else:
        print(f"⚠️  Last.fm source file not found: {lastfm_path}")

    # Load Spotify legacy data
    if os.path.exists(spotify_path):
        df_spotify = pd.read_csv(spotify_path, sep='|', encoding='utf-8', low_memory=False)
        df_spotify['timestamp'] = pd.to_datetime(df_spotify['timestamp'])
        print(f"✅ Loaded {len(df_spotify):,} tracks from Spotify legacy")
        dfs_to_merge.append(df_spotify)
    else:
        print(f"⚠️  Spotify source file not found: {spotify_path}")

    if not dfs_to_merge:
        print("❌ No source data found")
        return pd.DataFrame()

    # Merge all sources
    df_combined = pd.concat(dfs_to_merge, ignore_index=True)

    # Remove duplicates (same song_key + timestamp)
    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['song_key', 'timestamp'], keep='first')
    final_count = len(df_combined)

    duplicates_removed = initial_count - final_count
    if duplicates_removed > 0:
        print(f"🔄 Removed {duplicates_removed:,} duplicate tracks")

    # Sort by timestamp (newest first)
    df_combined = df_combined.sort_values('timestamp', ascending=False)

    print(f"✅ Combined dataset: {len(df_combined):,} unique tracks")
    print(f"📅 Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")

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
    print("\n🎧 Enriching with Spotify API metadata...")

    # Get Spotify API credentials
    client_id = os.environ.get('Spotify_API_Client_ID')
    client_secret = os.environ.get('Spotify_API_Client_Secret')

    if not client_id or not client_secret:
        print("⚠️  Spotify API credentials not found in environment variables")
        print("   Proceeding without Spotify enrichment...")
        return df

    # Authenticate with Spotify API
    print("🔐 Authenticating with Spotify API...")
    token = spotify_authentication(client_id, client_secret)

    # Prepare unique artists and tracks
    unique_artists = list(df['artist_name'].astype(str).replace("nan", "nan_").unique())
    unique_tracks = list(df['song_key'].astype(str).unique())

    print(f"📊 Found {len(unique_artists):,} unique artists and {len(unique_tracks):,} unique tracks")

    # Get artist information
    print("🎤 Gathering artist information from Spotify API...")
    artists_work_file = 'files/work_files/lastfm_work_files/artists_infos.csv'
    artist_df = get_artist_info(token, unique_artists, artists_work_file)

    # Get track information
    print("🎶 Gathering track information from Spotify API...")
    tracks_work_file = 'files/work_files/lastfm_work_files/tracks_infos.csv'
    track_df = get_track_info(token, unique_tracks, tracks_work_file)

    # Merge artist info
    print("🔄 Merging artist metadata...")
    df_merge_artist = pd.merge(df, artist_df, how='left', on='artist_name')

    # Merge track info (only new columns)
    print("🔄 Merging track metadata...")
    cols_to_use = list(track_df.columns.difference(df_merge_artist.columns))
    cols_to_use.append('song_key')
    df_enriched = pd.merge(df_merge_artist, track_df[cols_to_use], how='left', on='song_key')

    print(f"✅ Enrichment complete: {len(df_enriched.columns)} total columns")

    return df_enriched


# ============================================================================
# LISTENING STATISTICS
# ============================================================================

def compute_completion(df):
    """
    Calculate listening completion percentage and skip detection.

    Completion is calculated by comparing consecutive plays of the same track:
    - If track A plays after track B, but track B duration hasn't elapsed, B was likely skipped
    - Completion % = time_elapsed / track_duration

    Args:
        df (pandas.DataFrame): Data sorted by timestamp descending

    Returns:
        pandas.DataFrame: Data with 'completion' and 'skip_next_track' columns
    """
    print("\n📊 Calculating listening statistics...")

    df = df.copy()
    df['completion'] = 0.0
    df['skip_next_track'] = 0

    # Sort by timestamp ascending for chronological processing
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    total_tracks = len(df)

    for i in range(len(df) - 1):
        current_track = df.loc[i]
        next_track = df.loc[i + 1]

        # Get track duration in seconds
        try:
            duration_ms = float(current_track['track_duration'])
            if pd.isna(duration_ms) or duration_ms == 0 or str(duration_ms) == 'No API result':
                # Can't calculate completion without duration
                df.loc[i, 'completion'] = 1.0  # Assume full listen
                continue
            duration_seconds = duration_ms / 1000
        except (ValueError, TypeError):
            df.loc[i, 'completion'] = 1.0
            continue

        # Calculate time between tracks
        time_diff = (next_track['timestamp'] - current_track['timestamp']).total_seconds()

        # Calculate completion percentage
        if time_diff >= duration_seconds:
            # Track played completely (or beyond)
            completion = 1.0
            skip_next = 0
        else:
            # Track was interrupted
            completion = min(time_diff / duration_seconds, 1.0)
            skip_next = 1

        df.loc[i, 'completion'] = completion
        df.loc[i, 'skip_next_track'] = skip_next

    # Last track - assume completed
    df.loc[len(df) - 1, 'completion'] = 1.0
    df.loc[len(df) - 1, 'skip_next_track'] = 0

    # Sort back to descending timestamp
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    # Calculate statistics
    avg_completion = df['completion'].mean() * 100
    skip_rate = df['skip_next_track'].mean() * 100
    print(f"✅ Average completion: {avg_completion:.1f}%")
    print(f"✅ Skip rate: {skip_rate:.1f}%")

    return df


def calculate_discovery_flags(df):
    """
    Calculate new artist/track discovery flags.

    Flags:
    - new_artist_yn: First time listening to this artist
    - new_track_yn: First time listening to this track
    - new_recurring_artist_yn: 10th listen of this artist
    - new_recurring_track_yn: 5th listen of this track

    Args:
        df (pandas.DataFrame): Data with timestamp column

    Returns:
        pandas.DataFrame: Data with discovery flag columns
    """
    print("🔍 Calculating discovery flags...")

    # Sort chronologically for cumulative counting
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # Calculate flags
    df['new_artist_yn'] = df.groupby('artist_name').cumcount() == 0
    df['new_recurring_artist_yn'] = df.groupby('artist_name').cumcount() == 10
    df['new_track_yn'] = df.groupby('track_name').cumcount() == 0
    df['new_recurring_track_yn'] = df.groupby('track_name').cumcount() == 5

    # Convert boolean to integer
    df['new_artist_yn'] = df['new_artist_yn'].astype(int)
    df['new_recurring_artist_yn'] = df['new_recurring_artist_yn'].astype(int)
    df['new_track_yn'] = df['new_track_yn'].astype(int)
    df['new_recurring_track_yn'] = df['new_recurring_track_yn'].astype(int)

    # Sort back to descending timestamp
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    new_artists = df['new_artist_yn'].sum()
    new_tracks = df['new_track_yn'].sum()
    print(f"✅ Discovered {new_artists:,} new artists and {new_tracks:,} new tracks")

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
        'completion', 'skip_next_track',

        # Discovery flags
        'new_artist_yn', 'new_track_yn', 'new_recurring_artist_yn', 'new_recurring_track_yn',

        # Source tracking
        'source'
    ]

    # Keep only columns that exist in the dataframe
    existing_output_columns = [col for col in output_columns if col in df.columns]

    # Log how many columns are being filtered out
    removed_count = len(df.columns) - len(existing_output_columns)
    if removed_count > 0:
        print(f"📊 Filtered output: keeping {len(existing_output_columns)} columns, removing {removed_count} work-file-only columns")

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

    # Convert track_duration to float (replace 'No API result' with 0)
    if 'track_duration' in df.columns:
        df['track_duration'] = df['track_duration'].replace('No API result', '0').astype(float)

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
    print("\n🌐 Generating website files for Music page...")

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
        print(f"✅ Added toggle_id column ({len(df_web):,} toggles)")

        # Add listening_seconds - calculated from completion and track_duration
        # listening_seconds = (completion * track_duration_ms) / 1000
        df_web['listening_seconds'] = (df_web['completion'] * df_web['track_duration']) / 1000
        df_web['listening_seconds'] = df_web['listening_seconds'].fillna(0).astype(int)
        print(f"✅ Added listening_seconds column")

        # Combine genres from genre_1 through genre_14 into single 'genres' column
        # Use comma separator (not pipe, which is the CSV delimiter)
        genre_cols = [f'genre_{i}' for i in range(1, 15)]
        df_web['genres'] = df_web[genre_cols].apply(
            lambda row: ', '.join([str(g) for g in row if pd.notna(g) and str(g) != '' and str(g) != 'nan']),
            axis=1
        )
        print(f"✅ Combined genres from genre_1-14 into single column (comma-separated)")

        # Add simplified_genre column using genre_1
        df_web['simplified_genre'] = df_web['genre_1'].apply(get_simplified_genre)
        print(f"✅ Created simplified_genre from genre_1 using genre mapping")

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
            'skip_next_track',
            'listening_seconds',
            'new_artist_yn',
            'new_track_yn'
        ]

        # Add album_artwork_url if it exists in the data
        if 'album_artwork_url' in df_web.columns:
            website_columns.append('album_artwork_url')
            print(f"✅ Including album_artwork_url column")

        # Add artist_artwork_url if it exists in the data
        if 'artist_artwork_url' in df_web.columns:
            website_columns.append('artist_artwork_url')
            print(f"✅ Including artist_artwork_url column")

        # Filter to only website columns
        df_web = df_web[website_columns]
        print(f"✅ Filtered to {len(website_columns)} website columns")

        # Save website file
        website_path = f'{website_dir}/music_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Website file: {len(df_web):,} records → {website_path}")

        return True

    except Exception as e:
        print(f"❌ Error generating website files: {e}")
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
    print("\n" + "="*70)
    print("🎵 MUSIC TOPIC COORDINATOR PROCESSING")
    print("="*70)

    try:
        # Step 1: Load source data
        df = load_source_data()

        if df.empty:
            print("❌ No source data available")
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

        print(f"\n💾 Saving topic processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed music data!")
        print(f"📊 Output: {len(df):,} tracks")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"🎤 Unique artists: {df['artist_name'].nunique():,}")
        print(f"🎵 Unique tracks: {df['song_key'].nunique():,}")

        # Show sample
        print(f"\n📋 Sample records:")
        sample_df = df.head(3)[['timestamp', 'artist_name', 'track_name', 'completion', 'skip_next_track']]
        for _, row in sample_df.iterrows():
            completion_pct = row['completion'] * 100
            skip_status = "⏭️ " if row['skip_next_track'] else "✓"
            print(f"  {skip_status} {str(row['timestamp'])[:16]} | {row['artist_name']} - {row['track_name']} ({completion_pct:.0f}%)")

        # Step 9: Generate website files
        website_success = generate_music_website_files(df)

        if not website_success:
            print("⚠️  Warning: Website file generation had issues")
            return False

        return True

    except Exception as e:
        print(f"❌ Error in music topic processing: {e}")
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
    print("\n⬆️  Uploading music results to Google Drive...")

    files_to_upload = ['files/website_files/music/music_page_data.csv']

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} file(s)...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Music results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_music_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Music TOPIC coordinator pipeline.

    Options:
    1. Process source data, enrich with Spotify, generate website files, and upload to Drive
    2. Process existing topic data and upload to Drive
    3. Upload existing processed files to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*70)
    print("🎵 MUSIC TOPIC COORDINATOR PIPELINE")
    print("="*70)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data and uploading...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Full processing from source data...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Process source data, enrich with Spotify, and upload to Drive")
        print("2. Process existing topic data and upload to Drive")
        print("3. Upload existing processed files to Drive")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Option 1: Full processing from source data...")

        # Process data
        process_success = create_music_file()

        if not process_success:
            print("❌ Processing failed, skipping upload")
            return False

        # Test drive connection before upload
        if not verify_drive_connection():
            print("⚠️  Warning: Google Drive connection issues detected")
            proceed = input("Continue with upload anyway? (Y/N): ").upper() == 'Y'
            if not proceed:
                print("✅ Processing completed successfully (upload skipped)")
                return True

        # Upload results
        upload_success = upload_music_results()
        success = upload_success

        if success:
            record_successful_run('music_topic', 'coordination')

    elif choice == "2":
        print("\n⚙️  Option 2: Process existing topic data and upload...")

        # Check if topic processed file exists
        topic_file = 'files/topic_processed_files/music/music_processed.csv'

        if os.path.exists(topic_file):
            # Regenerate website files from existing topic data
            df = pd.read_csv(topic_file, sep='|', encoding='utf-8', low_memory=False)
            website_success = generate_music_website_files(df)

            if not website_success:
                print("❌ Website file generation failed")
                return False
        else:
            print(f"❌ Topic processed file not found: {topic_file}")
            print("   Run option 1 first to process source data")
            return False

        # Test drive connection before upload
        if not verify_drive_connection():
            print("⚠️  Warning: Google Drive connection issues detected")
            proceed = input("Continue with upload anyway? (Y/N): ").upper() == 'Y'
            if not proceed:
                print("✅ Processing completed successfully (upload skipped)")
                return True

        # Upload results
        upload_success = upload_music_results()
        success = upload_success

        if success:
            record_successful_run('music_topic', 'coordination')

    elif choice == "3":
        print("\n📤 Option 3: Upload existing processed files...")

        # Test drive connection before upload
        if not verify_drive_connection():
            print("⚠️  Warning: Google Drive connection issues detected")
            proceed = input("Continue with upload anyway? (Y/N): ").upper() == 'Y'
            if not proceed:
                print("✅ Upload skipped")
                return True

        # Upload results
        upload_success = upload_music_results()
        success = upload_success

        if success:
            record_successful_run('music_topic', 'coordination')

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*70)
    if success:
        print("✅ Music topic coordinator completed successfully!")
        print("📊 Topic output: files/topic_processed_files/music/music_processed.csv")
        print("🌐 Website output: files/website_files/music/music_page_data.csv")
    else:
        print("❌ Music topic coordinator failed")
    print("="*70)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("🎵 Music Topic Coordinator")
    print("This tool processes music data from multiple sources and enriches with Spotify metadata.")
    print("Note: Run source processors (Last.fm, Spotify) first to generate source data\n")

    # Run the pipeline
    full_music_pipeline(auto_full=False)
