"""
Last.fm Source Processor

This module handles Last.fm API data fetching and basic processing.
Follows the source processor pattern: 2 options (download+process, process only).
Does NOT upload to Drive (handled by Music topic coordinator).

Output: files/source_processed_files/lastfm/lastfm_processed.csv
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from src.utils.utils_functions import enforce_snake_case, time_difference_correction

load_dotenv()


class LastFmAPIProcessor:
    """Handles Last.fm API data fetching with incremental updates."""

    def __init__(self):
        """Initialize the processor with API credentials from environment variables."""
        # Load LastFM credentials from environment variables
        self.api_key = os.environ.get('LAST_FM_API_KEY')
        self.api_secret = os.environ.get('LAST_FM_API_SECRET')
        self.username = os.environ.get('LAST_FM_API_USERNAME')

        if not self.api_key:
            raise ValueError("LAST_FM_API_KEY environment variable is required")
        if not self.username:
            raise ValueError("LAST_FM_API_USERNAME environment variable is required")

        self.base_url = "http://ws.audioscrobbler.com/2.0/"

        # File paths
        self.export_file_path = "files/exports/lastfm_exports/lastfm_export.csv"
        self.processed_file_path = "files/source_processed_files/lastfm/lastfm_processed.csv"

    def get_latest_timestamp_from_file(self):
        """
        Read the existing raw export file and return the latest timestamp.
        This is used to determine the checkpoint for resuming downloads.

        Returns:
            datetime: Latest timestamp from the file, or None if file doesn't exist
        """
        try:
            if not os.path.exists(self.export_file_path):
                print(f"File {self.export_file_path} not found. Will fetch all available data.")
                return None

            # Try different encodings to read the file (UTF-8 first as it's the new standard)
            encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be']
            df = None

            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(self.export_file_path, sep='|', encoding=encoding, low_memory=False)
                    print(f"Successfully read export file with {encoding} encoding")
                    break
                except UnicodeError:
                    continue
                except Exception as e:
                    if encoding == encodings_to_try[-1]:  # Last encoding attempt
                        raise e
                    continue

            if df is None:
                raise Exception("Could not read file with any supported encoding")

            if df.empty:
                print("Existing export file is empty. Will fetch all available data.")
                return None

            # Convert timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Get the latest timestamp
            latest_timestamp = df['timestamp'].max()
            print(f"Latest timestamp in export file: {latest_timestamp}")

            return latest_timestamp

        except Exception as e:
            print(f"Error reading existing export file: {e}")
            print("Will fetch all available data.")
            return None

    def fetch_tracks_since_timestamp(self, since_timestamp=None):
        """
        Fetch tracks from Last.fm API since the given timestamp.
        Saves each page incrementally to lastfm_export.csv for crash recovery.

        Args:
            since_timestamp (datetime): Fetch tracks after this timestamp

        Returns:
            bool: True if successful, False otherwise
        """
        # Ensure export directory exists
        os.makedirs(os.path.dirname(self.export_file_path), exist_ok=True)

        page = 1
        total_pages = None
        printed_currently_playing = set()  # Track already-printed "currently playing" messages
        total_saved = 0

        print(f"Fetching tracks from Last.fm API with incremental checkpointing...")
        if since_timestamp:
            print(f"Fetching tracks since: {since_timestamp}")
        else:
            print("Fetching all available tracks (no existing data found)")

        # Check if export file exists to determine if we should write header
        file_exists = os.path.exists(self.export_file_path)

        while True:
            retry_count = 0
            max_retries = 3
            success = False

            while retry_count < max_retries and not success:
                try:
                    params = {
                        'method': 'user.getrecenttracks',
                        'user': self.username,
                        'api_key': self.api_key,
                        'format': 'json',
                        'limit': 200,  # Maximum allowed by API
                        'page': page
                    }

                    # Add timestamp filter if provided
                    if since_timestamp:
                        # Convert to Unix timestamp
                        unix_timestamp = int(since_timestamp.timestamp())
                        params['from'] = unix_timestamp

                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()

                    data = response.json()

                    # Check for API errors
                    if 'error' in data:
                        print(f"API Error: {data['message']}")
                        print(f"✅ Saved {total_saved} tracks before error")
                        return True  # Partial success - data is checkpointed

                    # Check if we have track data
                    if 'recenttracks' not in data or 'track' not in data['recenttracks']:
                        print("No track data found in API response")
                        print(f"✅ Saved {total_saved} tracks before response issue")
                        return True  # Partial success - data is checkpointed

                    tracks = data['recenttracks']['track']
                    attr = data['recenttracks']['@attr']

                    # Get total pages from first response
                    if total_pages is None:
                        total_pages = int(attr['totalPages'])
                        total_tracks = int(attr['total'])
                        print(f"Total tracks to fetch: {total_tracks} across {total_pages} pages")

                    # If no tracks on this page, we're done
                    if not tracks:
                        print(f"✅ Complete: Saved {total_saved} tracks")
                        return True  # Success

                    # Handle case where only one track is returned (not a list)
                    if isinstance(tracks, dict):
                        tracks = [tracks]

                    # Filter out currently playing tracks (they have no timestamp)
                    valid_tracks = []
                    for track in tracks:
                        if 'date' in track and 'uts' in track['date']:
                            valid_tracks.append(track)
                        else:
                            # Only print message once per track to avoid spam
                            track_name = track.get('name', 'Unknown')
                            if track_name not in printed_currently_playing:
                                print(f"Skipping currently playing track: {track_name}")
                                printed_currently_playing.add(track_name)

                    # Save this page immediately to export file (checkpoint)
                    if valid_tracks:
                        page_df = self.parse_api_tracks_to_dataframe(valid_tracks)

                        # Append to CSV (write header only if file doesn't exist or is first page)
                        mode = 'w' if not file_exists else 'a'
                        header = not file_exists
                        page_df.to_csv(self.export_file_path, sep='|', encoding='utf-8',
                                      index=False, mode=mode, header=header)

                        # After first write, file exists
                        file_exists = True
                        total_saved += len(valid_tracks)

                    print(f"Fetched page {page}/{total_pages} ({len(valid_tracks)} tracks) ✓ Saved checkpoint")

                    success = True  # Mark as successful

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"❌ Network error on page {page} after {max_retries} retries: {e}")
                        if total_pages:
                            estimated_missing = (total_pages - page + 1) * 200
                            print(f"⚠️  WARNING: Incomplete fetch - successfully saved {total_saved} tracks")
                            print(f"    Failed at page {page} of {total_pages}")
                            print(f"    Estimated missing tracks: ~{estimated_missing}")
                            print(f"    ✓ Checkpointed data saved - next run will resume from {total_saved} tracks")
                        return True  # Partial success - checkpointed data is recoverable
                    else:
                        wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                        print(f"⚠️  Error on page {page}, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                        time.sleep(wait_time)

                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"❌ Error processing page {page} after {max_retries} retries: {e}")
                        if total_pages:
                            print(f"⚠️  WARNING: Saved {total_saved} tracks before failure")
                            print(f"    ✓ Checkpointed data saved - next run will resume")
                        return True  # Partial success - checkpointed data is recoverable
                    else:
                        wait_time = 2 ** retry_count
                        print(f"⚠️  Error on page {page}, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                        time.sleep(wait_time)

            # If we didn't succeed after retries, move on
            if not success:
                break

            # Check if we've reached the end
            if page >= total_pages:
                break

            page += 1

            # Be nice to the API - small delay between requests
            time.sleep(0.2)

        print(f"✅ Successfully fetched and saved {total_saved} new tracks to {self.export_file_path}")

        # Sort the export file by timestamp (newest first) for better usability
        print("🔄 Sorting export file by timestamp (newest first)...")
        try:
            df_export = pd.read_csv(self.export_file_path, sep='|', encoding='utf-8', low_memory=False)
            df_export['timestamp'] = pd.to_datetime(df_export['timestamp'])
            df_export = df_export.sort_values('timestamp', ascending=False)
            df_export.to_csv(self.export_file_path, sep='|', encoding='utf-8', index=False)
            print(f"✅ Export file sorted - newest tracks are now at the top")
        except Exception as e:
            print(f"⚠️  Warning: Could not sort export file: {e}")
            print(f"   File is still usable, just not sorted")

        return True

    def parse_api_tracks_to_dataframe(self, tracks):
        """
        Parse raw API track data into a DataFrame with the basic structure.

        Args:
            tracks (list): List of track dictionaries from API

        Returns:
            pandas.DataFrame: Parsed track data in basic format
        """
        parsed_tracks = []

        for track in tracks:
            try:
                # Extract basic track information
                track_data = {
                    'artist_name': track['artist']['#text'],
                    'album_name': track['album']['#text'] if track['album']['#text'] else 'Unknown Album',
                    'track_name': track['name'],
                    'timestamp': datetime.fromtimestamp(int(track['date']['uts']))
                }

                parsed_tracks.append(track_data)

            except Exception as e:
                print(f"Error parsing track {track.get('name', 'Unknown')}: {e}")
                continue

        df = pd.DataFrame(parsed_tracks)
        return df


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_lastfm_data():
    """
    Fetches new data from Last.fm API.
    Returns True if successful, False otherwise.
    """
    print("🎵 Starting Last.fm API data fetch...")

    try:
        processor = LastFmAPIProcessor()

        # Get latest timestamp from existing export file
        latest_timestamp = processor.get_latest_timestamp_from_file()

        # Fetch tracks since that timestamp (or all if no existing data)
        success = processor.fetch_tracks_since_timestamp(latest_timestamp)

        if success:
            print("✅ Last.fm API fetch completed successfully!")
        else:
            print("❌ Last.fm API fetch failed")

        return success

    except Exception as e:
        print(f"❌ Error in Last.fm download: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def create_lastfm_file():
    """
    Process the Last.fm export file and create the source processed file.
    Basic processing only - enrichment happens in topic coordinator.

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("🎵 LAST.FM DATA PROCESSING")
    print("="*70)

    export_path = "files/exports/lastfm_exports/lastfm_export.csv"
    output_path = "files/source_processed_files/lastfm/lastfm_processed.csv"

    try:
        # Check if input file exists
        if not os.path.exists(export_path):
            print(f"❌ Export file not found: {export_path}")
            return False

        print(f"📊 Reading Last.fm export data...")

        # Try different encodings
        encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be']
        df = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(export_path, sep='|', encoding=encoding, low_memory=False)
                print(f"✅ Read export file with {encoding} encoding")
                break
            except UnicodeError:
                continue
            except Exception as e:
                if encoding == encodings_to_try[-1]:
                    raise e
                continue

        if df is None:
            raise Exception("Could not read file with any supported encoding")

        print(f"✅ Loaded {len(df):,} tracks from export")

        # Basic processing
        print("🔄 Processing data...")

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Apply timezone correction (Last.fm returns UTC timestamps)
        print("🌍 Applying timezone correction (UTC → local time)...")
        df = time_difference_correction(df, 'timestamp', 'UTC')

        # Create song_key for deduplication (used by topic coordinator)
        df['song_key'] = df['track_name'] + ' /: ' + df['artist_name']

        # Add source column
        df['source'] = 'lastfm'

        # Sort by timestamp (newest first)
        df = df.sort_values('timestamp', ascending=False)

        # Remove duplicates (same song at same timestamp)
        df = df.drop_duplicates(subset=['song_key', 'timestamp'], keep='first')

        print(f"✅ After deduplication: {len(df):,} tracks")

        # Enforce snake_case
        df = enforce_snake_case(df, "lastfm_processed")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV with pipe separator and UTF-8 encoding
        print(f"💾 Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed Last.fm data!")
        print(f"📊 Output: {len(df):,} unique tracks")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"🎤 Unique artists: {df['artist_name'].nunique()}")
        print(f"🎵 Unique tracks: {df['song_key'].nunique()}")

        # Show sample
        print(f"\n📋 Sample records:")
        sample_df = df.head(3)[['timestamp', 'artist_name', 'track_name', 'album_name']]
        for _, row in sample_df.iterrows():
            print(f"  • {row['timestamp']} | {row['artist_name']} - {row['track_name']}")

        return True

    except Exception as e:
        print(f"❌ Error processing Last.fm data: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_lastfm_pipeline(auto_full=False):
    """
    Complete Last.fm SOURCE processor pipeline.

    Options:
    1. Download new data from API and process
    2. Process existing data

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*70)
    print("🎵 LAST.FM SOURCE PROCESSOR PIPELINE")
    print("="*70)

    if auto_full:
        print("🤖 Auto mode: Downloading and processing...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new data from API and process")
        print("2. Process existing data")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Option 1: Download and process...")

        # Step 1: Download from API
        download_success = download_lastfm_data()

        if not download_success:
            print("⚠️  Download had issues, but will attempt to process existing data...")

        # Step 2: Process
        success = create_lastfm_file()

    elif choice == "2":
        print("\n⚙️  Option 2: Process existing data...")
        success = create_lastfm_file()

    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    # Final status
    print("\n" + "="*70)
    if success:
        print("✅ Last.fm source processor completed successfully!")
        print("📊 Output: files/source_processed_files/lastfm/lastfm_processed.csv")
        print("📝 Next: Run Music topic coordinator for Spotify enrichment and upload")
    else:
        print("❌ Last.fm processing failed")
    print("="*70)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("🎵 Last.fm Source Processing Tool")
    print("This tool downloads and processes Last.fm listening history.")
    print("Note: Upload is handled by the Music topic coordinator\n")

    # Run the pipeline
    full_lastfm_pipeline(auto_full=False)
