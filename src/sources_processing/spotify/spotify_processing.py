"""
Spotify Legacy Source Processor

This module processes legacy Spotify JSON exports (2013-2023).
Follows the source processor pattern: 1 option (process only - no API download).
Does NOT upload to Drive (handled by Music topic coordinator).

Input: files/exports/spotify_exports/*.json
Output: files/source_processed_files/spotify/spotify_processed.csv
"""

import os
import pandas as pd
import json

from src.utils.utils_functions import time_difference_correction, enforce_snake_case


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def create_spotify_file():
    """
    Process legacy Spotify JSON exports and create the source processed file.

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("🎧 SPOTIFY LEGACY DATA PROCESSING")
    print("="*70)

    folder_path = 'files/exports/spotify_exports/'
    output_path = 'files/source_processed_files/spotify/spotify_processed.csv'

    try:
        # Check if export folder exists
        if not os.path.exists(folder_path):
            print(f"❌ Spotify exports folder not found: {folder_path}")
            return False

        # Get all JSON files
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

        if not json_files:
            print(f"❌ No JSON files found in {folder_path}")
            return False

        print(f"📊 Found {len(json_files)} Spotify export file(s)")

        # Parse all JSON files
        artist_names = []
        album_names = []
        track_names = []
        timestamps = []

        for filename in json_files:
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Processing {filename}...")

            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                tracks = len(data)
                print(f"   • {tracks} tracks found")

                for t in range(tracks):
                    artist_names.append(data[t]['master_metadata_album_artist_name'])
                    album_names.append(data[t]['master_metadata_album_album_name'])
                    track_names.append(data[t]['master_metadata_track_name'])
                    timestamps.append(data[t]['ts'])

        # Create DataFrame
        data_dict = {
            'artist_name': artist_names,
            'album_name': album_names,
            'track_name': track_names,
            'timestamp': timestamps
        }

        df_spot = pd.DataFrame(data_dict)

        print(f"✅ Loaded {len(df_spot):,} tracks from Spotify exports")

        # Process timestamps with timezone correction
        print("🔄 Processing timestamps (GMT → local time)...")
        df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'], utc=True)
        df_spot = time_difference_correction(df_spot, 'timestamp', 'GMT')

        # Create song_key for matching with Last.fm
        df_spot['song_key'] = df_spot['track_name'] + ' /: ' + df_spot['artist_name']

        # Add source column
        df_spot['source'] = 'spotify_legacy'

        # Sort by timestamp (newest first)
        df_spot = df_spot.sort_values('timestamp', ascending=False)

        # Remove duplicates
        df_spot = df_spot.drop_duplicates(subset=['song_key', 'timestamp'], keep='first')

        print(f"✅ After deduplication: {len(df_spot):,} tracks")

        # Enforce snake_case
        df_spot = enforce_snake_case(df_spot, "spotify_processed")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV with pipe separator and UTF-8 encoding
        print(f"💾 Saving processed data to {output_path}...")
        df_spot.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed Spotify legacy data!")
        print(f"📊 Output: {len(df_spot):,} unique tracks")
        print(f"📅 Date range: {df_spot['timestamp'].min()} to {df_spot['timestamp'].max()}")
        print(f"🎤 Unique artists: {df_spot['artist_name'].nunique()}")
        print(f"🎵 Unique tracks: {df_spot['song_key'].nunique()}")

        # Show sample
        print(f"\n📋 Sample records:")
        sample_df = df_spot.head(3)[['timestamp', 'artist_name', 'track_name', 'album_name']]
        for _, row in sample_df.iterrows():
            print(f"  • {row['timestamp']} | {row['artist_name']} - {row['track_name']}")

        return True

    except Exception as e:
        print(f"❌ Error processing Spotify legacy data: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_spotify_pipeline():
    """
    Complete Spotify SOURCE processor pipeline.

    Spotify legacy data is already downloaded (JSON files), so there is only one option:
    1. Process existing data

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*70)
    print("🎧 SPOTIFY LEGACY SOURCE PROCESSOR PIPELINE")
    print("="*70)

    print("\n⚙️  Processing Spotify legacy export files...")

    success = create_spotify_file()

    # Final status
    print("\n" + "="*70)
    if success:
        print("✅ Spotify legacy source processor completed successfully!")
        print("📊 Output: files/source_processed_files/spotify/spotify_processed.csv")
        print("📝 Next: Run Music topic coordinator to merge with Last.fm and upload")
    else:
        print("❌ Spotify legacy processing failed")
    print("="*70)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("🎧 Spotify Legacy Source Processing Tool")
    print("This tool processes legacy Spotify JSON exports (2013-2023).")
    print("Note: Upload is handled by the Music topic coordinator\n")

    # Run the pipeline
    full_spotify_pipeline()
