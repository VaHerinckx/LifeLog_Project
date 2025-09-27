import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from pathlib import Path

from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run


def reverse_geocode_coordinates(lat: float, lon: float) -> Dict[str, str]:
    """
    Reverse geocode coordinates to get city and country information.
    Uses Nominatim (OpenStreetMap) API which is free and doesn't require API key.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dictionary with city and country information
    """
    try:
        # Use Nominatim API (free, no API key required)
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1,
            'zoom': 10  # City level zoom
        }

        headers = {
            'User-Agent': 'LifeLog-Location-Processor/1.0'
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})

            # Extract city (try multiple possible fields)
            city = (address.get('city') or
                   address.get('town') or
                   address.get('village') or
                   address.get('municipality') or
                   address.get('suburb') or
                   'Unknown City')

            # Extract country
            country = address.get('country', 'Unknown Country')

            return {
                'city': city,
                'country': country
            }
        else:
            print(f"⚠️  Geocoding failed for {lat}, {lon}: HTTP {response.status_code}")
            return {
                'city': 'Unknown City',
                'country': 'Unknown Country'
            }

    except Exception as e:
        print(f"⚠️  Error geocoding {lat}, {lon}: {e}")
        return {
            'city': 'Unknown City',
            'country': 'Unknown Country'
        }


def parse_coordinates(geo_string: str) -> Tuple[float, float]:
    """
    Parse coordinates from geo string format 'geo:lat,lon'

    Args:
        geo_string: String in format 'geo:50.837750,4.424380'

    Returns:
        Tuple of (latitude, longitude)
    """
    try:
        # Remove 'geo:' prefix and split by comma
        coords = geo_string.replace('geo:', '').split(',')
        lat = float(coords[0])
        lon = float(coords[1])
        return lat, lon
    except Exception as e:
        print(f"⚠️  Error parsing coordinates from '{geo_string}': {e}")
        return 0.0, 0.0


def extract_timezone_from_timestamp(timestamp_str: str) -> str:
    """
    Extract timezone offset from timestamp string.

    Args:
        timestamp_str: Timestamp like '2025-03-07T18:41:18.869+01:00'

    Returns:
        Timezone string like 'UTC+01:00'
    """
    try:
        if '+' in timestamp_str:
            offset = timestamp_str.split('+')[-1]
            return f"UTC+{offset}"
        elif timestamp_str.count('-') > 2:  # Has negative offset
            parts = timestamp_str.split('-')
            offset = parts[-1]
            return f"UTC-{offset}"
        else:
            return "UTC+00:00"  # Default to UTC
    except Exception as e:
        print(f"⚠️  Error extracting timezone from '{timestamp_str}': {e}")
        return "UTC+00:00"


def download_google_data():
    """
    Provides instructions for downloading Google Maps Timeline data from iPhone.
    Returns True if user confirms download, False otherwise.
    """
    print("📱 Starting Google Maps Timeline data download...")

    print("📝 Instructions for iPhone export:")
    print("   1. Open Google Maps app on your iPhone")
    print("   2. Tap your profile picture (top right)")
    print("   3. Go to 'Your timeline'")
    print("   4. Tap the settings/menu button")
    print("   5. Select 'Export timeline data'")
    print("   6. Choose your date range")
    print("   7. Download the JSON file")
    print("   8. AirDrop the 'location-history.json' file to your Mac Downloads folder")

    response = prompt_user_download_status("Google Maps Timeline")
    return response


def move_google_files():
    """
    Moves the downloaded Google Maps file from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving Google Maps files...")

    success = clean_rename_move_file(
        export_folder="files/exports/google_exports",
        download_folder="/Users/valen/Downloads",
        file_name="location-history.json",
        new_file_name="location_history.json"
    )

    if success:
        print("✅ Successfully moved Google Maps export to exports folder")
    else:
        print("❌ Failed to move Google Maps files")

    return success


def create_hourly_timezone_records(location_data: List[Dict]) -> List[Dict]:
    """
    Create hourly timezone and location records from Google Maps timeline data.

    Args:
        location_data: List of timeline objects from Google Maps JSON

    Returns:
        List of hourly records with timezone and location information
    """
    print("🌍 Creating hourly timezone and location records...")

    # Extract all timeline points with locations
    timeline_points = []

    for item in location_data:
        start_time_str = item.get('startTime')
        end_time_str = item.get('endTime')

        if not start_time_str or not end_time_str:
            continue

        try:
            # Parse timestamps
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))

            # Extract timezone
            timezone = extract_timezone_from_timestamp(start_time_str)

            # Extract location information
            coordinates = None
            is_home = False

            # Check for place visits (more reliable for location)
            if 'visit' in item:
                visit = item['visit']
                top_candidate = visit.get('topCandidate', {})
                place_location = top_candidate.get('placeLocation', '')
                semantic_type = top_candidate.get('semanticType', 'Unknown')

                if place_location:
                    coordinates = place_location
                    is_home = (semantic_type == 'Home')

            # Check for activity segments if no visit data
            elif 'activity' in item and not coordinates:
                activity = item['activity']
                # Use start location for activities
                start_location = activity.get('start', '')
                if start_location:
                    coordinates = start_location
                    is_home = False  # Activities are not home by definition

            # Add timeline point
            if coordinates:
                timeline_points.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'timezone': timezone,
                    'coordinates': coordinates,
                    'is_home': is_home
                })

        except Exception as e:
            print(f"⚠️  Error processing timeline item: {e}")
            continue

    if not timeline_points:
        print("❌ No valid timeline points found")
        return []

    # Sort by start time
    timeline_points.sort(key=lambda x: x['start_time'])

    print(f"📊 Found {len(timeline_points)} valid timeline points")

    # Find the overall date range
    min_time = timeline_points[0]['start_time']
    max_time = timeline_points[-1]['end_time']

    print(f"📅 Date range: {min_time.date()} to {max_time.date()}")

    # Create hourly records
    hourly_records = []
    current_hour = min_time.replace(minute=0, second=0, microsecond=0)

    # Cache for geocoding to avoid repeated API calls
    geocoding_cache = {}

    while current_hour <= max_time:
        # Find which timeline point covers this hour
        covering_point = None

        for point in timeline_points:
            if point['start_time'] <= current_hour <= point['end_time']:
                covering_point = point
                break

        if covering_point:
            # We have data for this hour
            coordinates = covering_point['coordinates']
            timezone = covering_point['timezone']
            is_home = covering_point['is_home']

            # Geocode coordinates if not cached
            if coordinates not in geocoding_cache:
                lat, lon = parse_coordinates(coordinates)
                if lat != 0.0 or lon != 0.0:
                    # Add small delay to be respectful to geocoding API
                    time.sleep(0.1)
                    location_info = reverse_geocode_coordinates(lat, lon)
                    geocoding_cache[coordinates] = location_info
                else:
                    geocoding_cache[coordinates] = {
                        'city': 'Unknown City',
                        'country': 'Unknown Country'
                    }

            location_info = geocoding_cache[coordinates]

            record = {
                'timestamp': current_hour.isoformat(),
                'timezone': timezone,
                'city': location_info['city'],
                'country': location_info['country'],
                'is_home': is_home,
                'coordinates': coordinates
            }

        else:
            # No data for this hour - interpolate from nearest points
            # Find the closest points before and after
            before_point = None
            after_point = None

            for point in timeline_points:
                if point['end_time'] < current_hour:
                    before_point = point
                elif point['start_time'] > current_hour and after_point is None:
                    after_point = point
                    break

            # Use the closer point, preferring the after point for timezone changes
            chosen_point = after_point if after_point else before_point

            if chosen_point:
                coordinates = chosen_point['coordinates']
                timezone = chosen_point['timezone']
                is_home = chosen_point['is_home']

                # Use cached geocoding
                if coordinates not in geocoding_cache:
                    lat, lon = parse_coordinates(coordinates)
                    if lat != 0.0 or lon != 0.0:
                        time.sleep(0.1)
                        location_info = reverse_geocode_coordinates(lat, lon)
                        geocoding_cache[coordinates] = location_info
                    else:
                        geocoding_cache[coordinates] = {
                            'city': 'Unknown City',
                            'country': 'Unknown Country'
                        }

                location_info = geocoding_cache[coordinates]

                record = {
                    'timestamp': current_hour.isoformat(),
                    'timezone': timezone,
                    'city': location_info['city'],
                    'country': location_info['country'],
                    'is_home': is_home,
                    'coordinates': coordinates
                }
            else:
                # Fallback record
                record = {
                    'timestamp': current_hour.isoformat(),
                    'timezone': 'UTC+00:00',
                    'city': 'Unknown City',
                    'country': 'Unknown Country',
                    'is_home': False,
                    'coordinates': 'geo:0.0,0.0'
                }

        hourly_records.append(record)
        current_hour += timedelta(hours=1)

    print(f"✅ Created {len(hourly_records)} hourly records")
    return hourly_records


def create_google_file():
    """
    Main processing function that creates the processed Google location file.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Google Maps location data...")

    # Define file paths
    input_path = "files/exports/google_exports/location_history.json"
    output_path = 'files/processed_files/location/google_processed.csv'

    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"❌ Google Maps file not found: {input_path}")
            return False

        # Load the JSON data
        print(f"📱 Reading Google Maps timeline data...")
        with open(input_path, 'r', encoding='utf-8') as f:
            location_data = json.load(f)

        print(f"✅ Loaded {len(location_data)} timeline items")

        # Create hourly records
        hourly_records = create_hourly_timezone_records(location_data)

        if not hourly_records:
            print("❌ No hourly records created")
            return False

        # Convert to DataFrame
        df = pd.DataFrame(hourly_records)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV with pipe separator
        print(f"💾 Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False)

        print(f"✅ Successfully processed Google Maps data!")
        print(f"📊 Created {len(df)} hourly location records")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"🌍 Countries visited: {df['country'].nunique()}")
        print(f"🏙️  Cities visited: {df['city'].nunique()}")
        print(f"🏠 Hours at home: {df['is_home'].sum()}")

        # Show sample of the data
        print(f"\n📋 Sample records:")
        sample_df = df.head(5)[['timestamp', 'timezone', 'city', 'country', 'is_home']]
        for _, row in sample_df.iterrows():
            home_status = "🏠 Home" if row['is_home'] else "📍 Away"
            print(f"  • {row['timestamp'][:16]} | {row['timezone']} | {row['city']}, {row['country']} | {home_status}")

        return True

    except Exception as e:
        print(f"❌ Error processing Google Maps data: {e}")
        import traceback
        traceback.print_exc()
        return False

def merge_timezone_files():
    """
    Optional function to merge manual and Google timezone files.
    Returns True if successful, False otherwise.
    """
    print("🔗 Merging manual and Google timezone files...")

    manual_path = 'files/processed_files/location/manual_timezone_processed.csv'
    google_path = 'files/processed_files/location/google_processed.csv'
    merged_path = 'files/processed_files/location/combined_timezone_processed.csv'

    try:
        files_to_merge = []

        # Check which files exist
        if os.path.exists(manual_path):
            manual_df = pd.read_csv(manual_path, sep='|')
            manual_df['source'] = 'manual'
            files_to_merge.append(manual_df)
            print(f"✅ Loaded manual file: {len(manual_df):,} records")

        if os.path.exists(google_path):
            google_df = pd.read_csv(google_path, sep='|')
            google_df['source'] = 'google'
            files_to_merge.append(google_df)
            print(f"✅ Loaded Google file: {len(google_df):,} records")

        if not files_to_merge:
            print("❌ No timezone files found to merge")
            return False

        # Merge dataframes
        combined_df = pd.concat(files_to_merge, ignore_index=True)

        # Convert timestamp to datetime, handling timezone issues
        print("🕐 Processing timestamps...")

        # Convert all timestamps to naive datetime (remove timezone info)
        def normalize_timestamp(timestamp_str):
            try:
                # Parse the timestamp
                dt = pd.to_datetime(timestamp_str)

                # If it's timezone-aware, convert to naive (remove timezone)
                if dt.tz is not None:
                    dt = dt.tz_localize(None)

                return dt
            except Exception as e:
                print(f"⚠️  Error parsing timestamp {timestamp_str}: {e}")
                return pd.NaT

        combined_df['timestamp_parsed'] = combined_df['timestamp'].apply(normalize_timestamp)

        # Remove any rows with invalid timestamps
        valid_rows = combined_df['timestamp_parsed'].notna()
        combined_df = combined_df[valid_rows]

        if len(combined_df) == 0:
            print("❌ No valid timestamps found after processing")
            return False

        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp_parsed')

        # Remove duplicates (prefer Google data over manual if same timestamp)
        # Group by hour to handle potential overlaps
        combined_df['hour_key'] = combined_df['timestamp_parsed'].dt.strftime('%Y-%m-%d-%H')

        # For each hour, prefer google data if available
        def choose_best_record(group):
            if len(group) == 1:
                return group.iloc[0]

            # Prefer google data over manual
            google_records = group[group['source'] == 'google']
            if len(google_records) > 0:
                return google_records.iloc[0]  # Take first google record
            else:
                return group.iloc[0]  # Take first manual record

        # Apply the selection logic
        print("🔄 Resolving duplicate timestamps...")
        deduplicated_df = combined_df.groupby('hour_key').apply(choose_best_record).reset_index(drop=True)

        # Clean up temporary columns
        deduplicated_df = deduplicated_df.drop(['timestamp_parsed', 'hour_key', 'source'], axis=1)

        # Ensure timestamps are in string format
        if 'timestamp' not in deduplicated_df.columns:
            print("❌ Timestamp column missing after processing")
            return False

        # Sort final dataframe by timestamp string
        deduplicated_df = deduplicated_df.sort_values('timestamp')

        # Save merged file
        print(f"💾 Saving merged file to {merged_path}...")
        deduplicated_df.to_csv(merged_path, sep='|', index=False)

        print(f"✅ Successfully merged timezone files!")
        print(f"📊 Combined records: {len(deduplicated_df):,}")
        print(f"📅 Date range: {deduplicated_df['timestamp'].min()[:10]} to {deduplicated_df['timestamp'].max()[:10]}")
        print(f"🌍 Countries: {', '.join(deduplicated_df['country'].unique()[:5])}{'...' if len(deduplicated_df['country'].unique()) > 5 else ''}")
        print(f"💾 Saved to: {merged_path}")

        # Show sample of merged data
        print(f"\n📋 Sample merged records:")
        sample_df = deduplicated_df.head(5)[['timestamp', 'timezone', 'city', 'country', 'is_home']]
        for _, row in sample_df.iterrows():
            home_status = "🏠" if row['is_home'] else "📍"
            print(f"  • {row['timestamp'][:16]} | {row['timezone']} | {row['city']}, {row['country']} {home_status}")

        return True

    except Exception as e:
        print(f"❌ Error merging timezone files: {e}")
        import traceback
        traceback.print_exc()
        return False



#def upload_google_results():
#    """
#    Uploads the processed Google location files to Google Drive.
#    Returns True if successful, False otherwise.
#    """
#    print("☁️  Uploading Google location results to Google Drive...")
#
#    files_to_upload = [
#        'files/processed_files/location/google_processed.csv'
#    ]
#
#    # Filter to only existing files
#    existing_files = [f for f in files_to_upload if os.path.exists(f)]
#
#    if not existing_files:
#        print("❌ No files found to upload")
#        return False
#
#    print(f"📤 Uploading {len(existing_files)} files...")
#    success = upload_multiple_files(existing_files)
#
#    if success:
#        print("✅ Google location results uploaded successfully!")
#    else:
#        print("❌ Some files failed to upload")
#
#    return success



def process_google_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_google_pipeline(auto_full=True)
    else:
        return create_google_file()


def full_google_pipeline(auto_full=False):
    """
    Complete Google Maps pipeline with multiple options.

    Options:
    1. Full pipeline (download → move → process → upload)
    2. Download data only (instructions + move files)
    3. Process existing file only (create processed file)
    4. Process existing file and upload to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🌍 GOOGLE MAPS LOCATION PIPELINE")
    print("="*60)

    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → move → process)")
        print("2. Download data only (instructions + move files)")
        print("3. Process existing file only")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full Google Maps pipeline...")

        # Step 1: Download
        download_success = download_google_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_google_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_google_files()

        # Step 3: Process (fallback to option 3 if no new files)
        if move_success:
            process_success = create_google_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_google_file()

    elif choice == "2":
        print("\n📥 Download Google Maps data only...")
        download_success = download_google_data()
        if download_success:
            success = move_google_files()
        else:
            success = False

    elif choice == "3":
        print("\n⚙️  Processing existing Google Maps file only...")
        success = create_google_file() & merge_timezone_files()

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Google Maps pipeline completed successfully!")
        # Record successful run
        record_successful_run('location_google', 'active')
    else:
        print("❌ Google Maps pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("🌍 Google Maps Location Processing Tool")
    print("This tool helps you process Google Maps Timeline data into hourly timezone and location records.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_google_pipeline(auto_full=False)
