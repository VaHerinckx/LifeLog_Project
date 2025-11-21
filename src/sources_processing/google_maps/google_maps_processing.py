"""
Google Maps Timeline Source Processor

This module processes Google Maps Timeline exports to extract hourly location data.
Follows the source processor pattern: 2 options (download+process, process only).
Does NOT upload to Drive (handled by Location topic coordinator).

Output: files/source_processed_files/google_maps/google_maps_processed.csv
"""

import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List

from src.utils.file_operations import clean_rename_move_file
from src.utils.web_operations import prompt_user_download_status
from src.utils.utils_functions import enforce_snake_case
from src.utils.geocoding_utils import (
    load_geocoding_cache,
    save_geocoding_cache,
    reverse_geocode_coordinates,
    parse_coordinates,
    extract_timezone_from_timestamp
)


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

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


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

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
            # Parse timestamps (strip timezone info for consistent format)
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00')).replace(tzinfo=None)

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

    # OPTIMIZATION 1: Load existing geocoding cache and geocode only new coordinates
    print("🌍 Loading geocoding cache and processing coordinates...")
    geocoding_cache = load_geocoding_cache()
    unique_coordinates = set(point['coordinates'] for point in timeline_points)

    # Find coordinates that need geocoding (not in cache)
    coordinates_to_geocode = [coord for coord in unique_coordinates if coord not in geocoding_cache]

    if coordinates_to_geocode:
        print(f"📍 Need to geocode {len(coordinates_to_geocode)} new locations (already have {len(geocoding_cache)} cached)")

        for idx, coordinates in enumerate(coordinates_to_geocode, 1):
            if idx % 10 == 0:
                print(f"   Geocoded {idx}/{len(coordinates_to_geocode)} new locations...")

            lat, lon = parse_coordinates(coordinates)
            if lat != 0.0 or lon != 0.0:
                time.sleep(1)  # Nominatim requires 1 second between requests
                location_info = reverse_geocode_coordinates(lat, lon)
                geocoding_cache[coordinates] = location_info
            else:
                geocoding_cache[coordinates] = {
                    'city': 'Unknown City',
                    'country': 'Unknown Country',
                    'place_name': None,
                    'address': ''
                }

            # Save cache every 50 geocoded locations to prevent data loss
            if idx % 50 == 0:
                save_geocoding_cache(geocoding_cache)
                print(f"   💾 Saved cache checkpoint at {idx} locations")

        # Final save
        save_geocoding_cache(geocoding_cache)
        print(f"✅ Geocoded {len(coordinates_to_geocode)} new locations")
    else:
        print(f"✅ All {len(unique_coordinates)} locations already cached - no geocoding needed!")

    # OPTIMIZATION 2: Build index for fast timeline lookup
    print("⚡ Building timeline index for fast lookup...")
    # Create a list of (hour, point) tuples for all hours covered by each point
    timeline_index = {}
    for point in timeline_points:
        point_start = point['start_time'].replace(minute=0, second=0, microsecond=0)
        point_end = point['end_time'].replace(minute=0, second=0, microsecond=0)

        current = point_start
        while current <= point_end:
            timeline_index[current] = point
            current += timedelta(hours=1)

    print(f"✅ Indexed {len(timeline_index)} hours")

    # OPTIMIZATION 3: Generate hourly records with O(1) lookup
    print("📝 Generating hourly records...")
    hourly_records = []
    current_hour = min_time.replace(minute=0, second=0, microsecond=0)
    total_hours = int((max_time - min_time).total_seconds() / 3600) + 1

    # Track last known point for interpolation
    last_known_point = None

    while current_hour <= max_time:
        # Progress indicator every 500 hours
        hours_processed = len(hourly_records)
        if hours_processed % 500 == 0 and hours_processed > 0:
            print(f"   Processed {hours_processed}/{total_hours} hours...")

        # O(1) lookup instead of O(n) search
        covering_point = timeline_index.get(current_hour)

        if covering_point:
            last_known_point = covering_point
            coordinates = covering_point['coordinates']
            timezone = covering_point['timezone']
            is_home = covering_point['is_home']
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
            # No data for this hour - use last known point or fallback
            if last_known_point:
                coordinates = last_known_point['coordinates']
                timezone = last_known_point['timezone']
                is_home = last_known_point['is_home']
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


def create_minute_location_records(location_data: List[Dict]) -> List[Dict]:
    """
    Create minute-by-minute location records from Google Maps timeline data.
    Uses timelinePath waypoints, visit records, and activity records.

    Args:
        location_data: List of timeline objects from Google Maps JSON

    Returns:
        List of minute records with detailed location information
    """
    print("🌍 Creating minute-by-minute location records...")

    # Step 1: Process all record types
    timeline_path_records = []
    visit_records = []
    activity_records = []

    for item in location_data:
        start_time_str = item.get('startTime')
        end_time_str = item.get('endTime')

        if not start_time_str or not end_time_str:
            continue

        try:
            # Parse timestamps
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
            timezone = extract_timezone_from_timestamp(start_time_str)

            # Process timelinePath records (highest accuracy)
            if 'timelinePath' in item:
                waypoints = item['timelinePath']
                for waypoint in waypoints:
                    point = waypoint.get('point')
                    offset_minutes = int(waypoint.get('durationMinutesOffsetFromStartTime', 0))

                    if point:
                        waypoint_time = start_time + timedelta(minutes=offset_minutes)
                        timeline_path_records.append({
                            'timestamp': waypoint_time,
                            'timezone': timezone,
                            'coordinates': point,
                            'record_type': 'waypoint',
                            'data_quality': 'actual',
                            'confidence': 1.0,
                            'is_home': False,
                            'activity_type': None
                        })

            # Process visit records
            elif 'visit' in item:
                visit = item['visit']
                top_candidate = visit.get('topCandidate', {})
                place_location = top_candidate.get('placeLocation', '')
                semantic_type = top_candidate.get('semanticType', 'Unknown')
                probability = float(visit.get('probability', 0))

                if place_location:
                    visit_records.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'timezone': timezone,
                        'coordinates': place_location,
                        'record_type': 'visit',
                        'data_quality': 'repeated',
                        'confidence': probability,
                        'is_home': (semantic_type == 'Home'),
                        'activity_type': None
                    })

            # Process activity records
            elif 'activity' in item:
                activity = item['activity']
                start_location = activity.get('start', '')
                end_location = activity.get('end', '')
                activity_type = activity.get('topCandidate', {}).get('type', 'unknown')
                probability = float(activity.get('probability', 0))

                if start_location and end_location:
                    activity_records.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'timezone': timezone,
                        'start_coordinates': start_location,
                        'end_coordinates': end_location,
                        'record_type': 'activity',
                        'data_quality': 'interpolated',
                        'confidence': probability,
                        'is_home': False,
                        'activity_type': activity_type
                    })

        except Exception as e:
            print(f"⚠️  Error processing timeline item: {e}")
            continue

    print(f"📊 Found {len(timeline_path_records)} waypoints, {len(visit_records)} visits, {len(activity_records)} activities")

    # Step 2: Geocode all unique coordinates
    print("🌍 Loading geocoding cache and processing coordinates...")
    geocoding_cache = load_geocoding_cache()

    # Collect unique coordinates
    unique_coordinates = set()
    for record in timeline_path_records:
        unique_coordinates.add(record['coordinates'])
    for record in visit_records:
        unique_coordinates.add(record['coordinates'])
    for record in activity_records:
        unique_coordinates.add(record['start_coordinates'])
        unique_coordinates.add(record['end_coordinates'])

    # Geocode new coordinates
    coordinates_to_geocode = [coord for coord in unique_coordinates if coord not in geocoding_cache]

    if coordinates_to_geocode:
        print(f"📍 Need to geocode {len(coordinates_to_geocode)} new locations (already have {len(geocoding_cache)} cached)")

        for idx, coordinates in enumerate(coordinates_to_geocode, 1):
            print(f"   Geocoded {idx}/{len(coordinates_to_geocode)} new locations...")

            lat, lon = parse_coordinates(coordinates)
            if lat != 0.0 or lon != 0.0:
                time.sleep(1)  # Nominatim requires 1 second between requests
                location_info = reverse_geocode_coordinates(lat, lon)
                geocoding_cache[coordinates] = location_info
            else:
                geocoding_cache[coordinates] = {
                    'city': 'Unknown City',
                    'country': 'Unknown Country',
                    'place_name': None,
                    'address': ''
                }

            # Save cache every 50 geocoded locations to prevent data loss
            if idx % 50 == 0:
                save_geocoding_cache(geocoding_cache)
                print(f"   💾 Saved cache checkpoint at {idx} locations")

        # Final save
        save_geocoding_cache(geocoding_cache)
        print(f"✅ Geocoded {len(coordinates_to_geocode)} new locations")
    else:
        print(f"✅ All {len(unique_coordinates)} locations already cached!")

    # Step 3: Generate minute-by-minute records
    print("📝 Generating minute-by-minute records...")

    # Find overall date range
    all_times = []
    for record in timeline_path_records:
        all_times.append(record['timestamp'])
    for record in visit_records:
        all_times.append(record['start_time'])
        all_times.append(record['end_time'])
    for record in activity_records:
        all_times.append(record['start_time'])
        all_times.append(record['end_time'])

    if not all_times:
        print("❌ No valid timeline data found")
        return []

    min_time = min(all_times)
    max_time = max(all_times)

    print(f"📅 Date range: {min_time.date()} to {max_time.date()}")

    # Build minute-level index
    minute_records = {}

    # Add waypoint records (highest priority - actual data)
    for record in timeline_path_records:
        minute_key = record['timestamp'].replace(second=0, microsecond=0)
        if minute_key not in minute_records:
            lat, lon = parse_coordinates(record['coordinates'])
            location_info = geocoding_cache[record['coordinates']]

            minute_records[minute_key] = {
                'timestamp': minute_key.isoformat(),
                'latitude': lat,
                'longitude': lon,
                'coordinates': record['coordinates'],
                'timezone': record['timezone'],
                'city': location_info['city'],
                'country': location_info['country'],
                'place_name': location_info.get('place_name'),
                'address': location_info.get('address', ''),
                'is_home': record['is_home'],
                'record_type': record['record_type'],
                'activity_type': record['activity_type'],
                'data_quality': record['data_quality'],
                'confidence': record['confidence']
            }

    # Expand visit records (medium priority - repeated data)
    for record in visit_records:
        current_minute = record['start_time'].replace(second=0, microsecond=0)
        end_minute = record['end_time'].replace(second=0, microsecond=0)

        lat, lon = parse_coordinates(record['coordinates'])
        location_info = geocoding_cache[record['coordinates']]

        while current_minute <= end_minute:
            if current_minute not in minute_records:  # Don't overwrite waypoint data
                minute_records[current_minute] = {
                    'timestamp': current_minute.isoformat(),
                    'latitude': lat,
                    'longitude': lon,
                    'coordinates': record['coordinates'],
                    'timezone': record['timezone'],
                    'city': location_info['city'],
                    'country': location_info['country'],
                    'place_name': location_info.get('place_name'),
                    'address': location_info.get('address', ''),
                    'is_home': record['is_home'],
                    'record_type': record['record_type'],
                    'activity_type': record['activity_type'],
                    'data_quality': record['data_quality'],
                    'confidence': record['confidence']
                }
            current_minute += timedelta(minutes=1)

    # Interpolate activity records (lowest priority - synthetic data)
    # OPTIMIZATION: Only geocode start/end points, interpolate city/country
    for record in activity_records:
        start_minute = record['start_time'].replace(second=0, microsecond=0)
        end_minute = record['end_time'].replace(second=0, microsecond=0)

        start_lat, start_lon = parse_coordinates(record['start_coordinates'])
        end_lat, end_lon = parse_coordinates(record['end_coordinates'])

        # Get location info for start and end (from cache)
        start_location_info = geocoding_cache[record['start_coordinates']]
        end_location_info = geocoding_cache[record['end_coordinates']]

        total_minutes = int((end_minute - start_minute).total_seconds() / 60)
        if total_minutes == 0:
            total_minutes = 1

        current_minute = start_minute
        minute_index = 0

        while current_minute <= end_minute:
            if current_minute not in minute_records:  # Don't overwrite better data
                # Linear interpolation of coordinates
                progress = minute_index / total_minutes if total_minutes > 0 else 0
                interp_lat = start_lat + (end_lat - start_lat) * progress
                interp_lon = start_lon + (end_lon - start_lon) * progress
                interp_coords = f"geo:{interp_lat},{interp_lon}"

                # Use start location for first half, end location for second half
                # (Avoids geocoding thousands of interpolated points)
                if progress < 0.5:
                    location_info = start_location_info
                else:
                    location_info = end_location_info

                minute_records[current_minute] = {
                    'timestamp': current_minute.isoformat(),
                    'latitude': interp_lat,
                    'longitude': interp_lon,
                    'coordinates': interp_coords,
                    'timezone': record['timezone'],
                    'city': location_info['city'],
                    'country': location_info['country'],
                    'place_name': location_info.get('place_name'),
                    'address': location_info.get('address', ''),
                    'is_home': record['is_home'],
                    'record_type': record['record_type'],
                    'activity_type': record['activity_type'],
                    'data_quality': record['data_quality'],
                    'confidence': record['confidence']
                }

            current_minute += timedelta(minutes=1)
            minute_index += 1

    # Convert to sorted list
    sorted_records = sorted(minute_records.values(), key=lambda x: x['timestamp'])

    print(f"✅ Created {len(sorted_records)} minute records")

    # Show data quality distribution
    quality_counts = {}
    for record in sorted_records:
        quality = record['data_quality']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

    print(f"📊 Data quality distribution:")
    for quality, count in sorted(quality_counts.items()):
        percentage = (count / len(sorted_records)) * 100
        print(f"   • {quality}: {count} records ({percentage:.1f}%)")

    return sorted_records


def create_google_maps_file():
    """
    Main processing function that creates both hourly and minute-level Google Maps location files.

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("📱 GOOGLE MAPS TIMELINE PROCESSING")
    print("="*70)

    # Define file paths
    input_path = "files/exports/google_exports/location_history.json"
    hourly_output_path = 'files/source_processed_files/google_maps/google_maps_processed.csv'
    minute_output_path = 'files/source_processed_files/google_maps/google_maps_minute_processed.csv'

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

        # ========================================================================
        # PART 1: Create hourly records (for timezone correction)
        # ========================================================================
        print("\n" + "="*70)
        print("PART 1: HOURLY LOCATION DATA (for timezone correction)")
        print("="*70)

        hourly_records = create_hourly_timezone_records(location_data)

        if not hourly_records:
            print("❌ No hourly records created")
            return False

        # Convert to DataFrame
        df_hourly = pd.DataFrame(hourly_records)

        # Add source column
        df_hourly['source'] = 'google_maps'

        # Enforce snake_case
        df_hourly = enforce_snake_case(df_hourly, "google_maps_processed")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(hourly_output_path), exist_ok=True)

        # Save to CSV with pipe separator and UTF-8 encoding
        print(f"💾 Saving hourly data to {hourly_output_path}...")
        df_hourly.to_csv(hourly_output_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed hourly Google Maps data!")
        print(f"📊 Created {len(df_hourly)} hourly location records")
        print(f"📅 Date range: {df_hourly['timestamp'].min()} to {df_hourly['timestamp'].max()}")
        print(f"🌍 Countries visited: {df_hourly['country'].nunique()}")
        print(f"🏙️  Cities visited: {df_hourly['city'].nunique()}")
        print(f"🏠 Hours at home: {df_hourly['is_home'].sum()}")

        # Show sample of the data
        print(f"\n📋 Sample hourly records:")
        sample_df = df_hourly.head(3)[['timestamp', 'timezone', 'city', 'country', 'is_home']]
        for _, row in sample_df.iterrows():
            home_status = "🏠 Home" if row['is_home'] else "📍 Away"
            print(f"  • {row['timestamp'][:16]} | {row['timezone']} | {row['city']}, {row['country']} | {home_status}")

        # ========================================================================
        # PART 2: Create minute-level records (detailed location tracking)
        # ========================================================================
        print("\n" + "="*70)
        print("PART 2: MINUTE-LEVEL LOCATION DATA (detailed tracking)")
        print("="*70)

        minute_records = create_minute_location_records(location_data)

        if not minute_records:
            print("❌ No minute records created")
            return False

        # Convert to DataFrame
        df_minute = pd.DataFrame(minute_records)

        # Add source column
        df_minute['source'] = 'google_maps'

        # Enforce snake_case
        df_minute = enforce_snake_case(df_minute, "google_maps_minute_processed")

        # Save to CSV with pipe separator and UTF-8 encoding
        print(f"💾 Saving minute data to {minute_output_path}...")
        df_minute.to_csv(minute_output_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed minute-level Google Maps data!")
        print(f"📊 Created {len(df_minute)} minute location records")
        print(f"📅 Date range: {df_minute['timestamp'].min()} to {df_minute['timestamp'].max()}")
        print(f"🌍 Countries visited: {df_minute['country'].nunique()}")
        print(f"🏙️  Cities visited: {df_minute['city'].nunique()}")
        print(f"📍 Unique places: {df_minute['place_name'].nunique()}")
        print(f"🏠 Minutes at home: {df_minute['is_home'].sum()}")

        # Show sample of the data
        print(f"\n📋 Sample minute records:")
        sample_df = df_minute.head(3)[['timestamp', 'city', 'place_name', 'data_quality', 'is_home']]
        for _, row in sample_df.iterrows():
            home_status = "🏠" if row['is_home'] else "📍"
            place = row['place_name'] if row['place_name'] else row['city']
            print(f"  {home_status} {row['timestamp'][:16]} | {place} | {row['data_quality']}")

        # ========================================================================
        # SUMMARY
        # ========================================================================
        print("\n" + "="*70)
        print("✅ GOOGLE MAPS PROCESSING COMPLETE!")
        print("="*70)
        print(f"📄 Hourly file: {hourly_output_path}")
        print(f"   • {len(df_hourly)} records | {len(df_hourly.columns)} columns")
        print(f"📄 Minute file: {minute_output_path}")
        print(f"   • {len(df_minute)} records | {len(df_minute.columns)} columns")
        print("="*70)

        return True

    except Exception as e:
        print(f"❌ Error processing Google Maps data: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_google_maps_pipeline(auto_full=False):
    """
    Complete Google Maps SOURCE processor pipeline.

    Options:
    1. Download new data and process
    2. Process existing data

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*70)
    print("📱 GOOGLE MAPS SOURCE PROCESSOR PIPELINE")
    print("="*70)

    if auto_full:
        print("🤖 Auto mode: Downloading and processing...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new data and process")
        print("2. Process existing data")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Option 1: Download and process...")

        # Step 1: Download
        download_success = download_google_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_google_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_google_files()

        # Step 3: Process
        success = create_google_maps_file()

    elif choice == "2":
        print("\n⚙️  Option 2: Process existing data...")
        success = create_google_maps_file()

    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    # Final status
    print("\n" + "="*70)
    if success:
        print("✅ Google Maps source processor completed successfully!")
        print("📊 Outputs:")
        print("   • Hourly: files/source_processed_files/google_maps/google_maps_processed.csv")
        print("   • Minute: files/source_processed_files/google_maps/google_maps_minute_processed.csv")
        print("📝 Next: Run Location topic coordinator to merge and upload")
    else:
        print("❌ Google Maps processing failed")
    print("="*70)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("📱 Google Maps Timeline Source Processor")
    print("This processor handles Google Maps Timeline data.")
    print("Upload is handled by Location topic coordinator.\n")

    # Run the pipeline
    full_google_maps_pipeline(auto_full=False)
