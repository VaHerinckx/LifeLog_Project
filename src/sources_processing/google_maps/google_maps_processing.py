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
# CACHE MAINTENANCE FUNCTIONS
# ============================================================================

def clean_geocoding_cache():
    """
    Remove incomplete geocoding cache entries that are missing place_name or address.
    This forces the pipeline to re-geocode these coordinates with complete data.

    Returns:
        tuple: (entries_removed, total_entries_before, total_entries_after)
    """
    print("🧹 Cleaning geocoding cache...")

    # Load existing cache
    geocoding_cache = load_geocoding_cache()
    initial_count = len(geocoding_cache)

    if initial_count == 0:
        print("   Cache is empty, nothing to clean")
        return (0, 0, 0)

    # Identify incomplete entries
    incomplete_entries = []
    for coordinates, location_info in geocoding_cache.items():
        # Check if missing place_name or address fields
        if 'place_name' not in location_info or 'address' not in location_info:
            incomplete_entries.append(coordinates)

    # Remove incomplete entries
    for coordinates in incomplete_entries:
        del geocoding_cache[coordinates]

    # Save cleaned cache
    save_geocoding_cache(geocoding_cache)

    final_count = len(geocoding_cache)
    removed_count = len(incomplete_entries)

    print(f"✅ Cache cleaning complete:")
    print(f"   • Removed {removed_count} incomplete entries")
    print(f"   • Before: {initial_count} entries")
    print(f"   • After: {final_count} entries")
    print(f"   • These will be re-geocoded on next pipeline run")

    return (removed_count, initial_count, final_count)


# ============================================================================
# PLACE NAME MAPPING FUNCTIONS
# ============================================================================

PLACE_NAME_MAPPINGS_PATH = 'files/work_files/google_maps_work_files/place_name_mappings.json'


def load_place_name_mappings(coordinate_mapping: Dict = None) -> Dict:
    """
    Load custom place name mappings from work file.
    If coordinate_mapping is provided, migrates old coordinates to canonical coordinates.

    Args:
        coordinate_mapping: Optional dict mapping original coords to canonical coords

    Returns:
        dict: Mapping of coordinates to custom place name info
    """
    if not os.path.exists(PLACE_NAME_MAPPINGS_PATH):
        return {}

    try:
        with open(PLACE_NAME_MAPPINGS_PATH, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading place name mappings: {e}")
        return {}

    # Migrate old coordinates to canonical coordinates if mapping provided
    if coordinate_mapping:
        migrated_mappings = {}
        migrations_needed = False

        for old_coord, mapping_info in mappings.items():
            canonical_coord = coordinate_mapping.get(old_coord, old_coord)

            if canonical_coord != old_coord:
                migrations_needed = True

            # Check if we already have an entry for this canonical coordinate
            if canonical_coord in migrated_mappings:
                # Conflict: keep the most recently updated one
                existing_updated = migrated_mappings[canonical_coord].get('last_updated', '')
                new_updated = mapping_info.get('last_updated', '')
                if new_updated > existing_updated:
                    migrated_mappings[canonical_coord] = mapping_info
            else:
                migrated_mappings[canonical_coord] = mapping_info

        if migrations_needed:
            # Save migrated mappings
            print(f"🔄 Migrating {len(mappings)} place name mappings to canonical coordinates...")
            save_place_name_mappings(migrated_mappings)
            return migrated_mappings

    return mappings


def save_place_name_mappings(mappings: Dict):
    """
    Save custom place name mappings to work file.

    Args:
        mappings: Dict of coordinates to custom place name info
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(PLACE_NAME_MAPPINGS_PATH), exist_ok=True)

    try:
        with open(PLACE_NAME_MAPPINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(mappings)} place name mappings")
    except Exception as e:
        print(f"❌ Error saving place name mappings: {e}")


def apply_place_name_mappings(geocoding_cache: Dict, mappings: Dict) -> Dict:
    """
    Apply custom place names to geocoding cache entries.
    Returns updated geocoding cache with custom place names applied.

    Args:
        geocoding_cache: Dict of coordinates to location info
        mappings: Dict of coordinates to custom place name info

    Returns:
        Updated geocoding cache with custom place_names
    """
    updated_count = 0
    for coordinates, mapping_info in mappings.items():
        if coordinates in geocoding_cache:
            custom_name = mapping_info.get('custom_place_name')
            if custom_name:
                geocoding_cache[coordinates]['place_name'] = custom_name
                updated_count += 1

    if updated_count > 0:
        print(f"📍 Applied {updated_count} custom place names")

    return geocoding_cache


def consolidate_geocoding_cache(geocoding_cache: Dict) -> tuple:
    """
    Consolidate coordinates that share the same address into a single canonical coordinate.
    This handles GPS drift where the same physical location has slightly different coordinates.

    Args:
        geocoding_cache: Dict of coordinates to location info

    Returns:
        tuple: (updated_geocoding_cache, coordinate_mapping)
            - updated_geocoding_cache: Cache with canonical coordinates added
            - coordinate_mapping: Dict mapping original coordinates to canonical coordinates
    """
    from collections import defaultdict

    # Group coordinates by address
    address_to_coords = defaultdict(list)
    for coordinates, info in geocoding_cache.items():
        address = info.get('address', '')
        if address:  # Only consolidate if there's an actual address
            address_to_coords[address].append(coordinates)

    # Find addresses with multiple coordinates
    coordinate_mapping = {}
    consolidated_count = 0

    for address, coords_list in address_to_coords.items():
        if len(coords_list) > 1:
            # Calculate average coordinates
            lats = []
            lons = []
            for coord in coords_list:
                lat, lon = parse_coordinates(coord)
                if lat != 0.0 or lon != 0.0:
                    lats.append(lat)
                    lons.append(lon)

            if lats and lons:
                avg_lat = sum(lats) / len(lats)
                avg_lon = sum(lons) / len(lons)
                canonical_coord = f"geo:{avg_lat:.6f},{avg_lon:.6f}"

                # Get info from first coordinate (they should all have same address/city)
                first_info = geocoding_cache[coords_list[0]]

                # Check if any of the original coordinates has a custom place name
                # (from place_name_mappings applied earlier)
                best_place_name = first_info.get('place_name')
                for coord in coords_list:
                    coord_info = geocoding_cache[coord]
                    place_name = coord_info.get('place_name')
                    # Prefer non-numeric place names (custom names are usually not just numbers)
                    if place_name and not str(place_name).isdigit():
                        best_place_name = place_name
                        break

                # Add canonical coordinate to cache
                geocoding_cache[canonical_coord] = {
                    'city': first_info.get('city', ''),
                    'country': first_info.get('country', ''),
                    'place_name': best_place_name,
                    'address': address
                }

                # Map all original coordinates to canonical
                for coord in coords_list:
                    coordinate_mapping[coord] = canonical_coord

                consolidated_count += 1

    if consolidated_count > 0:
        print(f"🔗 Consolidated {consolidated_count} addresses with multiple coordinates")

    return geocoding_cache, coordinate_mapping


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
    Processes both visit records (places where you stopped) and activity records (movement types).

    Args:
        location_data: List of timeline objects from Google Maps JSON

    Returns:
        List of minute records with detailed location information and activity types
    """
    print("🌍 Creating minute-by-minute location records (visits + activities)...")

    # Step 1: Process visit and activity records
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

            # Process visit records (places where you stopped)
            if 'visit' in item:
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
                        'data_quality': 'actual',
                        'confidence': probability,
                        'is_home': (semantic_type == 'Home'),
                        'activity_type': None,
                        'distance_meters': None
                    })

            # Process activity records (movement between places)
            elif 'activity' in item:
                activity = item['activity']
                start_location = activity.get('start', '')
                activity_type = activity.get('topCandidate', {}).get('type', 'unknown')
                probability = float(activity.get('probability', 0))
                distance = float(activity.get('distanceMeters', 0))

                # Only process activities with significant distance (skip GPS noise)
                if start_location and distance > 10:
                    activity_records.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'timezone': timezone,
                        'coordinates': start_location,
                        'record_type': 'activity',
                        'data_quality': 'transit',
                        'confidence': probability,
                        'is_home': False,
                        'activity_type': activity_type,
                        'distance_meters': distance
                    })

        except Exception as e:
            print(f"⚠️  Error processing timeline item: {e}")
            continue

    print(f"📊 Found {len(visit_records)} visits, {len(activity_records)} activities")

    # Step 2: Geocode visit and activity start coordinates
    print("🌍 Loading geocoding cache and processing locations...")
    geocoding_cache = load_geocoding_cache()

    # Collect unique coordinates from visits and activity start locations
    unique_coordinates = set()
    for record in visit_records:
        unique_coordinates.add(record['coordinates'])
    for record in activity_records:
        unique_coordinates.add(record['coordinates'])

    # Geocode new coordinates
    coordinates_to_geocode = [coord for coord in unique_coordinates if coord not in geocoding_cache]

    if coordinates_to_geocode:
        print(f"📍 Need to geocode {len(coordinates_to_geocode)} new locations (already have {len(geocoding_cache)} cached)")

        for idx, coordinates in enumerate(coordinates_to_geocode, 1):
            if idx % 10 == 0 or idx == len(coordinates_to_geocode):
                print(f"   Geocoding {idx}/{len(coordinates_to_geocode)} locations...")

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

    # Consolidate coordinates that share the same address (GPS drift fix)
    # This must happen BEFORE applying place name mappings so mappings use canonical coordinates
    geocoding_cache, coordinate_mapping = consolidate_geocoding_cache(geocoding_cache)

    # Apply custom place name mappings (uses canonical coordinates)
    place_name_mappings = load_place_name_mappings(coordinate_mapping)
    if place_name_mappings:
        geocoding_cache = apply_place_name_mappings(geocoding_cache, place_name_mappings)

    # Step 3: Generate minute-by-minute records (visits + activities)
    print("📝 Generating minute-by-minute records from visits and activities...")

    # Find overall date range
    if not visit_records and not activity_records:
        print("❌ No visit or activity records found")
        return []

    all_times = []
    for record in visit_records:
        all_times.append(record['start_time'])
        all_times.append(record['end_time'])
    for record in activity_records:
        all_times.append(record['start_time'])
        all_times.append(record['end_time'])

    min_time = min(all_times)
    max_time = max(all_times)

    print(f"📅 Date range: {min_time.date()} to {max_time.date()}")

    # Build minute-level index
    minute_records = {}

    # Expand visit records to minute-by-minute (highest priority)
    for record in visit_records:
        current_minute = record['start_time'].replace(second=0, microsecond=0)
        end_minute = record['end_time'].replace(second=0, microsecond=0)

        # Use canonical coordinate if available (consolidation)
        original_coord = record['coordinates']
        canonical_coord = coordinate_mapping.get(original_coord, original_coord)
        lat, lon = parse_coordinates(canonical_coord)
        location_info = geocoding_cache[canonical_coord]

        # Get place_name, replacing numeric-only names with address
        place_name = location_info.get('place_name')
        address = location_info.get('address', '')
        if place_name and str(place_name).isdigit():
            place_name = address if address else None

        while current_minute <= end_minute:
            if current_minute not in minute_records:  # Don't duplicate overlapping visits
                minute_records[current_minute] = {
                    'timestamp': current_minute.isoformat(),
                    'latitude': lat,
                    'longitude': lon,
                    'coordinates': canonical_coord,
                    'timezone': record['timezone'],
                    'city': location_info['city'],
                    'country': location_info['country'],
                    'place_name': place_name,
                    'address': address,
                    'is_home': record['is_home'],
                    'record_type': record['record_type'],
                    'activity_type': record['activity_type'],
                    'data_quality': record['data_quality'],
                    'confidence': record['confidence'],
                    'distance_meters': record['distance_meters'],
                    'is_moving': False
                }
            current_minute += timedelta(minutes=1)

    # Expand activity records to minute-by-minute (fill gaps)
    for record in activity_records:
        current_minute = record['start_time'].replace(second=0, microsecond=0)
        end_minute = record['end_time'].replace(second=0, microsecond=0)

        # Use canonical coordinate if available (consolidation)
        original_coord = record['coordinates']
        canonical_coord = coordinate_mapping.get(original_coord, original_coord)
        lat, lon = parse_coordinates(canonical_coord)
        location_info = geocoding_cache[canonical_coord]

        # Calculate distance per minute (distribute total distance across all minutes)
        total_minutes = int((record['end_time'] - record['start_time']).total_seconds() / 60) + 1
        if record['distance_meters'] and total_minutes > 0:
            distance_per_minute = round(record['distance_meters'] / total_minutes, 2)
        else:
            distance_per_minute = None

        while current_minute <= end_minute:
            if current_minute not in minute_records:  # Don't overwrite visits
                minute_records[current_minute] = {
                    'timestamp': current_minute.isoformat(),
                    'latitude': lat,
                    'longitude': lon,
                    'coordinates': canonical_coord,
                    'timezone': record['timezone'],
                    'city': location_info['city'],
                    'country': location_info['country'],
                    'place_name': None,  # No place name for transit
                    'address': '',  # No address for transit
                    'is_home': record['is_home'],
                    'record_type': record['record_type'],
                    'activity_type': record['activity_type'],
                    'data_quality': record['data_quality'],
                    'confidence': record['confidence'],
                    'distance_meters': distance_per_minute,
                    'is_moving': True
                }
            current_minute += timedelta(minutes=1)

    # Convert to sorted list
    sorted_records = sorted(minute_records.values(), key=lambda x: x['timestamp'])

    print(f"✅ Created {len(sorted_records)} minute records from {len(visit_records)} visits + {len(activity_records)} activities")

    # Show statistics
    visit_minutes = sum(1 for record in sorted_records if record['record_type'] == 'visit')
    activity_minutes = sum(1 for record in sorted_records if record['record_type'] == 'activity')
    home_minutes = sum(1 for record in sorted_records if record['is_home'])
    unique_places = len(set(record['place_name'] for record in sorted_records if record['place_name']))

    # Activity type breakdown
    activity_types = {}
    for record in sorted_records:
        if record['activity_type']:
            activity_types[record['activity_type']] = activity_types.get(record['activity_type'], 0) + 1

    print(f"📊 Location statistics:")
    print(f"   • Total minutes: {len(sorted_records)}")
    print(f"   • Visit minutes: {visit_minutes} ({(visit_minutes/len(sorted_records)*100):.1f}%)")
    print(f"   • Activity minutes: {activity_minutes} ({(activity_minutes/len(sorted_records)*100):.1f}%)")
    print(f"   • Minutes at home: {home_minutes} ({(home_minutes/len(sorted_records)*100):.1f}%)")
    print(f"   • Unique places visited: {unique_places}")

    if activity_types:
        print(f"📊 Activity type breakdown:")
        sorted_activities = sorted(activity_types.items(), key=lambda x: x[1], reverse=True)
        for activity_type, count in sorted_activities[:5]:  # Show top 5
            percentage = (count / activity_minutes * 100) if activity_minutes > 0 else 0
            print(f"   • {activity_type}: {count} minutes ({percentage:.1f}%)")

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

        # Clean geocoding cache first to ensure all entries have complete data
        print("\n📋 Step 0: Cleaning geocoding cache...")
        clean_geocoding_cache()

        # Load the JSON data
        print(f"\n📱 Reading Google Maps timeline data...")
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
        sample_df = df_minute.head(5)[['timestamp', 'city', 'place_name', 'record_type', 'activity_type', 'is_moving']]
        for _, row in sample_df.iterrows():
            icon = "🏠" if row.get('place_name') and 'home' in str(row['place_name']).lower() else ("🚶" if row['is_moving'] else "📍")
            place = row['place_name'] if row['place_name'] else row['city']
            activity = f" ({row['activity_type']})" if row['activity_type'] else ""
            status = "moving" if row['is_moving'] else "stationary"
            print(f"  {icon} {row['timestamp'][:16]} | {place}{activity} | {status}")

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
# PLACE NAME MANAGEMENT FUNCTIONS
# ============================================================================

def identify_frequent_places():
    """
    Identify addresses actually visited 3+ times and prompt user to name them.
    Only includes stationary visits (not transit/activity records).
    Shows dates visited to help user remember the place.
    Excludes places without a proper address (just city name).

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("📍 IDENTIFY AND NAME FREQUENT PLACES")
    print("="*70)

    # Load existing minute-level data to count visits
    minute_file = 'files/source_processed_files/google_maps/google_maps_minute_processed.csv'
    if not os.path.exists(minute_file):
        print(f"❌ Minute-level file not found: {minute_file}")
        print("   Run option 2 first to process existing data.")
        return False

    try:
        df = pd.read_csv(minute_file, sep='|', encoding='utf-8')
        print(f"✅ Loaded {len(df):,} minute records")
    except Exception as e:
        print(f"❌ Error loading minute file: {e}")
        return False

    # Filter for actual visits only (not transit/activity records)
    # record_type == 'visit' means stationary visits to places
    visits_only = df[df['record_type'] == 'visit'].copy()
    print(f"📍 Filtered to {len(visits_only):,} visit records (excluding transit)")

    if len(visits_only) == 0:
        print("❌ No visit records found")
        return False

    # Filter out records without a proper address
    visits_only = visits_only[visits_only['address'].notna() & (visits_only['address'] != '')]
    print(f"📍 {len(visits_only):,} records have actual addresses")

    if len(visits_only) == 0:
        print("❌ No visit records with addresses found")
        return False

    # Parse timestamp to extract date
    visits_only['date'] = pd.to_datetime(visits_only['timestamp']).dt.date

    # Count unique visit dates per coordinates (actual visits, not minutes)
    visit_stats = visits_only.groupby('coordinates').agg({
        'place_name': 'first',
        'address': 'first',
        'city': 'first',
        'date': lambda x: sorted(list(x.unique())),  # Sorted list of unique dates visited
        'timestamp': 'count'  # Total minutes at location
    }).rename(columns={'timestamp': 'minute_count', 'date': 'dates_visited'})

    # Add visit count (number of unique dates)
    visit_stats['visit_count'] = visit_stats['dates_visited'].apply(len)

    # Filter for places visited on at least 3 different dates
    frequent_places = visit_stats[visit_stats['visit_count'] >= 3].copy()
    frequent_places = frequent_places.sort_values('visit_count', ascending=False)

    print(f"📊 Found {len(frequent_places)} places visited on 3+ different dates")

    # Load existing mappings
    existing_mappings = load_place_name_mappings()
    print(f"📂 {len(existing_mappings)} existing custom place names")

    # Filter out already-named places
    unnamed_places = []
    for coordinates in frequent_places.index:
        if coordinates not in existing_mappings:
            place_info = frequent_places.loc[coordinates]
            unnamed_places.append({
                'coordinates': coordinates,
                'place_name': place_info['place_name'],
                'address': place_info['address'],
                'city': place_info['city'],
                'visit_count': place_info['visit_count'],
                'minute_count': place_info['minute_count'],
                'dates_visited': place_info['dates_visited']
            })

    if not unnamed_places:
        print("✅ All frequent places already have custom names!")
        return True

    print(f"\n📝 {len(unnamed_places)} frequent places need naming:")
    print("-" * 70)

    # Sort by visit_count descending (most visited first)
    unnamed_places.sort(key=lambda x: x['visit_count'], reverse=True)

    named_count = 0
    for i, place in enumerate(unnamed_places, 1):
        current_name = place['place_name'] if place['place_name'] else '(no name)'

        # Format dates for display (show up to 5 most recent)
        dates = place['dates_visited']
        dates_str = ', '.join(str(d) for d in dates[-5:])  # Last 5 dates
        if len(dates) > 5:
            dates_str = f"... {dates_str} (and {len(dates) - 5} earlier)"

        print(f"\n[{i}/{len(unnamed_places)}] Visited on {place['visit_count']} different dates ({place['minute_count']:,} total minutes)")
        print(f"   Current name: {current_name}")
        print(f"   Address: {place['address']}")
        print(f"   City: {place['city']}")
        print(f"   Dates: {dates_str}")

        new_name = input("   Enter a name (or press Enter to skip, 'quit' to stop): ").strip()

        if new_name.lower() == 'quit':
            print("\n⏹️  Stopping naming process...")
            break

        if new_name:
            existing_mappings[place['coordinates']] = {
                'original_place_name': place['place_name'],
                'custom_place_name': new_name,
                'address': place['address'],
                'city': place['city'],
                'visit_count': int(place['visit_count']),
                'last_updated': datetime.now().isoformat()
            }
            named_count += 1
            print(f"   ✅ Named as: {new_name}")

    if named_count > 0:
        # Save updated mappings
        save_place_name_mappings(existing_mappings)
        print(f"\n✅ Named {named_count} places")

        # Re-run processing to apply new names
        print("\n🔄 Re-processing data to apply new place names...")
        success = create_google_maps_file()
        return success
    else:
        print("\n❌ No places were named")
        return True


def update_place_name():
    """
    Manually update place name for a specific address.
    Similar to update_cover_url in reading_processing.py.
    Uses canonical coordinates (after consolidation) to avoid duplicates.

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("📝 MANUALLY UPDATE PLACE NAME")
    print("="*70)

    # Load geocoding cache for searching
    geocoding_cache = load_geocoding_cache()
    if not geocoding_cache:
        print("❌ Geocoding cache is empty. Run processing first.")
        return False

    print(f"✅ Loaded {len(geocoding_cache)} cached locations")

    # Consolidate to get canonical coordinates
    geocoding_cache, coordinate_mapping = consolidate_geocoding_cache(geocoding_cache)

    # Load existing mappings (with migration if needed)
    existing_mappings = load_place_name_mappings(coordinate_mapping)

    # Build set of canonical coordinates (either from mapping or unchanged)
    canonical_coords = set()
    for coord in geocoding_cache.keys():
        # Use the canonical coordinate if this coord maps to one, otherwise use as-is
        canonical = coordinate_mapping.get(coord, coord)
        canonical_coords.add(canonical)

    # Convert cache to searchable list (only canonical coordinates)
    locations_list = []
    for coordinates in canonical_coords:
        if coordinates not in geocoding_cache:
            continue
        info = geocoding_cache[coordinates]
        # Check if there's already a custom name
        custom_name = None
        if coordinates in existing_mappings:
            custom_name = existing_mappings[coordinates].get('custom_place_name')

        locations_list.append({
            'coordinates': coordinates,
            'place_name': info.get('place_name', ''),
            'custom_place_name': custom_name,
            'address': info.get('address', ''),
            'city': info.get('city', '')
        })

    updated_places = []

    # Loop to allow multiple updates
    while True:
        search_term = input("\nEnter part of address to search (or 'quit' to finish): ").strip()
        if not search_term or search_term.lower() == 'quit':
            break

        # Find matching locations (case-insensitive search in address)
        matches = [loc for loc in locations_list
                   if search_term.lower() in str(loc['address']).lower()
                   or search_term.lower() in str(loc['place_name']).lower()
                   or search_term.lower() in str(loc['city']).lower()]

        if not matches:
            print(f"❌ No locations found containing '{search_term}'")
            continue

        print(f"\n📍 Found {len(matches)} locations:")
        print("-" * 70)

        # Display matches with numbers
        for i, loc in enumerate(matches[:20], 1):  # Limit to 20 results
            current_name = loc['custom_place_name'] or loc['place_name'] or '(no name)'
            display_name = current_name[:40] + "..." if len(str(current_name)) > 40 else current_name
            address = loc['address'][:50] + "..." if len(str(loc['address'])) > 50 else loc['address']
            custom_indicator = " [custom]" if loc['custom_place_name'] else ""
            print(f"{i:2}. {display_name}{custom_indicator}")
            print(f"    Address: {address}")
            print()

        if len(matches) > 20:
            print(f"   ... and {len(matches) - 20} more. Refine your search to see others.")

        # Get user selection
        try:
            choice = int(input(f"Select location (1-{min(len(matches), 20)}): "))
            if choice < 1 or choice > min(len(matches), 20):
                print(f"❌ Invalid choice. Please select 1-{min(len(matches), 20)}")
                continue
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
            continue

        # Get selected location
        selected = matches[choice - 1]
        current_display = selected['custom_place_name'] or selected['place_name'] or 'None'

        print(f"\n✅ Selected: {selected['address']}")
        print(f"   Current name: {current_display}")
        print(f"   Coordinates: {selected['coordinates']}")

        # Get new name
        new_name = input("\nEnter new place name: ").strip()
        if not new_name:
            print("❌ No name provided, skipping")
            continue

        # Update mapping
        existing_mappings[selected['coordinates']] = {
            'original_place_name': selected['place_name'],
            'custom_place_name': new_name,
            'address': selected['address'],
            'city': selected['city'],
            'last_updated': datetime.now().isoformat()
        }

        # Also update the local list so subsequent searches show new name
        selected['custom_place_name'] = new_name

        print(f"✅ Updated place name to: {new_name}")
        updated_places.append(new_name)

        # Ask if user wants to continue
        continue_choice = input("\nUpdate another place? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break

    if not updated_places:
        print("❌ No places were updated")
        return False

    # Save updated mappings
    save_place_name_mappings(existing_mappings)

    # Re-run processing to apply new names
    print(f"\n🔄 Re-processing data to apply {len(updated_places)} updated place names...")
    success = create_google_maps_file()

    if success:
        print(f"\n✅ Successfully updated {len(updated_places)} place names:")
        for name in updated_places:
            print(f"   • {name}")

    return success


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_google_maps_pipeline(auto_full=False):
    """
    Complete Google Maps SOURCE processor pipeline.

    Options:
    1. Download new data and process
    2. Process existing data
    3. Identify and name frequent places
    4. Manually update a place name

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
        print("3. Identify and name frequent places")
        print("4. Manually update a place name")

        choice = input("\nEnter your choice (1-4): ").strip()

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

    elif choice == "3":
        print("\n📍 Option 3: Identify and name frequent places...")
        success = identify_frequent_places()

    elif choice == "4":
        print("\n📝 Option 4: Manually update a place name...")
        success = update_place_name()

    else:
        print("❌ Invalid choice. Please select 1-4.")
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
