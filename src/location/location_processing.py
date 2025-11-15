import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from pathlib import Path
import pytz

from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run

# Geocoding cache file path (shared between Google and Manual processing)
GEOCODING_CACHE_FILE = 'files/work_files/geocoding_cache.json'

# Global geocoding cache for manual processing
_geocoding_cache = {}


# ============================================================================
# GEOCODING FUNCTIONS (Shared by both Google Maps and Manual processing)
# ============================================================================

def load_geocoding_cache() -> Dict[str, Dict[str, str]]:
    """
    Load geocoding cache from JSON file.

    Returns:
        Dictionary mapping coordinates to location info
    """
    if os.path.exists(GEOCODING_CACHE_FILE):
        try:
            with open(GEOCODING_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"📦 Loaded geocoding cache with {len(cache)} locations")
            return cache
        except Exception as e:
            print(f"⚠️  Error loading geocoding cache: {e}")
            return {}
    return {}


def save_geocoding_cache(cache: Dict[str, Dict[str, str]]) -> None:
    """
    Save geocoding cache to JSON file.

    Args:
        cache: Dictionary mapping coordinates to location info
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(GEOCODING_CACHE_FILE), exist_ok=True)

        with open(GEOCODING_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved geocoding cache with {len(cache)} locations")
    except Exception as e:
        print(f"⚠️  Error saving geocoding cache: {e}")


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


def geocode_location(city: str, country: str) -> Tuple[str, str]:
    """
    Programmatically get coordinates and timezone for any city using APIs.

    Args:
        city: City name
        country: Country name

    Returns:
        Tuple of (coordinates_string, timezone_id)
    """
    try:
        # Use Nominatim (OpenStreetMap) for geocoding - free and no API key required
        print(f"🌍 Geocoding {city}, {country}...")

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': f"{city}, {country}",
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }

        headers = {
            'User-Agent': 'LifeLog-Location-Processor/1.0'
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data and len(data) > 0:
                location = data[0]
                lat = float(location['lat'])
                lon = float(location['lon'])
                coordinates = f"{lat},{lon}"

                print(f"  ✅ Found coordinates: {coordinates}")

                # Get timezone using coordinate-based methods
                timezone_id = get_timezone_from_coordinates(lat, lon, city, country)

                return coordinates, timezone_id
            else:
                print(f"  ❌ No results found for {city}, {country}")
                return "0.0,0.0", "UTC"
        else:
            print(f"  ❌ Geocoding failed: HTTP {response.status_code}")
            return "0.0,0.0", "UTC"

    except Exception as e:
        print(f"  ❌ Error geocoding {city}, {country}: {e}")
        return "0.0,0.0", "UTC"


def get_timezone_from_coordinates(lat: float, lon: float, city: str, country: str) -> str:
    """
    Get timezone from coordinates using multiple API methods with robust fallbacks.

    Args:
        lat: Latitude
        lon: Longitude
        city: City name (for fallback)
        country: Country name (for fallback)

    Returns:
        Timezone identifier (e.g., 'Europe/Brussels')
    """
    print(f"  🕐 Getting timezone for coordinates {lat}, {lon}...")

    # Method 1: Try WorldTimeAPI (very reliable, free, no registration)
    try:
        url = f"http://worldtimeapi.org/api/timezone"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            timezones = response.json()

            # Find the best timezone match based on coordinates
            # For now, let's use a geographic approach
            timezone_id = get_timezone_by_geographic_rules(lat, lon, country)
            if timezone_id != 'UTC':
                print(f"  ✅ Found timezone via geographic rules: {timezone_id}")
                return timezone_id

    except Exception as e:
        print(f"  ⚠️  WorldTimeAPI failed: {e}")

    # Method 2: Try GeoNames with shorter timeout and better error handling
    try:
        print(f"  🔄 Trying GeoNames API...")

        url = "http://api.geonames.org/timezoneJSON"
        params = {
            'lat': lat,
            'lng': lon,
            'username': 'demo'
        }

        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            timezone_id = data.get('timezoneId')

            if timezone_id and timezone_id != 'unknown':
                print(f"  ✅ Found timezone via GeoNames: {timezone_id}")
                return timezone_id

    except Exception as e:
        print(f"  ⚠️  GeoNames failed: {e}")

    # Method 3: Geographic rules based on coordinates (most reliable)
    print(f"  🔄 Using geographic coordinate rules...")
    timezone_id = get_timezone_by_geographic_rules(lat, lon, country)
    if timezone_id != 'UTC':
        print(f"  ✅ Found timezone via coordinate rules: {timezone_id}")
        return timezone_id

    # Method 4: Enhanced country fallback
    print(f"  🔄 Using enhanced country fallback...")
    return get_timezone_fallback(country, city)


def get_timezone_by_geographic_rules(lat: float, lon: float, country: str) -> str:
    """
    Determine timezone using geographic coordinate rules.
    This is more reliable than API calls for most locations.

    Args:
        lat: Latitude
        lon: Longitude
        country: Country name for additional context

    Returns:
        Timezone identifier
    """
    # Convert country to standard format
    country_lower = country.lower()

    # Asia-Pacific region rules
    if 'philippines' in country_lower or 'phil' in country_lower:
        return 'Asia/Manila'
    elif 'taiwan' in country_lower:
        return 'Asia/Taipei'
    elif 'japan' in country_lower:
        return 'Asia/Tokyo'
    elif 'thailand' in country_lower:
        return 'Asia/Bangkok'
    elif 'vietnam' in country_lower:
        return 'Asia/Ho_Chi_Minh'
    elif 'cambodia' in country_lower:
        return 'Asia/Phnom_Penh'
    elif 'laos' in country_lower:
        return 'Asia/Vientiane'
    elif 'malaysia' in country_lower:
        return 'Asia/Kuala_Lumpur'
    elif 'singapore' in country_lower:
        return 'Asia/Singapore'
    elif 'indonesia' in country_lower:
        # Indonesia has multiple timezones
        if lon < 120:  # Western Indonesia
            return 'Asia/Jakarta'
        elif lon < 135:  # Central Indonesia
            return 'Asia/Makassar'
        else:  # Eastern Indonesia
            return 'Asia/Jayapura'
    elif 'hong kong' in country_lower:
        return 'Asia/Hong_Kong'
    elif 'china' in country_lower or 'beijing' in country_lower or 'shanghai' in country_lower:
        return 'Asia/Shanghai'
    elif 'south korea' in country_lower or 'korea' in country_lower:
        return 'Asia/Seoul'
    elif 'india' in country_lower:
        return 'Asia/Kolkata'
    elif 'australia' in country_lower:
        # Australia timezone rules based on longitude
        if lon < 130:  # Western Australia
            return 'Australia/Perth'
        elif lon < 140:  # Central Australia
            return 'Australia/Adelaide'
        else:  # Eastern Australia
            return 'Australia/Sydney'
    elif 'new zealand' in country_lower:
        return 'Pacific/Auckland'

    # Europe rules
    elif 'belgium' in country_lower:
        return 'Europe/Brussels'
    elif 'france' in country_lower:
        return 'Europe/Paris'
    elif 'germany' in country_lower:
        return 'Europe/Berlin'
    elif 'netherlands' in country_lower:
        return 'Europe/Amsterdam'
    elif 'switzerland' in country_lower:
        return 'Europe/Zurich'
    elif 'italy' in country_lower:
        return 'Europe/Rome'
    elif 'spain' in country_lower:
        return 'Europe/Madrid'
    elif 'austria' in country_lower:
        return 'Europe/Vienna'
    elif 'united kingdom' in country_lower or 'uk' in country_lower or 'britain' in country_lower:
        return 'Europe/London'
    elif 'portugal' in country_lower:
        return 'Europe/Lisbon'
    elif 'greece' in country_lower:
        return 'Europe/Athens'
    elif 'turkey' in country_lower:
        return 'Europe/Istanbul'
    elif 'russia' in country_lower:
        # Russia timezone rules based on longitude
        if lon < 40:  # Western Russia
            return 'Europe/Moscow'
        elif lon < 60:  # Ekaterinburg
            return 'Asia/Yekaterinburg'
        elif lon < 105:  # Omsk/Novosibirsk
            return 'Asia/Novosibirsk'
        elif lon < 120:  # Krasnoyarsk
            return 'Asia/Krasnoyarsk'
        elif lon < 135:  # Irkutsk
            return 'Asia/Irkutsk'
        elif lon < 150:  # Yakutsk
            return 'Asia/Yakutsk'
        else:  # Vladivostok
            return 'Asia/Vladivostok'

    # Americas rules
    elif 'united states' in country_lower or 'usa' in country_lower or 'us' in country_lower:
        # US timezone rules based on longitude
        if lon > -70:  # Eastern
            return 'America/New_York'
        elif lon > -90:  # Central
            return 'America/Chicago'
        elif lon > -115:  # Mountain
            return 'America/Denver'
        else:  # Pacific
            return 'America/Los_Angeles'
    elif 'canada' in country_lower:
        # Canada timezone rules
        if lon > -60:  # Atlantic
            return 'America/Halifax'
        elif lon > -90:  # Eastern
            return 'America/Toronto'
        elif lon > -105:  # Central
            return 'America/Winnipeg'
        elif lon > -120:  # Mountain
            return 'America/Edmonton'
        else:  # Pacific
            return 'America/Vancouver'
    elif 'mexico' in country_lower:
        return 'America/Mexico_City'
    elif 'brazil' in country_lower:
        # Brazil timezone rules
        if lon > -45:  # Brasilia time
            return 'America/Sao_Paulo'
        else:  # Amazon time
            return 'America/Manaus'
    elif 'argentina' in country_lower:
        return 'America/Argentina/Buenos_Aires'
    elif 'chile' in country_lower:
        return 'America/Santiago'
    elif 'colombia' in country_lower:
        return 'America/Bogota'
    elif 'venezuela' in country_lower:
        return 'America/Caracas'
    elif 'peru' in country_lower:
        return 'America/Lima'
    elif 'ecuador' in country_lower:
        return 'America/Guayaquil'

    # Middle East & Africa
    elif 'uae' in country_lower or 'emirates' in country_lower or 'dubai' in country_lower:
        return 'Asia/Dubai'
    elif 'saudi arabia' in country_lower or 'saudi' in country_lower:
        return 'Asia/Riyadh'
    elif 'israel' in country_lower:
        return 'Asia/Jerusalem'
    elif 'egypt' in country_lower:
        return 'Africa/Cairo'
    elif 'south africa' in country_lower:
        return 'Africa/Johannesburg'
    elif 'kenya' in country_lower:
        return 'Africa/Nairobi'
    elif 'nigeria' in country_lower:
        return 'Africa/Lagos'
    elif 'morocco' in country_lower:
        return 'Africa/Casablanca'
    elif 'ethiopia' in country_lower:
        return 'Africa/Addis_Ababa'

    # If no specific rule found, return UTC
    return 'UTC'


def get_timezone_fallback(country: str, city: str) -> str:
    """
    Fallback timezone mapping for when APIs fail.

    Args:
        country: Country name
        city: City name

    Returns:
        Timezone identifier
    """
    # Basic mapping for common countries
    country_timezones = {
        'Belgium': 'Europe/Brussels',
        'France': 'Europe/Paris',
        'Germany': 'Europe/Berlin',
        'Netherlands': 'Europe/Amsterdam',
        'Switzerland': 'Europe/Zurich',
        'Italy': 'Europe/Rome',
        'Spain': 'Europe/Madrid',
        'Austria': 'Europe/Vienna',
        'United Kingdom': 'Europe/London',
        'UK': 'Europe/London',
        'Taiwan': 'Asia/Taipei',
        'Japan': 'Asia/Tokyo',
        'South Korea': 'Asia/Seoul',
        'Korea': 'Asia/Seoul',
        'China': 'Asia/Shanghai',
        'Hong Kong': 'Asia/Hong_Kong',
        'Singapore': 'Asia/Singapore',
        'Thailand': 'Asia/Bangkok',
        'Malaysia': 'Asia/Kuala_Lumpur',
        'Indonesia': 'Asia/Jakarta',
        'Philippines': 'Asia/Manila',
        'United States': 'America/New_York',
        'USA': 'America/New_York',
        'US': 'America/New_York',
        'Canada': 'America/Toronto',
        'Mexico': 'America/Mexico_City',
        'Brazil': 'America/Sao_Paulo',
        'Argentina': 'America/Argentina/Buenos_Aires',
        'Australia': 'Australia/Sydney',
        'New Zealand': 'Pacific/Auckland',
        'UAE': 'Asia/Dubai',
        'Egypt': 'Africa/Cairo',
        'South Africa': 'Africa/Johannesburg',
        'India': 'Asia/Kolkata',
        'Pakistan': 'Asia/Karachi',
        'Russia': 'Europe/Moscow',
        'Turkey': 'Europe/Istanbul',
        'Israel': 'Asia/Jerusalem',
        'Saudi Arabia': 'Asia/Riyadh',
        'Vietnam': 'Asia/Ho_Chi_Minh',
        'Cambodia': 'Asia/Phnom_Penh',
        'Laos': 'Asia/Vientiane',
        'Myanmar': 'Asia/Yangon',
        'Bangladesh': 'Asia/Dhaka',
        'Sri Lanka': 'Asia/Colombo',
        'Nepal': 'Asia/Kathmandu',
        'Iran': 'Asia/Tehran',
        'Iraq': 'Asia/Baghdad',
        'Jordan': 'Asia/Amman',
        'Lebanon': 'Asia/Beirut',
        'Syria': 'Asia/Damascus',
        'Kenya': 'Africa/Nairobi',
        'Nigeria': 'Africa/Lagos',
        'Morocco': 'Africa/Casablanca',
        'Algeria': 'Africa/Algiers',
        'Tunisia': 'Africa/Tunis',
        'Libya': 'Africa/Tripoli',
        'Ethiopia': 'Africa/Addis_Ababa',
        'Ghana': 'Africa/Accra',
        'Senegal': 'Africa/Dakar',
        'Tanzania': 'Africa/Dar_es_Salaam',
        'Uganda': 'Africa/Kampala',
        'Zimbabwe': 'Africa/Harare',
        'Botswana': 'Africa/Gaborone',
        'Zambia': 'Africa/Lusaka',
        'Madagascar': 'Indian/Antananarivo'
    }

    timezone_id = country_timezones.get(country, 'UTC')
    print(f"  📍 Using fallback timezone: {timezone_id}")
    return timezone_id


def calculate_timezone_offset(timezone_id: str, date: datetime) -> str:
    """
    Calculate the UTC offset for a specific timezone on a specific date.
    Handles DST automatically.

    Args:
        timezone_id: Timezone identifier (e.g., 'Europe/Brussels')
        date: Date to calculate offset for

    Returns:
        UTC offset string (e.g., 'UTC+01:00')
    """
    try:
        tz = pytz.timezone(timezone_id)

        # Create a datetime in the target timezone
        localized_dt = tz.localize(date.replace(hour=12))  # Use noon to avoid DST edge cases

        # Get the UTC offset
        offset = localized_dt.utcoffset()

        # Convert to hours and minutes
        total_seconds = int(offset.total_seconds())
        hours = total_seconds // 3600
        minutes = (abs(total_seconds) % 3600) // 60

        # Format as UTC+/-HH:MM
        if total_seconds >= 0:
            return f"UTC+{hours:02d}:{minutes:02d}"
        else:
            return f"UTC{hours:03d}:{minutes:02d}"

    except Exception as e:
        print(f"⚠️  Error calculating timezone offset for {timezone_id} on {date}: {e}")
        return "UTC+00:00"


# ============================================================================
# GOOGLE MAPS PROCESSING FUNCTIONS
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
                time.sleep(0.1)  # Respectful API delay
                location_info = reverse_geocode_coordinates(lat, lon)
                geocoding_cache[coordinates] = location_info
            else:
                geocoding_cache[coordinates] = {
                    'city': 'Unknown City',
                    'country': 'Unknown Country'
                }

        # Save updated cache
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

        # Save to CSV with pipe separator and UTF-8 encoding
        print(f"💾 Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

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


# ============================================================================
# MANUAL EXCEL PROCESSING FUNCTIONS
# ============================================================================

def read_manual_timezone_excel(file_path: str) -> List[Dict]:
    """
    Read the manual timezone Excel file and parse travel records.

    Args:
        file_path: Path to the Excel file

    Returns:
        List of travel records with parsed dates and locations
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path)

        print(f"📊 Loaded {len(df)} travel records from Excel")
        print(f"📋 Columns: {list(df.columns)}")

        # Parse records
        travel_records = []

        for _, row in df.iterrows():
            try:
                # Parse date (handle different formats)
                date_str = str(row['Date'])

                # Try different date parsing approaches
                try:
                    # Try pandas date parsing first
                    date = pd.to_datetime(date_str).date()
                except:
                    # Try manual parsing for formats like "1/1/23"
                    if '/' in date_str:
                        parts = date_str.split('/')
                        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])

                        # Handle 2-digit years
                        if year < 100:
                            if year < 50:  # Assume 2000s
                                year += 2000
                            else:  # Assume 1900s
                                year += 1900

                        date = datetime(year, month, day).date()
                    else:
                        print(f"⚠️  Could not parse date: {date_str}")
                        continue

                # Extract location information
                country = str(row['Location']).strip()
                city = str(row['City']).strip() if 'City' in row else country

                travel_records.append({
                    'date': date,
                    'country': country,
                    'city': city
                })

            except Exception as e:
                print(f"⚠️  Error parsing row: {row} - {e}")
                continue

        # Sort by date
        travel_records.sort(key=lambda x: x['date'])

        print(f"✅ Successfully parsed {len(travel_records)} travel records")

        # Show sample
        if travel_records:
            print("📋 Sample records:")
            for record in travel_records[:3]:
                print(f"  • {record['date']} - {record['city']}, {record['country']}")

        return travel_records

    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return []


def create_hourly_records_from_travel_data(travel_records: List[Dict]) -> List[Dict]:
    """
    Create hourly timezone records from travel data.
    Fills gaps between travel dates.

    Args:
        travel_records: List of travel records from Excel

    Returns:
        List of hourly records with timezone and location information
    """
    global _geocoding_cache

    if not travel_records:
        print("❌ No travel records to process")
        return []

    print("🕐 Creating hourly records from travel data...")

    # Pre-process all locations to get coordinates and timezones
    print("🌍 Pre-processing location data...")
    location_data = {}

    for record in travel_records:
        location_key = f"{record['city']}, {record['country']}"

        if location_key not in location_data:
            # Check cache first
            if location_key in _geocoding_cache:
                print(f"📋 Using cached data for {location_key}")
                location_data[location_key] = _geocoding_cache[location_key]
            else:
                # Get fresh data
                coordinates, timezone_id = geocode_location(record['city'], record['country'])
                location_data[location_key] = {
                    'coordinates': coordinates,
                    'timezone_id': timezone_id
                }
                # Cache the result
                _geocoding_cache[location_key] = location_data[location_key]

                # Be respectful to APIs
                time.sleep(1)

    print(f"✅ Pre-processed {len(location_data)} unique locations")

    # Determine the overall date range
    start_date = travel_records[0]['date']
    end_date = travel_records[-1]['date']

    print(f"📅 Date range: {start_date} to {end_date}")

    # Create hourly records
    hourly_records = []
    current_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Keep track of current location
    current_location = travel_records[0]
    travel_index = 0

    while current_datetime <= end_datetime:
        # Check if we need to update location based on travel dates
        while (travel_index + 1 < len(travel_records) and
               current_datetime.date() >= travel_records[travel_index + 1]['date']):
            travel_index += 1
            current_location = travel_records[travel_index]
            print(f"📍 Location changed to {current_location['city']}, {current_location['country']} on {current_datetime.date()}")

        # Get location data
        location_key = f"{current_location['city']}, {current_location['country']}"
        loc_data = location_data[location_key]

        # Calculate timezone offset for this specific datetime
        timezone_offset = calculate_timezone_offset(loc_data['timezone_id'], current_datetime)

        # Create record
        record = {
            'timestamp': current_datetime.isoformat(),
            'timezone': timezone_offset,
            'city': current_location['city'],
            'country': current_location['country'],
            'is_home': False,  # Always False for manual entries
            'coordinates': f"geo:{loc_data['coordinates']}"
        }

        hourly_records.append(record)
        current_datetime += timedelta(hours=1)

    print(f"✅ Created {len(hourly_records):,} hourly records")

    # Show timezone summary
    timezone_summary = {}
    for record in hourly_records:
        location_key = f"{record['city']}, {record['country']}"
        if location_key not in timezone_summary:
            timezone_summary[location_key] = set()
        timezone_summary[location_key].add(record['timezone'])

    print("🌍 Timezone summary:")
    for location, timezones in timezone_summary.items():
        timezones_str = ", ".join(sorted(timezones))
        print(f"  • {location}: {timezones_str}")

    return hourly_records


def create_manual_timezone_file():
    """
    Main function to process manual timezone Excel file and create processed CSV.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing manual timezone data...")

    # Define file paths
    input_path = "files/work_files/GMT_timediff.xlsx"
    output_path = 'files/processed_files/location/manual_timezone_processed.csv'

    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"❌ Excel file not found: {input_path}")
            return False

        # Read and parse travel data
        travel_records = read_manual_timezone_excel(input_path)

        if not travel_records:
            print("❌ No valid travel records found")
            return False

        # Create hourly records
        hourly_records = create_hourly_records_from_travel_data(travel_records)

        if not hourly_records:
            print("❌ No hourly records created")
            return False

        # Convert to DataFrame
        df = pd.DataFrame(hourly_records)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV with pipe separator and UTF-8 encoding (FIXED from UTF-16)
        print(f"💾 Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed manual timezone data!")
        print(f"📊 Created {len(df):,} hourly records")
        print(f"📅 Date range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
        print(f"🌍 Countries: {', '.join(df['country'].unique())}")
        print(f"🏙️  Cities: {', '.join(df['city'].unique())}")

        # Show sample records
        print(f"\n📋 Sample records:")
        sample_df = df.head(5)[['timestamp', 'timezone', 'city', 'country']]
        for _, row in sample_df.iterrows():
            print(f"  • {row['timestamp'][:16]} | {row['timezone']} | {row['city']}, {row['country']}")

        return True

    except Exception as e:
        print(f"❌ Error processing manual timezone data: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MERGE FUNCTION (Combines Google and Manual data)
# ============================================================================

def merge_timezone_files():
    """
    Merge manual and Google timezone files into combined file.
    Returns True if successful, False otherwise.
    """
    print("🔗 Merging manual and Google timezone files...")

    manual_path = 'files/processed_files/location/manual_timezone_processed.csv'
    google_path = 'files/processed_files/location/google_processed.csv'
    merged_path = 'files/processed_files/location/combined_timezone_processed.csv'

    try:
        files_to_merge = []

        # Check which files exist (both now UTF-8 encoded)
        if os.path.exists(manual_path):
            manual_df = pd.read_csv(manual_path, sep='|', encoding='utf-8')
            manual_df['source'] = 'manual'
            files_to_merge.append(manual_df)
            print(f"✅ Loaded manual file: {len(manual_df):,} records")

        if os.path.exists(google_path):
            google_df = pd.read_csv(google_path, sep='|', encoding='utf-8')
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

        # Save merged file with UTF-8 encoding (FIXED from no encoding spec)
        print(f"💾 Saving merged file to {merged_path}...")
        deduplicated_df.to_csv(merged_path, sep='|', index=False, encoding='utf-8')

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


# ============================================================================
# UPLOAD FUNCTION
# ============================================================================

def upload_location_results():
    """
    Upload the combined timezone file to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading location results to Google Drive...")

    files_to_upload = [
        'files/processed_files/location/combined_timezone_processed.csv'
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Location results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


# ============================================================================
# MAIN PIPELINE (Standard 3 options)
# ============================================================================

def full_location_pipeline(auto_full=False):
    """
    Complete location processing pipeline with standard 3 options.

    Options:
    1. Download new data, process, and upload to Drive
    2. Process existing data and upload to Drive
    3. Upload existing processed files to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("📍 LOCATION PROCESSING PIPELINE")
    print("="*60)

    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new data, process, and upload to Drive")
        print("2. Process existing data and upload to Drive")
        print("3. Upload existing processed files to Drive")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full location pipeline...")

        # Step 1: Download Google Maps data
        download_success = download_google_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_google_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_google_files()

        # Step 3: Process Google Maps data
        print("\n📱 Processing Google Maps data...")
        google_success = create_google_file()

        # Step 4: Process Manual Excel data
        print("\n📝 Processing manual Excel data...")
        manual_success = create_manual_timezone_file()

        # Step 5: Merge both sources
        print("\n🔗 Merging location data...")
        merge_success = merge_timezone_files()

        # Step 6: Upload combined file
        if merge_success:
            print("\n☁️  Uploading to Google Drive...")
            upload_success = upload_location_results()
            success = upload_success
        else:
            print("❌ Merge failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n⚙️  Processing existing data and uploading...")

        # Process Google Maps data
        print("\n📱 Processing Google Maps data...")
        google_success = create_google_file()

        # Process Manual Excel data
        print("\n📝 Processing manual Excel data...")
        manual_success = create_manual_timezone_file()

        # Merge both sources
        print("\n🔗 Merging location data...")
        merge_success = merge_timezone_files()

        # Upload combined file
        if merge_success:
            print("\n☁️  Uploading to Google Drive...")
            upload_success = upload_location_results()
            success = upload_success
        else:
            print("❌ Merge failed, skipping upload")
            success = False

    elif choice == "3":
        print("\n☁️  Uploading existing processed files to Drive...")
        success = upload_location_results()

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Location pipeline completed successfully!")
        # Record successful run
        record_successful_run('location_combined', 'active')
    else:
        print("❌ Location pipeline failed")
    print("="*60)

    return success


# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

def process_google_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_location_pipeline(auto_full=True)
    else:
        # Process only, no upload
        google_success = create_google_file()
        manual_success = create_manual_timezone_file()
        return merge_timezone_files()


def process_manual_timezone_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    """
    if upload == "Y":
        return full_location_pipeline(auto_full=True)
    else:
        return create_manual_timezone_file()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Allow running this file directly
    print("📍 Location Processing Tool")
    print("This tool processes both Google Maps Timeline and manual Excel travel data.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_location_pipeline(auto_full=False)
