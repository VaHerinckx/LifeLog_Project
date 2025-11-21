"""
Shared geocoding utilities for location processing.
Used by both Google Maps and Manual Location source processors.
"""

import json
import os
import time
import requests
from datetime import datetime
from typing import Dict, Tuple
import pytz


# Geocoding cache file path (shared between all location processors)
GEOCODING_CACHE_FILE = 'files/work_files/geocoding_cache.json'


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
    Reverse geocode coordinates to get city, country, place name, and address information.
    Uses Nominatim (OpenStreetMap) API which is free and doesn't require API key.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dictionary with city, country, place_name, and address information
    """
    try:
        # Use Nominatim API (free, no API key required)
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1,
            'zoom': 18  # Higher zoom for more detailed place information
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

            # Extract place name (POI, business, landmark, etc.)
            # Priority: name field, then specific address components
            place_name = (data.get('name') or
                         address.get('amenity') or
                         address.get('shop') or
                         address.get('tourism') or
                         address.get('leisure') or
                         address.get('building') or
                         address.get('house_name') or
                         address.get('house_number'))

            # Extract formatted address
            # Build a readable address from components
            address_parts = []
            if address.get('road'):
                road = address.get('road')
                if address.get('house_number'):
                    road = f"{address.get('house_number')} {road}"
                address_parts.append(road)
            if address.get('suburb') and address.get('suburb') != city:
                address_parts.append(address.get('suburb'))
            if address.get('postcode'):
                address_parts.append(address.get('postcode'))
            if city and city != 'Unknown City':
                address_parts.append(city)

            formatted_address = ', '.join(address_parts) if address_parts else data.get('display_name', '')

            # If no place name found, use display name or formatted address
            if not place_name:
                place_name = data.get('display_name', '').split(',')[0] if data.get('display_name') else None

            return {
                'city': city,
                'country': country,
                'place_name': place_name,
                'address': formatted_address
            }
        else:
            print(f"⚠️  Geocoding failed for {lat}, {lon}: HTTP {response.status_code}")
            return {
                'city': 'Unknown City',
                'country': 'Unknown Country',
                'place_name': None,
                'address': ''
            }

    except Exception as e:
        print(f"⚠️  Error geocoding {lat}, {lon}: {e}")
        return {
            'city': 'Unknown City',
            'country': 'Unknown Country',
            'place_name': None,
            'address': ''
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
