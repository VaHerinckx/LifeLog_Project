import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pytz
import requests
import time
from pathlib import Path

from src.utils.drive_operations import upload_multiple_files, verify_drive_connection


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

                # Now get timezone using TimeZoneDB API (free tier: 1000 requests/month)
                # Alternative: use coordinates to get timezone
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


def get_city_coordinates(city: str, country: str = None) -> str:
    """
    Get coordinates for any city using programmatic geocoding.
    This function now acts as a wrapper for the new geocoding system.

    Args:
        city: City name
        country: Country name (optional, but recommended for accuracy)

    Returns:
        Coordinate string in 'lat,lon' format
    """
    if country:
        coordinates, _ = geocode_location(city, country)
        return coordinates
    else:
        # Fallback for when country is not provided
        return geocode_location(city, "")[0]


def get_timezone_for_location(country: str, city: str = None) -> str:
    """
    Get timezone identifier for a given location using programmatic lookup.

    Args:
        country: Country name
        city: City name (optional, for more precision)

    Returns:
        Timezone identifier (e.g., 'Europe/Brussels')
    """
    if city:
        # Use full geocoding to get precise timezone
        _, timezone_id = geocode_location(city, country)
        return timezone_id
    else:
        # Use fallback mapping
        return get_timezone_fallback(country, city or "")


# Add a geocoding cache to avoid repeated API calls
_geocoding_cache = {}


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
            'is_home': False,  # Always False as requested
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

        # Save to CSV with pipe separator (same format as Google data)
        print(f"💾 Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False, encoding='utf-16')

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


def upload_manual_timezone_results():
    """
    Upload the processed manual timezone file to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading manual timezone results to Google Drive...")

    files_to_upload = [
        'files/processed_files/location/manual_timezone_processed.csv'
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Manual timezone results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


def process_manual_timezone_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    """
    if upload == "Y":
        return full_manual_timezone_pipeline(auto_full=True)
    else:
        return create_manual_timezone_file()


def full_manual_timezone_pipeline(auto_full=False):
    """
    Complete manual timezone pipeline.

    Options:
    1. Process Excel file and upload to Drive
    2. Process Excel file only

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🕐 MANUAL TIMEZONE PROCESSING PIPELINE")
    print("="*60)

    if auto_full:
        print("🤖 Auto mode: Processing Excel file and uploading...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Process Excel file and upload to Drive")
        print("2. Process Excel file only")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Processing Excel file and uploading...")

        # Process the Excel file
        process_success = create_manual_timezone_file()

        if process_success:
            # Upload results
            upload_success = upload_manual_timezone_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n⚙️  Processing Excel file only...")
        success = create_manual_timezone_file()

    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Manual timezone pipeline completed successfully!")
    else:
        print("❌ Manual timezone pipeline failed")
    print("="*60)

    return success


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

        # Sort by timestamp
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp')

        # Remove duplicates (prefer Google data over manual)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')

        # Convert timestamp back to string format
        combined_df['timestamp'] = combined_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Remove source column
        combined_df = combined_df.drop('source', axis=1)

        # Save merged file
        combined_df.to_csv(merged_path, sep='|', index=False, encoding='utf-16')

        print(f"✅ Successfully merged timezone files!")
        print(f"📊 Combined records: {len(combined_df):,}")
        print(f"📅 Date range: {combined_df['timestamp'].min()[:10]} to {combined_df['timestamp'].max()[:10]}")
        print(f"💾 Saved to: {merged_path}")

        return True

    except Exception as e:
        print(f"❌ Error merging timezone files: {e}")
        return False


if __name__ == "__main__":
    # Allow running this file directly
    print("🕐 Manual Timezone Processing Tool")
    print("This tool processes Excel travel data into hourly timezone records.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_manual_timezone_pipeline(auto_full=False)
