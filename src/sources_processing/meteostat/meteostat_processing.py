"""
Meteostat Source Processor

Fetches weather data from the Meteostat API based on location data from the location pipeline.
Uses coordinates from the processed location file to query weather stations.

Input: files/topic_processed_files/location/location_processed.csv
Output: files/source_processed_files/meteostat/meteostat_processed.csv
"""

import requests
import json
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

# Load Meteostat API credentials from .env file
api_key = os.environ.get('Meteostats_API_KEY')
api_host = os.environ.get('Meteostats_API_HOST')

# API limit: max 30 days per request
ONE_MONTH = pd.Timedelta('30 days')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_location_data():
    """
    Load processed location data as input for weather queries.

    Returns:
        DataFrame with columns: timestamp, city, country, latitude, longitude
        or None if file not found
    """
    location_path = 'files/topic_processed_files/location/location_processed.csv'

    if not os.path.exists(location_path):
        print(f"❌ Location file not found: {location_path}")
        print("   Run the location pipeline first.")
        return None

    print(f"📍 Loading location data from {location_path}...")
    df = pd.read_csv(location_path, sep='|', encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Parse coordinates from "geo:lat,long" format
    coords = df['coordinates'].str.replace('geo:', '', regex=False).str.split(',', expand=True)
    df['latitude'] = coords[0].astype(float)
    df['longitude'] = coords[1].astype(float)

    print(f"✅ Loaded {len(df):,} location records")
    print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    return df


def load_existing_weather_data():
    """
    Load already processed weather data to avoid re-fetching.

    Returns:
        DataFrame of existing weather data, or empty DataFrame if none exists
    """
    output_path = 'files/source_processed_files/meteostat/meteostat_processed.csv'

    if os.path.exists(output_path):
        df = pd.read_csv(output_path, sep='|', encoding='utf-8')
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
        print(f"📂 Found {len(df):,} existing weather records")
        return df

    print("📂 No existing weather data found, starting fresh")
    return pd.DataFrame()


# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_weather_code_mapping():
    """Returns mapping of Meteostat weather condition codes to descriptions."""
    return {
        1: 'Clear', 2: 'Fair', 3: 'Cloudy', 4: 'Overcast',
        5: 'Fog', 6: 'Freezing Fog',
        7: 'Light Rain', 8: 'Rain', 9: 'Heavy Rain',
        10: 'Freezing Rain', 11: 'Heavy Freezing Rain',
        12: 'Sleet', 13: 'Heavy Sleet',
        14: 'Light Snowfall', 15: 'Snowfall', 16: 'Heavy Snowfall',
        17: 'Rain Shower', 18: 'Heavy Rain Shower',
        19: 'Sleet Shower', 20: 'Heavy Sleet Shower',
        21: 'Snow Shower', 22: 'Heavy Snow Shower',
        23: 'Lightning', 24: 'Hail',
        25: 'Thunderstorm', 26: 'Heavy Thunderstorm', 27: 'Storm'
    }


def api_call_nearest_station(latitude: float, longitude: float):
    """
    Get the nearest weather station for given coordinates.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        dict: Mapping of station_id -> station_name for nearby stations
    """
    url = "https://meteostat.p.rapidapi.com/stations/nearby"
    querystring = {"lat": latitude, "lon": longitude, "radius": 100000}
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": api_host
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = json.loads(response.text)

    if 'data' not in data or not data['data']:
        return {}

    return {station['id']: station['name'].get('en', station['id']) for station in data['data']}


def api_call_weather_data(station_id: str, start_date: str, end_date: str):
    """
    Get hourly weather data for a station within a date range.

    Args:
        station_id: Meteostat station ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        list: Hourly weather data points, or empty list if failed
    """
    url = "https://meteostat.p.rapidapi.com/stations/hourly"
    querystring = {"station": station_id, "start": start_date, "end": end_date}
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": api_host
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = json.loads(response.text)

    if 'data' not in data:
        return []

    return data['data']


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def get_location_date_ranges(location_df):
    """
    Group location data into date ranges by unique location to minimize API calls.

    Args:
        location_df: DataFrame with location records

    Returns:
        DataFrame with columns: city, latitude, longitude, start_date, end_date
    """
    # Group by city and coordinates, get date range for each
    grouped = location_df.groupby(['city', 'latitude', 'longitude']).agg(
        start_date=('timestamp', 'min'),
        end_date=('timestamp', 'max')
    ).reset_index()

    # Split ranges larger than 30 days (API limit)
    result_rows = []
    for _, row in grouped.iterrows():
        start = row['start_date']
        end = row['end_date']

        while start < end:
            chunk_end = min(start + ONE_MONTH, end)
            result_rows.append({
                'city': row['city'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'start_date': start,
                'end_date': chunk_end
            })
            start = chunk_end + pd.Timedelta('1 hour')

    return pd.DataFrame(result_rows)


def fetch_weather_for_location(city, latitude, longitude, start_date, end_date):
    """
    Fetch weather data for a specific location and date range.

    Returns:
        DataFrame with weather data, or None if failed
    """
    # Get nearest station
    stations = api_call_nearest_station(latitude, longitude)
    if not stations:
        print(f"   ⚠️  No stations found near {city}")
        return None

    # Try stations until we get data
    for station_id, station_name in stations.items():
        weather_data = api_call_weather_data(
            station_id,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

        if weather_data:
            # Convert to DataFrame
            records = []
            for dp in weather_data:
                records.append({
                    'location': city,
                    'closest_station_id': station_id,
                    'closest_station_name': station_name,
                    'utc_timestamp': dp['time'],
                    'temperature': dp.get('temp'),
                    'dew_point': dp.get('dwpt'),
                    'relative_humidity_%': dp.get('rhum'),
                    'precipitation': dp.get('prcp'),
                    'snow_depth': dp.get('snow'),
                    'wind_direction': dp.get('wdir'),
                    'wind_speed': dp.get('wspd'),
                    'peak_wind_gust': dp.get('wpgt'),
                    'sea_level_air_pressure': dp.get('pres'),
                    'sunshine_total_time': dp.get('tsun'),
                    'weather_condition_code': dp.get('coco')
                })

            df = pd.DataFrame(records)
            df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
            return df

    print(f"   ⚠️  No weather data available for {city}")
    return None


def create_meteostat_file():
    """
    Main processing function: fetch weather data for all locations.

    Reads location data, determines what weather data is missing,
    fetches from Meteostat API, and saves combined results.
    """
    # Load location data
    location_df = load_location_data()
    if location_df is None:
        return False

    # Load existing weather data
    existing_df = load_existing_weather_data()

    # Determine what dates we already have weather for
    existing_timestamps = set()
    if len(existing_df) > 0:
        existing_timestamps = set(existing_df['utc_timestamp'].dt.floor('h'))

    # Filter location data to only include timestamps we don't have weather for
    location_df['timestamp_hour'] = location_df['timestamp'].dt.floor('h')
    missing_df = location_df[~location_df['timestamp_hour'].isin(existing_timestamps)]

    if len(missing_df) == 0:
        print("✅ All weather data already up to date!")
        return True

    print(f"🔍 Need weather data for {len(missing_df):,} location records")

    # Group into date ranges by location
    date_ranges = get_location_date_ranges(missing_df)
    print(f"📅 Fetching weather for {len(date_ranges)} location/date range combinations...")

    # Fetch weather for each range
    all_weather = []
    for i, row in date_ranges.iterrows():
        print(f"   [{i+1}/{len(date_ranges)}] {row['city']}: {row['start_date'].date()} to {row['end_date'].date()}")

        weather_df = fetch_weather_for_location(
            row['city'],
            row['latitude'],
            row['longitude'],
            row['start_date'],
            row['end_date']
        )

        if weather_df is not None:
            all_weather.append(weather_df)

    if not all_weather:
        print("⚠️  No new weather data fetched")
        return len(existing_df) > 0  # Success if we have existing data

    # Combine new weather data
    new_weather_df = pd.concat(all_weather, ignore_index=True)
    print(f"✅ Fetched {len(new_weather_df):,} new weather records")

    # Add weather assessment column
    weather_codes = get_weather_code_mapping()
    new_weather_df['weather_assessment'] = new_weather_df['weather_condition_code'].map(weather_codes)

    # Combine with existing data
    if len(existing_df) > 0:
        combined_df = pd.concat([existing_df, new_weather_df], ignore_index=True)
    else:
        combined_df = new_weather_df

    # Remove duplicates (by timestamp)
    combined_df = combined_df.drop_duplicates(subset=['utc_timestamp', 'location'], keep='last')
    combined_df = combined_df.sort_values('utc_timestamp')

    # Save output
    os.makedirs('files/source_processed_files/meteostat', exist_ok=True)
    output_path = 'files/source_processed_files/meteostat/meteostat_processed.csv'
    combined_df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

    print(f"💾 Saved {len(combined_df):,} total weather records to {output_path}")
    return True


# ============================================================================
# PIPELINE FUNCTION
# ============================================================================

def full_meteostat_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Meteostat SOURCE pipeline.

    Fetches weather data from Meteostat API based on processed location data.
    Requires location pipeline to have run first.

    Args:
        auto_full (bool): If True, automatically runs without prompts
        auto_process_only (bool): If True, automatically runs without prompts

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("🌤️  METEOSTAT SOURCE PIPELINE")
    print("="*60)

    # Check API credentials
    if not api_key or not api_host:
        print("❌ Meteostat API credentials not found in .env file")
        print("   Required: Meteostats_API_KEY, Meteostats_API_HOST")
        return False

    print("\n📊 Fetching weather data based on location history...")
    print("   Input: Location pipeline output (location_processed.csv)")

    try:
        success = create_meteostat_file()

        print("\n" + "="*60)
        if success:
            print("✅ Meteostat source pipeline completed!")
            print(f"📁 Output: files/source_processed_files/meteostat/meteostat_processed.csv")
        else:
            print("❌ Meteostat source pipeline failed")
        print("="*60)

        return success

    except Exception as e:
        print(f"\n❌ Meteostat processing failed: {e}")
        import traceback
        traceback.print_exc()
        print("="*60)
        return False


if __name__ == "__main__":
    print("🌤️  Meteostat Source Processing Tool")
    print("This tool fetches weather data from Meteostat API")
    print("based on your location history from the location pipeline.")
    full_meteostat_pipeline(auto_full=True)
