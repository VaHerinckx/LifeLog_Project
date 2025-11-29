import pandas as pd
import numpy as np
import os
import json
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.apple.apple_processing import full_apple_pipeline, download_apple_data, move_apple_files
from src.sources_processing.nutrilio.nutrilio_processing import full_nutrilio_pipeline, download_nutrilio_data, move_nutrilio_files
from src.topic_processing.location.location_processing import full_location_pipeline
from src.sources_processing.google_maps.google_maps_processing import download_google_data, move_google_files
from src.sources_processing.offscreen.offscreen_processing import full_offscreen_pipeline, download_offscreen_data, move_offscreen_files


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_apple_minute_data():
    """
    Load Apple Health minute-level data.

    Returns:
        DataFrame: Apple Health data with minute-level granularity
    """
    apple_path = 'files/source_processed_files/apple/apple_processed.csv'

    if not os.path.exists(apple_path):
        print(f"❌ Apple Health file not found: {apple_path}")
        return None

    print(f"📱 Loading Apple Health data...")
    df = pd.read_csv(apple_path, sep='|', encoding='utf-8', low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    print(f"✅ Loaded Apple Health: {len(df):,} minute-level records")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def load_nutrilio_mental_health():
    """
    Load Nutrilio mental health data (daily summary rows only).
    Columns: sleep_quality, sleep_quality_text, dream_description, dreams,
             sleep_rest_feeling, fitness_feeling, overall_evaluation, notes_summary

    Returns:
        DataFrame: Nutrilio mental health data with daily granularity
    """
    nutrilio_path = 'files/source_processed_files/nutrilio/nutrilio_health_processed.csv'

    if not os.path.exists(nutrilio_path):
        print(f"❌ Nutrilio health file not found: {nutrilio_path}")
        return None

    print(f"🥗 Loading Nutrilio mental health data...")
    df_daily = pd.read_csv(nutrilio_path, sep='|', encoding='utf-8')

    # File already contains only daily summary rows (pre-filtered by Nutrilio source processor)

    # Parse date
    df_daily['date'] = pd.to_datetime(df_daily['date']).dt.normalize()

    # Select mental health columns (excluding productivity)
    columns_to_keep = [
        'date',
        'sleep_-_quality',
        'sleep_-_quality_text',
        'dream_description',
        'dreams',
        'sleep_-_rest_feeling_(points)',
        'fitness_feeling_(points)',
        'overall_evaluation_(points)',
        'notes_summary',
        'weight_(kg)'
    ]

    # Filter columns that exist
    existing_columns = [col for col in columns_to_keep if col in df_daily.columns]
    df_daily = df_daily[existing_columns].copy()

    # Rename columns to cleaner names
    rename_dict = {
        'sleep_-_quality': 'sleep_quality',
        'sleep_-_quality_text': 'sleep_quality_text',
        'sleep_-_rest_feeling_(points)': 'sleep_rest_feeling',
        'fitness_feeling_(points)': 'fitness_feeling',
        'overall_evaluation_(points)': 'overall_evaluation'
    }

    df_daily = df_daily.rename(columns=rename_dict)

    print(f"✅ Loaded Nutrilio: {len(df_daily):,} daily mental health records")
    print(f"   Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")

    return df_daily


def load_location_data():
    """
    Load location data with location_type enrichment.

    Returns:
        DataFrame: Location data with minute-level granularity
    """
    location_path = 'files/topic_processed_files/location/location_processed.csv'

    if not os.path.exists(location_path):
        print(f"⚠️  Location file not found: {location_path}")
        return None

    print(f"📍 Loading location data...")
    df = pd.read_csv(location_path, sep='|', encoding='utf-8')

    # Parse timestamp to datetime (timezone-naive guaranteed by upstream processing)
    df['date'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S')

    print(f"✅ Loaded location: {len(df):,} records")
    if len(df) > 0:
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def load_screentime_data():
    """
    Load screen time data.

    Returns:
        DataFrame: Screen time data with minute-level granularity, or None if not available
    """
    screen_path = 'files/source_processed_files/offscreen/offscreen_processed.csv'

    if not os.path.exists(screen_path):
        print(f"⚠️  Screen time file not found: {screen_path}")
        return None

    try:
        print(f"📱 Loading screen time data...")
        df = pd.read_csv(screen_path, sep='|', encoding='utf-8')
        df['date'] = pd.to_datetime(df['date']).dt.floor('T')  # Floor to minute

        print(f"✅ Loaded screen time: {len(df):,} minute-level records")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    except Exception as e:
        print(f"❌ Error loading screen time data: {e}")
        return None


def load_google_maps_data():
    """
    Load Google Maps minute-level data for richer location details.

    Returns:
        DataFrame: Google Maps data with minute-level granularity, or None if not available
    """
    gmaps_path = 'files/source_processed_files/google_maps/google_maps_minute_processed.csv'

    if not os.path.exists(gmaps_path):
        print(f"⚠️  Google Maps file not found: {gmaps_path}")
        return None

    try:
        print(f"🗺️  Loading Google Maps data...")
        df = pd.read_csv(gmaps_path, sep='|', encoding='utf-8', low_memory=False)
        df['date'] = pd.to_datetime(df['timestamp']).dt.floor('T')  # Floor to minute

        print(f"✅ Loaded Google Maps: {len(df):,} minute-level records")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    except Exception as e:
        print(f"❌ Error loading Google Maps data: {e}")
        return None


# ============================================================================
# HOURLY AGGREGATION FUNCTIONS
# ============================================================================

def get_time_period(hour):
    """Convert hour (0-23) to time period string."""
    if 6 <= hour <= 11:
        return 'MORNING'
    elif 12 <= hour <= 17:
        return 'AFTERNOON'
    elif 18 <= hour <= 23:
        return 'EVENING'
    else:  # 0-5
        return 'NIGHT'


def aggregate_to_hourly(minute_df, google_maps_df=None, existing_hourly_path=None, nutrilio_cutoff_date=None):
    """
    Aggregate minute-level data to hourly with rich JSON detail columns.

    Supports incremental processing: if existing_hourly_path and nutrilio_cutoff_date
    are provided, only reprocesses data from cutoff_date onwards. Historical data
    (before the cutoff) is preserved from the existing file.

    Args:
        minute_df: Minute-level merged DataFrame (Apple Health + Location + Screen)
        google_maps_df: Optional Google Maps minute-level DataFrame for richer location data
        existing_hourly_path: Path to existing hourly file (for incremental processing)
        nutrilio_cutoff_date: Date object - only reprocess from this date onwards

    Returns:
        DataFrame: Hourly aggregated DataFrame with JSON detail columns
    """
    print("\n⏰ Aggregating to hourly level with JSON details...")

    # Work with a copy
    df = minute_df.copy()

    # Check for incremental processing
    historical_df = None
    if existing_hourly_path and nutrilio_cutoff_date and os.path.exists(existing_hourly_path):
        print(f"   ⚡ Incremental mode: Keeping data before {nutrilio_cutoff_date}")

        # Load existing hourly file
        existing_hourly = pd.read_csv(existing_hourly_path, sep='|', encoding='utf-8')
        existing_hourly['datetime'] = pd.to_datetime(existing_hourly['datetime'])

        # Split: keep historical records (before cutoff date)
        historical_df = existing_hourly[existing_hourly['datetime'].dt.date < nutrilio_cutoff_date].copy()

        # Filter minute_df to only process data from cutoff date onwards
        df = df[df['date'].dt.date >= nutrilio_cutoff_date].copy()

        print(f"      Historical records preserved: {len(historical_df):,}")
        print(f"      Minute records to reprocess: {len(df):,}")

    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Create hour column (floor to hour)
    df['hour_dt'] = df['date'].dt.floor('H')

    # Merge Google Maps data if available (for richer location details)
    if google_maps_df is not None:
        print("   🗺️  Merging Google Maps data for richer location details...")
        gm = google_maps_df.copy()
        gm['hour_dt'] = gm['date'].dt.floor('H')

        # Select Google Maps columns to merge (prefix with gm_ to avoid conflicts)
        gm_cols = ['hour_dt', 'place_name', 'address', 'activity_type', 'distance_meters', 'is_moving', 'record_type']
        gm_cols = [c for c in gm_cols if c in gm.columns]
        gm_subset = gm[gm_cols].copy()

        # Rename to avoid conflicts
        gm_subset = gm_subset.rename(columns={
            'place_name': 'gm_place_name',
            'address': 'gm_address',
            'activity_type': 'gm_activity_type',
            'distance_meters': 'gm_distance_meters',
            'is_moving': 'gm_is_moving',
            'record_type': 'gm_record_type'
        })

        # Merge on hour (many-to-many, will aggregate later)
        df = df.merge(gm_subset, on='hour_dt', how='left')
        print(f"      Merged {len(google_maps_df):,} Google Maps records")

    # Initialize result list
    hourly_records = []

    # Group by hour
    print("   📊 Processing hourly aggregations...")
    grouped = df.groupby('hour_dt')
    total_hours = len(grouped)

    for i, (hour_dt, group) in enumerate(grouped):
        if i % 5000 == 0:
            print(f"      Processing hour {i:,}/{total_hours:,}...")

        record = {
            'datetime': hour_dt,
            'date': hour_dt.date(),
            'hour': hour_dt.hour,
            'weekday': hour_dt.weekday(),  # Monday=0
            'time_period': get_time_period(hour_dt.hour)
        }

        # =====================================================================
        # FLAT COLUMNS - Simple aggregations for filtering/heatmap
        # =====================================================================

        # Movement metrics (sum)
        record['total_steps'] = group['step_count'].sum() if 'step_count' in group.columns else 0
        record['total_distance_m'] = group['walking_dist'].sum() * 1000 if 'walking_dist' in group.columns else 0  # km to m
        record['total_flights_climbed'] = group['flights_climbed'].sum() if 'flights_climbed' in group.columns else 0
        record['total_active_energy'] = group['active_energy'].sum() if 'active_energy' in group.columns else 0
        record['total_resting_energy'] = group['resting_energy'].sum() if 'resting_energy' in group.columns else 0

        # Body metrics (average, excluding nulls)
        record['avg_heart_rate'] = group['heart_rate'].mean() if 'heart_rate' in group.columns and group['heart_rate'].notna().any() else None
        record['avg_walking_speed'] = group['walking_speed'].mean() if 'walking_speed' in group.columns and group['walking_speed'].notna().any() else None
        record['avg_step_length'] = group['step_length'].mean() if 'step_length' in group.columns and group['step_length'].notna().any() else None
        record['avg_body_weight'] = group['body_weight'].mean() if 'body_weight' in group.columns and group['body_weight'].notna().any() else None
        record['avg_body_fat_percent'] = group['body_fat_percent'].mean() if 'body_fat_percent' in group.columns and group['body_fat_percent'].notna().any() else None
        record['avg_audio_exposure'] = group['audio_exposure'].mean() if 'audio_exposure' in group.columns and group['audio_exposure'].notna().any() else None

        # Sleep metrics
        sleep_minutes = 0
        if 'sleep_analysis' in group.columns:
            sleep_rows = group[group['sleep_analysis'].notna() & (group['sleep_analysis'] != '')]
            sleep_minutes = len(sleep_rows)
        record['total_sleep_minutes'] = sleep_minutes

        # Screen time metrics
        record['total_screen_minutes'] = group['screen_time'].sum() / 60 if 'screen_time' in group.columns else 0  # seconds to minutes
        record['total_pickups'] = int(group['pickups'].sum()) if 'pickups' in group.columns else 0
        record['screen_before_sleep_minutes'] = group['within_hour_before_sleep'].sum() / 60 if 'within_hour_before_sleep' in group.columns else 0

        # Location dominant values
        if 'city' in group.columns and group['city'].notna().any():
            record['dominant_city'] = group['city'].mode().iloc[0] if len(group['city'].mode()) > 0 else None
        else:
            record['dominant_city'] = None

        if 'country' in group.columns and group['country'].notna().any():
            record['dominant_country'] = group['country'].mode().iloc[0] if len(group['country'].mode()) > 0 else None
        else:
            record['dominant_country'] = None

        if 'timezone' in group.columns and group['timezone'].notna().any():
            record['timezone'] = group['timezone'].mode().iloc[0] if len(group['timezone'].mode()) > 0 else None
        else:
            record['timezone'] = None

        # Activity type (from Google Maps if available)
        if 'gm_activity_type' in group.columns and group['gm_activity_type'].notna().any():
            record['dominant_activity'] = group['gm_activity_type'].mode().iloc[0] if len(group['gm_activity_type'].mode()) > 0 else None
        else:
            record['dominant_activity'] = None

        # Boolean flags
        if 'is_home' in group.columns:
            home_minutes = (group['is_home'] == True).sum() | (group['is_home'] == 'True').sum()
            record['is_home_hour'] = home_minutes > len(group) / 2
        else:
            record['is_home_hour'] = None

        record['is_sleeping_hour'] = sleep_minutes > 30  # More than half hour asleep
        record['is_moving_hour'] = record['total_steps'] > 100 or record['total_distance_m'] > 50

        # =====================================================================
        # JSON COLUMNS - Rich detail aggregations
        # =====================================================================

        # 1. locations_json - Location breakdown
        locations = []
        if 'city' in group.columns:
            loc_groups = group.groupby(['city', 'country'], dropna=False)
            for (city, country), loc_group in loc_groups:
                if pd.isna(city):
                    continue
                loc_entry = {
                    'city': city,
                    'country': country if pd.notna(country) else None,
                    'minutes': len(loc_group),
                    'is_home': bool((loc_group['is_home'] == True).any() | (loc_group['is_home'] == 'True').any()) if 'is_home' in loc_group.columns else False
                }
                # Add Google Maps details if available
                if 'gm_place_name' in loc_group.columns:
                    place_names = loc_group['gm_place_name'].dropna().unique()
                    if len(place_names) > 0:
                        loc_entry['place_name'] = place_names[0]  # First place name
                if 'gm_address' in loc_group.columns:
                    addresses = loc_group['gm_address'].dropna().unique()
                    if len(addresses) > 0:
                        loc_entry['address'] = addresses[0]
                if 'coordinates' in loc_group.columns:
                    coords = loc_group['coordinates'].dropna().unique()
                    if len(coords) > 0:
                        loc_entry['coordinates'] = coords[0]
                if 'location_type' in loc_group.columns:
                    loc_types = loc_group['location_type'].dropna().unique()
                    if len(loc_types) > 0:
                        loc_entry['location_type'] = loc_types[0]
                locations.append(loc_entry)
        record['locations_json'] = json.dumps(locations) if locations else '[]'

        # 2. activities_json - Activity/Movement breakdown (from Google Maps)
        activities = {}
        if 'gm_activity_type' in group.columns:
            activity_counts = group['gm_activity_type'].value_counts()
            for activity, count in activity_counts.items():
                if pd.isna(activity) or activity == '':
                    continue
                activity_group = group[group['gm_activity_type'] == activity]
                activities[activity] = {
                    'minutes': int(count),
                    'distance_m': float(activity_group['gm_distance_meters'].sum()) if 'gm_distance_meters' in activity_group.columns else 0
                }
        # Add stationary time (minutes without activity type)
        if 'gm_activity_type' in group.columns:
            stationary_count = group['gm_activity_type'].isna().sum()
            if stationary_count > 0:
                activities['stationary'] = {'minutes': int(stationary_count), 'distance_m': 0}
        record['activities_json'] = json.dumps(activities) if activities else '{}'

        # 3. sleep_json - Sleep stage breakdown
        sleep_data = {}
        if 'sleep_analysis' in group.columns:
            sleep_counts = group['sleep_analysis'].value_counts()
            phase_mapping = {
                'Deep sleep': 'deep',
                'REM sleep': 'rem',
                'Core sleep': 'core',
                'Awake': 'awake',
                'In bed': 'in_bed',
                'Unspecified': 'unspecified'
            }
            for phase, count in sleep_counts.items():
                if pd.isna(phase) or phase == '':
                    continue
                key = phase_mapping.get(phase, phase.lower().replace(' ', '_'))
                sleep_data[key] = int(count)
            if sleep_data:
                sleep_data['total'] = sum(sleep_data.values())
        record['sleep_json'] = json.dumps(sleep_data) if sleep_data else '{}'

        # 4. screen_json - Screen time breakdown
        screen_data = {
            'total_minutes': round(record['total_screen_minutes'], 1),
            'total_pickups': record['total_pickups'],
            'before_sleep_minutes': round(record['screen_before_sleep_minutes'], 1),
            'is_before_sleep_hour': bool(group['is_within_hour_before_sleep'].any()) if 'is_within_hour_before_sleep' in group.columns else False
        }
        record['screen_json'] = json.dumps(screen_data)

        # 5. body_metrics_json - Body measurements
        body_metrics = {}
        if 'heart_rate' in group.columns and group['heart_rate'].notna().any():
            hr_data = group['heart_rate'].dropna()
            body_metrics['heart_rate'] = {
                'min': float(hr_data.min()),
                'max': float(hr_data.max()),
                'avg': round(float(hr_data.mean()), 1),
                'readings': len(hr_data)
            }
        if 'body_weight' in group.columns and group['body_weight'].notna().any():
            bw_data = group['body_weight'].dropna()
            body_metrics['body_weight'] = {
                'avg': round(float(bw_data.mean()), 1),
                'readings': len(bw_data)
            }
        if 'body_fat_percent' in group.columns and group['body_fat_percent'].notna().any():
            bf_data = group['body_fat_percent'].dropna()
            body_metrics['body_fat_percent'] = {
                'avg': round(float(bf_data.mean()), 1),
                'readings': len(bf_data)
            }
        if 'audio_exposure' in group.columns and group['audio_exposure'].notna().any():
            ae_data = group['audio_exposure'].dropna()
            body_metrics['audio_exposure'] = {
                'avg': round(float(ae_data.mean()), 1),
                'max': float(ae_data.max()),
                'readings': len(ae_data)
            }
        record['body_metrics_json'] = json.dumps(body_metrics) if body_metrics else '{}'

        # 6. movement_metrics_json - Movement statistics
        movement_metrics = {
            'total_steps': round(record['total_steps'], 0),
            'total_distance_m': round(record['total_distance_m'], 1),
            'total_flights_climbed': round(record['total_flights_climbed'], 1),
            'avg_step_length_cm': round(record['avg_step_length'] * 100, 1) if record['avg_step_length'] else None,
            'avg_walking_speed_kmh': round(record['avg_walking_speed'], 1) if record['avg_walking_speed'] else None,
            'total_active_energy_kcal': round(record['total_active_energy'], 1),
            'total_resting_energy_kcal': round(record['total_resting_energy'], 1)
        }
        record['movement_metrics_json'] = json.dumps(movement_metrics)

        # 7. countries_visited_json - Countries in that hour
        countries = []
        if 'country' in group.columns:
            countries = group['country'].dropna().unique().tolist()
        record['countries_visited_json'] = json.dumps(countries)

        # 8. cities_visited_json - Cities in that hour
        cities = []
        if 'city' in group.columns:
            cities = group['city'].dropna().unique().tolist()
        record['cities_visited_json'] = json.dumps(cities)

        # 9. timezones_json - Timezone changes
        timezones = []
        if 'timezone' in group.columns:
            timezones = group['timezone'].dropna().unique().tolist()
        record['timezones_json'] = json.dumps(timezones)

        hourly_records.append(record)

    # Create DataFrame
    hourly_df = pd.DataFrame(hourly_records)

    # Merge historical data if incremental processing was used
    if historical_df is not None and len(historical_df) > 0:
        print(f"   🔗 Merging {len(historical_df):,} historical records with {len(hourly_df):,} new records...")
        hourly_df = pd.concat([historical_df, hourly_df], ignore_index=True)

    # Sort by datetime descending
    hourly_df = hourly_df.sort_values('datetime', ascending=False)

    print(f"✅ Hourly aggregation complete: {len(hourly_df):,} hourly records")

    return hourly_df


# ============================================================================
# MERGE FUNCTIONS
# ============================================================================

def merge_minute_level_data(apple_df, location_df, screen_df):
    """
    Merge Location and Screen Time into Apple Health at minute-level.

    Args:
        apple_df: Apple Health minute-level DataFrame
        location_df: Location minute-level DataFrame
        screen_df: Screen time minute-level DataFrame (can be None)

    Returns:
        DataFrame: Merged minute-level DataFrame
    """
    print("\n🔗 Merging minute-level data...")

    # Start with Apple data
    merged_df = apple_df.copy()

    # Merge location data (HOURLY → MINUTE mapping)
    if location_df is not None:
        location_cols = ['date', 'timezone', 'city', 'country', 'is_home', 'coordinates', 'location_type']
        existing_loc_cols = [col for col in location_cols if col in location_df.columns]

        # Create hour column for merging (floor to hour)
        # Location data is hourly, so 09:00:00 applies to all minutes 09:00-09:59
        merged_df['hour'] = merged_df['date'].dt.floor('H')
        location_df_copy = location_df.copy()
        location_df_copy['hour'] = location_df_copy['date'].dt.floor('H')

        # Merge on hour (maps all 60 minutes to their hourly location record)
        merge_cols = ['hour'] + [col for col in existing_loc_cols if col != 'date']
        merged_df = merged_df.merge(
            location_df_copy[merge_cols],
            on='hour',
            how='left'
        )

        # Drop helper column
        merged_df = merged_df.drop(columns=['hour'])

        print(f"✅ Merged location data: {len(location_df):,} hourly records → {len(merged_df):,} minute records")

    # Merge screen time data
    if screen_df is not None:
        screen_cols = ['date', 'screen_time', 'pickups', 'within_hour_before_sleep', 'is_within_hour_before_sleep']
        existing_screen_cols = [col for col in screen_cols if col in screen_df.columns]

        merged_df = merged_df.merge(
            screen_df[existing_screen_cols],
            on='date',
            how='left'
        )
        print(f"✅ Merged screen time data: {len(screen_df):,} records")

    print(f"✅ Minute-level merge complete: {len(merged_df):,} records")

    return merged_df


def aggregate_to_daily(minute_df):
    """
    Aggregate minute-level data to daily summaries.

    Args:
        minute_df: Minute-level merged DataFrame

    Returns:
        DataFrame: Daily aggregated DataFrame
    """
    print("\n📊 Aggregating to daily level...")

    # Extract date only (remove time)
    minute_df_copy = minute_df.copy()
    minute_df_copy['date_only'] = minute_df_copy['date'].dt.date
    minute_df_copy['date_only'] = pd.to_datetime(minute_df_copy['date_only'])

    # Aggregate Apple activity metrics
    agg_dict = {}

    # Activity metrics (sum)
    for col in ['step_count', 'walking_dist', 'flights_climbed', 'resting_energy', 'active_energy']:
        if col in minute_df_copy.columns:
            agg_dict[col] = 'sum'

    # Body metrics (mean)
    for col in ['step_length', 'walking_speed', 'heart_rate', 'body_weight', 'body_fat_percent', 'audio_exposure']:
        if col in minute_df_copy.columns:
            agg_dict[col] = 'mean'

    # Perform aggregation
    daily_df = minute_df_copy.groupby('date_only').agg(agg_dict).reset_index()

    # Rename columns
    rename_dict = {
        'date_only': 'date',
        'step_count': 'daily_steps',
        'step_length': 'avg_step_length',
        'walking_dist': 'daily_walking_dist',
        'flights_climbed': 'daily_flights_climbed',
        'walking_speed': 'avg_walking_speed',
        'resting_energy': 'daily_resting_energy',
        'active_energy': 'daily_active_energy',
        'heart_rate': 'avg_heart_rate',
        'body_weight': 'avg_body_weight',
        'body_fat_percent': 'avg_body_fat_percent',
        'audio_exposure': 'avg_audio_exposure'
    }

    daily_df = daily_df.rename(columns=rename_dict)

    # Handle sleep_analysis (pivot to count minutes per phase)
    if 'sleep_analysis' in minute_df_copy.columns:
        print("   💤 Processing sleep phases...")
        sleep_df = minute_df_copy[minute_df_copy['sleep_analysis'].notna()].copy()

        if len(sleep_df) > 0:
            # Pivot to get minutes per sleep phase
            sleep_pivot = sleep_df.groupby(['date_only', 'sleep_analysis']).size().unstack(fill_value=0)
            sleep_pivot = sleep_pivot.reset_index()

            # Rename columns based on actual phases present
            phase_mapping = {
                'Deep sleep': 'sleep_deep_sleep_minutes',
                'REM sleep': 'sleep_rem_sleep_minutes',
                'Core sleep': 'sleep_core_sleep_minutes',
                'Awake': 'sleep_awake_minutes',
                'In bed': 'sleep_in_bed_minutes',
                'Unspecified': 'sleep_unspecified_minutes'
            }

            # Rename columns that exist
            rename_sleep = {'date_only': 'date'}
            for old_name, new_name in phase_mapping.items():
                if old_name in sleep_pivot.columns:
                    rename_sleep[old_name] = new_name

            sleep_pivot = sleep_pivot.rename(columns=rename_sleep)

            # Calculate total sleep minutes (sum of all phases)
            sleep_cols = [col for col in sleep_pivot.columns if col.startswith('sleep_') and col.endswith('_minutes')]
            if sleep_cols:
                sleep_pivot['sleep_minutes'] = sleep_pivot[sleep_cols].sum(axis=1)

            # Merge sleep data with daily aggregation
            daily_df = daily_df.merge(sleep_pivot, on='date', how='left')

            print(f"      Sleep phases tracked: {', '.join([col.replace('sleep_', '').replace('_minutes', '') for col in sleep_cols])}")

    # Aggregate location metrics
    if 'city' in minute_df_copy.columns:
        print("   📍 Processing location metrics...")
        location_daily = minute_df_copy.groupby('date_only').agg({
            'city': ['nunique', lambda x: x.mode()[0] if len(x.mode()) > 0 and len(x) > 0 else None],
            'country': 'nunique',
            'timezone': ['nunique', lambda x: x.mode()[0] if len(x.mode()) > 0 and len(x) > 0 else None]
        }).reset_index()

        location_daily.columns = [
            'date',
            'cities_visited',
            'dominant_city',
            'countries_visited',
            'timezone_changes',
            'dominant_timezone'
        ]

        # Calculate location_type percentages
        if 'location_type' in minute_df_copy.columns:
            location_type_daily = minute_df_copy.groupby('date_only')['location_type'].apply(
                lambda x: x.value_counts(normalize=True) * 100
            ).unstack(fill_value=0).reset_index()

            location_type_daily.columns = ['date'] + [f'percent_time_{col}' for col in location_type_daily.columns[1:]]

            # Merge location type percentages
            location_daily = location_daily.merge(location_type_daily, on='date', how='left')

        # Merge with main daily dataframe
        daily_df = daily_df.merge(location_daily, on='date', how='left')

    # Aggregate screen time metrics
    if 'screen_time' in minute_df_copy.columns:
        print("   📱 Processing screen time metrics...")
        screen_daily = minute_df_copy.groupby('date_only').agg({
            'screen_time': 'sum',
            'pickups': 'sum',
            'within_hour_before_sleep': 'sum'
        }).reset_index()

        # Rename aggregation columns
        screen_daily = screen_daily.rename(columns={
            'date_only': 'date',
            'pickups': 'total_pickups'
        })

        # Convert screen_time from seconds to minutes and hours
        screen_daily['total_screen_minutes'] = (screen_daily['screen_time'] / 60).round(0).astype(int)
        screen_daily['total_screen_hours'] = (screen_daily['screen_time'] / 3600).round(2)

        # Convert sleep screen time from seconds to minutes
        screen_daily['screen_before_sleep_minutes'] = (screen_daily['within_hour_before_sleep'] / 60).round(0).astype(int)

        # Drop intermediate columns
        screen_daily = screen_daily.drop(columns=['screen_time', 'within_hour_before_sleep'])

        # Merge with main daily dataframe
        daily_df = daily_df.merge(screen_daily, on='date', how='left')

    print(f"✅ Daily aggregation complete: {len(daily_df):,} daily records")

    return daily_df


def merge_daily_nutrilio(daily_df, nutrilio_df):
    """
    Merge Nutrilio mental health data into daily aggregated data.

    Args:
        daily_df: Daily aggregated DataFrame
        nutrilio_df: Nutrilio mental health DataFrame (daily)

    Returns:
        DataFrame: Final daily DataFrame with Nutrilio data merged
    """
    if nutrilio_df is None:
        print("⚠️  No Nutrilio data to merge")
        return daily_df

    print("\n🔗 Merging Nutrilio mental health data...")

    # Merge on date
    final_df = daily_df.merge(nutrilio_df, on='date', how='left')

    # Coalesce weight: Apple Health primary, Nutrilio fallback
    if 'avg_body_weight' in final_df.columns and 'weight_(kg)' in final_df.columns:
        # Count how many gaps will be filled
        apple_nulls = final_df['avg_body_weight'].isna().sum()
        nutrilio_fills = final_df[final_df['avg_body_weight'].isna() & final_df['weight_(kg)'].notna()].shape[0]

        # Fill Apple Health gaps with Nutrilio data
        final_df['avg_body_weight'] = final_df['avg_body_weight'].fillna(final_df['weight_(kg)'])

        # Drop the separate Nutrilio weight column (now merged into avg_body_weight)
        final_df = final_df.drop(columns=['weight_(kg)'])

        if nutrilio_fills > 0:
            print(f"   ⚖️  Filled {nutrilio_fills} body weight gaps using Nutrilio data (out of {apple_nulls} missing)")

    print(f"✅ Nutrilio data merged: {len(nutrilio_df):,} daily mental health records added")

    return final_df


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def create_health_files():
    """
    Main processing function that merges all health data sources
    and generates minute-level, daily aggregated, and hourly aggregated files.

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("🏥 HEALTH DATA PROCESSING - MULTI-SOURCE MERGE")
    print("="*70)

    try:
        # Step 1: Load all data sources
        print("\n📥 STEP 1: Loading data sources...")
        apple_df = load_apple_minute_data()

        if apple_df is None:
            print("❌ Cannot proceed without Apple Health data")
            return False

        nutrilio_df = load_nutrilio_mental_health()
        location_df = load_location_data()
        screen_df = load_screentime_data()
        google_maps_df = load_google_maps_data()  # Load Google Maps for hourly aggregation

        # Determine cutoff date for incremental hourly processing
        # Only data from Nutrilio date onwards needs reprocessing (older Apple Health data is static)
        nutrilio_cutoff_date = None
        existing_hourly_path = 'files/website_files/health/health_page_hourly.csv'

        if nutrilio_df is not None and os.path.exists(existing_hourly_path):
            nutrilio_cutoff_date = nutrilio_df['date'].min().date()
            print(f"   ⚡ Incremental processing enabled: Cutoff date = {nutrilio_cutoff_date}")
        else:
            if nutrilio_df is None:
                print("   ℹ️  Full processing: No Nutrilio data available")
            else:
                print("   ℹ️  Full processing: No existing hourly file found")

        # Step 2: Merge at minute-level
        print("\n🔗 STEP 2: Merging minute-level data...")
        minute_merged_df = merge_minute_level_data(apple_df, location_df, screen_df)

        # Step 3: Aggregate to daily
        print("\n📊 STEP 3: Aggregating to daily level...")
        daily_df = aggregate_to_daily(minute_merged_df)

        # Step 4: Merge Nutrilio (daily)
        print("\n🔗 STEP 4: Merging Nutrilio mental health data...")
        final_daily_df = merge_daily_nutrilio(daily_df, nutrilio_df)

        # Step 5: Aggregate to hourly (with Google Maps data for richer details)
        # Uses incremental processing if existing hourly file and Nutrilio data are available
        print("\n⏰ STEP 5: Aggregating to hourly level with JSON details...")
        hourly_df = aggregate_to_hourly(
            minute_merged_df,
            google_maps_df,
            existing_hourly_path=existing_hourly_path,
            nutrilio_cutoff_date=nutrilio_cutoff_date
        )

        # Step 6: Enforce snake_case
        print("\n🔤 STEP 6: Enforcing snake_case...")
        minute_merged_df = enforce_snake_case(minute_merged_df, "health_minute_level")
        final_daily_df = enforce_snake_case(final_daily_df, "health_daily_aggregated")
        hourly_df = enforce_snake_case(hourly_df, "health_hourly_aggregated")

        # Step 7: Save files
        print("\n💾 STEP 7: Saving processed files...")
        health_dir = 'files/topic_processed_files/health'
        os.makedirs(health_dir, exist_ok=True)

        # Sort by date descending (most recent first)
        minute_merged_df = minute_merged_df.sort_values('date', ascending=False)
        final_daily_df = final_daily_df.sort_values('date', ascending=False)
        hourly_df = hourly_df.sort_values('datetime', ascending=False)

        # Save minute-level file
        minute_path = f'{health_dir}/health_minute_processed.csv'
        minute_merged_df.to_csv(minute_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Minute-level: {len(minute_merged_df):,} records → {minute_path}")

        # Save daily aggregated file
        daily_path = f'{health_dir}/health_daily_processed.csv'
        final_daily_df.to_csv(daily_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Daily aggregated: {len(final_daily_df):,} records → {daily_path}")

        # Save hourly aggregated file
        hourly_path = f'{health_dir}/health_hourly_processed.csv'
        hourly_df.to_csv(hourly_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Hourly aggregated: {len(hourly_df):,} records → {hourly_path}")

        # Generate website files (including hourly)
        generate_health_website_files(minute_merged_df, final_daily_df, hourly_df)

        print("\n" + "="*70)
        print("🎉 HEALTH DATA PROCESSING COMPLETE!")
        print("="*70)
        print(f"📊 Summary:")
        print(f"   • Minute-level: {len(minute_merged_df):,} records")
        print(f"   • Hourly aggregated: {len(hourly_df):,} records")
        print(f"   • Daily aggregated: {len(final_daily_df):,} records")
        print(f"   • Date range: {final_daily_df['date'].min()} to {final_daily_df['date'].max()}")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n❌ Error in health data processing: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# WEBSITE FILE GENERATION
# ============================================================================

def generate_health_website_files(minute_df, daily_df, hourly_df=None):
    """
    Generate website-optimized files for the Health page.

    Args:
        minute_df: Minute-level merged DataFrame (already in snake_case)
        daily_df: Daily aggregated DataFrame (already in snake_case)
        hourly_df: Hourly aggregated DataFrame with JSON columns (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🌐 Generating website files for Health page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/health'
        os.makedirs(website_dir, exist_ok=True)

        # Generate minute-level website file
        minute_web = minute_df.copy()
        minute_web = enforce_snake_case(minute_web, "health_page_minute")
        minute_web = minute_web.sort_values('date', ascending=False)
        minute_path = f'{website_dir}/health_page_minute.csv'
        minute_web.to_csv(minute_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Minute-level website file: {len(minute_web):,} records → {minute_path}")

        # Generate daily aggregated website file
        daily_web = daily_df.copy()
        daily_web = enforce_snake_case(daily_web, "health_page_daily")
        daily_web = daily_web.sort_values('date', ascending=False)
        daily_path = f'{website_dir}/health_page_daily.csv'
        daily_web.to_csv(daily_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Daily website file: {len(daily_web):,} records → {daily_path}")

        # Generate hourly aggregated website file (for IntensityHeatmap)
        if hourly_df is not None:
            hourly_web = hourly_df.copy()
            hourly_web = enforce_snake_case(hourly_web, "health_page_hourly")
            hourly_web = hourly_web.sort_values('datetime', ascending=False)
            hourly_path = f'{website_dir}/health_page_hourly.csv'
            hourly_web.to_csv(hourly_path, sep='|', index=False, encoding='utf-8')
            print(f"✅ Hourly website file: {len(hourly_web):,} records → {hourly_path}")

        return True

    except Exception as e:
        print(f"❌ Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# UPLOAD FUNCTION
# ============================================================================

def upload_health_results():
    """
    Uploads the processed health files to Google Drive.

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n☁️  Uploading health results to Google Drive...")

    files_to_upload = [
        'files/website_files/health/health_page_daily.csv',
        #'files/website_files/health/health_page_minute.csv',
        'files/website_files/health/health_page_hourly.csv'
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} file(s)...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Health results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


# ============================================================================
# FULL PIPELINE
# ============================================================================

def full_health_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Health pipeline with 3 standard options.

    Options:
    1. Download new data, process, and upload to Drive
    2. Process existing data and upload to Drive
    3. Upload existing processed files to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*70)
    print("🏥 HEALTH DATA COORDINATION PIPELINE")
    print("="*70)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data and uploading...")
        choice = "2"
    elif auto_full:
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
        print("\n🚀 Option 1: Download, process, and upload...")

        # ============================================================
        # PHASE 1: DOWNLOAD - All user interaction upfront
        # ============================================================
        print("\n" + "="*60)
        print("📥 DOWNLOAD PHASE - User interaction required")
        print("="*60)
        print("\nYou will be prompted to download each data source.")
        print("After all downloads are complete, processing will run automatically.\n")

        download_status = {}

        # Download 1: Apple Health (CRITICAL)
        print("\n🍎 Download 1/4: Apple Health...")
        print("-" * 40)
        download_confirmed = download_apple_data()
        if download_confirmed:
            move_success = move_apple_files()
            download_status['apple'] = move_success
            if move_success:
                print("✅ Apple Health files ready")
            else:
                print("⚠️  Apple Health file move failed")
        else:
            download_status['apple'] = False
            print("⏭️  Skipping Apple Health download")

        # Download 2: Nutrilio (OPTIONAL)
        print("\n🥗 Download 2/4: Nutrilio...")
        print("-" * 40)
        download_confirmed = download_nutrilio_data()
        if download_confirmed:
            move_success = move_nutrilio_files()
            download_status['nutrilio'] = move_success
            if move_success:
                print("✅ Nutrilio files ready")
            else:
                print("⚠️  Nutrilio file move failed")
        else:
            download_status['nutrilio'] = False
            print("⏭️  Skipping Nutrilio download")

        # Download 3: Google Maps (for Location - OPTIONAL)
        print("\n📍 Download 3/4: Google Maps (for Location)...")
        print("-" * 40)
        download_confirmed = download_google_data()
        if download_confirmed:
            move_success = move_google_files()
            download_status['google_maps'] = move_success
            if move_success:
                print("✅ Google Maps files ready")
            else:
                print("⚠️  Google Maps file move failed")
        else:
            download_status['google_maps'] = False
            print("⏭️  Skipping Google Maps download")

        # Download 4: Offscreen/Screen Time (OPTIONAL)
        print("\n📱 Download 4/4: Screen Time (Offscreen)...")
        print("-" * 40)
        download_confirmed = download_offscreen_data()
        if download_confirmed:
            move_success = move_offscreen_files()
            download_status['offscreen'] = move_success
            if move_success:
                print("✅ Screen Time files ready")
            else:
                print("⚠️  Screen Time file move failed")
        else:
            download_status['offscreen'] = False
            print("⏭️  Skipping Screen Time download")

        # Summary of downloads
        print("\n" + "="*60)
        print("📋 DOWNLOAD SUMMARY")
        print("="*60)
        for source, status in download_status.items():
            status_icon = "✅" if status else "⏭️"
            print(f"   {status_icon} {source.replace('_', ' ').title()}: {'Ready' if status else 'Skipped'}")

        # Check if Apple Health was downloaded (critical)
        if not download_status.get('apple', False):
            print("\n⚠️  Warning: Apple Health is the critical data source.")
            print("   The pipeline may fail without Apple Health data.")

        # ============================================================
        # PHASE 2: PROCESSING - Automated, no user interaction
        # ============================================================
        print("\n" + "="*60)
        print("⚙️  AUTOMATED PROCESSING PHASE - No more user input needed")
        print("="*60)

        # Step 1: Process Apple Health (CRITICAL)
        print("\n🍎 Step 1/5: Processing Apple Health data...")
        try:
            apple_success = full_apple_pipeline(auto_process_only=True)
            if not apple_success:
                print("❌ Apple Health pipeline failed, stopping health coordination pipeline")
                return False
        except Exception as e:
            print(f"❌ Error in Apple Health pipeline: {e}")
            return False

        # Step 2: Process Nutrilio (OPTIONAL)
        print("\n🥗 Step 2/5: Processing Nutrilio mental health data...")
        try:
            nutrilio_success = full_nutrilio_pipeline(auto_process_only=True)
            if not nutrilio_success:
                print("⚠️  Nutrilio pipeline failed, but continuing (optional data source)")
        except Exception as e:
            print(f"⚠️  Error in Nutrilio pipeline: {e}, but continuing (optional data source)")

        # Step 3: Process Location (OPTIONAL)
        print("\n📍 Step 3/5: Processing Location data...")
        try:
            location_success = full_location_pipeline(auto_process_only=True)
            if not location_success:
                print("⚠️  Location pipeline failed, but continuing (optional data source)")
        except Exception as e:
            print(f"⚠️  Error in Location pipeline: {e}, but continuing (optional data source)")

        # Step 4: Process Screen Time (OPTIONAL)
        print("\n📱 Step 4/5: Processing Screen Time data...")
        try:
            screen_success = full_offscreen_pipeline(auto_process_only=True)
            if not screen_success:
                print("⚠️  Screen Time pipeline failed, but continuing (optional data source)")
        except Exception as e:
            print(f"⚠️  Error in Screen Time pipeline: {e}, but continuing (optional data source)")

        # Step 5: Merge all health data
        print("\n🔗 Step 5/5: Merging all health data sources...")
        process_success = create_health_files()
        if not process_success:
            print("❌ Health coordination merge failed, stopping pipeline")
            return False

        # Upload results
        print("\n☁️  Uploading results to Google Drive...")
        success = upload_health_results()

    elif choice == "2":
        print("\n⚙️  Option 2: Process existing data and upload...")
        print("   (Merges already-processed source files)")

        # Merge all health data from existing processed files
        process_success = create_health_files()

        if process_success:
            success = upload_health_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "3":
        print("\n☁️  Option 3: Upload existing processed files...")
        success = upload_health_results()

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*70)
    if success:
        print("✅ Health coordination pipeline completed successfully!")
        # Record successful run
        record_successful_run('topic_health', 'active')
    else:
        print("❌ Health coordination pipeline failed")
    print("="*70)

    return success


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Allow running this file directly
    print("🏥 Health Data Coordination Tool")
    print("This tool merges health data from multiple sources.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_health_pipeline(auto_full=False)
