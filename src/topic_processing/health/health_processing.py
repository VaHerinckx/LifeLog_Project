"""
Health Topic Coordinator - Reorganized Version

Creates two clean, flat CSV files:
1. health_daily.csv - Daily subjective metrics + sleep timing + daily totals
2. health_hourly.csv - Granular activity/location segments with minute-level metric attribution

Data Sources:
- Apple Health (minute-level metrics)
- Google Maps (segment data with activity types)
- Nutrilio (daily subjective ratings)
- Offscreen (screen time data)
- Location pipeline (fallback for pre-Google Maps dates)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.apple.apple_processing import full_apple_pipeline, download_apple_data, move_apple_files
from src.sources_processing.nutrilio.nutrilio_processing import full_nutrilio_pipeline, download_nutrilio_data, move_nutrilio_files
from src.topic_processing.location.location_processing import full_location_pipeline
from src.sources_processing.google_maps.google_maps_processing import download_google_data, move_google_files
from src.sources_processing.offscreen.offscreen_processing import full_offscreen_pipeline, download_offscreen_data, move_offscreen_files
from src.topic_processing.website_maintenance.website_maintenance_processing import full_website_maintenance_pipeline


# ============================================================================
# CONSTANTS
# ============================================================================

def get_time_period(hour):
    """Convert hour (0-23) to time period string."""
    if 6 <= hour <= 11:
        return 'MORNING'
    elif 12 <= hour <= 17:
        return 'AFTERNOON'
    elif 18 <= hour <= 21:
        return 'EVENING'
    else:  # 22-5
        return 'NIGHT'


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_apple_minute_data():
    """Load Apple Health minute-level data."""
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
    """Load Nutrilio mental health data (daily summary rows only)."""
    nutrilio_path = 'files/source_processed_files/nutrilio/nutrilio_health_processed.csv'

    if not os.path.exists(nutrilio_path):
        print(f"❌ Nutrilio health file not found: {nutrilio_path}")
        return None

    print(f"🥗 Loading Nutrilio mental health data...")
    df_daily = pd.read_csv(nutrilio_path, sep='|', encoding='utf-8')
    df_daily['date'] = pd.to_datetime(df_daily['date']).dt.normalize()

    # Select and rename mental health columns
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

    existing_columns = [col for col in columns_to_keep if col in df_daily.columns]
    df_daily = df_daily[existing_columns].copy()

    rename_dict = {
        'sleep_-_quality': 'sleep_quality',
        'sleep_-_quality_text': 'sleep_quality_text',
        'sleep_-_rest_feeling_(points)': 'sleep_rest_feeling',
        'fitness_feeling_(points)': 'fitness_feeling',
        'overall_evaluation_(points)': 'overall_evaluation'
    }
    df_daily = df_daily.rename(columns=rename_dict)

    print(f"✅ Loaded Nutrilio: {len(df_daily):,} daily mental health records")
    return df_daily


def load_google_maps_minute_data():
    """Load Google Maps minute-level data with activity segments."""
    gmaps_path = 'files/source_processed_files/google_maps/google_maps_minute_processed.csv'

    if not os.path.exists(gmaps_path):
        print(f"⚠️  Google Maps minute file not found: {gmaps_path}")
        return None

    print(f"🗺️  Loading Google Maps minute data...")
    df = pd.read_csv(gmaps_path, sep='|', encoding='utf-8', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"✅ Loaded Google Maps: {len(df):,} minute-level records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def load_location_daily_data():
    """Load location data for daily fallback (pre-Google Maps dates)."""
    location_path = 'files/topic_processed_files/location/location_processed.csv'

    if not os.path.exists(location_path):
        print(f"⚠️  Location file not found: {location_path}")
        return None

    print(f"📍 Loading location data for fallback...")
    df = pd.read_csv(location_path, sep='|', encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Aggregate to daily level (take most common city/country per day)
    daily_location = df.groupby('date').agg({
        'city': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
        'country': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
    }).reset_index()

    print(f"✅ Loaded location fallback: {len(daily_location):,} daily records")
    return daily_location


def load_screentime_data():
    """Load screen time data."""
    screen_path = 'files/source_processed_files/offscreen/offscreen_processed.csv'

    if not os.path.exists(screen_path):
        print(f"⚠️  Screen time file not found: {screen_path}")
        return None

    print(f"📱 Loading screen time data...")
    df = pd.read_csv(screen_path, sep='|', encoding='utf-8')
    df['date'] = pd.to_datetime(df['date']).dt.floor('min')

    print(f"✅ Loaded screen time: {len(df):,} minute-level records")
    return df


def load_garmin_sleep_data():
    """Load Garmin sleep data for backfilling Apple Health gaps and extra metrics."""
    garmin_sleep_path = 'files/source_processed_files/garmin/garmin_sleep_processed.csv'

    if not os.path.exists(garmin_sleep_path):
        print(f"⚠️  Garmin sleep file not found: {garmin_sleep_path}")
        return None

    print(f"⌚ Loading Garmin sleep data...")
    df = pd.read_csv(garmin_sleep_path, sep='|', encoding='utf-8')

    # Parse timestamps
    df['sleepStartTimestampLocal'] = pd.to_datetime(df['sleepStartTimestampLocal'])
    df['sleepEndTimestampLocal'] = pd.to_datetime(df['sleepEndTimestampLocal'])
    df['calendarDate'] = pd.to_datetime(df['calendarDate']).dt.date

    # Convert seconds to minutes for sleep stages
    df['deep_sleep_minutes'] = df['deepSleepSeconds'].fillna(0) / 60
    df['core_sleep_minutes'] = df['lightSleepSeconds'].fillna(0) / 60  # Garmin light = Apple core
    df['rem_sleep_minutes'] = df['remSleepSeconds'].fillna(0) / 60
    df['awake_minutes'] = df['awakeSleepSeconds'].fillna(0) / 60
    df['total_sleep_minutes'] = df['deep_sleep_minutes'] + df['core_sleep_minutes'] + df['rem_sleep_minutes']

    # Filter for valid confirmed sleep records
    valid_types = ['ENHANCED_CONFIRMED_FINAL', 'ENHANCED_CONFIRMED']
    df = df[df['sleepWindowConfirmationType'].isin(valid_types)]

    print(f"✅ Loaded Garmin sleep: {len(df):,} records")
    print(f"   Date range: {df['calendarDate'].min()} to {df['calendarDate'].max()}")

    return df


def load_garmin_training_history():
    """Load Garmin training history data."""
    training_path = 'files/source_processed_files/garmin/garmin_training_history_processed.csv'

    if not os.path.exists(training_path):
        print(f"⚠️  Garmin training history file not found: {training_path}")
        return None

    print(f"🏋️ Loading Garmin training history...")
    df = pd.read_csv(training_path, sep='|', encoding='utf-8')
    df['calendarDate'] = pd.to_datetime(df['calendarDate']).dt.date
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"✅ Loaded Garmin training: {len(df):,} records")
    print(f"   Date range: {df['calendarDate'].min()} to {df['calendarDate'].max()}")

    return df


def load_garmin_stress_data():
    """Load Garmin stress level data."""
    stress_path = 'files/source_processed_files/garmin/garmin_stress_level_processed.csv'

    if not os.path.exists(stress_path):
        print(f"⚠️  Garmin stress file not found: {stress_path}")
        return None

    print(f"😰 Loading Garmin stress data...")
    df = pd.read_csv(stress_path, sep='|', encoding='utf-8', low_memory=False)
    df['stress_level_time'] = pd.to_datetime(df['stress_level_time'])

    # Filter out invalid readings (negative values indicate unavailable)
    df = df[df['stress_level_value'] >= 0].copy()

    # Add date and hour columns for aggregation
    df['date'] = df['stress_level_time'].dt.date
    df['hour'] = df['stress_level_time'].dt.hour

    print(f"✅ Loaded Garmin stress: {len(df):,} valid records")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    return df


# ============================================================================
# GARMIN DATA AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_training_history_daily(training_df):
    """Aggregate training history to daily level - take latest entry per day."""
    if training_df is None or len(training_df) == 0:
        return None

    print("📊 Aggregating training history to daily level...")

    # Sort by timestamp and take the last entry per day (most recent status)
    training_df = training_df.sort_values('timestamp')
    daily = training_df.groupby('calendarDate').last().reset_index()

    # Keep relevant columns and rename
    columns_to_keep = {
        'calendarDate': 'date',
        'weeklyTrainingLoadSum': 'training_load_weekly',
        'trainingStatus': 'training_status',
        'fitnessLevelTrend': 'fitness_trend',
        'loadLevelTrend': 'load_trend'
    }

    daily = daily[[c for c in columns_to_keep.keys() if c in daily.columns]].copy()
    daily = daily.rename(columns=columns_to_keep)

    print(f"✅ Aggregated training history: {len(daily):,} daily records")
    return daily


def aggregate_stress_hourly(stress_df):
    """Aggregate stress data to hourly level."""
    if stress_df is None or len(stress_df) == 0:
        return None

    print("📊 Aggregating stress data to hourly level...")

    # Group by date and hour
    hourly = stress_df.groupby(['date', 'hour']).agg(
        stress_avg=('stress_level_value', 'mean'),
        stress_max=('stress_level_value', 'max'),
        stress_minutes_high=('stress_level_value', lambda x: (x >= 76).sum()),  # High stress
        stress_minutes_medium=('stress_level_value', lambda x: ((x >= 51) & (x < 76)).sum())  # Medium stress
    ).reset_index()

    # Round averages
    hourly['stress_avg'] = hourly['stress_avg'].round(1)

    print(f"✅ Aggregated stress hourly: {len(hourly):,} records")
    return hourly


def aggregate_stress_daily(stress_df):
    """Aggregate stress data to daily level."""
    if stress_df is None or len(stress_df) == 0:
        return None

    print("📊 Aggregating stress data to daily level...")

    # Group by date
    daily = stress_df.groupby('date').agg(
        daily_stress_avg=('stress_level_value', 'mean'),
        daily_stress_max=('stress_level_value', 'max'),
        daily_stress_minutes_high=('stress_level_value', lambda x: (x >= 76).sum()),
        daily_stress_minutes_medium=('stress_level_value', lambda x: ((x >= 51) & (x < 76)).sum())
    ).reset_index()

    # Round averages
    daily['daily_stress_avg'] = daily['daily_stress_avg'].round(1)

    print(f"✅ Aggregated stress daily: {len(daily):,} records")
    return daily


# ============================================================================
# SEGMENT PROCESSING FUNCTIONS
# ============================================================================

def create_segments_from_google_maps(gmaps_df):
    """
    Create segments from Google Maps minute data.
    A segment is a continuous activity/location that may span multiple hours.

    Returns DataFrame with columns:
    - segment_id: Unique ID for continuous activity spanning hours
    - start_time, end_time: Segment boundaries
    - segment_type: 'stationary' or 'moving'
    - activity_type: 'idle' for stationary, actual activity for moving
    - place_name, address, city, country, coordinates, is_home
    - distance_meters: Total distance for moving segments
    """
    if gmaps_df is None or len(gmaps_df) == 0:
        return None

    print("🔀 Creating segments from Google Maps data...")

    # Sort by timestamp
    df = gmaps_df.sort_values('timestamp').copy()

    # Fill NaN values with sentinel strings for proper comparison
    # (NaN != NaN is True in pandas, which would create a new segment for each minute)
    df['_record_type'] = df['record_type'].fillna('__NONE__')
    df['_activity_type'] = df['activity_type'].fillna('__NONE__')
    df['_place_name'] = df['place_name'].fillna('__NONE__')

    # Create segment boundaries when record_type, activity_type, place, or DATE changes
    df['prev_record_type'] = df['_record_type'].shift(1)
    df['prev_activity_type'] = df['_activity_type'].shift(1)
    df['prev_place_name'] = df['_place_name'].shift(1)

    # CRITICAL: Also break segments at day boundaries to ensure each day has its own segments
    df['current_date'] = df['timestamp'].dt.date
    df['prev_date'] = df['current_date'].shift(1)

    # New segment when record_type changes, OR activity_type changes, OR place changes, OR date changes
    df['new_segment'] = (
        (df['_record_type'] != df['prev_record_type']) |
        (df['_activity_type'] != df['prev_activity_type']) |
        ((df['_record_type'] == 'visit') & (df['_place_name'] != df['prev_place_name'])) |
        (df['current_date'] != df['prev_date'])  # Break at day boundaries
    )

    # Assign segment IDs
    df['segment_id'] = df['new_segment'].cumsum()

    # Aggregate each segment
    segments = df.groupby('segment_id').agg({
        'timestamp': ['min', 'max'],
        'record_type': 'first',
        'activity_type': 'first',
        'place_name': 'first',
        'address': 'first',
        'city': 'first',
        'country': 'first',
        'coordinates': 'first',
        'is_home': 'first',
        'is_moving': 'first',
        'distance_meters': 'sum'
    }).reset_index()

    # Flatten column names
    segments.columns = [
        'segment_id', 'start_time', 'end_time', 'record_type', 'activity_type',
        'place_name', 'address', 'city', 'country', 'coordinates', 'is_home',
        'is_moving', 'distance_meters'
    ]

    # Determine segment_type
    segments['segment_type'] = np.where(segments['is_moving'] == True, 'moving', 'stationary')

    # Set activity_type to 'idle' for stationary segments
    segments.loc[segments['segment_type'] == 'stationary', 'activity_type'] = 'idle'

    # Convert segment_id to string format: YYYYMMDD_HHMMSS_seq (vectorized)
    segments['segment_id'] = (
        segments['start_time'].dt.strftime('%Y%m%d_%H%M%S') + '_' + segments.index.astype(str)
    )

    print(f"✅ Created {len(segments):,} segments")
    print(f"   Stationary: {(segments['segment_type'] == 'stationary').sum():,}")
    print(f"   Moving: {(segments['segment_type'] == 'moving').sum():,}")

    return segments


def expand_segments_to_hourly(segments_df):
    """
    Expand segments to hourly records.
    One row per segment per hour it spans.

    Returns DataFrame with hour_segment_id for uniqueness within each hour.
    hour_segment_id format: YYYYMMDD_HH_N where N is sequential within each hour.
    """
    if segments_df is None or len(segments_df) == 0:
        return None

    print("⏰ Expanding segments to hourly records...")

    # Process segments in batches using list comprehension for speed
    all_hourly_records = []

    # Convert to list of dicts for faster iteration (faster than iterrows)
    segments_list = segments_df.to_dict('records')

    for segment in segments_list:
        # Get start and end hours
        start_hour = segment['start_time'].replace(minute=0, second=0, microsecond=0)
        end_hour = segment['end_time'].replace(minute=0, second=0, microsecond=0)

        # Calculate total segment duration for proportional distance distribution
        total_segment_minutes = (segment['end_time'] - segment['start_time']).total_seconds() / 60

        current_hour = start_hour

        while current_hour <= end_hour:
            # Calculate duration within this hour
            hour_start = current_hour
            hour_end = current_hour + timedelta(hours=1)

            seg_start_in_hour = max(segment['start_time'], hour_start)
            seg_end_in_hour = min(segment['end_time'], hour_end)

            duration_minutes = (seg_end_in_hour - seg_start_in_hour).total_seconds() / 60

            if duration_minutes > 0:
                # Calculate proportional distance for this hour slice
                if segment['segment_type'] == 'moving' and segment['distance_meters'] and total_segment_minutes > 0:
                    proportion = duration_minutes / total_segment_minutes
                    proportional_distance = round(segment['distance_meters'] * proportion, 2)
                else:
                    proportional_distance = None

                # Calculate duration in different units
                duration_hours = round(duration_minutes / 60, 4)
                duration_days = round(duration_hours / 24, 6)

                all_hourly_records.append({
                    'date': current_hour.date(),
                    'hour': current_hour.hour,
                    'weekday': current_hour.weekday(),
                    'time_period': get_time_period(current_hour.hour),
                    'segment_id': segment['segment_id'],
                    'segment_start_time': segment['start_time'].strftime('%H:%M'),
                    'segment_end_time': segment['end_time'].strftime('%H:%M'),
                    '_hour_key': current_hour.strftime('%Y%m%d_%H'),  # Temp key for sequencing
                    'segment_type': segment['segment_type'],
                    'segment_duration_minutes': round(duration_minutes, 1),
                    'segment_duration_hours': duration_hours,
                    'segment_duration_days': duration_days,
                    # Preserve ALL location fields from Google Maps
                    'place_name': segment['place_name'] if segment['segment_type'] == 'stationary' else None,
                    'address': segment['address'] if segment['segment_type'] == 'stationary' else None,
                    'city': segment['city'],
                    'country': segment['country'],
                    'is_home': segment['is_home'],
                    'location_type': None,  # Will be enriched later
                    'coordinates': segment['coordinates'],
                    'activity_type': segment['activity_type'],
                    'distance_meters': proportional_distance,
                    'data_source': 'google_maps'
                })

            current_hour += timedelta(hours=1)

    hourly_df = pd.DataFrame(all_hourly_records)

    # Generate hour_segment_id with proper per-hour sequential numbering
    # Group by hour_key and assign sequential numbers within each hour
    if len(hourly_df) > 0:
        hourly_df = hourly_df.sort_values(['date', 'hour', 'segment_id'])
        hourly_df['_seq'] = hourly_df.groupby('_hour_key').cumcount() + 1
        hourly_df['hour_segment_id'] = hourly_df['_hour_key'] + '_' + hourly_df['_seq'].astype(str)
        hourly_df = hourly_df.drop(columns=['_hour_key', '_seq'])

    print(f"✅ Created {len(hourly_df):,} hourly segment records")

    return hourly_df


def create_fallback_hourly_records(apple_df, location_daily_df, gmaps_dates):
    """
    Create hourly records for dates without Google Maps data.
    One row per hour, using daily location from location pipeline.
    """
    print("📋 Creating fallback hourly records for pre-Google Maps dates...")

    # Get all dates from Apple Health
    apple_dates = set(apple_df['date'].dt.date.unique())

    # Get dates that have Google Maps data
    if gmaps_dates is not None:
        covered_dates = set(gmaps_dates)
    else:
        covered_dates = set()

    # Find dates needing fallback
    fallback_dates = apple_dates - covered_dates

    if not fallback_dates:
        print("   No fallback dates needed")
        return None

    print(f"   {len(fallback_dates)} dates need fallback records")

    # Create location lookup (vectorized)
    location_lookup = {}
    if location_daily_df is not None and len(location_daily_df) > 0:
        for row in location_daily_df.to_dict('records'):
            location_lookup[row['date']] = {
                'city': row.get('city'),
                'country': row.get('country')
            }

    hourly_records = []

    for date in sorted(fallback_dates):
        location = location_lookup.get(date, {'city': None, 'country': None})

        for hour in range(24):
            dt = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)

            hourly_records.append({
                'date': date,
                'hour': hour,
                'weekday': dt.weekday(),
                'time_period': get_time_period(hour),
                'segment_id': f"{date.strftime('%Y%m%d')}_{hour:02d}_default",
                'hour_segment_id': f"{date.strftime('%Y%m%d')}_{hour:02d}_1",
                'segment_type': 'stationary',
                'segment_duration_minutes': 60,
                'segment_duration_hours': 1.0,
                'segment_duration_days': round(1.0 / 24, 6),
                'place_name': None,
                'address': None,
                'city': location['city'],
                'country': location['country'],
                'is_home': None,  # Do not derive
                'location_type': None,
                'coordinates': None,
                'activity_type': 'idle',
                'distance_meters': None,
                'data_source': 'manual_location_input'
            })

    fallback_df = pd.DataFrame(hourly_records)
    print(f"✅ Created {len(fallback_df):,} fallback hourly records")

    return fallback_df


def fill_missing_hours(hourly_df, location_daily_df):
    """
    Fill in missing hours for dates that have partial Google Maps coverage.

    For dates that have SOME Google Maps segments but not all 24 hours,
    creates placeholder records for the missing hours.
    """
    if hourly_df is None or len(hourly_df) == 0:
        return hourly_df

    print("🔧 Filling in missing hours for dates with partial coverage...")

    # Get all unique dates in hourly data
    all_dates = hourly_df['date'].unique()

    # Create location lookup
    location_lookup = {}
    if location_daily_df is not None and len(location_daily_df) > 0:
        for row in location_daily_df.to_dict('records'):
            location_lookup[row['date']] = {
                'city': row.get('city'),
                'country': row.get('country')
            }

    # Find missing hours for each date
    missing_records = []
    dates_with_gaps = 0

    for date_val in all_dates:
        # Convert to Python date object if it's a numpy/pandas date
        if hasattr(date_val, 'date'):
            date_obj = date_val.date() if callable(date_val.date) else date_val
        elif isinstance(date_val, date):
            date_obj = date_val
        else:
            # Attempt to parse if it's a string
            date_obj = pd.to_datetime(date_val).date()

        # Get hours that exist for this date
        existing_hours = set(hourly_df[hourly_df['date'] == date_val]['hour'].unique())
        all_hours = set(range(24))
        missing_hours = all_hours - existing_hours

        if missing_hours:
            dates_with_gaps += 1
            location = location_lookup.get(date_obj, {'city': None, 'country': None})

            for hour in missing_hours:
                dt = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=hour)
                date_str = date_obj.strftime('%Y%m%d')

                missing_records.append({
                    'date': date_obj,
                    'hour': hour,
                    'weekday': dt.weekday(),
                    'time_period': get_time_period(hour),
                    'segment_id': f"{date_str}_{hour:02d}_gap",
                    'hour_segment_id': f"{date_str}_{hour:02d}_gap_1",
                    'segment_type': 'stationary',
                    'segment_duration_minutes': 60,
                    'segment_duration_hours': 1.0,
                    'segment_duration_days': round(1.0 / 24, 6),
                    'place_name': None,
                    'address': None,
                    'city': location['city'],
                    'country': location['country'],
                    'is_home': None,
                    'location_type': None,
                    'coordinates': None,
                    'activity_type': 'idle',
                    'distance_meters': None,
                    'data_source': 'gap_fill'
                })

    if missing_records:
        gap_df = pd.DataFrame(missing_records)
        hourly_df = pd.concat([hourly_df, gap_df], ignore_index=True)
        print(f"✅ Filled {len(missing_records)} missing hours across {dates_with_gaps} dates")
    else:
        print("   No missing hours found")

    return hourly_df


# ============================================================================
# METRIC ATTRIBUTION FUNCTIONS
# ============================================================================

def attribute_apple_metrics_to_segments(hourly_df, apple_df, gmaps_minute_df):
    """
    Attribute Apple Health metrics to hourly segments using vectorized operations.
    Uses pre-aggregation by date+hour and then merges, which is much faster.
    """
    print("📊 Attributing Apple Health metrics to segments...")

    if hourly_df is None or len(hourly_df) == 0:
        return hourly_df

    # Initialize metric columns
    metric_columns = [
        'steps', 'apple_distance_meters', 'avg_step_length_cm', 'avg_walking_speed_kmh',
        'avg_heart_rate', 'avg_audio_exposure', 'active_energy_kcal', 'resting_energy_kcal',
        'flights_climbed', 'body_weight_kg', 'body_fat_percent',
        'sleep_minutes', 'deep_sleep_minutes', 'rem_sleep_minutes',
        'core_sleep_minutes', 'awake_minutes'
    ]

    for col in metric_columns:
        hourly_df[col] = None

    if apple_df is None or len(apple_df) == 0:
        print("⚠️  No Apple data to attribute")
        return hourly_df

    # Create date and hour columns in apple_df for grouping
    apple_df = apple_df.copy()
    apple_df['_date'] = apple_df['date'].dt.date
    apple_df['_hour'] = apple_df['date'].dt.hour

    # Pre-aggregate Apple data by date+hour (MUCH faster than row-by-row)
    print("   Pre-aggregating Apple data by hour...")

    # Aggregation functions for different metric types
    agg_dict = {}

    # Sum metrics
    if 'step_count' in apple_df.columns:
        agg_dict['step_count'] = 'sum'
    if 'walking_dist' in apple_df.columns:
        agg_dict['walking_dist'] = 'sum'
    if 'flights_climbed' in apple_df.columns:
        agg_dict['flights_climbed'] = 'sum'
    if 'active_energy' in apple_df.columns:
        agg_dict['active_energy'] = 'sum'
    if 'resting_energy' in apple_df.columns:
        agg_dict['resting_energy'] = 'sum'

    # Average metrics
    if 'step_length' in apple_df.columns:
        agg_dict['step_length'] = 'mean'
    if 'walking_speed' in apple_df.columns:
        agg_dict['walking_speed'] = 'mean'
    if 'heart_rate' in apple_df.columns:
        agg_dict['heart_rate'] = 'mean'
    if 'audio_exposure' in apple_df.columns:
        agg_dict['audio_exposure'] = 'mean'
    if 'body_weight' in apple_df.columns:
        agg_dict['body_weight'] = 'mean'
    if 'body_fat_percent' in apple_df.columns:
        agg_dict['body_fat_percent'] = 'mean'

    # Aggregate main metrics
    if agg_dict:
        apple_hourly = apple_df.groupby(['_date', '_hour']).agg(agg_dict).reset_index()
        apple_hourly.columns = ['_date', '_hour'] + [f'_{c}' for c in agg_dict.keys()]
    else:
        apple_hourly = pd.DataFrame(columns=['_date', '_hour'])

    # Handle sleep separately (need to pivot by phase)
    if 'sleep_analysis' in apple_df.columns:
        print("   Processing sleep phases...")
        sleep_df = apple_df[apple_df['sleep_analysis'].notna() & (apple_df['sleep_analysis'] != '')]

        if len(sleep_df) > 0:
            # Count minutes per phase per hour
            sleep_counts = sleep_df.groupby(['_date', '_hour', 'sleep_analysis']).size().reset_index(name='minutes')

            # Pivot to get one column per phase
            sleep_pivot = sleep_counts.pivot_table(
                index=['_date', '_hour'],
                columns='sleep_analysis',
                values='minutes',
                fill_value=0
            ).reset_index()

            # Rename columns
            phase_mapping = {
                'Deep sleep': '_deep_sleep',
                'REM sleep': '_rem_sleep',
                'Core sleep': '_core_sleep',
                'Awake': '_awake'
            }

            for old_name, new_name in phase_mapping.items():
                if old_name in sleep_pivot.columns:
                    sleep_pivot = sleep_pivot.rename(columns={old_name: new_name})

            # Calculate total sleep
            sleep_cols = [c for c in sleep_pivot.columns if c.startswith('_') and c != '_date' and c != '_hour']
            if sleep_cols:
                sleep_pivot['_total_sleep'] = sleep_pivot[sleep_cols].sum(axis=1)

            # Merge sleep with other metrics
            if len(apple_hourly) > 0:
                apple_hourly = apple_hourly.merge(sleep_pivot, on=['_date', '_hour'], how='outer')
            else:
                apple_hourly = sleep_pivot

    # Now merge with hourly_df
    print("   Merging metrics with hourly segments...")
    hourly_df['_date'] = hourly_df['date']
    hourly_df['_hour'] = hourly_df['hour']

    # Merge
    hourly_df = hourly_df.merge(apple_hourly, on=['_date', '_hour'], how='left')

    # Calculate proportion for Google Maps segments (vectorized)
    hourly_df['_proportion'] = np.where(
        hourly_df['data_source'] == 'google_maps',
        hourly_df['segment_duration_minutes'] / 60.0,
        1.0
    )

    # Apply proportion and rename columns
    if '_step_count' in hourly_df.columns:
        hourly_df['steps'] = (hourly_df['_step_count'].fillna(0) * hourly_df['_proportion']).astype(int)

    if '_walking_dist' in hourly_df.columns:
        hourly_df['apple_distance_meters'] = (hourly_df['_walking_dist'].fillna(0) * 1000 * hourly_df['_proportion']).round(1)

    if '_flights_climbed' in hourly_df.columns:
        hourly_df['flights_climbed'] = (hourly_df['_flights_climbed'].fillna(0) * hourly_df['_proportion']).astype(int)

    if '_active_energy' in hourly_df.columns:
        hourly_df['active_energy_kcal'] = (hourly_df['_active_energy'].fillna(0) * hourly_df['_proportion']).round(1)

    if '_resting_energy' in hourly_df.columns:
        hourly_df['resting_energy_kcal'] = (hourly_df['_resting_energy'].fillna(0) * hourly_df['_proportion']).round(1)

    # Average metrics (no proportion needed)
    if '_step_length' in hourly_df.columns:
        hourly_df['avg_step_length_cm'] = (hourly_df['_step_length'] * 100).round(1)

    if '_walking_speed' in hourly_df.columns:
        hourly_df['avg_walking_speed_kmh'] = hourly_df['_walking_speed'].round(2)

    if '_heart_rate' in hourly_df.columns:
        hourly_df['avg_heart_rate'] = hourly_df['_heart_rate'].round(1)

    if '_audio_exposure' in hourly_df.columns:
        hourly_df['avg_audio_exposure'] = hourly_df['_audio_exposure'].round(1)

    if '_body_weight' in hourly_df.columns:
        hourly_df['body_weight_kg'] = hourly_df['_body_weight'].round(1)

    if '_body_fat_percent' in hourly_df.columns:
        hourly_df['body_fat_percent'] = hourly_df['_body_fat_percent'].round(1)

    # Sleep metrics (with proportion)
    if '_total_sleep' in hourly_df.columns:
        hourly_df['sleep_minutes'] = (hourly_df['_total_sleep'].fillna(0) * hourly_df['_proportion']).astype(int)

    if '_deep_sleep' in hourly_df.columns:
        hourly_df['deep_sleep_minutes'] = (hourly_df['_deep_sleep'].fillna(0) * hourly_df['_proportion']).astype(int)

    if '_rem_sleep' in hourly_df.columns:
        hourly_df['rem_sleep_minutes'] = (hourly_df['_rem_sleep'].fillna(0) * hourly_df['_proportion']).astype(int)

    if '_core_sleep' in hourly_df.columns:
        hourly_df['core_sleep_minutes'] = (hourly_df['_core_sleep'].fillna(0) * hourly_df['_proportion']).astype(int)

    if '_awake' in hourly_df.columns:
        hourly_df['awake_minutes'] = (hourly_df['_awake'].fillna(0) * hourly_df['_proportion']).astype(int)

    # Clean up temporary columns
    temp_cols = [c for c in hourly_df.columns if c.startswith('_')]
    hourly_df = hourly_df.drop(columns=temp_cols)

    print(f"✅ Attributed metrics to {len(hourly_df):,} hourly records")
    return hourly_df


def attribute_garmin_sleep_to_hourly(hourly_df, garmin_sleep_df):
    """
    Attribute Garmin sleep data to hourly records for dates where Apple data is not available.

    Uses Garmin sleep sessions and distributes sleep stages proportionally across sleep hours.
    Only fills in for dates before Apple sleep stages start (2023-12-28).
    """
    if garmin_sleep_df is None or len(garmin_sleep_df) == 0:
        return hourly_df

    APPLE_SLEEP_STAGES_START = date(2023, 12, 28)

    print("😴 Attributing Garmin sleep data to hourly records...")

    # Initialize sleep columns if they don't exist
    sleep_cols = ['sleep_minutes', 'deep_sleep_minutes', 'rem_sleep_minutes', 'core_sleep_minutes', 'awake_minutes', 'sleep_data_source']
    for col in sleep_cols:
        if col not in hourly_df.columns:
            hourly_df[col] = None if col == 'sleep_data_source' else 0

    # Create a mapping of (date, hour) -> hourly_df index for faster lookup
    hourly_df['_date_obj'] = pd.to_datetime(hourly_df['date']).dt.date
    hourly_lookup = {}
    for idx, row in hourly_df.iterrows():
        key = (row['_date_obj'], row['hour'])
        if key not in hourly_lookup:
            hourly_lookup[key] = []
        hourly_lookup[key].append(idx)

    records_updated = 0

    for _, sleep_row in garmin_sleep_df.iterrows():
        sleep_date = sleep_row['calendarDate']

        # Only process dates before Apple sleep stages start
        if sleep_date >= APPLE_SLEEP_STAGES_START:
            continue

        sleep_start = sleep_row['sleepStartTimestampLocal']
        sleep_end = sleep_row['sleepEndTimestampLocal']
        total_sleep_duration = (sleep_end - sleep_start).total_seconds() / 60  # in minutes

        if total_sleep_duration <= 0:
            continue

        # Get sleep stage totals from Garmin
        deep_total = sleep_row['deep_sleep_minutes']
        core_total = sleep_row['core_sleep_minutes']  # Garmin light → Apple core
        rem_total = sleep_row['rem_sleep_minutes']
        awake_total = sleep_row['awake_minutes']
        sleep_total = deep_total + core_total + rem_total

        # Distribute across hours that the sleep session spans
        current_time = sleep_start
        while current_time < sleep_end:
            hour_start = current_time.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + timedelta(hours=1)

            # Calculate overlap with this hour
            overlap_start = max(current_time, hour_start)
            overlap_end = min(sleep_end, hour_end)
            overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60

            if overlap_minutes > 0:
                # Proportion of this hour's overlap vs total sleep duration
                proportion = overlap_minutes / total_sleep_duration

                # The date for this hour - sleep hours before midnight belong to previous day
                hour_date = hour_start.date()
                if hour_start.hour < 12 and hour_start.date() > sleep_date:
                    # This is morning hours of the next calendar day, but belongs to sleep_date
                    hour_date = sleep_date

                # Find matching hourly records
                key = (hour_date, hour_start.hour)
                if key in hourly_lookup:
                    for idx in hourly_lookup[key]:
                        # Only update if no Apple sleep data
                        if pd.isna(hourly_df.at[idx, 'sleep_data_source']) or hourly_df.at[idx, 'sleep_data_source'] is None:
                            hourly_df.at[idx, 'sleep_minutes'] = int(round(sleep_total * proportion))
                            hourly_df.at[idx, 'deep_sleep_minutes'] = int(round(deep_total * proportion))
                            hourly_df.at[idx, 'core_sleep_minutes'] = int(round(core_total * proportion))
                            hourly_df.at[idx, 'rem_sleep_minutes'] = int(round(rem_total * proportion))
                            hourly_df.at[idx, 'awake_minutes'] = int(round(awake_total * proportion))
                            hourly_df.at[idx, 'sleep_data_source'] = 'garmin'
                            records_updated += 1

            current_time = hour_end

    # Clean up temp column
    hourly_df = hourly_df.drop(columns=['_date_obj'])

    print(f"✅ Attributed Garmin sleep to {records_updated:,} hourly records")
    return hourly_df


def attribute_stress_to_hourly(hourly_df, stress_hourly_df):
    """
    Merge hourly stress aggregations into hourly health data.
    """
    if stress_hourly_df is None or len(stress_hourly_df) == 0:
        return hourly_df

    print("😰 Merging stress data with hourly records...")

    # Ensure date types match
    hourly_df['_date_obj'] = pd.to_datetime(hourly_df['date']).dt.date
    stress_hourly_df['date'] = pd.to_datetime(stress_hourly_df['date']).dt.date if not isinstance(stress_hourly_df['date'].iloc[0], date) else stress_hourly_df['date']

    # Merge on date and hour
    hourly_df = hourly_df.merge(
        stress_hourly_df,
        left_on=['_date_obj', 'hour'],
        right_on=['date', 'hour'],
        how='left',
        suffixes=('', '_stress')
    )

    # Clean up duplicate columns and temp columns
    if 'date_stress' in hourly_df.columns:
        hourly_df = hourly_df.drop(columns=['date_stress'])
    hourly_df = hourly_df.drop(columns=['_date_obj'])

    print(f"✅ Merged stress data with hourly records")
    return hourly_df


def attribute_screen_time_to_segments(hourly_df, screen_df, sleep_times):
    """
    Attribute screen time metrics to hourly segments using vectorized operations.
    Also calculates screen_time_minutes_before_sleep - screen time within 60 minutes of sleep start.

    Args:
        hourly_df: DataFrame with hourly segments
        screen_df: DataFrame with minute-level screen time data
        sleep_times: Dict mapping date -> sleep_start_datetime (full datetime, not just time)
    """
    print("📱 Attributing screen time to segments...")

    if hourly_df is None:
        return hourly_df

    # Initialize columns
    hourly_df['screen_time_minutes'] = 0.0
    hourly_df['phone_pickups'] = 0
    hourly_df['screen_time_minutes_before_sleep'] = 0.0

    if screen_df is None or len(screen_df) == 0:
        print("⚠️  No screen time data to attribute")
        return hourly_df

    # Work with a copy of screen data
    screen_df = screen_df.copy()
    screen_df['_date'] = screen_df['date'].dt.date
    screen_df['_hour'] = screen_df['date'].dt.hour

    # Pre-aggregate screen data by date+hour for regular screen time
    print("   Pre-aggregating screen time by hour...")
    agg_dict = {}
    if 'screen_time' in screen_df.columns:
        agg_dict['screen_time'] = 'sum'
    if 'pickups' in screen_df.columns:
        agg_dict['pickups'] = 'sum'

    if agg_dict:
        screen_hourly = screen_df.groupby(['_date', '_hour']).agg(agg_dict).reset_index()
        # Convert screen_time from seconds to minutes
        if 'screen_time' in screen_hourly.columns:
            screen_hourly['screen_time'] = (screen_hourly['screen_time'] / 60).round(1)
    else:
        screen_hourly = pd.DataFrame(columns=['_date', '_hour'])

    # Calculate screen time before sleep at minute level using pre-calculated sleep_times
    before_sleep_hourly = {}  # (date, hour) -> screen_time_minutes_before_sleep
    if sleep_times and 'screen_time' in screen_df.columns:
        print("   Calculating screen time before sleep (minute-level)...")

        # Map each screen minute's date to its sleep_start_datetime
        screen_df['_sleep_start'] = screen_df['_date'].map(sleep_times)

        # For each minute, check if it's within 60 minutes before sleep
        mask_has_sleep = screen_df['_sleep_start'].notna()
        screen_with_sleep = screen_df[mask_has_sleep].copy()

        if len(screen_with_sleep) > 0:
            # Calculate time difference in minutes (sleep_start - current_minute)
            screen_with_sleep['_minutes_until_sleep'] = (
                screen_with_sleep['_sleep_start'] - screen_with_sleep['date']
            ).dt.total_seconds() / 60

            # Screen time counts as "before sleep" if:
            # - It's BEFORE sleep (minutes_until_sleep > 0)
            # - It's within 60 minutes of sleep (minutes_until_sleep <= 60)
            mask_before_sleep = (
                (screen_with_sleep['_minutes_until_sleep'] > 0) &
                (screen_with_sleep['_minutes_until_sleep'] <= 60)
            )

            before_sleep_minutes = screen_with_sleep[mask_before_sleep].copy()

            if len(before_sleep_minutes) > 0:
                # Aggregate by date+hour
                before_sleep_agg = before_sleep_minutes.groupby(['_date', '_hour'])['screen_time'].sum().reset_index()
                # Convert seconds to minutes
                before_sleep_agg['screen_time'] = (before_sleep_agg['screen_time'] / 60).round(1)

                for _, row in before_sleep_agg.iterrows():
                    before_sleep_hourly[(row['_date'], row['_hour'])] = row['screen_time']

                print(f"   Found {len(before_sleep_minutes):,} screen time minutes within 60 min of sleep")

    # Merge screen data with hourly_df
    print("   Merging screen time with hourly segments...")
    hourly_df['_date'] = hourly_df['date']
    hourly_df['_hour'] = hourly_df['hour']

    hourly_df = hourly_df.merge(
        screen_hourly.rename(columns={'screen_time': '_screen_time', 'pickups': '_pickups'}),
        on=['_date', '_hour'],
        how='left'
    )

    # Apply values
    if '_screen_time' in hourly_df.columns:
        hourly_df['screen_time_minutes'] = hourly_df['_screen_time'].fillna(0.0)

    if '_pickups' in hourly_df.columns:
        hourly_df['phone_pickups'] = hourly_df['_pickups'].fillna(0).astype(int)

    # Apply before-sleep screen time
    if before_sleep_hourly:
        before_sleep_df = pd.DataFrame([
            {'_date': k[0], '_hour': k[1], '_before_sleep_screen': v}
            for k, v in before_sleep_hourly.items()
        ])
        hourly_df = hourly_df.merge(before_sleep_df, on=['_date', '_hour'], how='left')
        hourly_df['screen_time_minutes_before_sleep'] = hourly_df['_before_sleep_screen'].fillna(0.0)

    # Clean up temporary columns
    temp_cols = [c for c in hourly_df.columns if c.startswith('_')]
    hourly_df = hourly_df.drop(columns=temp_cols)

    print(f"✅ Attributed screen time to {len(hourly_df):,} hourly records")
    return hourly_df


# ============================================================================
# DAILY AGGREGATION FUNCTIONS
# ============================================================================

def create_daily_file(nutrilio_df, apple_df, hourly_df, sleep_times_df=None,
                      training_daily_df=None, stress_daily_df=None):
    """
    Create the daily health file with:
    - Nutrilio subjective metrics
    - Sleep start/wake up times (from Apple + Garmin)
    - Daily totals for weighted averages
    - Training history from Garmin
    - Stress aggregates from Garmin

    Uses vectorized operations for performance.

    Args:
        sleep_times_df: Pre-calculated sleep times DataFrame (includes Garmin extras)
        training_daily_df: Daily training history from Garmin
        stress_daily_df: Daily stress aggregates from Garmin
    """
    print("📅 Creating daily health file...")

    # Start with all dates from Apple Health
    all_dates = sorted(apple_df['date'].dt.date.unique())
    daily_df = pd.DataFrame({'date': all_dates})

    # 1. Merge Nutrilio data (vectorized)
    print("   Merging Nutrilio data...")
    if nutrilio_df is not None and len(nutrilio_df) > 0:
        nutrilio_cols = ['sleep_quality', 'sleep_quality_text', 'dreams', 'dream_description',
                         'sleep_rest_feeling', 'fitness_feeling', 'overall_evaluation', 'notes_summary']
        nutrilio_subset = nutrilio_df.copy()
        nutrilio_subset['_date'] = nutrilio_subset['date'].dt.date

        # Keep only columns that exist
        existing_cols = ['_date'] + [c for c in nutrilio_cols if c in nutrilio_subset.columns]
        nutrilio_subset = nutrilio_subset[existing_cols].drop_duplicates(subset=['_date'])

        daily_df = daily_df.merge(nutrilio_subset, left_on='date', right_on='_date', how='left')
        if '_date' in daily_df.columns:
            daily_df = daily_df.drop(columns=['_date'])

    # 2. Merge sleep times (includes Garmin extras like SPO2, HR, respiration)
    print("   Merging sleep times...")
    if sleep_times_df is None:
        sleep_times_df = calculate_all_sleep_times(apple_df, all_dates)
    if sleep_times_df is not None:
        # Keep all sleep columns including Garmin extras (exclude sleep_start_datetime)
        sleep_cols = ['date', 'sleep_start_time', 'sleep_start_time_minutes', 'wake_up_time',
                      'wake_up_time_minutes', 'sleep_data_source', 'sleep_avg_spo2', 'sleep_avg_hr',
                      'sleep_avg_respiration', 'sleep_lowest_respiration', 'sleep_highest_respiration']
        sleep_subset = sleep_times_df[[c for c in sleep_cols if c in sleep_times_df.columns]]
        daily_df = daily_df.merge(sleep_subset, on='date', how='left')

    # 3. Calculate daily totals from hourly data (vectorized)
    print("   Aggregating daily totals from hourly data...")
    if hourly_df is not None and len(hourly_df) > 0:
        # Aggregate hourly to daily
        agg_cols = {
            'steps': 'sum',
            'apple_distance_meters': 'sum',
            'flights_climbed': 'sum',
            'active_energy_kcal': 'sum',
            'resting_energy_kcal': 'sum',
            'sleep_minutes': 'sum',
            'deep_sleep_minutes': 'sum',
            'rem_sleep_minutes': 'sum',
            'core_sleep_minutes': 'sum',
            'awake_minutes': 'sum',
            'screen_time_minutes': 'sum',
            'phone_pickups': 'sum',
            'screen_time_minutes_before_sleep': 'sum'
        }

        # Only aggregate columns that exist
        existing_agg = {k: v for k, v in agg_cols.items() if k in hourly_df.columns}

        if existing_agg:
            hourly_daily = hourly_df.groupby('date').agg(existing_agg).reset_index()

            # Rename columns with total_ prefix
            rename_map = {col: f'total_{col}' for col in existing_agg.keys()}
            hourly_daily = hourly_daily.rename(columns=rename_map)

            daily_df = daily_df.merge(hourly_daily, on='date', how='left')

    # 4. Merge Garmin training history
    print("   Merging training history...")
    if training_daily_df is not None and len(training_daily_df) > 0:
        daily_df = daily_df.merge(training_daily_df, on='date', how='left')
        print(f"      Merged training data for {len(training_daily_df):,} dates")

    # 5. Merge Garmin stress daily aggregates
    print("   Merging stress data...")
    if stress_daily_df is not None and len(stress_daily_df) > 0:
        daily_df = daily_df.merge(stress_daily_df, on='date', how='left')
        print(f"      Merged stress data for {len(stress_daily_df):,} dates")

    # Convert date to datetime
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    print(f"✅ Created daily file: {len(daily_df):,} records")
    return daily_df


def calculate_all_sleep_times(apple_df, all_dates, garmin_sleep_df=None):
    """
    Calculate sleep start and wake up times for all dates at once.
    Uses Apple Health data as primary source for dates >= 2023-12-28,
    falls back to Garmin for earlier dates.

    Args:
        apple_df: Apple Health minute-level data
        all_dates: List of all dates to process
        garmin_sleep_df: Optional Garmin sleep data for backfilling

    Returns:
        DataFrame with date, sleep_start_time, wake_up_time, sleep_data_source, and Garmin extras.
    """
    # Cutoff date - Apple sleep stages only start from this date
    APPLE_SLEEP_STAGES_START = date(2023, 12, 28)

    print("   Finding sleep sessions...")

    results = []
    apple_dates_with_sleep = set()

    # Process Apple sleep data first
    if 'sleep_analysis' in apple_df.columns:
        # Sleep phases (excluding 'In bed' and 'Awake')
        sleep_phases = ['Deep sleep', 'REM sleep', 'Core sleep', 'Unspecified']

        # Filter to only sleep records
        sleep_df = apple_df[apple_df['sleep_analysis'].isin(sleep_phases)].copy()

        if len(sleep_df) > 0:
            sleep_df = sleep_df.sort_values('date')

            # Assign each sleep record to a "sleep date" (the night it belongs to)
            hours = sleep_df['date'].dt.hour
            sleep_df['sleep_date'] = np.where(
                hours < 12,
                (sleep_df['date'] - pd.Timedelta(days=1)).dt.date,
                sleep_df['date'].dt.date
            )

            # Process each sleep date
            for sleep_date in sleep_df['sleep_date'].unique():
                # Skip dates before Apple sleep stages start if we have Garmin data
                if garmin_sleep_df is not None and sleep_date < APPLE_SLEEP_STAGES_START:
                    continue

                date_records = sleep_df[sleep_df['sleep_date'] == sleep_date].sort_values('date')

                if len(date_records) == 0:
                    continue

                # Find sessions using time differences
                times = date_records['date'].values

                if len(times) > 1:
                    gaps = np.diff(times).astype('timedelta64[m]').astype(float)
                    session_breaks = np.where(gaps > 30)[0] + 1
                    session_ids = np.zeros(len(times), dtype=int)
                    for i, brk in enumerate(session_breaks):
                        session_ids[brk:] = i + 1
                else:
                    session_ids = np.zeros(len(times), dtype=int)

                date_records = date_records.copy()
                date_records['session_id'] = session_ids

                sessions = date_records.groupby('session_id').agg(
                    start=('date', 'min'),
                    end=('date', 'max')
                ).reset_index()

                sessions['duration'] = (sessions['end'] - sessions['start']).dt.total_seconds()

                if len(sessions) > 0:
                    longest = sessions.loc[sessions['duration'].idxmax()]

                    start_hour = longest['start'].hour
                    end_hour = longest['end'].hour

                    is_valid_start = start_hour >= 20 or start_hour <= 3
                    is_valid_end = end_hour <= 12

                    if is_valid_start and is_valid_end:
                        sleep_start_minutes = longest['start'].hour * 60 + longest['start'].minute
                        wake_up_minutes = longest['end'].hour * 60 + longest['end'].minute

                        results.append({
                            'date': sleep_date,
                            'sleep_start_time': longest['start'].strftime('%H:%M'),
                            'sleep_start_time_minutes': sleep_start_minutes,
                            'sleep_start_datetime': longest['start'],
                            'wake_up_time': longest['end'].strftime('%H:%M'),
                            'wake_up_time_minutes': wake_up_minutes,
                            'sleep_data_source': 'apple'
                        })
                        apple_dates_with_sleep.add(sleep_date)

    # Process Garmin sleep data for gap period and extras
    if garmin_sleep_df is not None and len(garmin_sleep_df) > 0:
        print("   Processing Garmin sleep data for backfill and extras...")

        for _, row in garmin_sleep_df.iterrows():
            sleep_date = row['calendarDate']

            # For dates before Apple cutoff OR dates where Apple has no data, use Garmin as primary
            if sleep_date < APPLE_SLEEP_STAGES_START or sleep_date not in apple_dates_with_sleep:
                sleep_start = row['sleepStartTimestampLocal']
                sleep_end = row['sleepEndTimestampLocal']

                sleep_start_minutes = sleep_start.hour * 60 + sleep_start.minute
                wake_up_minutes = sleep_end.hour * 60 + sleep_end.minute

                results.append({
                    'date': sleep_date,
                    'sleep_start_time': sleep_start.strftime('%H:%M'),
                    'sleep_start_time_minutes': sleep_start_minutes,
                    'sleep_start_datetime': sleep_start,
                    'wake_up_time': sleep_end.strftime('%H:%M'),
                    'wake_up_time_minutes': wake_up_minutes,
                    'sleep_data_source': 'garmin',
                    # Garmin extras
                    'sleep_avg_spo2': row.get('averageSPO2'),
                    'sleep_avg_hr': row.get('averageHR'),
                    'sleep_avg_respiration': row.get('averageRespiration'),
                    'sleep_lowest_respiration': row.get('lowestRespiration'),
                    'sleep_highest_respiration': row.get('highestRespiration')
                })
            else:
                # For overlap period, merge Garmin extras into existing Apple record
                for result in results:
                    if result['date'] == sleep_date and result['sleep_data_source'] == 'apple':
                        result['sleep_data_source'] = 'apple+garmin'
                        result['sleep_avg_spo2'] = row.get('averageSPO2')
                        result['sleep_avg_hr'] = row.get('averageHR')
                        result['sleep_avg_respiration'] = row.get('averageRespiration')
                        result['sleep_lowest_respiration'] = row.get('lowestRespiration')
                        result['sleep_highest_respiration'] = row.get('highestRespiration')
                        break

    if results:
        df = pd.DataFrame(results)
        print(f"   Found sleep times for {len(df):,} dates")
        source_counts = df['sleep_data_source'].value_counts()
        for source, count in source_counts.items():
            print(f"      - {source}: {count:,}")
        return df
    return None


def safe_sum(df, column):
    """Safely sum a column, returning None if all values are None."""
    if column not in df.columns:
        return None
    values = df[column].dropna()
    if len(values) == 0:
        return None
    return round(values.sum(), 1)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def create_health_files():
    """
    Main processing function that creates two clean CSV files:
    1. health_daily.csv - Daily metrics
    2. health_hourly.csv - Hourly segment data
    """
    print("\n" + "="*70)
    print("🏥 HEALTH DATA PROCESSING - REORGANIZED VERSION")
    print("="*70)

    try:
        # Step 1: Load all data sources
        print("\n📥 STEP 1: Loading data sources...")
        apple_df = load_apple_minute_data()

        if apple_df is None:
            print("❌ Cannot proceed without Apple Health data")
            return False

        nutrilio_df = load_nutrilio_mental_health()
        gmaps_minute_df = load_google_maps_minute_data()
        location_daily_df = load_location_daily_data()
        screen_df = load_screentime_data()

        # Load Garmin data sources
        garmin_sleep_df = load_garmin_sleep_data()
        garmin_training_df = load_garmin_training_history()
        garmin_stress_df = load_garmin_stress_data()

        # Step 2: Create segments from Google Maps
        print("\n🔀 STEP 2: Creating segments from Google Maps...")
        segments_df = create_segments_from_google_maps(gmaps_minute_df)

        # Step 3: Expand segments to hourly
        print("\n⏰ STEP 3: Expanding segments to hourly records...")
        hourly_gmaps_df = expand_segments_to_hourly(segments_df)

        # Get dates covered by Google Maps
        gmaps_dates = None
        if hourly_gmaps_df is not None:
            gmaps_dates = set(hourly_gmaps_df['date'].unique())

        # Step 4: Create fallback hourly records for pre-Google Maps dates
        print("\n📋 STEP 4: Creating fallback hourly records...")
        hourly_fallback_df = create_fallback_hourly_records(
            apple_df, location_daily_df, gmaps_dates
        )

        # Step 5: Combine hourly data
        print("\n🔗 STEP 5: Combining hourly data...")
        hourly_dfs = [df for df in [hourly_gmaps_df, hourly_fallback_df] if df is not None]

        if hourly_dfs:
            hourly_df = pd.concat(hourly_dfs, ignore_index=True)
            hourly_df = hourly_df.sort_values(['date', 'hour', 'hour_segment_id'])
        else:
            print("❌ No hourly data created")
            return False

        print(f"✅ Combined hourly data: {len(hourly_df):,} records")

        # Step 5b: Fill in missing hours for dates with partial coverage
        print("\n🔧 STEP 5b: Filling gaps in hourly data...")
        hourly_df = fill_missing_hours(hourly_df, location_daily_df)
        hourly_df = hourly_df.sort_values(['date', 'hour', 'hour_segment_id'])
        print(f"✅ Hourly data after gap fill: {len(hourly_df):,} records")

        # Step 6: Calculate sleep times first (needed for screen time before sleep)
        # Now includes Garmin sleep data for backfill and extras
        print("\n😴 STEP 6: Calculating sleep times (Apple + Garmin)...")
        all_dates = sorted(apple_df['date'].dt.date.unique())
        sleep_times_df = calculate_all_sleep_times(apple_df, all_dates, garmin_sleep_df)

        # Create sleep_times dict: date -> sleep_start_datetime
        sleep_times = {}
        if sleep_times_df is not None and 'sleep_start_datetime' in sleep_times_df.columns:
            for _, row in sleep_times_df.iterrows():
                sleep_times[row['date']] = row['sleep_start_datetime']
            print(f"✅ Found sleep times for {len(sleep_times):,} dates")

        # Step 7: Attribute metrics to segments
        print("\n📊 STEP 7: Attributing metrics to segments...")
        hourly_df = attribute_apple_metrics_to_segments(hourly_df, apple_df, gmaps_minute_df)
        hourly_df = attribute_screen_time_to_segments(hourly_df, screen_df, sleep_times)

        # Step 7b: Attribute Garmin sleep to hourly (for gap period)
        print("\n😴 STEP 7b: Attributing Garmin sleep data...")
        hourly_df = attribute_garmin_sleep_to_hourly(hourly_df, garmin_sleep_df)

        # Step 7c: Aggregate and attribute stress data
        print("\n😰 STEP 7c: Processing stress data...")
        stress_hourly_df = aggregate_stress_hourly(garmin_stress_df)
        stress_daily_df = aggregate_stress_daily(garmin_stress_df)
        hourly_df = attribute_stress_to_hourly(hourly_df, stress_hourly_df)

        # Step 7d: Aggregate training history
        print("\n🏋️ STEP 7d: Processing training history...")
        training_daily_df = aggregate_training_history_daily(garmin_training_df)

        # Step 8: Create daily file
        print("\n📅 STEP 8: Creating daily file...")
        daily_df = create_daily_file(nutrilio_df, apple_df, hourly_df, sleep_times_df,
                                     training_daily_df, stress_daily_df)

        # Step 9: Enforce snake_case
        print("\n🔤 STEP 9: Enforcing snake_case...")
        daily_df = enforce_snake_case(daily_df, "health_daily")
        hourly_df = enforce_snake_case(hourly_df, "health_hourly")

        # Step 10: Save files
        print("\n💾 STEP 10: Saving processed files...")

        # Save to topic_processed_files
        health_dir = 'files/topic_processed_files/health'
        os.makedirs(health_dir, exist_ok=True)

        daily_df = daily_df.sort_values('date', ascending=False)
        hourly_df = hourly_df.sort_values(['date', 'hour', 'hour_segment_id'], ascending=[False, False, True])

        daily_path = f'{health_dir}/health_daily_processed.csv'
        hourly_path = f'{health_dir}/health_hourly_processed.csv'

        daily_df.to_csv(daily_path, sep='|', index=False, encoding='utf-8')
        hourly_df.to_csv(hourly_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Daily file: {len(daily_df):,} records → {daily_path}")
        print(f"✅ Hourly file: {len(hourly_df):,} records → {hourly_path}")

        # Generate website files
        generate_health_website_files(daily_df, hourly_df)

        print("\n" + "="*70)
        print("🎉 HEALTH DATA PROCESSING COMPLETE!")
        print("="*70)
        print(f"📊 Summary:")
        print(f"   • Daily file: {len(daily_df):,} records ({len(daily_df.columns)} columns)")
        print(f"   • Hourly file: {len(hourly_df):,} records ({len(hourly_df.columns)} columns)")
        print(f"   • Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")

        # Show data source breakdown
        if 'data_source' in hourly_df.columns:
            source_counts = hourly_df['data_source'].value_counts()
            print(f"   • Location data sources:")
            for source, count in source_counts.items():
                print(f"      - {source}: {count:,} records")

        # Show sleep data source breakdown
        if 'sleep_data_source' in daily_df.columns:
            sleep_source_counts = daily_df['sleep_data_source'].value_counts()
            print(f"   • Sleep data sources:")
            for source, count in sleep_source_counts.items():
                print(f"      - {source}: {count:,} days")

        # Show Garmin integration stats
        garmin_cols = ['training_load_weekly', 'daily_stress_avg', 'sleep_avg_spo2']
        for col in garmin_cols:
            if col in daily_df.columns:
                non_null = daily_df[col].notna().sum()
                if non_null > 0:
                    print(f"   • {col}: {non_null:,} days with data")

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

def generate_health_website_files(daily_df, hourly_df):
    """Generate website-optimized files for the Health page."""
    print("\n🌐 Generating website files for Health page...")

    try:
        website_dir = 'files/website_files/health'
        os.makedirs(website_dir, exist_ok=True)

        # Daily file
        daily_web = enforce_snake_case(daily_df.copy(), "health_page_daily")
        daily_web = daily_web.sort_values('date', ascending=False)
        daily_path = f'{website_dir}/health_page_daily.csv'
        daily_web.to_csv(daily_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Daily website file: {len(daily_web):,} records → {daily_path}")

        # Hourly file
        hourly_web = enforce_snake_case(hourly_df.copy(), "health_page_hourly")
        # Add datetime column combining date and hour for filtering purposes
        hourly_web['datetime'] = pd.to_datetime(hourly_web['date']).dt.strftime('%Y-%m-%d') + ' ' + hourly_web['hour'].astype(str).str.zfill(2) + ':00:00'
        hourly_web = hourly_web.sort_values(['date', 'hour', 'hour_segment_id'], ascending=[False, False, True])
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
    """Uploads the processed health files to Google Drive."""
    print("\n☁️  Uploading health results to Google Drive...")

    files_to_upload = [
        'files/website_files/health/health_page_daily.csv',
        'files/website_files/health/health_page_hourly.csv'
    ]

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

        # DOWNLOAD PHASE
        print("\n" + "="*60)
        print("📥 DOWNLOAD PHASE - User interaction required")
        print("="*60)

        download_status = {}

        # Download Apple Health
        print("\n🍎 Download 1/4: Apple Health...")
        download_confirmed = download_apple_data()
        if download_confirmed:
            move_success = move_apple_files()
            download_status['apple'] = move_success
        else:
            download_status['apple'] = False

        # Download Nutrilio
        print("\n🥗 Download 2/4: Nutrilio...")
        download_confirmed = download_nutrilio_data()
        if download_confirmed:
            move_success = move_nutrilio_files()
            download_status['nutrilio'] = move_success
        else:
            download_status['nutrilio'] = False

        # Download Google Maps
        print("\n📍 Download 3/4: Google Maps...")
        download_confirmed = download_google_data()
        if download_confirmed:
            move_success = move_google_files()
            download_status['google_maps'] = move_success
        else:
            download_status['google_maps'] = False

        # Download Screen Time
        print("\n📱 Download 4/4: Screen Time...")
        download_confirmed = download_offscreen_data()
        if download_confirmed:
            move_success = move_offscreen_files()
            download_status['offscreen'] = move_success
        else:
            download_status['offscreen'] = False

        # PROCESSING PHASE
        print("\n" + "="*60)
        print("⚙️  AUTOMATED PROCESSING PHASE")
        print("="*60)

        # Process Apple Health
        print("\n🍎 Step 1/5: Processing Apple Health...")
        try:
            apple_success = full_apple_pipeline(auto_process_only=True)
            if not apple_success:
                print("❌ Apple Health pipeline failed")
                return False
        except Exception as e:
            print(f"❌ Error in Apple Health pipeline: {e}")
            return False

        # Process Nutrilio
        print("\n🥗 Step 2/5: Processing Nutrilio...")
        try:
            full_nutrilio_pipeline(auto_process_only=True)
        except Exception as e:
            print(f"⚠️  Nutrilio pipeline error: {e}")

        # Process Location
        print("\n📍 Step 3/5: Processing Location...")
        try:
            full_location_pipeline(auto_process_only=True)
        except Exception as e:
            print(f"⚠️  Location pipeline error: {e}")

        # Process Screen Time
        print("\n📱 Step 4/5: Processing Screen Time...")
        try:
            full_offscreen_pipeline(auto_process_only=True)
        except Exception as e:
            print(f"⚠️  Screen Time pipeline error: {e}")

        # Create health files
        print("\n🔗 Step 5/5: Creating health files...")
        process_success = create_health_files()
        if not process_success:
            return False

        # Upload
        success = upload_health_results()

    elif choice == "2":
        print("\n⚙️  Option 2: Process existing data and upload...")
        process_success = create_health_files()

        if process_success:
            success = upload_health_results()
        else:
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
        record_successful_run('topic_health', 'active')
        # Update website tracking file
        full_website_maintenance_pipeline(auto_mode=True, quiet=True)
    else:
        print("❌ Health coordination pipeline failed")
    print("="*70)

    return success


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("🏥 Health Data Coordination Tool")
    print("This tool creates clean daily and hourly health files.")

    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    full_health_pipeline(auto_full=False)
