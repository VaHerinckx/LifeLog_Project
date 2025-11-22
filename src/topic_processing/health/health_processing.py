import pandas as pd
import numpy as np
import os
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.apple.apple_processing import full_apple_pipeline
from src.sources_processing.nutrilio.nutrilio_processing import full_nutrilio_pipeline
from src.topic_processing.location.location_processing import full_location_pipeline
from src.sources_processing.offscreen.offscreen_processing import full_offscreen_pipeline


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
    location_path = 'files/processed_files/location/combined_timezone_processed.csv'

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
    and generates both minute-level and daily aggregated files.

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

        # Step 2: Merge at minute-level
        print("\n🔗 STEP 2: Merging minute-level data...")
        minute_merged_df = merge_minute_level_data(apple_df, location_df, screen_df)

        # Step 3: Aggregate to daily
        print("\n📊 STEP 3: Aggregating to daily level...")
        daily_df = aggregate_to_daily(minute_merged_df)

        # Step 4: Merge Nutrilio (daily)
        print("\n🔗 STEP 4: Merging Nutrilio mental health data...")
        final_daily_df = merge_daily_nutrilio(daily_df, nutrilio_df)

        # Step 5: Enforce snake_case
        print("\n🔤 STEP 5: Enforcing snake_case...")
        minute_merged_df = enforce_snake_case(minute_merged_df, "health_minute_level")
        final_daily_df = enforce_snake_case(final_daily_df, "health_daily_aggregated")

        # Step 6: Save files
        print("\n💾 STEP 6: Saving processed files...")
        health_dir = 'files/topic_processed_files/health'
        os.makedirs(health_dir, exist_ok=True)

        # Sort by date descending (most recent first)
        minute_merged_df = minute_merged_df.sort_values('date', ascending=False)
        final_daily_df = final_daily_df.sort_values('date', ascending=False)

        # Save minute-level file
        minute_path = f'{health_dir}/health_minute_processed.csv'
        minute_merged_df.to_csv(minute_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Minute-level: {len(minute_merged_df):,} records → {minute_path}")

        # Save daily aggregated file
        daily_path = f'{health_dir}/health_daily_processed.csv'
        final_daily_df.to_csv(daily_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Daily aggregated: {len(final_daily_df):,} records → {daily_path}")

        # Generate website files
        generate_health_website_files(minute_merged_df, final_daily_df)

        print("\n" + "="*70)
        print("🎉 HEALTH DATA PROCESSING COMPLETE!")
        print("="*70)
        print(f"📊 Summary:")
        print(f"   • Minute-level: {len(minute_merged_df):,} records")
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

def generate_health_website_files(minute_df, daily_df):
    """
    Generate website-optimized files for the Health page.

    Args:
        minute_df: Minute-level merged DataFrame (already in snake_case)
        daily_df: Daily aggregated DataFrame (already in snake_case)

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
        #'files/website_files/health/health_page_minute.csv',
        'files/website_files/health/health_page_daily.csv'
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
        print("   (Note: Triggers Apple Health pipeline first)")
        print("2. Process existing data and upload to Drive")
        print("3. Upload existing processed files to Drive")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Option 1: Download, process, and upload...")

        # Step 1: Run Apple Health pipeline (CRITICAL)
        print("\n🍎 Step 1/5: Processing Apple Health data...")
        try:
            apple_success = full_apple_pipeline(auto_full=True)
            if not apple_success:
                print("❌ Apple Health pipeline failed, stopping health coordination pipeline")
                return False
        except Exception as e:
            print(f"❌ Error in Apple Health pipeline: {e}")
            return False

        # Step 2: Run Nutrilio pipeline (OPTIONAL)
        print("\n🥗 Step 2/5: Processing Nutrilio mental health data...")
        try:
            nutrilio_success = full_nutrilio_pipeline(auto_full=True)
            if not nutrilio_success:
                print("⚠️  Nutrilio pipeline failed, but continuing (optional data source)")
        except Exception as e:
            print(f"⚠️  Error in Nutrilio pipeline: {e}, but continuing (optional data source)")

        # Step 3: Run Location pipeline (OPTIONAL)
        print("\n📍 Step 3/5: Processing Location data...")
        try:
            location_success = full_location_pipeline(auto_full=True)
            if not location_success:
                print("⚠️  Location pipeline failed, but continuing (optional data source)")
        except Exception as e:
            print(f"⚠️  Error in Location pipeline: {e}, but continuing (optional data source)")

        # Step 4: Run Screen Time pipeline (OPTIONAL)
        print("\n📱 Step 4/5: Processing Screen Time data...")
        try:
            screen_success = full_offscreen_pipeline(auto_full=True)
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
