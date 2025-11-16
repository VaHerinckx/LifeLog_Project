import pandas as pd
import os
import re
import traceback
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, time_difference_correction


def row_expander_seconds(row):
    """Expands the dataframe to have one row per minute, to remove the aggregation in the original df"""
    lower_bound = (row['start']-pd.Timedelta(seconds=30)).round('T')
    upper_bound = (row['end']-pd.Timedelta(seconds=30)).round('T')
    rows_needed = (int((upper_bound-lower_bound).total_seconds()/60))+1
    if rows_needed == 1:
        date_df = pd.DataFrame(columns=['date', 'screen_time'])
        new_row = {'date': lower_bound, 'screen_time': row["duration_s"], 'pickups': 1}
        date_df = pd.concat([date_df, pd.DataFrame([new_row])], ignore_index=True)
        return date_df
    dates = pd.date_range(lower_bound, upper_bound, freq='T')
    date_df = pd.DataFrame({'date': dates})
    for minute in range(1, rows_needed+1, 1):
        if minute == 1:
            seconds = (lower_bound + pd.Timedelta(minutes=1) - row["start"]).total_seconds()
            pickup = 1
        elif minute != rows_needed:
            seconds = 60
            pickup = 0
        else:
            seconds = (row["end"] - upper_bound).total_seconds()
            pickup = 0
        date_df.loc[minute-1, 'screen_time'] = seconds
        date_df.loc[minute-1, 'pickups'] = pickup
    return date_df


def expand_df(df, processed_file_path):
    """Expands the dataframe to have one row per minute, to remove the aggregation in the original df"""

    # Apply timezone correction based on location data
    # Offscreen exports are in UTC (+0000), need to convert to local timezone based on where you were
    print("🌍 Applying location-based timezone correction...")
    df = time_difference_correction(df, 'start', source_timezone='UTC')
    df = time_difference_correction(df, 'end', source_timezone='UTC')

    # Calculate duration
    df['duration_s'] = (df['end'] - df['start']).dt.total_seconds().astype(float)

    new_df = pd.DataFrame(columns=['date', 'screen_time', 'pickups'])

    # Try to load existing processed file
    if os.path.exists(processed_file_path):
        print(f"📁 Loading existing processed file: {processed_file_path}")
        old_df = pd.read_csv(processed_file_path, sep='|', encoding='utf-8')
        old_df['date'] = pd.to_datetime(old_df['date'])
        new_df = pd.concat([new_df, old_df], ignore_index=True)

        # Filter to only new data - both sides should now be timezone-naive
        max_date = max(new_df["date"])
        df = df[(df['start']-pd.Timedelta(seconds=30)).round('T') > max_date].reset_index(drop=True)
        print(f"📊 Found {df.shape[0]} new rows to expand")
    else:
        print(f"⚠️  No existing processed file found, processing all {df.shape[0]} rows")

    # Expand each row
    for _, row in df.iterrows():
        new_df = pd.concat([new_df, row_expander_seconds(row)], ignore_index=True)

    # Aggregate by date
    new_df = new_df.groupby('date').sum().reset_index()
    return new_df


def screentime_before_sleep(df):
    """Computes the screentime in the hour before sleep"""
    sleep_file = 'files/processed_files/health/garmin_sleep_processed.csv'

    # Check if sleep file exists
    if not os.path.exists(sleep_file):
        print(f"⚠️  Warning: Sleep data file not found: {sleep_file}")
        print("   Adding empty sleep-related columns")
        df['is_within_hour_before_sleep'] = 0
        df['sleep_start_timestamp_local'] = None
        return df

    print(f"📁 Loading sleep data from {sleep_file}")
    df2 = pd.read_csv(sleep_file, sep='|', encoding='utf-8')

    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)

    # Calculate sleep date (date before sleep, accounting for early morning hours)
    df['sleep_date'] = df['date'].apply(lambda x: x - pd.Timedelta(days=1) if x.hour < 6 else x).dt.date
    df['sleep_date'] = pd.to_datetime(df['sleep_date'], utc=True).dt.tz_localize(None)

    # Process sleep data
    df2['sleep_date'] = pd.to_datetime(df2['calendar_date'], utc=True).dt.tz_localize(None) - pd.Timedelta(days=1)

    # Merge with sleep data
    merged_df = pd.merge(
        df,
        df2[['sleep_date', "sleep_start_timestamp_local"]],
        on='sleep_date',
        how='left'
    )

    # Calculate time difference and flag within hour before sleep
    merged_df['time_diff'] = (
        pd.to_datetime(merged_df['sleep_start_timestamp_local']).dt.tz_localize(None) -
        merged_df['date']
    )
    merged_df['is_within_hour_before_sleep'] = merged_df['time_diff'].apply(
        lambda x: 1 if pd.notna(x) and x <= pd.Timedelta(hours=1) and x >= pd.Timedelta(0) else 0
    )

    # Drop temporary columns
    return merged_df.drop(["sleep_date", "time_diff"], axis=1)


def download_offscreen_data():
    """
    Prompts user to download Offscreen data from the app.
    Returns True if user confirms download, False otherwise.
    """
    try:
        print("📱 Starting Offscreen data download...")
        print("📝 Instructions:")
        print("   1. Open Offscreen app on your phone")
        print("   2. Go to Settings > Export Data")
        print("   3. Select 'Pickup.csv' for export")
        print("   4. Share/export the file to your computer")
        print("   5. Save the file to Downloads folder")
        print("   6. File should be named with timestamp (e.g., '2024-01-15 12:30:45 000-Pickup.csv')")

        response = prompt_user_download_status("Offscreen")

        if response:
            print("✅ Download confirmed")

        return response

    except Exception as e:
        print(f"❌ Error during download process: {e}")
        return False


def move_offscreen_files():
    """
    Moves the downloaded Offscreen Pickup.csv file from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    try:
        print("📁 Moving Offscreen files...")

        download_folder = os.path.expanduser("~/Downloads")
        export_folder = "files/exports/offscreen_exports"
        csv_regex = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*\.csv$'

        # Ensure export folder exists
        os.makedirs(export_folder, exist_ok=True)

        # Find and move matching files
        count_file = 0
        for f in os.listdir(download_folder):
            if re.match(csv_regex, f):
                count_file += 1
                # Extract the actual filename after the timestamp
                new_file_name = f.split('000-')[1] if '000-' in f else f
                clean_rename_move_file(export_folder, download_folder, f, new_file_name, count_file)

        if count_file == 0:
            print(f"❌ No Offscreen export files found in {download_folder}")
            print(f"   Looking for pattern: YYYY-MM-DD HH:MM:SS*.csv")
            return False

        print(f"✅ Processed {count_file} Offscreen export file(s)")
        return True

    except Exception as e:
        print(f"❌ Error moving files: {e}")
        return False


def create_offscreen_file():
    """
    Main processing logic for Offscreen data.
    Reads the Pickup.csv file, processes it, and saves as CSV with snake_case columns.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Offscreen data...")

    input_file = "files/exports/offscreen_exports/Pickup.csv"
    output_file = "files/processed_files/screentime/offscreen_processed.csv"

    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return False

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Read the input CSV file
        print(f"📖 Reading data from {input_file}...")
        df = pd.read_csv(input_file)

        print(f"📊 Loaded {len(df)} pickup records")

        # Expand dataframe (one row per minute)
        print("🔄 Expanding data to minute-level granularity...")
        df = expand_df(df, output_file)

        # Add sleep-related features
        print("😴 Computing screentime before sleep...")
        df = screentime_before_sleep(df)

        # Sort by date (descending)
        df = df.sort_values("date", ascending=False)

        # Save as CSV with UTF-8 encoding and pipe delimiter
        print(f"💾 Saving processed data to {output_file}...")
        df.to_csv(output_file, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed {len(df)} records")
        print(f"📊 Data range: {df['date'].min()} to {df['date'].max()}")
        print(f"📊 Total screen time: {df['screen_time'].sum() / 3600:.1f} hours")
        print(f"📊 Total pickups: {int(df['pickups'].sum())}")

        return True

    except Exception as e:
        print(f"❌ Error processing Offscreen data: {e}")
        traceback.print_exc()
        return False


def upload_offscreen_results():
    """
    Uploads the processed Offscreen files to Google Drive.
    Returns True if successful, False otherwise.
    """
    try:
        print("⬆️  Uploading Offscreen results to Google Drive...")

        files_to_upload = ['files/processed_files/screentime/offscreen_processed.csv']

        # Filter to only existing files
        existing_files = [f for f in files_to_upload if os.path.exists(f)]

        if not existing_files:
            print("❌ No files found to upload")
            return False

        print(f"📤 Uploading {len(existing_files)} file(s)...")
        success = upload_multiple_files(existing_files)

        if success:
            print("✅ Offscreen results uploaded successfully!")
        else:
            print("❌ Some files failed to upload")

        return success

    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        return False


def full_offscreen_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Offscreen pipeline with 3 standard options.

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
    print("\n" + "="*60)
    print("📱 OFFSCREEN DATA PIPELINE")
    print("="*60)

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
        print("\n🚀 Download new data, process, and upload to Drive...")

        # Step 1: Download
        download_success = download_offscreen_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_offscreen_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_offscreen_files()

        # Step 3: Process (fallback to existing files if no new files)
        if move_success:
            process_success = create_offscreen_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_offscreen_file()

        # Step 4: Upload
        if process_success:
            upload_success = upload_offscreen_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n⚙️  Process existing data and upload to Drive...")
        process_success = create_offscreen_file()
        if process_success:
            success = upload_offscreen_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "3":
        print("\n⬆️  Upload existing processed files to Drive...")
        success = upload_offscreen_results()

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Offscreen pipeline completed successfully!")
        # Record successful run
        record_successful_run('screentime_offscreen', 'active')
    else:
        print("❌ Offscreen pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("📱 Offscreen Processing Tool")
    print("This tool helps you download, process, and upload Offscreen screentime data.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_offscreen_pipeline(auto_full=False)
