import pandas as pd
import os
import re
import traceback
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import prompt_user_download_status
# Drive operations not needed - source processor doesn't upload
from src.utils.utils_functions import record_successful_run, time_difference_correction, enforce_snake_case


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


# NOTE: screentime_before_sleep calculation has been moved to health_processing.py
# It now uses Apple Health sleep_analysis data instead of Garmin sleep data.
# This ensures accurate calculation using the authoritative sleep source.
# The function below is kept for reference but is no longer called.

def _legacy_screentime_before_sleep(df):
    """
    DEPRECATED: This function is no longer used.
    Screen time before sleep is now calculated in health_processing.py
    using Apple Health sleep_analysis data for accuracy.

    The calculation is done AFTER Apple Health data is merged,
    which provides the actual first sleep timestamp.
    """
    print("⚠️  This function is deprecated - sleep calculation moved to health_processing.py")
    return df


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
    output_file = "files/source_processed_files/offscreen/offscreen_processed.csv"

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

        # NOTE: Sleep-related features (screen_before_sleep) are now calculated
        # in health_processing.py using Apple Health sleep data for accuracy.
        # This ensures we use the authoritative sleep source.

        # Sort by date (descending)
        df = df.sort_values("date", ascending=False)

        # Enforce snake_case before saving
        df = enforce_snake_case(df, "processed file")

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




def full_offscreen_pipeline(auto_full=False, auto_process_only=False):
    """
    Offscreen (Screen Time) SOURCE processor with 2 standard options.

    NOTE: This is a source processor - does NOT upload to Drive.
    Upload is handled by the Health topic coordinator.

    Options:
    1. Download new data and process
    2. Process existing data

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("📱 OFFSCREEN (SCREEN TIME) SOURCE PROCESSOR")
    print("="*60)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new data and process")
        print("2. Process existing data")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Download new data and process...")

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
            success = create_offscreen_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            success = create_offscreen_file()

    elif choice == "2":
        print("\n⚙️  Process existing data...")
        success = create_offscreen_file()

    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Offscreen source processing completed successfully!")
        print("📁 Output: files/source_processed_files/offscreen/offscreen_processed.csv")
        # Record successful run
        record_successful_run('source_offscreen', 'active')
    else:
        print("❌ Offscreen source processing failed")
    print("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("📱 Offscreen Source Processing Tool")
    print("This tool downloads and processes Offscreen screentime data.")
    print("Note: Upload is handled by the Health topic coordinator")

    # Run the pipeline
    full_offscreen_pipeline(auto_full=False)
