import pandas as pd
import os
from datetime import date
from utils.file_operations import clean_rename_move_file, check_file_exists
from utils.web_operations import open_web_urls, prompt_user_download_status
from utils.drive_operations import upload_multiple_files, verify_drive_connection


def add_sorting_columns(df):
    """Adds some columns used for sorting in the PBI report"""
    df['Year_Week'] = df['Period'].apply(lambda x: str(x.year) + ' - ' + str(str(x.week)))
    df['Year_Month'] = df['Period'].apply(lambda x: str(x.year) + ' - ' + str(str(x.month)))
    df['sorting_week'] = df['Period'].dt.year * 100 + df['Period'].dt.isocalendar().week
    df['sorting_month'] = df['Period'].dt.year * 100 + df['Period'].dt.month
    df['sorting_day'] = df['Period'].dt.year * 100 + df['Period'].dt.isocalendar().day
    return df


def download_moneymgr_data():
    """
    Opens Money Manager export page and prompts user to download data.
    Returns True if user confirms download, False otherwise.
    """
    print("💰 Starting Money Manager data download...")
    print("📝 Instructions:")
    print("   1. Open Money Manager app")
    print("   2. Go to Settings > Export Data")
    print("   3. Export as Excel (.xlsx) format")
    print("   4. Save the file to Downloads folder")
    print("   5. The file should be named with today's date")

    # Note: Money Manager is a mobile app, so no web URL to open
    response = prompt_user_download_status("Money Manager")

    if response:
        print(f"✅ Expected file name: {date.today().strftime('%Y-%m-%d')}.xlsx")

    return response


def move_moneymgr_files():
    """
    Moves the downloaded Money Manager file from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving Money Manager files...")

    download_folder = "/Users/valen/Downloads"
    export_folder = "files/exports/moneymgr_exports"
    expected_file = f"{date.today().strftime('%Y-%m-%d')}.xlsx"
    target_file = "moneymgr_export.xlsx"

    # Check if the expected file exists
    if not check_file_exists(download_folder, expected_file):
        print(f"❌ Expected file {expected_file} not found in Downloads")

        # Look for any Excel files with today's date pattern
        print("🔍 Looking for alternative Excel files...")
        excel_files = [f for f in os.listdir(download_folder)
                      if f.endswith('.xlsx') and date.today().strftime('%Y-%m-%d') in f]

        if excel_files:
            print(f"📋 Found potential files: {excel_files}")
            chosen_file = input(f"Enter the correct filename (or press Enter for {excel_files[0]}): ").strip()
            expected_file = chosen_file if chosen_file else excel_files[0]
        else:
            print("❌ No suitable Excel files found")
            return False

    # Move the file
    success = clean_rename_move_file(
        export_folder=export_folder,
        download_folder=download_folder,
        file_name=expected_file,
        new_file_name=target_file
    )

    if success:
        print(f"✅ Successfully moved {expected_file} to {export_folder}/{target_file}")
    else:
        print(f"❌ Failed to move {expected_file}")

    return success


def process_moneymgr_export():
    """
    Main processing logic for Money Manager data.
    Reads the Excel file, processes it, and saves as CSV.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Money Manager data...")

    input_file = "files/exports/moneymgr_exports/moneymgr_export.xlsx"
    output_file = "files/processed_files/moneymgr_processed.csv"

    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return False

        # Read the Excel file
        print(f"📖 Reading data from {input_file}...")
        df = pd.read_excel(input_file)

        # Sort by Period
        df.sort_values(by="Period", inplace=True)

        # Drop unnecessary column if it exists
        if 'Accounts.1' in df.columns:
            df.drop('Accounts.1', axis=1, inplace=True)

        # Add sorting columns
        df = add_sorting_columns(df)

        # Save as CSV
        print(f"💾 Saving processed data to {output_file}...")
        df.to_csv(output_file, sep='|', index=False)

        print(f"✅ Successfully processed {len(df)} records")
        print(f"📊 Data range: {df['Period'].min()} to {df['Period'].max()}")

        return True

    except Exception as e:
        print(f"❌ Error processing Money Manager data: {e}")
        return False


def upload_moneymgr_results():
    """
    Uploads the processed Money Manager files to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading Money Manager results to Google Drive...")

    files_to_upload = [
        'files/processed_files/moneymgr_processed.csv',
        'files/work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx',  # Related file
        'files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx',  # Related file
        'files/work_files/Objectives.xlsx'  # Related file
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Money Manager results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


def full_moneymgr_pipeline(auto_full=False):
    """
    Complete Money Manager pipeline with 4 options.

    Options:
    1. Full pipeline (download → move → process → upload)
    2. Download data only (open app instructions + move files)
    3. Process existing file only (just processing)
    4. Process existing file and upload (process → upload)

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("💰 MONEY MANAGER DATA PIPELINE")
    print("="*60)

    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → move → process → upload)")
        print("2. Download data only (get files from app)")
        print("3. Process existing file only")
        print("4. Process existing file and upload to Drive")

        choice = input("\nEnter your choice (1-4): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full Money Manager pipeline...")

        # Step 1: Download
        download_success = download_moneymgr_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_moneymgr_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_moneymgr_files()

        # Step 3: Process (fallback to option 3 if no new files)
        if move_success:
            process_success = process_moneymgr_export()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = process_moneymgr_export()

        # Step 4: Upload
        if process_success:
            upload_success = upload_moneymgr_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n📥 Download Money Manager data only...")
        download_success = download_moneymgr_data()
        if download_success:
            success = move_moneymgr_files()
        else:
            success = False

    elif choice == "3":
        print("\n⚙️  Processing existing Money Manager file only...")
        success = process_moneymgr_export()

    elif choice == "4":
        print("\n⚙️  Processing existing file and uploading...")
        process_success = process_moneymgr_export()
        if process_success:
            success = upload_moneymgr_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    else:
        print("❌ Invalid choice. Please select 1-4.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Money Manager pipeline completed successfully!")
    else:
        print("❌ Money Manager pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("💰 Money Manager Processing Tool")
    print("This tool helps you download, process, and upload Money Manager data.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_moneymgr_pipeline(auto_full=False)
