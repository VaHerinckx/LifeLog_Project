import pandas as pd
import os
from datetime import date
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run


def add_sorting_columns(df):
    """Adds some columns used for sorting in the PBI report"""
    df['year_week'] = df['date'].apply(lambda x: str(x.year) + ' - ' + str(str(x.week)))
    df['year_month'] = df['date'].apply(lambda x: str(x.year) + ' - ' + str(str(x.month)))
    df['sorting_week'] = df['date'].dt.year * 100 + df['date'].dt.isocalendar().week
    df['sorting_month'] = df['date'].dt.year * 100 + df['date'].dt.month
    df['sorting_day'] = df['date'].dt.year * 100 + df['date'].dt.isocalendar().day
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

    download_folder = os.path.expanduser("~/Downloads")
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


def create_moneymgr_file():
    """
    Main processing logic for Money Manager data.
    Reads the Excel file, processes it, and saves as CSV.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Money Manager data...")

    input_file = "files/exports/moneymgr_exports/moneymgr_export.xlsx"
    output_file = "files/processed_files/finance/moneymgr_processed.csv"

    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return False

        # Read the Excel file
        print(f"📖 Reading data from {input_file}...")
        df = pd.read_excel(input_file)

        # Rename Period to date immediately after reading
        df = df.rename(columns={'Period': 'date'})

        # Sort by date
        df.sort_values(by="date", inplace=True)

        # Drop unnecessary column if it exists
        if 'Accounts.1' in df.columns:
            df.drop('Accounts.1', axis=1, inplace=True)

        # Remove specific accounts that are no longer needed
        print("🗑️  Filtering out removed accounts...")
        accounts_to_remove = ["Argenta Life Longer Life", "Argenta Life DP Dynamic Allocation", "Savings account"]
        initial_count = len(df)
        df = df[~df['Accounts'].isin(accounts_to_remove)]
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"   Removed {removed_count} records from: {', '.join(accounts_to_remove)}")

        # Add sorting columns
        df = add_sorting_columns(df)

        print("Use unique currency rate for NTD expenses")
        df['corrected_EUR'] = df.apply(lambda row: row['Amount'] * 0.029
                                       if (row.get('Currency') == 'NTD') and (row.get('Accounts') not in ['Personal account', 'Cash'])
                                       else row['EUR'], axis = 1)

        print("Adjusting for tricount expenses")
        df['corrected_EUR'] = df.apply(lambda row:
            abs(float(row['corrected_EUR'])) / 2 if (row.get('Accounts') == 'Tricount Taiwan') and (row.get('Category') != "Cash swap")
            else abs(float(row['corrected_EUR'])), axis=1)


        # Add transaction_type column with mapped values
        print("📝 Adding transaction_type column...")
        transaction_type_dict = {
            "Income": "income",
            "Exp.": "expense",
            "Transfer-In": "incoming_transfer",
            "Transfer-Out": "outgoing_transfer"
        }

        # Check for unmapped Income/Expense values before filtering
        unmapped_values = df[~df['Income/Expense'].isin(transaction_type_dict.keys())]['Income/Expense'].unique()
        if len(unmapped_values) > 0:
            print(f"⚠️  WARNING: Found unmapped Income/Expense values that will be removed:")
            for val in unmapped_values:
                count = len(df[df['Income/Expense'] == val])
                print(f"   - '{val}': {count} records")

        # Map transaction types
        df['transaction_type'] = df['Income/Expense'].map(transaction_type_dict)

        # Remove records with unmapped values (NaN in transaction_type)
        before_filter = len(df)
        df = df[df['transaction_type'].notna()]
        removed_unmapped = before_filter - len(df)
        if removed_unmapped > 0:
            print(f"🗑️  Removed {removed_unmapped} records with unmapped Income/Expense values")

        # Add movement column (positive for income/incoming transfers, negative for expenses/outgoing transfers)
        print("➕ Adding movement column...")
        df['movement'] = df.apply(
            lambda x: x['corrected_EUR'] if x['transaction_type'] in ['income', 'incoming_transfer']
            else x['corrected_EUR'] * -1,
            axis=1
        )
        print("✅ Movement column added")

        # Drop the old Income/Expense column - now replaced by transaction_type
        df = df.drop(columns=['Income/Expense'])
        print("✅ Removed old Income/Expense column")

        # Save as CSV with UTF-8 encoding (easier for website to handle)
        print(f"💾 Saving processed data to {output_file}...")
        df.to_csv(output_file, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed {len(df)} records")
        print(f"📊 Data range: {df['date'].min()} to {df['date'].max()}")

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

    files_to_upload = ['files/processed_files/finance/moneymgr_processed.csv']

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


def process_moneymgr_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_moneymgr_pipeline(auto_full=True)
    else:
        return create_moneymgr_file()


def full_moneymgr_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Money Manager pipeline with 3 standard options.

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
    print("💰 MONEY MANAGER DATA PIPELINE")
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
        download_success = download_moneymgr_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_moneymgr_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_moneymgr_files()

        # Step 3: Process (fallback to existing files if no new files)
        if move_success:
            process_success = create_moneymgr_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_moneymgr_file()

        # Step 4: Upload
        if process_success:
            upload_success = upload_moneymgr_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n⚙️  Process existing data and upload to Drive...")
        process_success = create_moneymgr_file()
        if process_success:
            success = upload_moneymgr_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "3":
        print("\n⬆️  Upload existing processed files to Drive...")
        success = upload_moneymgr_results()

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Money Manager pipeline completed successfully!")
        # Record successful run
        record_successful_run('finance_moneymgr', 'active')
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
