import pandas as pd
import os
from datetime import date
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import prompt_user_download_status
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.utils.logger import log


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
    log.progress("Starting Money Manager data download...")
    log.info("Instructions:")
    log.info(" 1. Open Money Manager app")
    log.info(" 2. Go to Settings > Export Data")
    log.info(" 3. Export as Excel (.xlsx) format")
    log.info(" 4. Save the file to Downloads folder")
    log.info(" 5. The file should be named with today's date")

    # Note: Money Manager is a mobile app, so no web URL to open
    response = prompt_user_download_status("Money Manager")

    if response:
        log.success(f"Expected file name: {date.today().strftime('%Y-%m-%d')}.xlsx")

    return response


def move_moneymgr_files():
    """
    Moves the downloaded Money Manager file from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    log.info("📁 Moving Money Manager files...")

    download_folder = os.path.expanduser("~/Downloads")
    export_folder = "files/exports/moneymgr_exports"
    expected_file = f"{date.today().strftime('%Y-%m-%d')}.xlsx"
    target_file = "moneymgr_export.xlsx"

    # Check if the expected file exists
    if not check_file_exists(download_folder, expected_file):
        log.error(f"Expected file {expected_file} not found in Downloads")

        # Look for any Excel files with today's date pattern
        log.progress("Looking for alternative Excel files...")
        excel_files = [f for f in os.listdir(download_folder)
                      if f.endswith('.xlsx') and date.today().strftime('%Y-%m-%d') in f]

        if excel_files:
            log.progress(f"Found potential files: {excel_files}")
            chosen_file = input(f"Enter the correct filename (or press Enter for {excel_files[0]}): ").strip()
            expected_file = chosen_file if chosen_file else excel_files[0]
        else:
            log.error("No suitable Excel files found")
            return False

    # Move the file
    success = clean_rename_move_file(
        export_folder=export_folder,
        download_folder=download_folder,
        file_name=expected_file,
        new_file_name=target_file
    )

    if success:
        log.success(f"Successfully moved {expected_file} to {export_folder}/{target_file}")
    else:
        log.error(f"Failed to move {expected_file}")

    return success


def create_moneymgr_file():
    """
    Main processing logic for Money Manager data.
    Reads the Excel file, processes it, and saves as CSV.
    Returns True if successful, False otherwise.

    This is the SOURCE-level processor - outputs raw processed data only.
    Website file generation happens in the topic coordinator.
    """
    log.info("⚙️  Processing Money Manager source data...")

    input_file = "files/exports/moneymgr_exports/moneymgr_export.xlsx"
    output_file = "files/source_processed_files/moneymgr/moneymgr_processed.csv"

    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            log.error(f"Input file not found: {input_file}")
            return False

        # Read the Excel file
        log.progress(f"Reading data from {input_file}...")
        df = pd.read_excel(input_file)

        # Rename Period to date immediately after reading
        df = df.rename(columns={'Period': 'date'})

        # Sort by date
        df.sort_values(by="date", inplace=True)

        # Drop unnecessary column if it exists
        if 'Accounts.1' in df.columns:
            df.drop('Accounts.1', axis=1, inplace=True)

        # Remove specific accounts that are no longer needed
        log.info("🗑️  Filtering out removed accounts...")
        accounts_to_remove = ["Argenta Life Longer Life", "Argenta Life DP Dynamic Allocation", "Savings account"]
        initial_count = len(df)
        df = df[~df['Accounts'].isin(accounts_to_remove)]
        removed_count = initial_count - len(df)
        if removed_count > 0:
            log.info(f"Removed {removed_count} records from: {', '.join(accounts_to_remove)}")

        # Add sorting columns
        df = add_sorting_columns(df)

        log.info("Use unique currency rate for NTD expenses")
        df['corrected_EUR'] = df.apply(lambda row: row['Amount'] * 0.029
                                       if (row.get('Currency') == 'NTD') and (row.get('Accounts') not in ['Personal account', 'Cash'])
                                       else row['EUR'], axis = 1)

        log.info("Adjusting for tricount expenses")
        df['corrected_EUR'] = df.apply(lambda row:
            abs(float(row['corrected_EUR'])) / 2 if (row.get('Accounts') == 'Tricount Taiwan') and (row.get('Category') != "Cash swap")
            else abs(float(row['corrected_EUR'])), axis=1)


        # Add transaction_type column with mapped values
        log.info("Adding transaction_type column...")
        transaction_type_dict = {
            "Income": "income",
            "Exp.": "expense",
            "Transfer-In": "incoming_transfer",
            "Transfer-Out": "outgoing_transfer"
        }

        # Check for unmapped Income/Expense values before filtering
        unmapped_values = df[~df['Income/Expense'].isin(transaction_type_dict.keys())]['Income/Expense'].unique()
        if len(unmapped_values) > 0:
            log.warning(f"WARNING: Found unmapped Income/Expense values that will be removed:")
            for val in unmapped_values:
                count = len(df[df['Income/Expense'] == val])
                log.info(f"- '{val}': {count} records")

        # Map transaction types
        df['transaction_type'] = df['Income/Expense'].map(transaction_type_dict)

        # Remove records with unmapped values (NaN in transaction_type)
        before_filter = len(df)
        df = df[df['transaction_type'].notna()]
        removed_unmapped = before_filter - len(df)
        if removed_unmapped > 0:
            log.info(f"🗑️  Removed {removed_unmapped} records with unmapped Income/Expense values")

        # Add movement column (positive for income/incoming transfers, negative for expenses/outgoing transfers)
        log.info("➕ Adding movement column...")
        df['movement'] = df.apply(
            lambda x: x['corrected_EUR'] if x['transaction_type'] in ['income', 'incoming_transfer']
            else x['corrected_EUR'] * -1,
            axis=1
        )
        log.success("Movement column added")

        # Drop the old Income/Expense column - now replaced by transaction_type
        df = df.drop(columns=['Income/Expense'])
        log.success("Removed old Income/Expense column")

        # Enforce snake_case before saving
        df = enforce_snake_case(df, "processed file")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save as CSV with UTF-8 encoding (easier for website to handle)
        log.success(f"Saving source processed data to {output_file}...")
        df.to_csv(output_file, sep='|', index=False, encoding='utf-8')

        log.normal_success(f"Processed {len(df)} records (range: {df['date'].min()} to {df['date'].max()})")
        log.progress(f"Data range: {df['date'].min()} to {df['date'].max()}")

        return True

    except Exception as e:
        log.error(f"Error processing Money Manager data: {e}")
        return False


def full_moneymgr_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Money Manager SOURCE pipeline with 2 options.

    Options:
    1. Download new data and process
    2. Process existing data

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    log.info("\n" + "="*60)
    log.progress("MONEY MANAGER SOURCE DATA PIPELINE")
    log.info("="*60)

    if auto_process_only:
        log.info("🤖 Auto process mode: Processing existing data...")
        choice = "2"
    elif auto_full:
        log.info("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        log.prompt("\nSelect an option:")
        log.prompt("1. Download new data and process")
        log.prompt("2. Process existing data")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        log.milestone("\n Download new data and process...")

        # Step 1: Download
        download_success = download_moneymgr_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_moneymgr_files()
        else:
            log.warning("Download not confirmed, but checking for existing files...")
            move_success = move_moneymgr_files()

        # Step 3: Process (fallback to existing files if no new files)
        if move_success:
            process_success = create_moneymgr_file()
        else:
            log.warning("No new files found, attempting to process existing files...")
            process_success = create_moneymgr_file()

        success = process_success

    elif choice == "2":
        log.info("\n⚙️  Process existing data...")
        success = create_moneymgr_file()

    else:
        log.error("Invalid choice. Please select 1-2.")
        return False

    # Final status
    log.info("\n" + "="*60)
    if success:
        log.success("Money Manager source pipeline completed successfully!")
        log.info("Note: To upload to Drive, run the Finance topic pipeline.")
        # Record successful run
        record_successful_run('source_moneymgr', 'active')
    else:
        log.error("Money Manager source pipeline failed")
    log.info("="*60)

    return success


# Legacy function for backward compatibility - redirects to new structure
def process_moneymgr_export(upload="Y"):
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use full_moneymgr_pipeline() for source processing or
    the Finance topic coordinator for website generation and upload.
    """
    log.warning("process_moneymgr_export() is deprecated.")
    log.info(" Using new source pipeline...")
    return full_moneymgr_pipeline(auto_process_only=True)


if __name__ == "__main__":
    # Allow running this file directly
    log.progress("Money Manager Source Processing Tool")
    log.info("This tool processes Money Manager exports into source data files.")
    log.info("For website generation and upload, use the Finance topic coordinator.")

    # Run the pipeline
    full_moneymgr_pipeline(auto_full=False)
