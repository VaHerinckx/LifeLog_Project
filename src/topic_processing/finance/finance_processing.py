import pandas as pd
import os
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.moneymgr.moneymgr_processing import full_moneymgr_pipeline


def generate_finance_website_page_files(df):
    """
    Generate website-optimized files for the Finance page.

    Args:
        df: Processed dataframe (already in snake_case)

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🌐 Generating website files for Finance page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/finance'
        os.makedirs(website_dir, exist_ok=True)

        # Work with copy to avoid modifying original
        df_web = df.copy()

        # Set category to null for transfer transactions
        print("🔄 Clearing category for transfer transactions...")
        transfer_mask = df_web['transaction_type'].isin(['incoming_transfer', 'outgoing_transfer'])
        transfer_count = transfer_mask.sum()
        df_web.loc[transfer_mask, 'category'] = pd.NA
        print(f"   Cleared category for {transfer_count} transfer transactions")

        # Enforce snake_case before saving
        df_web = enforce_snake_case(df_web, "finance_page_data")

        # Save website file
        website_path = f'{website_dir}/finance_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Website file: {len(df_web):,} records → {website_path}")

        return True

    except Exception as e:
        print(f"❌ Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_finance_topic_file():
    """
    Creates the Finance topic file by reading Money Manager source data
    and generating website files.

    For Finance, we only have one source (Money Manager), so this is a pass-through
    with website file generation.

    Returns:
        bool: True if successful, False otherwise
    """
    print("⚙️  Creating Finance topic files...")

    source_file = "files/source_processed_files/moneymgr/moneymgr_processed.csv"
    topic_output_file = "files/topic_processed_files/finance/finance_processed.csv"

    try:
        # Check if source file exists
        if not os.path.exists(source_file):
            print(f"❌ Source file not found: {source_file}")
            print("   Run the Money Manager source pipeline first.")
            return False

        # Read source data
        print(f"📖 Reading source data from {source_file}...")
        df = pd.read_csv(source_file, sep='|', encoding='utf-8')
        print(f"✅ Loaded {len(df)} records")

        # For Finance, the topic file is the same as source (single source, no merging needed)
        # But we save it to topic_processed_files for consistency
        os.makedirs(os.path.dirname(topic_output_file), exist_ok=True)
        df.to_csv(topic_output_file, sep='|', index=False, encoding='utf-8')
        print(f"💾 Saved topic file to {topic_output_file}")

        # Generate website files
        website_success = generate_finance_website_page_files(df)

        return website_success

    except Exception as e:
        print(f"❌ Error creating Finance topic files: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_finance_results():
    """
    Uploads the processed Finance files to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading Finance results to Google Drive...")

    files_to_upload = ['files/website_files/finance/finance_page_data.csv']

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Finance results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


def full_finance_pipeline(auto_full=False, auto_process_only=False, skip_source=False):
    """
    Complete Finance TOPIC pipeline with 3 standard options.

    Options:
    1. Run source pipeline, create topic files, and upload to Drive
    2. Create topic files from existing source data and upload to Drive
    3. Upload existing topic/website files to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input
        skip_source (bool): If True, skips running the source pipeline (assumes source data exists)

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("💰 FINANCE TOPIC PIPELINE")
    print("="*60)

    if auto_process_only:
        print("🤖 Auto process mode: Creating topic files and uploading...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Run source pipeline, create topic files, and upload to Drive")
        print("2. Create topic files from existing source data and upload to Drive")
        print("3. Upload existing topic/website files to Drive")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Running full Finance pipeline...")

        # Step 1: Run Money Manager source pipeline (unless skipped)
        if not skip_source:
            print("\n📥 Step 1: Running Money Manager source pipeline...")
            source_success = full_moneymgr_pipeline(auto_process_only=True)
            if not source_success:
                print("⚠️  Source pipeline failed, but attempting to use existing source data...")
        else:
            print("\n⏭️  Skipping source pipeline (using existing data)...")

        # Step 2: Create topic files
        print("\n📊 Step 2: Creating Finance topic files...")
        topic_success = create_finance_topic_file()

        # Step 3: Upload
        if topic_success:
            print("\n☁️  Step 3: Uploading to Drive...")
            upload_success = upload_finance_results()
            success = upload_success
        else:
            print("❌ Topic file creation failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n⚙️  Creating topic files from existing source data and uploading...")

        # Step 1: Create topic files
        topic_success = create_finance_topic_file()

        # Step 2: Upload
        if topic_success:
            success = upload_finance_results()
        else:
            print("❌ Topic file creation failed, skipping upload")
            success = False

    elif choice == "3":
        print("\n⬆️  Uploading existing files to Drive...")
        success = upload_finance_results()

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Finance topic pipeline completed successfully!")
        # Record successful run
        record_successful_run('topic_finance', 'active')
    else:
        print("❌ Finance topic pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("💰 Finance Topic Processing Tool")
    print("This tool coordinates Finance data sources and generates website files.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_finance_pipeline(auto_full=False)
