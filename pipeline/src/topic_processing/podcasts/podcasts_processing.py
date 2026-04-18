import pandas as pd
import os
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.pocket_casts.pocket_casts_processing import full_pocket_casts_pipeline
from src.topic_processing.website_maintenance.website_maintenance_processing import full_website_maintenance_pipeline
from src.utils.logger import log


def generate_podcasts_website_page_files(df):
    """Generate website-optimized files for the Podcasts page."""
    log.progress(f"\n Generating website files for Podcasts page...")

    try:
        website_dir = 'files/website_files/podcasts'
        os.makedirs(website_dir, exist_ok=True)

        df_web = df.copy()
        df_web = enforce_snake_case(df_web, "podcasts_page_data")

        # Add derived date columns
        log.info("📅 Adding derived date columns...")
        df_web['listened_year'] = pd.to_datetime(df_web['listened_date']).dt.year.astype('Int64').astype(str)
        df_web['listened_month'] = pd.to_datetime(df_web['listened_date']).dt.month.astype('Int64').astype(str)
        df_web['listened_quarter'] = pd.to_datetime(df_web['listened_date']).dt.quarter.astype('Int64').astype(str)

        website_path = f'{website_dir}/podcasts_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        log.success(f"Website file: {len(df_web):,} records  {website_path}")

        return True

    except Exception as e:
        log.error(f"Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_podcasts_topic_file():
    """Creates the Podcasts topic file by reading pocket_casts source data."""
    log.info(f"⚙️  Creating Podcasts topic files...")

    source_file = "files/source_processed_files/pocket_casts/pocket_casts_processed.csv"
    topic_output_file = "files/topic_processed_files/podcasts/podcasts_processed.csv"

    try:
        if not os.path.exists(source_file):
            log.error(f"Source file not found: {source_file}")
            log.info(f"Run the pocket_casts source pipeline first.")
            return False

        log.progress(f"Reading source data from {source_file}...")
        df = pd.read_csv(source_file, sep='|', encoding='utf-8').drop(["is_new_podcast", "is_recurring_podcast"], axis = 1)
        log.success(f"Loaded {len(df)} records")

        os.makedirs(os.path.dirname(topic_output_file), exist_ok=True)
        df["listened_minutes"] = df["listened_seconds"] / 60
        df["listened_hours"] = df["listened_minutes"] / 60

        # First listen flags (True only for the first occurrence)
        df = df.sort_values("listened_date")
        df['is_new_podcast'] = df.groupby('podcast_name').cumcount() == 0

        # Milestone flags (True only at the 10th/5th listen)
        df['is_new_recurring_podcast'] = df.groupby('podcast_name').cumcount() == 9  # 10th listen (0-indexed)

        # Recurring status (True for ALL listens once threshold reached)
        # Use transform to get total count per podcast across all time
        podcast_total_count = df.groupby('podcast_name')['podcast_name'].transform('count')

        df['is_recurring_podcast'] = podcast_total_count >= 10

        bool_cols = ['is_new_podcast', 'is_recurring_podcast', 'is_new_recurring_podcast']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        log.success(f"Converted boolean columns to integers")

        df = df.sort_values("listened_date", ascending = False)
        df.to_csv(topic_output_file, sep='|', index=False, encoding='utf-8')
        log.success(f"Saved topic file to {topic_output_file}")

        website_success = generate_podcasts_website_page_files(df)
        return website_success

    except Exception as e:
        log.error(f"Error creating Podcasts topic files: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_podcasts_results():
    """Uploads the processed Podcasts files to Google Drive."""
    log.progress(f"Uploading Podcasts results to Google Drive...")

    files_to_upload = ['files/website_files/podcasts/podcasts_page_data.csv']
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        log.error("No files found to upload")
        return False

    log.info(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        log.success(f"Podcasts results uploaded successfully!")
    else:
        log.error("Some files failed to upload")

    return success


def full_podcasts_pipeline(auto_full=False, auto_process_only=False, skip_source=False, merge_only=False):
    """Complete Podcasts TOPIC pipeline with 3 standard options."""
    log.normal("Processing Podcasts...")
    log.info("\n" + "="*60)
    log.info("🎙️ PODCASTS TOPIC PIPELINE")
    log.info("="*60)

    if merge_only or auto_process_only:
        choice = "2"
    elif auto_full:
        choice = "1"
    else:
        log.prompt("\nSelect an option:")
        log.prompt("1. Run source pipeline, create topic files, and upload to Drive")
        log.prompt("2. Create topic files from existing source data and upload to Drive")
        log.prompt("3. Upload existing topic/website files to Drive")
        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        log.milestone(f"\n Running full Podcasts pipeline...")
        if not skip_source:
            log.info(f"\n📥 Step 1: Running pocket_casts source pipeline...")
            source_success = full_pocket_casts_pipeline(auto_process_only=True)
            if not source_success:
                log.warning("Source pipeline failed, but attempting to use existing source data...")

        log.progress(f"\n Step 2: Creating Podcasts topic files...")
        topic_success = create_podcasts_topic_file()

        if topic_success:
            log.progress("\n Step 3: Uploading to Drive...")
            success = upload_podcasts_results()
        else:
            log.error("Topic file creation failed, skipping upload")

    elif choice == "2":
        log.info("\n⚙️  Creating topic files from existing source data and uploading...")
        topic_success = create_podcasts_topic_file()
        if topic_success:
            success = upload_podcasts_results()

    elif choice == "3":
        log.info("\n⬆️  Uploading existing files to Drive...")
        success = upload_podcasts_results()

    else:
        log.error("Invalid choice. Please select 1-3.")
        return False

    log.info("\n" + "="*60)
    if success:
        log.success(f"Podcasts topic pipeline completed successfully!")
        log.normal_success("Podcasts processing completed successfully")
        record_successful_run('topic_podcasts', 'active')
        # Update website tracking file
        full_website_maintenance_pipeline(auto_mode=True, quiet=True)
    else:
        log.error(f"Podcasts topic pipeline failed")
        log.normal("Podcasts processing failed")
    log.info("="*60)

    return success


if __name__ == "__main__":
    log.info(f"🎙️ Podcasts Topic Processing Tool")
    if not verify_drive_connection():
        log.warning("Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()
    full_podcasts_pipeline(auto_full=False)
