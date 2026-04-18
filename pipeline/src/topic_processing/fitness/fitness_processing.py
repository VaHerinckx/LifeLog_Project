import pandas as pd
import os
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.garmin.garmin_processing import full_garmin_pipeline
from src.utils.logger import log


def generate_fitness_website_page_files(df):
    """Generate website-optimized files for the Fitness page."""
    log.progress(f"\n Generating website files for Fitness page...")

    try:
        website_dir = 'files/website_files/fitness'
        os.makedirs(website_dir, exist_ok=True)

        df_web = df.copy()
        df_web = enforce_snake_case(df_web, "fitness_page_data")

        website_path = f'{website_dir}/fitness_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        log.success(f"Website file: {len(df_web):,} records  {website_path}")

        return True

    except Exception as e:
        log.error(f"Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_fitness_topic_file():
    """Creates the Fitness topic file by merging Garmin and Polar source data."""
    log.info(f"⚙️  Creating Fitness topic files...")

    garmin_file = "files/source_processed_files/garmin/garmin_activities_list_processed.csv"
    polar_file = "files/source_processed_files/garmin/polar_summary_processed.csv"
    topic_output_file = "files/topic_processed_files/fitness/fitness_processed.csv"

    try:
        dataframes = []

        # Load Garmin data
        if os.path.exists(garmin_file):
            log.progress(f"Reading Garmin data from {garmin_file}...")
            df_garmin = pd.read_csv(garmin_file, sep='|', encoding='utf-8')
            dataframes.append(df_garmin)
            log.success(f"Loaded {len(df_garmin)} Garmin records")
        else:
            log.warning(f"Garmin file not found: {garmin_file}")

        # Load Polar data (optional - legacy data)
        if os.path.exists(polar_file):
            log.progress(f"Reading Polar data from {polar_file}...")
            df_polar = pd.read_csv(polar_file, sep='|', encoding='utf-8')
            dataframes.append(df_polar)
            log.success(f"Loaded {len(df_polar)} Polar records")

        if not dataframes:
            log.error(f"No source files found. Run source pipelines first.")
            return False

        # Merge all sources
        df = pd.concat(dataframes, ignore_index=True)
        log.progress(f"Total: {len(df)} fitness records")

        os.makedirs(os.path.dirname(topic_output_file), exist_ok=True)
        df.to_csv(topic_output_file, sep='|', index=False, encoding='utf-8')
        log.success(f"Saved topic file to {topic_output_file}")

        website_success = generate_fitness_website_page_files(df)
        return website_success

    except Exception as e:
        log.error(f"Error creating Fitness topic files: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_fitness_results():
    """Uploads the processed Fitness files to Google Drive."""
    log.progress(f"Uploading Fitness results to Google Drive...")

    files_to_upload = ['files/website_files/fitness/fitness_page_data.csv']
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        log.error("No files found to upload")
        return False

    log.info(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        log.success(f"Fitness results uploaded successfully!")
    else:
        log.error("Some files failed to upload")

    return success


def full_fitness_pipeline(auto_full=False, auto_process_only=False, skip_source=False, merge_only=False):
    """Complete Fitness TOPIC pipeline with 3 standard options."""
    log.normal("Processing Fitness...")
    log.info("\n" + "="*60)
    log.progress("FITNESS TOPIC PIPELINE")
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
        log.milestone(f"\n Running full Fitness pipeline...")
        if not skip_source:
            log.info(f"\n📥 Step 1: Running garmin source pipeline...")
            source_success = full_garmin_pipeline(auto_process_only=True)
            if not source_success:
                log.warning("Source pipeline failed, but attempting to use existing source data...")
        
        log.progress(f"\n Step 2: Creating Fitness topic files...")
        topic_success = create_fitness_topic_file()

        if topic_success:
            log.progress("\n Step 3: Uploading to Drive...")
            success = upload_fitness_results()
        else:
            log.error("Topic file creation failed, skipping upload")

    elif choice == "2":
        log.info("\n⚙️  Creating topic files from existing source data and uploading...")
        topic_success = create_fitness_topic_file()
        if topic_success:
            success = upload_fitness_results()

    elif choice == "3":
        log.info("\n⬆️  Uploading existing files to Drive...")
        success = upload_fitness_results()

    else:
        log.error("Invalid choice. Please select 1-3.")
        return False

    log.info("\n" + "="*60)
    if success:
        log.success(f"Fitness topic pipeline completed successfully!")
        log.normal_success("Fitness processing completed successfully")
        record_successful_run('topic_fitness', 'active')
    else:
        log.error(f"Fitness topic pipeline failed")
        log.normal("Fitness processing failed")
    log.info("="*60)

    return success


if __name__ == "__main__":
    log.progress(f"Fitness Topic Processing Tool")
    if not verify_drive_connection():
        log.warning("Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()
    full_fitness_pipeline(auto_full=False)
