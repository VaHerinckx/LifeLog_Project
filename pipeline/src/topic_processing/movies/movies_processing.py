import pandas as pd
import os
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.letterboxd.letterboxd_processing import full_letterboxd_pipeline
from src.topic_processing.website_maintenance.website_maintenance_processing import full_website_maintenance_pipeline
from src.utils.logger import log


def generate_movies_website_page_files(df):
    """
    Generate website-optimized files for the Movies page (Letterboxd data).

    Args:
        df: Processed dataframe (already in snake_case)

    Returns:
        bool: True if successful, False otherwise
    """
    log.progress("\n Generating website files for Movies page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/movies'
        os.makedirs(website_dir, exist_ok=True)

        # Work with copy to avoid modifying original
        df_web = df.copy()

        # Add movie_id column BEFORE genre splitting
        # movie_id uniquely identifies each watch (same movie watched multiple times = different IDs)
        log.info("Adding movie_id to website file...")
        df_web['movie_id'] = range(len(df_web))

        # Split comma-separated genres into individual rows for website filtering
        # This allows filtering by individual genres on the website
        log.progress("Splitting genres into individual rows for website...")
        rows = []
        for _, row in df_web.iterrows():
            genres = str(row['genre']).split(',')
            for genre in genres:
                genre = genre.strip()
                if genre and genre != 'Unknown':
                    new_row = row.copy()
                    new_row['genre'] = genre
                    rows.append(new_row)

        if rows:
            df_web = pd.DataFrame(rows)
            log.success(f"Expanded website file to {len(df_web)} rows (one per genre)")

        # Reorder columns to put movie_id first for clarity
        cols = df_web.columns.tolist()
        if 'movie_id' in cols:
            cols.remove('movie_id')
            cols = ['movie_id'] + cols
            df_web = df_web[cols]

        # Add extra columns to simplify website processing
        df_web["runtime_hour"] = (df_web["runtime"] / 60).astype(float)


        # Enforce snake_case before saving
        df_web = enforce_snake_case(df_web, "movies_page_letterboxd_data")

        # Add derived date columns
        log.info("📅 Adding derived date columns...")
        df_web['watch_year'] = pd.to_datetime(df_web['date']).dt.year.astype('Int64').astype(str)
        df_web['watch_month'] = pd.to_datetime(df_web['date']).dt.month.astype('Int64').astype(str)
        df_web['watch_quarter'] = pd.to_datetime(df_web['date']).dt.quarter.astype('Int64').astype(str)

        # Save website file
        website_path = f'{website_dir}/movies_page_letterboxd_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        log.success(f"Website file: {len(df_web):,} records  {website_path}")

        return True

    except Exception as e:
        log.error(f"Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_movies_topic_file():
    """
    Creates the Movies topic file by reading Letterboxd source data
    and generating website files.

    For Movies, we only have one source (Letterboxd), so this is a pass-through
    with website file generation.

    Returns:
        bool: True if successful, False otherwise
    """
    log.info("⚙️  Creating Movies topic files...")

    source_file = "files/source_processed_files/letterboxd/letterboxd_processed.csv"
    topic_output_file = "files/topic_processed_files/movies/movies_processed.csv"

    try:
        # Check if source file exists
        if not os.path.exists(source_file):
            log.error(f"Source file not found: {source_file}")
            log.info(" Run the Letterboxd source pipeline first.")
            return False

        # Read source data
        log.progress(f"Reading source data from {source_file}...")
        df = pd.read_csv(source_file, sep='|', encoding='utf-8')
        log.success(f"Loaded {len(df)} records")

        # For Movies, the topic file is the same as source (single source, no merging needed)
        # But we save it to topic_processed_files for consistency
        os.makedirs(os.path.dirname(topic_output_file), exist_ok=True)
        df.to_csv(topic_output_file, sep='|', index=False, encoding='utf-8')
        log.success(f"Saved topic file to {topic_output_file}")

        # Generate website files
        website_success = generate_movies_website_page_files(df)

        return website_success

    except Exception as e:
        log.error(f"Error creating Movies topic files: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_movies_results():
    """
    Uploads the processed Movies files to Google Drive.
    Returns True if successful, False otherwise.
    """
    log.progress("Uploading Movies results to Google Drive...")

    files_to_upload = ['files/website_files/movies/movies_page_letterboxd_data.csv']

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        log.error("No files found to upload")
        return False

    log.info(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        log.success("Movies results uploaded successfully!")
    else:
        log.error("Some files failed to upload")

    return success


def full_movies_pipeline(auto_full=False, auto_process_only=False, skip_source=False, merge_only=False):
    """
    Complete Movies TOPIC pipeline with 3 standard options.

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
    log.normal("Processing Movies...")
    log.info("\n" + "="*60)
    log.progress("MOVIES TOPIC PIPELINE")
    log.info("="*60)

    if merge_only or auto_process_only:
        log.info("🤖 Auto process mode: Creating topic files and uploading...")
        choice = "2"
    elif auto_full:
        log.info("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        log.prompt("\nSelect an option:")
        log.prompt("1. Run source pipeline, create topic files, and upload to Drive")
        log.prompt("2. Create topic files from existing source data and upload to Drive")
        log.prompt("3. Upload existing topic/website files to Drive")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        log.milestone("\n Running full Movies pipeline...")

        # Step 1: Run Letterboxd source pipeline (unless skipped)
        if not skip_source:
            log.info("\n📥 Step 1: Running Letterboxd source pipeline...")
            source_success = full_letterboxd_pipeline(auto_process_only=True)
            if not source_success:
                log.warning("Source pipeline failed, but attempting to use existing source data...")
        else:
            log.info("\n⏭️  Skipping source pipeline (using existing data)...")

        # Step 2: Create topic files
        log.progress("\n Step 2: Creating Movies topic files...")
        topic_success = create_movies_topic_file()

        # Step 3: Upload
        if topic_success:
            log.progress("\n Step 3: Uploading to Drive...")
            upload_success = upload_movies_results()
            success = upload_success
        else:
            log.error("Topic file creation failed, skipping upload")
            success = False

    elif choice == "2":
        log.info("\n⚙️  Creating topic files from existing source data and uploading...")

        # Step 1: Create topic files
        topic_success = create_movies_topic_file()

        # Step 2: Upload
        if topic_success:
            success = upload_movies_results()
        else:
            log.error("Topic file creation failed, skipping upload")
            success = False

    elif choice == "3":
        log.info("\n⬆️  Uploading existing files to Drive...")
        success = upload_movies_results()

    else:
        log.error("Invalid choice. Please select 1-3.")
        return False

    # Final status
    log.info("\n" + "="*60)
    if success:
        log.success("Movies topic pipeline completed successfully!")
        log.normal_success("Movies processing completed successfully")
        # Record successful run
        record_successful_run('topic_movies', 'active')
        # Update website tracking file
        full_website_maintenance_pipeline(auto_mode=True, quiet=True)
    else:
        log.error("Movies topic pipeline failed")
        log.normal("Movies processing failed")
    log.info("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    log.progress("Movies Topic Processing Tool")
    log.info("This tool coordinates Movies data sources and generates website files.")

    # Test drive connection first
    if not verify_drive_connection():
        log.warning("Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_movies_pipeline(auto_full=False)
