"""
Location Topic Coordinator

Coordinates location data from multiple sources (Google Maps, Manual Excel).
Merges data, enriches with location types, and generates website files.
Follows the topic coordinator pattern: 3 options (full, process+upload, upload only).

Sources:
- Google Maps Timeline (files/source_processed_files/google_maps/google_maps_processed.csv)
- Manual Excel (files/source_processed_files/manual_location/manual_location_processed.csv)

Output:
- files/topic_processed_files/location/location_processed.csv
- files/website_files/location/location_page_data.csv
"""

import pandas as pd
import os

from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.google_maps.google_maps_processing import full_google_maps_pipeline, download_google_data, move_google_files
from src.sources_processing.manual_location.manual_location_processing import full_manual_location_pipeline
from src.utils.logger import log


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_google_maps_data():
    """
    Load Google Maps location data.

    Returns:
        DataFrame: Google Maps location data or None if not found
    """
    google_path = 'files/source_processed_files/google_maps/google_maps_processed.csv'

    if not os.path.exists(google_path):
        log.warning(f"Google Maps file not found: {google_path}")
        return None

    log.progress(f"Loading Google Maps data...")
    df = pd.read_csv(google_path, sep='|', encoding='utf-8')

    log.success(f"Loaded Google Maps: {len(df):,} records")
    log.info(f"Date range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")

    return df


def load_manual_location_data():
    """
    Load Manual Excel location data.

    Returns:
        DataFrame: Manual location data or None if not found
    """
    manual_path = 'files/source_processed_files/manual_location/manual_location_processed.csv'

    if not os.path.exists(manual_path):
        log.warning(f"Manual location file not found: {manual_path}")
        return None

    log.info(f"Loading Manual location data...")
    df = pd.read_csv(manual_path, sep='|', encoding='utf-8')

    log.success(f"Loaded Manual location: {len(df):,} records")
    log.info(f"Date range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")

    return df


# ============================================================================
# ENRICHMENT FUNCTIONS
# ============================================================================

def enrich_location_with_type(location_df):
    """
    Add location_type column (home/work/other) based on is_home column
    and potentially coordinates or city patterns.

    Args:
        location_df: DataFrame with location data

    Returns:
        DataFrame: location_df with location_type column added
    """
    log.debug("️ Enriching location data with location_type...")

    # Start with 'other' as default
    location_df['location_type'] = 'other'

    # Set 'home' based on is_home column
    location_df.loc[location_df['is_home'] == True, 'location_type'] = 'home'

    # TODO: Add work location detection logic here
    # Example patterns that could be used:
    #
    # 1. Based on specific coordinates (if work location known):
    # WORK_COORDS = "geo:50.8503,4.3517"  # Example coordinates
    # location_df.loc[
    #     (location_df['coordinates'] == WORK_COORDS) &
    #     (location_df['is_home'] == False),
    #     'location_type'
    # ] = 'work'
    #
    # 2. Based on city patterns (if work city known and different from home):
    # location_df.loc[
    #     (location_df['city'] == 'Brussels') &
    #     (location_df['is_home'] == False),
    #     'location_type'
    # ] = 'work'
    #
    # 3. Based on time patterns (weekdays 9-5 at non-home location):
    # location_df['timestamp_dt'] = pd.to_datetime(location_df['timestamp'])
    # location_df['hour'] = location_df['timestamp_dt'].dt.hour
    # location_df['weekday'] = location_df['timestamp_dt'].dt.dayofweek  # 0=Monday
    # location_df.loc[
    #     (location_df['is_home'] == False) &
    #     (location_df['weekday'] < 5) &  # Monday-Friday
    #     (location_df['hour'] >= 9) &
    #     (location_df['hour'] <= 17),
    #     'location_type'
    # ] = 'work'

    home_count = (location_df['location_type'] == 'home').sum()
    work_count = (location_df['location_type'] == 'work').sum()
    other_count = (location_df['location_type'] == 'other').sum()

    log.success(f"Location types assigned:")
    log.info(f"🏠 Home: {home_count:,} records ({home_count/len(location_df)*100:.1f}%)")
    log.info(f"💼 Work: {work_count:,} records ({work_count/len(location_df)*100:.1f}%)")
    log.info(f"📍 Other: {other_count:,} records ({other_count/len(location_df)*100:.1f}%)")

    return location_df


# ============================================================================
# MERGING FUNCTIONS
# ============================================================================

def merge_location_files():
    """
    Merge Google Maps and Manual location files into combined file.

    Returns:
        DataFrame: Merged location data or None if failed
    """
    log.info("\n🔗 Merging location data from all sources...")

    try:
        files_to_merge = []

        # Load Google Maps data
        google_df = load_google_maps_data()
        if google_df is not None:
            files_to_merge.append(google_df)

        # Load Manual location data
        manual_df = load_manual_location_data()
        if manual_df is not None:
            files_to_merge.append(manual_df)

        if not files_to_merge:
            log.error("No location files found to merge")
            return None

        # Merge dataframes
        combined_df = pd.concat(files_to_merge, ignore_index=True)

        # Convert timestamp to datetime, handling timezone issues
        log.progress("Processing timestamps...")

        # Convert all timestamps to naive datetime (remove timezone info)
        def normalize_timestamp(timestamp_str):
            try:
                # Parse the timestamp
                dt = pd.to_datetime(timestamp_str)

                # If it's timezone-aware, convert to naive (remove timezone)
                if dt.tz is not None:
                    dt = dt.tz_localize(None)

                return dt
            except Exception as e:
                log.warning(f"Error parsing timestamp {timestamp_str}: {e}")
                return pd.NaT

        combined_df['timestamp_parsed'] = combined_df['timestamp'].apply(normalize_timestamp)

        # Remove any rows with invalid timestamps
        valid_rows = combined_df['timestamp_parsed'].notna()
        combined_df = combined_df[valid_rows]

        if len(combined_df) == 0:
            log.error("No valid timestamps found after processing")
            return None

        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp_parsed')

        # Remove duplicates (prefer Google data over manual if same timestamp)
        # Group by hour to handle potential overlaps
        combined_df['hour_key'] = combined_df['timestamp_parsed'].dt.strftime('%Y-%m-%d-%H')

        # For each hour, prefer google data if available
        def choose_best_record(group):
            if len(group) == 1:
                return group.iloc[0]

            # Prefer google_maps data over manual
            google_records = group[group['source'] == 'google_maps']
            if len(google_records) > 0:
                return google_records.iloc[0]  # Take first google record
            else:
                return group.iloc[0]  # Take first manual record

        # Apply the selection logic
        log.progress("Resolving duplicate timestamps...")
        deduplicated_df = combined_df.groupby('hour_key').apply(choose_best_record).reset_index(drop=True)

        # Clean up temporary columns
        deduplicated_df = deduplicated_df.drop(['timestamp_parsed', 'hour_key', 'source'], axis=1)

        # Ensure timestamps are in string format
        if 'timestamp' not in deduplicated_df.columns:
            log.error("Timestamp column missing after processing")
            return None

        # Sort final dataframe by timestamp string
        deduplicated_df = deduplicated_df.sort_values('timestamp')

        # Enrich with location_type (home/work/other)
        deduplicated_df = enrich_location_with_type(deduplicated_df)

        log.success(f"Successfully merged location files!")
        log.progress(f"Combined records: {len(deduplicated_df):,}")
        log.info(f"📅 Date range: {deduplicated_df['timestamp'].min()[:10]} to {deduplicated_df['timestamp'].max()[:10]}")
        log.progress(f"Countries: {', '.join(deduplicated_df['country'].unique()[:5])}{'...'if len(deduplicated_df['country'].unique()) > 5 else ''}")

        # Show sample of merged data
        log.debug(f"\n Sample merged records:")
        sample_df = deduplicated_df.head(5)[['timestamp', 'timezone', 'city', 'country', 'is_home']]
        for _, row in sample_df.iterrows():
            home_status = "🏠" if row['is_home'] else "📍"
            log.info(f"• {row['timestamp'][:16]} | {row['timezone']} | {row['city']}, {row['country']} {home_status}")

        return deduplicated_df

    except Exception as e:
        log.error(f"Error merging location files: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def create_location_files():
    """
    Main processing function that loads location data from sources and generates output files.

    Returns:
        bool: True if successful, False otherwise
    """
    log.info("\n" + "="*70)
    log.info("📍 LOCATION DATA PROCESSING")
    log.info("="*70)

    try:
        # STEP 1: Merge location data from all sources
        log.progress("\n STEP 1: Merging location data from sources...")
        location_df = merge_location_files()

        if location_df is None or len(location_df) == 0:
            log.error("No location data loaded from sources")
            return False

        # STEP 2: Enforce snake_case
        log.info("\n🔤 STEP 2: Enforcing snake_case column names...")
        location_df = enforce_snake_case(location_df, "location_processed")

        # STEP 3: Save processed file
        log.success("\n STEP 3: Saving processed files...")
        location_dir = 'files/topic_processed_files/location'
        os.makedirs(location_dir, exist_ok=True)

        # Main location file
        location_output = f'{location_dir}/location_processed.csv'
        location_df.to_csv(location_output, sep='|', index=False, encoding='utf-8')

        log.success(f"Saved location data: {len(location_df):,} records")
        log.info(f"Output: {location_output}")

        # STEP 4: Generate website files
        log.progress("\n STEP 4: Generating website files...")
        website_success = generate_location_website_page_files(location_df)

        if not website_success:
            log.warning("Website file generation had issues")

        return True

    except Exception as e:
        log.error(f"\n Error processing location data: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_location_website_page_files(df):
    """
    Generate website-optimized files for the Location page.

    Args:
        df: Processed dataframe (already in snake_case)

    Returns:
        bool: True if successful, False otherwise
    """
    log.progress("Generating website files for Location page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/location'
        os.makedirs(website_dir, exist_ok=True)

        # Work with copy to avoid modifying original
        df_web = df.copy()

        # Enforce snake_case before saving
        df_web = enforce_snake_case(df_web, "location_page_data")

        # Add derived date columns
        log.info("📅 Adding derived date columns...")
        df_web['location_year'] = pd.to_datetime(df_web['timestamp']).dt.year.astype('Int64').astype(str)
        df_web['location_month'] = pd.to_datetime(df_web['timestamp']).dt.month.astype('Int64').astype(str)
        df_web['location_quarter'] = pd.to_datetime(df_web['timestamp']).dt.quarter.astype('Int64').astype(str)

        # Save website file
        website_path = f'{website_dir}/location_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        log.success(f"Website file: {len(df_web):,} records  {website_path}")

        return True

    except Exception as e:
        log.error(f"Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_location_results():
    """
    Uploads processed location files to Google Drive.

    Returns:
        bool: True if successful, False otherwise
    """
    log.progress("\n Uploading location results to Google Drive...")

    files_to_upload = [
        "files/topic_processed_files/location/location_processed.csv",
        "files/website_files/location/location_page_data.csv"
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        log.error("No files found to upload")
        return False

    log.info(f"📤 Uploading {len(existing_files)} file(s)...")
    for f in existing_files:
        log.info(f"• {f}")

    success = upload_multiple_files(existing_files)

    if success:
        log.success("Location results uploaded successfully!")
    else:
        log.error("Some files failed to upload")

    return success


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_location_pipeline(auto_full=False, auto_process_only=False, merge_only=False):
    """
    Complete Location TOPIC COORDINATOR pipeline.

    Options:
    1. Full pipeline (download → process sources → merge → upload)
    2. Process existing data and upload to Drive
    3. Upload existing processed files to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input
        merge_only (bool): If True, skip source pipelines and just merge + upload
                           (used when sources are already processed by main orchestrator)

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    log.normal("Processing Location...")
    log.info("\n" + "="*70)
    log.info("📍 LOCATION TOPIC COORDINATOR PIPELINE")
    log.info("="*70)

    if merge_only:
        # Called from main orchestrator — sources already processed, just merge + upload
        log.info("🔗 Merge mode: Merging existing source data and uploading...")

        success = False
        process_success = create_location_files()

        if process_success:
            success = upload_location_results()
        else:
            log.error("Merge failed, skipping upload")
            success = False

    else:
        if auto_process_only:
            log.info("🤖 Auto process mode: Processing existing data and uploading...")
            choice = "2"
        elif auto_full:
            log.info("🤖 Auto mode: Running full pipeline...")
            choice = "1"
        else:
            log.prompt("\nSelect an option:")
            log.prompt("1. Full pipeline (download  process sources  merge  upload)")
            log.prompt("2. Process existing data and upload to Drive")
            log.prompt("3. Upload existing processed files to Drive")

            choice = input("\nEnter your choice (1-3): ").strip()

        success = False

        if choice == "1":
            log.milestone("\n Option 1: Full pipeline...")

            # ============================================================
            # PHASE 1: DOWNLOAD - All user interaction upfront
            # ============================================================
            log.info("\n" + "="*60)
            log.info("📥 DOWNLOAD PHASE - User interaction required")
            log.info("="*60)
            log.info("\nYou will be prompted to download Google Maps data.")
            log.info("After download is complete, processing will run automatically.\n")

            download_status = {}

            # Download 1: Google Maps
            log.progress("\n Download 1/1: Google Maps Timeline...")
            download_confirmed = download_google_data()
            if download_confirmed:
                move_success = move_google_files()
                download_status['google_maps'] = move_success
                if move_success:
                    log.success("Google Maps files ready")
                else:
                    log.warning("Google Maps file move failed")
            else:
                download_status['google_maps'] = False
                log.info("⏭️  Skipping Google Maps download")

            # Summary of downloads
            log.info("\n" + "="*60)
            log.progress("DOWNLOAD SUMMARY")
            log.info("="*60)
            for source, status in download_status.items():
                status_icon = "✅" if status else "⏭️"
                log.info(f"{status_icon} {source.replace('_', ' ').title()}: {'Ready' if status else 'Skipped'}")

            # ============================================================
            # PHASE 2: PROCESSING - Automated, no user interaction
            # ============================================================
            log.info("\n" + "="*60)
            log.info("⚙️  AUTOMATED PROCESSING PHASE - No more user input needed")
            log.info("="*60)

            # Step 1: Process Google Maps
            log.progress("\n Step 1/4: Processing Google Maps data...")
            try:
                google_success = full_google_maps_pipeline(auto_process_only=True)
                if not google_success:
                    log.warning("Google Maps pipeline failed, continuing with available data...")
            except Exception as e:
                log.warning(f"Error in Google Maps pipeline: {e}")

            # Step 2: Process Manual Location
            log.info("\n Step 2/4: Processing Manual location data...")
            try:
                manual_success = full_manual_location_pipeline(auto_process_only=True)
                if not manual_success:
                    log.warning("Manual location pipeline failed, continuing with available data...")
            except Exception as e:
                log.warning(f"Error in Manual location pipeline: {e}")

            # Step 3: Merge location data
            log.info("\n🔗 Step 3/4: Merging location data...")
            process_success = create_location_files()
            if not process_success:
                log.error("Location data merge failed, stopping pipeline")
                return False

            # Step 4: Upload results
            log.progress("\n Step 4/4: Uploading results...")
            success = upload_location_results()

        elif choice == "2":
            log.info("\n⚙️  Option 2: Process existing data and upload...")
            log.info(" (Merges already-processed source files)")

            # Merge location data from existing processed files
            process_success = create_location_files()

            if process_success:
                success = upload_location_results()
            else:
                log.error("Processing failed, skipping upload")
                success = False

        elif choice == "3":
            log.progress("\n Option 3: Upload existing processed files...")
            success = upload_location_results()

        else:
            log.error("Invalid choice. Please select 1-3.")
            return False

    # Final status
    log.info("\n" + "="*70)
    if success:
        log.success("Location topic coordinator completed successfully!")
        log.normal_success("Location processing completed successfully")
        log.progress("Your location dataset is ready for analysis!")
        # Record successful run
        record_successful_run('topic_location', 'active')
    else:
        log.error("Location coordination pipeline failed")
        log.normal("Location processing failed")
    log.info("="*70)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    log.info("📍 Location Topic Coordinator")
    log.info("This tool coordinates location data from multiple sources.")

    # Test drive connection first
    if not verify_drive_connection():
        log.warning("Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_location_pipeline(auto_full=False)
