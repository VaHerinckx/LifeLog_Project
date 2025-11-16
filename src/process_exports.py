import subprocess
import sys
from src.music.lastfm_processing import full_lfm_pipeline
from src.books.books_processing import full_books_pipeline
from src.books.goodreads_processing import download_goodreads_data, move_goodreads_files
from src.books.kindle_processing import download_kindle_data, move_kindle_files
from src.podcasts.pocket_casts_processing import full_pocket_casts_pipeline, move_pocket_casts_files, download_pocket_casts_data
from src.sport.garmin_processing import full_garmin_pipeline, move_garmin_files, download_garmin_data
from src.finance.moneymgr_processing import full_moneymgr_pipeline, move_moneymgr_files, download_moneymgr_data
from src.nutrilio.nutrilio_processing import full_nutrilio_pipeline, move_nutrilio_files, download_nutrilio_data
from src.health.apple_processing import full_apple_pipeline, move_apple_files, download_apple_data
from src.screentime.offscreen_processing import full_offscreen_pipeline, move_offscreen_files, download_offscreen_data
from src.weather.weather_processing import get_weather_data
from src.movies.letterboxd_processing import full_letterboxd_pipeline, move_letterboxd_files, download_letterboxd_data
from src.shows.trakt_processing import full_trakt_pipeline, move_trakt_files, download_trakt_data
from src.location.location_processing import full_location_pipeline, move_google_files

# Updated imports for enhanced authentication
from src.utils.drive_storage import (
    update_drive,
    check_credentials_status,
    test_drive_connection,
)


# Initialize Drive connection at startup
def initialize_drive_connection():
    """Initialize and test Google Drive connection with enhanced authentication"""
    print("🔗 Initializing Google Drive connection...")
    print("=" * 60)

    # Check current credential status
    print("📋 Checking credential status...")
    check_credentials_status()

    # Test the connection
    print("\n🧪 Testing Drive connection...")
    connection_success = test_drive_connection()

    if connection_success:
        print("✅ Google Drive initialization successful!")
        print("=" * 60)
        return True
    else:
        print("❌ Google Drive initialization failed!")
        print("=" * 60)
        return False

# Initialize connection once at startup
DRIVE_INITIALIZED = initialize_drive_connection()
if not DRIVE_INITIALIZED:
    print("❌ Cannot proceed without Google Drive connection. Exiting.")
    sys.exit(1)


# ============================================================================
# PIPELINE REGISTRY - Central configuration for all data sources
# ============================================================================

PIPELINE_REGISTRY = {
    'books': {
        'name': 'Books (Goodreads + Kindle)',
        'function': full_books_pipeline,
        'download_method': 'coordination',  # Handles both Goodreads and Kindle
        'requires_timezone': False,
        'move_function': None,  # Coordination pipeline handles this
        'download_function': None,  # Handled separately by Goodreads and Kindle
        'urls': [],
        'description': 'Processes Goodreads and Kindle reading data',
        'user_selectable': True
    },
    'goodreads': {
        'name': 'Goodreads (Reading)',
        'function': None,  # Part of books coordination
        'download_method': 'manual_web',
        'requires_timezone': False,
        'move_function': move_goodreads_files,
        'download_function': download_goodreads_data,
        'urls': ['https://www.goodreads.com/review/import'],
        'description': 'Downloads Goodreads library export',
        'user_selectable': False  # Hidden - part of books coordination
    },
    'kindle': {
        'name': 'Kindle (Reading)',
        'function': None,  # Part of books coordination
        'download_method': 'email',
        'requires_timezone': False,
        'move_function': move_kindle_files,
        'download_function': download_kindle_data,
        'urls': ['https://www.amazon.com/hz/privacy-central/data-requests/preview.html'],
        'description': 'Downloads Kindle data from Amazon',
        'user_selectable': False  # Hidden - part of books coordination
    },
    'music': {
        'name': 'Music (Last.fm)',
        'function': full_lfm_pipeline,
        'download_method': 'api',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Fetches listening history from Last.fm API',
        'user_selectable': True
    },
    'podcasts': {
        'name': 'Podcasts (Pocket Casts)',
        'function': full_pocket_casts_pipeline,
        'download_method': 'manual_web',
        'requires_timezone': True,
        'move_function': move_pocket_casts_files,
        'download_function': download_pocket_casts_data,
        'urls': ['https://pocketcasts.com/'],
        'description': 'Processes Pocket Casts listening history (requires timezone correction)',
        'user_selectable': True
    },
    'garmin': {
        'name': 'Fitness (Garmin)',
        'function': full_garmin_pipeline,
        'download_method': 'email',
        'requires_timezone': True,
        'move_function': move_garmin_files,
        'download_function': download_garmin_data,
        'urls': ['https://www.garmin.com/fr-BE/account/datamanagement/exportdata/'],
        'description': 'Processes Garmin fitness data (requires timezone correction)',
        'user_selectable': True
    },
    'apple_health': {
        'name': 'Health (Apple Health)',
        'function': full_apple_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': False,
        'move_function': move_apple_files,
        'download_function': download_apple_data,
        'urls': [],
        'description': 'Processes Apple Health export from iPhone',
        'user_selectable': True
    },
    'letterboxd': {
        'name': 'Movies (Letterboxd)',
        'function': full_letterboxd_pipeline,
        'download_method': 'manual_web',
        'requires_timezone': False,
        'move_function': move_letterboxd_files,
        'download_function': download_letterboxd_data,
        'urls': ['https://letterboxd.com/settings/data/'],
        'description': 'Processes Letterboxd movie viewing history',
        'user_selectable': True
    },
    'trakt': {
        'name': 'TV Shows (Trakt)',
        'function': full_trakt_pipeline,
        'download_method': 'manual_web',
        'requires_timezone': False,
        'move_function': move_trakt_files,
        'download_function': download_trakt_data,
        'urls': ['https://trakt.tv/settings/data'],
        'description': 'Processes Trakt TV show watching history',
        'user_selectable': True
    },
    'finance': {
        'name': 'Finance (Money Manager)',
        'function': full_moneymgr_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': False,
        'move_function': move_moneymgr_files,
        'download_function': download_moneymgr_data,
        'urls': [],
        'description': 'Processes Money Manager expense tracking data',
        'user_selectable': True
    },
    'nutrition': {
        'name': 'Nutrition (Nutrilio)',
        'function': full_nutrilio_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': False,
        'move_function': move_nutrilio_files,
        'download_function': download_nutrilio_data,
        'urls': [],
        'description': 'Processes Nutrilio meal and nutrition tracking data',
        'user_selectable': True
    },
    'offscreen': {
        'name': 'Screen Time (Offscreen)',
        'function': full_offscreen_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': True,
        'move_function': move_offscreen_files,
        'download_function': download_offscreen_data,
        'urls': [],
        'description': 'Processes Offscreen app usage data (requires timezone correction)',
        'user_selectable': True
    },
    'weather': {
        'name': 'Weather (Meteostat API)',
        'function': get_weather_data,
        'download_method': 'api',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Fetches weather data from Meteostat API',
        'user_selectable': True
    },
}


# ============================================================================
# WORKFLOW FUNCTIONS
# ============================================================================

def collect_user_choices(registry):
    """
    Present all available topics and collect user Y/N choices.
    Only shows topics where user_selectable is True.

    Args:
        registry (dict): Pipeline registry with all data sources

    Returns:
        list: Names of selected topics
    """
    print("\n" + "=" * 60)
    print("📋 DATA SOURCE SELECTION")
    print("=" * 60)
    print("\nAvailable data sources:\n")

    # Filter to only user-selectable topics
    selectable_topics = {k: v for k, v in registry.items() if v.get('user_selectable', True)}

    # Display all user-selectable topics with descriptions
    for i, (key, config) in enumerate(selectable_topics.items(), 1):
        tz_indicator = " ⚠️ [Timezone]" if config['requires_timezone'] else ""
        method_label = {
            'api': '[API]',
            'manual_web': '[Web]',
            'manual_app': '[App]',
            'email': '[Email]',
            'coordination': '[Multi]'
        }.get(config['download_method'], '')

        print(f"  {i:2d}. {config['name']:<30} {method_label:<8} {tz_indicator}")
        print(f"      {config['description']}")

    print("\n" + "-" * 60)
    print("Legend:")
    print("  [API]  - Automated API fetch")
    print("  [Web]  - Manual browser download")
    print("  [App]  - Manual app export")
    print("  [Email]- Request via web, receive email")
    print("  [Multi]- Coordinates multiple sources")
    print("  ⚠️ [Timezone] - Requires timezone correction")
    print("-" * 60)

    # Collect user choices
    selected = []
    print("\n🎯 Select which data sources to process:\n")

    for key, config in selectable_topics.items():
        while True:
            choice = input(f"Process {config['name']}? (Y/N): ").strip().upper()
            if choice in ['Y', 'N']:
                if choice == 'Y':
                    selected.append(key)
                    print(f"  ✅ Added to processing queue")
                break
            else:
                print("  ❌ Invalid input. Please enter Y or N.")

    print("\n" + "=" * 60)
    print(f"📊 Selected {len(selected)} data source(s) for processing")
    print("=" * 60)

    return selected


def handle_timezone_dependency(selected_topics, registry):
    """
    Check if any selected topics require timezone correction.
    If yes, automatically run location pipeline to ensure timezone data is current.

    Args:
        selected_topics (list): List of selected topic names
        registry (dict): Pipeline registry

    Returns:
        bool: True if timezone dependency handled successfully, False otherwise
    """
    # Check if any selected topic requires timezone correction
    needs_timezone = any(
        registry[topic]['requires_timezone']
        for topic in selected_topics
    )

    if not needs_timezone:
        print("\n✓ No timezone correction needed for selected topics")
        return True

    # Display which topics need timezone correction
    tz_topics = [
        registry[topic]['name']
        for topic in selected_topics
        if registry[topic]['requires_timezone']
    ]

    print("\n" + "=" * 60)
    print("⚠️  TIMEZONE CORRECTION REQUIRED")
    print("=" * 60)
    print("\nThe following selected topics require timezone correction:")
    for topic in tz_topics:
        print(f"  • {topic}")

    print("\n📍 Running Location Pipeline to ensure timezone data is current...")
    print("This will download Google Timeline data and process location information.")
    print("=" * 60)

    try:
        # Run location pipeline option 1 (download + process + upload)
        success = full_location_pipeline(auto_full=True)

        if success:
            print("\n✅ Location pipeline completed successfully!")
            print("Timezone correction data is ready for processing.")
            return True
        else:
            print("\n❌ Location pipeline failed!")
            print("Timezone-dependent topics may not process correctly.")
            return False

    except Exception as e:
        print(f"\n❌ ERROR during location pipeline: {str(e)}")
        print("Timezone-dependent topics may not process correctly.")
        return False


def handle_manual_downloads(selected_topics, registry):
    """
    Handle manual downloads individually with detailed instructions for each source.
    Calls each source's download_*_data() function and moves files immediately after confirmation.

    Args:
        selected_topics (list): List of selected topic names
        registry (dict): Pipeline registry
    """
    # Expand 'books' into 'goodreads' and 'kindle' for individual download handling
    download_topics = []
    for topic in selected_topics:
        if topic == 'books':
            # Books is a coordination pipeline - handle Goodreads and Kindle separately
            download_topics.extend(['goodreads', 'kindle'])
        else:
            download_topics.append(topic)

    # Filter topics that require manual downloads
    manual_topics = [t for t in download_topics
                     if t in registry and registry[t]['download_method'] in ['manual_web', 'manual_app', 'email']]

    if not manual_topics:
        print("\n✓ No manual downloads required (API sources only)")
        return

    print("\n" + "=" * 60)
    print("📥 MANUAL DOWNLOAD PHASE")
    print("=" * 60)
    print(f"\nProcessing {len(manual_topics)} source(s) requiring manual downloads\n")

    # Process each source individually
    for i, topic in enumerate(manual_topics, 1):
        config = registry[topic]

        print("\n" + "=" * 60)
        print(f"[{i}/{len(manual_topics)}] {config['name']}")
        print("=" * 60)

        # Call the download instruction function if it exists
        if config['download_function']:
            try:
                # The download function shows instructions and asks for user confirmation
                download_confirmed = config['download_function']()

                if download_confirmed:
                    # Move files immediately after user confirms download
                    if config['move_function']:
                        try:
                            print(f"\n📁 Moving {config['name']} files...")
                            config['move_function']()
                            print(f"✅ Files moved successfully")
                        except Exception as e:
                            print(f"⚠️  Error moving files: {str(e)}")
                            print(f"→ You may need to move files manually to the appropriate export folder")
                    else:
                        print(f"✅ Download confirmed (no file movement needed)")
                else:
                    print(f"⚠️  Skipping {config['name']} - download not confirmed")
                    print(f"→ You can process this source later by running its individual pipeline")

            except Exception as e:
                print(f"❌ Error during {config['name']} download process: {str(e)}")
                print(f"→ Continuing with next source...")
        else:
            # This shouldn't happen if registry is configured correctly
            print(f"⚠️  No download function configured for {config['name']}")

    print("\n" + "=" * 60)
    print("✅ Manual download phase complete")
    print("=" * 60)


def run_processing_pipelines(selected_topics, registry):
    """
    Run processing pipeline (option 2) for each selected topic.
    Continues processing even if individual pipelines fail.

    Args:
        selected_topics (list): List of selected topic names
        registry (dict): Pipeline registry

    Returns:
        dict: Results with 'success' and 'failed' lists
    """
    print("\n" + "=" * 60)
    print("⚙️  AUTOMATED PROCESSING PHASE")
    print("=" * 60)
    print(f"\nProcessing {len(selected_topics)} data source(s)...\n")

    results = {
        'success': [],
        'failed': []
    }

    for i, topic in enumerate(selected_topics, 1):
        config = registry[topic]

        print("\n" + "=" * 60)
        print(f"[{i}/{len(selected_topics)}] Processing: {config['name']}")
        print("=" * 60)

        try:
            # Call pipeline function with auto_process_only=True to run option 2
            # Note: weather uses different function signature
            if topic == 'weather':
                # Weather function doesn't have auto_process_only, just call it directly
                config['function'](upload="N")
                success = True
            else:
                # All other pipelines support auto_process_only parameter
                success = config['function'](auto_process_only=True)

            if success:
                results['success'].append(topic)
                print(f"\n✅ {config['name']} processing completed successfully")
            else:
                results['failed'].append({
                    'topic': topic,
                    'name': config['name'],
                    'error': 'Pipeline returned failure status'
                })
                print(f"\n❌ {config['name']} processing failed")

        except Exception as e:
            results['failed'].append({
                'topic': topic,
                'name': config['name'],
                'error': str(e)
            })
            print(f"\n❌ ERROR processing {config['name']}: {str(e)}")
            print("Continuing with next topic...")

    return results


def generate_report(results, registry):
    """
    Generate and display final processing report.

    Args:
        results (dict): Results from run_processing_pipelines
        registry (dict): Pipeline registry
    """
    print("\n\n" + "=" * 60)
    print("📊 PROCESSING REPORT")
    print("=" * 60)

    total = len(results['success']) + len(results['failed'])
    success_count = len(results['success'])
    failure_count = len(results['failed'])

    print(f"\nTotal: {total} | ✅ Success: {success_count} | ❌ Failed: {failure_count}")

    if results['success']:
        print("\n✅ SUCCESSFUL PROCESSING:")
        for topic in results['success']:
            print(f"  • {registry[topic]['name']}")

    if results['failed']:
        print("\n❌ FAILED PROCESSING:")
        for failure in results['failed']:
            print(f"  • {failure['name']}")
            print(f"    Error: {failure['error']}")

    print("\n" + "=" * 60)

    if failure_count == 0:
        print("🎉 All selected data sources processed successfully!")
    elif success_count == 0:
        print("⚠️  All processing attempts failed. Check errors above.")
    else:
        print("⚠️  Some processing attempts failed. Check errors above.")

    print("=" * 60)


def batch_download_process_upload():
    """
    Main orchestration function for batch processing workflow.

    Workflow:
    1. Collect user selections for all topics
    2. Handle timezone dependency (run location pipeline if needed)
    3. Handle all manual downloads in batch
    4. Run processing pipelines for all selected topics
    5. Generate final report
    """
    try:
        print("\n" + "=" * 60)
        print("🚀 LIFELOG BATCH PROCESSING SYSTEM")
        print("=" * 60)
        print("\nThis workflow will:")
        print("  1. Let you select which data sources to process")
        print("  2. Handle timezone correction if needed")
        print("  3. Guide you through any manual downloads")
        print("  4. Automatically process and upload all selected data")
        print("  5. Provide a summary report")

        # Step 1: Collect user choices
        selected_topics = collect_user_choices(PIPELINE_REGISTRY)

        if not selected_topics:
            print("\n⚠️  No topics selected. Exiting.")
            return

        # Step 2: Handle timezone dependency
        handle_timezone_dependency(selected_topics, PIPELINE_REGISTRY)

        # Step 3: Handle manual downloads
        handle_manual_downloads(selected_topics, PIPELINE_REGISTRY)

        # Step 4: Run all processing pipelines
        results = run_processing_pipelines(selected_topics, PIPELINE_REGISTRY)

        # Step 5: Generate report
        generate_report(results, PIPELINE_REGISTRY)

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        print("Partial processing may have occurred.")
    except Exception as e:
        print(f"\n\n💥 CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n📋 Processing session complete.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    batch_download_process_upload()
