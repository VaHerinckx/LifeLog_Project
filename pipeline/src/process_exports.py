# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import argparse
import subprocess
import sys
import time

# Logger
from src.utils.logger import log, Verbosity

# Utils
from src.utils.drive_storage import (
    update_drive,
    check_credentials_status,
    test_drive_connection,
)

# Source processors - Reading
from src.sources_processing.goodreads.goodreads_processing import download_goodreads_data, move_goodreads_files, full_goodreads_pipeline
from src.sources_processing.kindle.kindle_processing import download_kindle_data, move_kindle_files, full_kindle_pipeline

# Source processors - Music
from src.sources_processing.lastfm.lastfm_processing import full_lastfm_pipeline
from src.sources_processing.spotify.spotify_processing import full_spotify_pipeline

# Source processors - Media
from src.sources_processing.letterboxd.letterboxd_processing import full_letterboxd_pipeline, move_letterboxd_files, download_letterboxd_data
from src.sources_processing.trakt.trakt_processing import full_trakt_pipeline, move_trakt_files, download_trakt_data
from src.sources_processing.pocket_casts.pocket_casts_processing import full_pocket_casts_pipeline, move_pocket_casts_files, download_pocket_casts_data

# Source processors - Health/Fitness
from src.sources_processing.garmin.garmin_processing import full_garmin_pipeline, move_garmin_files, download_garmin_data
from src.sources_processing.apple.apple_processing import full_apple_pipeline, move_apple_files, download_apple_data
from src.sources_processing.nutrilio.nutrilio_processing import full_nutrilio_pipeline, move_nutrilio_files, download_nutrilio_data
from src.sources_processing.offscreen.offscreen_processing import full_offscreen_pipeline, move_offscreen_files, download_offscreen_data

# Source processors - Other
from src.sources_processing.moneymgr.moneymgr_processing import full_moneymgr_pipeline, move_moneymgr_files, download_moneymgr_data
from src.sources_processing.google_maps.google_maps_processing import download_google_data, move_google_files, full_google_maps_pipeline
from src.sources_processing.manual_location.manual_location_processing import full_manual_location_pipeline
from src.sources_processing.meteostat.meteostat_processing import full_meteostat_pipeline

# Topic coordinators
from src.topic_processing.reading.reading_processing import full_books_pipeline
from src.topic_processing.music.music_processing import full_music_pipeline
from src.topic_processing.health.health_processing import full_health_pipeline
from src.topic_processing.nutrition.nutrition_processing import full_nutrition_pipeline
from src.topic_processing.location.location_processing import full_location_pipeline
from src.topic_processing.podcasts.podcasts_processing import full_podcasts_pipeline
from src.topic_processing.movies.movies_processing import full_movies_pipeline
from src.topic_processing.shows.shows_processing import full_shows_pipeline
from src.topic_processing.fitness.fitness_processing import full_fitness_pipeline
from src.topic_processing.finance.finance_processing import full_finance_pipeline
from src.topic_processing.website_maintenance.website_maintenance_processing import full_website_maintenance_pipeline
from src.utils.logger import log


# Initialize Drive connection (called from main, not at import time)
def initialize_drive_connection():
    """Initialize and test Google Drive connection with enhanced authentication."""
    log.progress("🔗 Initializing Google Drive connection...")
    log.progress("Checking credential status...")
    check_credentials_status()

    log.progress("Testing Drive connection...")
    connection_success = test_drive_connection()

    if connection_success:
        log.success("Google Drive initialization successful!")
        return True
    else:
        log.error("Google Drive initialization failed!")
        return False


# ============================================================================
# PIPELINE REGISTRY - Central configuration for all data sources
# ============================================================================

PIPELINE_REGISTRY = {
    'goodreads': {
        'name': 'Goodreads (Reading) - SOURCE',
        'function': full_goodreads_pipeline,
        'download_method': 'manual_web',
        'requires_timezone': False,
        'move_function': move_goodreads_files,
        'download_function': download_goodreads_data,
        'urls': ['https://www.goodreads.com/review/import'],
        'description': 'Downloads Goodreads library export (source processor)',
        'user_selectable': True
    },
    'kindle': {
        'name': 'Kindle (Reading) - SOURCE',
        'function': full_kindle_pipeline,
        'download_method': 'email',
        'requires_timezone': False,
        'move_function': move_kindle_files,
        'download_function': download_kindle_data,
        'urls': ['https://www.amazon.com/hz/privacy-central/data-requests/preview.html'],
        'description': 'Downloads Kindle data from Amazon (source processor)',
        'user_selectable': True
    },
    'reading_topic': {
        'name': 'Reading (Complete) - TOPIC',
        'function': full_books_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Merges Goodreads and Kindle reading data',
        'user_selectable': False
    },
    'lastfm': {
        'name': 'Music (Last.fm) - SOURCE',
        'function': full_lastfm_pipeline,
        'download_method': 'api',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Fetches listening history from Last.fm API (source processor only)',
        'user_selectable': True
    },
    'spotify_legacy': {
        'name': 'Music (Spotify Legacy) - SOURCE',
        'function': full_spotify_pipeline,
        'download_method': 'manual',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Processes legacy Spotify JSON exports 2013-2023 (source processor only)',
        'user_selectable': False
    },
    'music_topic': {
        'name': 'Music (Complete) - TOPIC',
        'function': full_music_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Coordinates Last.fm + Spotify legacy data, enriches with Spotify API',
        'user_selectable': False
    },
    'podcasts': {
        'name': 'Podcasts (Pocket Casts) - SOURCE',
        'function': full_pocket_casts_pipeline,
        'download_method': 'manual_web',
        'requires_timezone': True,
        'move_function': move_pocket_casts_files,
        'download_function': download_pocket_casts_data,
        'urls': ['https://pocketcasts.com/'],
        'description': 'Processes Pocket Casts listening history (source processor, requires timezone correction)',
        'user_selectable': True
    },
    'podcasts_topic': {
        'name': 'Podcasts (Complete) - TOPIC',
        'function': full_podcasts_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Coordinates Pocket Casts data and generates website files',
        'user_selectable': False
    },
    'garmin': {
        'name': 'Fitness (Garmin) - SOURCE',
        'function': full_garmin_pipeline,
        'download_method': 'email',
        'requires_timezone': True,
        'move_function': move_garmin_files,
        'download_function': download_garmin_data,
        'urls': ['https://www.garmin.com/fr-BE/account/datamanagement/exportdata/'],
        'description': 'Processes Garmin fitness data (source processor, requires timezone correction)',
        'user_selectable': True
    },
    'fitness_topic': {
        'name': 'Fitness (Complete) - TOPIC',
        'function': full_fitness_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Coordinates Garmin data and generates website files',
        'user_selectable': False
    },
    'apple_health': {
        'name': 'Health (Apple Health) - SOURCE',
        'function': full_apple_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': False,
        'move_function': move_apple_files,
        'download_function': download_apple_data,
        'urls': [],
        'description': 'Processes Apple Health export from iPhone (source processor only)',
        'user_selectable': True
    },
    'health_topic': {
        'name': 'Health (Complete) - TOPIC',
        'function': full_health_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Merges Apple Health, Nutrilio, Screen Time into complete health dataset',
        'user_selectable': False
    },
    'letterboxd': {
        'name': 'Movies (Letterboxd) - SOURCE',
        'function': full_letterboxd_pipeline,
        'download_method': 'manual_web',
        'requires_timezone': False,
        'move_function': move_letterboxd_files,
        'download_function': download_letterboxd_data,
        'urls': ['https://letterboxd.com/settings/data/'],
        'description': 'Processes Letterboxd movie viewing history (source processor)',
        'user_selectable': True
    },
    'movies_topic': {
        'name': 'Movies (Complete) - TOPIC',
        'function': full_movies_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Coordinates Letterboxd data and generates website files',
        'user_selectable': False
    },
    'trakt': {
        'name': 'TV Shows (Trakt) - SOURCE',
        'function': full_trakt_pipeline,
        'download_method': 'manual_web',
        'requires_timezone': False,
        'move_function': move_trakt_files,
        'download_function': download_trakt_data,
        'urls': ['https://trakt.tv/settings/data'],
        'description': 'Processes Trakt TV show watching history (source processor)',
        'user_selectable': True
    },
    'shows_topic': {
        'name': 'TV Shows (Complete) - TOPIC',
        'function': full_shows_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Coordinates Trakt data and generates website files',
        'user_selectable': False
    },
    'finance': {
        'name': 'Finance (Money Manager) - SOURCE',
        'function': full_moneymgr_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': False,
        'move_function': move_moneymgr_files,
        'download_function': download_moneymgr_data,
        'urls': [],
        'description': 'Processes Money Manager expense tracking data (source processor)',
        'user_selectable': True
    },
    'finance_topic': {
        'name': 'Finance (Complete) - TOPIC',
        'function': full_finance_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Coordinates Money Manager data and generates website files',
        'user_selectable': False
    },
    'nutrition_source': {
        'name': 'Nutrition (Nutrilio) - SOURCE',
        'function': full_nutrilio_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': False,
        'move_function': move_nutrilio_files,
        'download_function': download_nutrilio_data,
        'urls': [],
        'description': 'Processes Nutrilio meal and nutrition tracking data (source processor, feeds Nutrition topic)',
        'user_selectable': True
    },
    'nutrition_topic': {
        'name': 'Nutrition (Complete) - TOPIC',
        'function': full_nutrition_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Merges Nutrilio nutrition data and generates website files for Nutrition page',
        'user_selectable': False
    },
    'offscreen': {
        'name': 'Screen Time (Offscreen) - SOURCE',
        'function': full_offscreen_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': True,
        'move_function': move_offscreen_files,
        'download_function': download_offscreen_data,
        'urls': [],
        'description': 'Processes Offscreen app usage data (source processor, feeds Health topic, requires timezone correction)',
        'user_selectable': True
    },
    'meteostat': {
        'name': 'Meteostat (Weather API) - SOURCE',
        'function': full_meteostat_pipeline,
        'download_method': 'api',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Fetches weather data from Meteostat API (source processor)',
        'user_selectable': True
    },
    'google_maps': {
        'name': 'Location (Google Maps) - SOURCE',
        'function': full_google_maps_pipeline,
        'download_method': 'manual_app',
        'requires_timezone': False,
        'move_function': move_google_files,
        'download_function': download_google_data,
        'urls': [],
        'description': 'Processes Google Maps Timeline data (source processor, feeds Location topic)',
        'user_selectable': True
    },
    'manual_location': {
        'name': 'Location (Manual Excel) - SOURCE',
        'function': full_manual_location_pipeline,
        'download_method': 'manual',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Processes manual Excel travel records (source processor, feeds Location topic)',
        'user_selectable': False
    },
    'location_topic': {
        'name': 'Location (Complete) - TOPIC',
        'function': full_location_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Merges Google Maps and Manual location data and generates website files for Location page',
        'user_selectable': False
    },
    'website_maintenance': {
        'name': 'Website Maintenance',
        'function': full_website_maintenance_pipeline,
        'download_method': 'coordination',
        'requires_timezone': False,
        'move_function': None,
        'download_function': None,
        'urls': [],
        'description': 'Website maintenance tasks (runs automatically at end of pipeline)',
        'user_selectable': False
    },
}


# ============================================================================
# SOURCE-TO-TOPIC DEPENDENCY MAPPING
# ============================================================================

SOURCE_TO_TOPIC_MAP = {
    'goodreads': ['reading_topic'],
    'kindle': ['reading_topic'],
    'lastfm': ['music_topic'],
    'spotify_legacy': ['music_topic'],
    'apple_health': ['health_topic'],
    'nutrition_source': ['health_topic', 'nutrition_topic'],
    'offscreen': ['health_topic'],
    'google_maps': ['location_topic'],
    'manual_location': ['location_topic'],
    'podcasts': ['podcasts_topic'],
    'letterboxd': ['movies_topic'],
    'trakt': ['shows_topic'],
    'garmin': ['fitness_topic'],
    'finance': ['finance_topic'],
}


def get_impacted_topics(processed_sources):
    """
    Determine which topic coordinators need to run based on successfully processed sources.

    Args:
        processed_sources (list): List of source names that were successfully processed

    Returns:
        list: Sorted unique list of topic names that need to be run

    Example:
        >>> get_impacted_topics(['nutrilio', 'apple'])
        ['health_topic', 'nutrition_topic']
    """
    impacted = set()
    for source in processed_sources:
        if source in SOURCE_TO_TOPIC_MAP:
            impacted.update(SOURCE_TO_TOPIC_MAP[source])
    return sorted(impacted)


# ============================================================================
# WORKFLOW FUNCTIONS
# ============================================================================

def collect_user_choices(registry):
    """
    Present all available sources and collect user Y/N choices.
    Only shows sources where user_selectable is True.
    Filters out topic coordinators (which have 'topic' in the key or 'TOPIC' in the name).

    Args:
        registry (dict): Pipeline registry with all data sources

    Returns:
        list: Names of selected sources
    """
    log.header("SOURCE SELECTION")
    log.prompt("\nAvailable data sources:\n")

    # Filter to only user-selectable sources (exclude topics and coordination entries)
    selectable_sources = {
        k: v for k, v in registry.items()
        if v.get('user_selectable', True)
        and '_topic' not in k
        and 'coordination' not in k
        and k != 'website_maintenance'
    }

    # Display all user-selectable sources with descriptions
    for i, (key, config) in enumerate(selectable_sources.items(), 1):
        tz_indicator = " ⚠️ [Timezone]" if config['requires_timezone'] else ""
        method_label = {
            'api': '[API]',
            'manual_web': '[Web]',
            'manual_app': '[App]',
            'email': '[Email]',
            'coordination': '[Multi]'
        }.get(config['download_method'], '')

        log.prompt(f"{i:2d}. {config['name']:<35} {method_label:<8} {tz_indicator}")
        log.prompt(f"   {config['description']}")

    log.prompt("\n" + "-" * 60)
    log.prompt("Legend:")
    log.prompt("[API]  - Automated API fetch")
    log.prompt("[Web]  - Manual browser download")
    log.prompt("[App]  - Manual app export")
    log.prompt("[Email]- Request via web, receive email")
    log.prompt(" [Timezone] - Requires timezone correction")
    log.prompt("-" * 60)

    # Collect user choices
    selected = []
    log.prompt("\n Select which data SOURCES to update:\n")

    for key, config in selectable_sources.items():
        while True:
            choice = input(f"Update {config['name']}? (Y/N): ").strip().upper()
            if choice in ['Y', 'N']:
                if choice == 'Y':
                    selected.append(key)
                    log.prompt(f"Added to processing queue")
                break
            else:
                log.prompt(" Invalid input. Please enter Y or N.")

    log.header(f"Selected {len(selected)} source(s) for processing")

    return selected


def handle_geocoding_dependency(selected_sources, registry):
    """
    Handle Google Maps timeline download and geocoding setup.

    Flow:
    - If any selected source requires timezone correction → auto-add Google Maps (no question)
    - If no timezone needed → ask user if they want to update Google Maps Timeline
    - Google Maps processing generates the geocoding file for timezone correction

    Args:
        selected_sources (list): List of selected source names
        registry (dict): Pipeline registry

    Returns:
        bool: True if geocoding handled successfully (or not needed), False on failure
    """
    needs_timezone = any(
        registry[source]['requires_timezone']
        for source in selected_sources
        if source in registry
    )

    process_google_maps = False

    if needs_timezone:
        # Auto-add Google Maps — required for timezone correction
        tz_sources = [
            registry[source]['name']
            for source in selected_sources
            if source in registry and registry[source]['requires_timezone']
        ]
        log.subheader("GEOCODING REQUIRED FOR TIMEZONE CORRECTION")
        log.milestone("The following sources require timezone correction:")
        for source in tz_sources:
            log.summary(f" • {source}")
        log.milestone("Google Maps Timeline will be downloaded automatically.")
        process_google_maps = True
    else:
        # Optional — ask user
        choice = input("\nWould you also like to update Google Maps Timeline? (Y/N): ").strip().upper()
        process_google_maps = (choice == 'Y')
        if not process_google_maps:
            log.info("Skipping Google Maps update")
            return True

    try:
        # Download Google Maps data
        google_maps_config = registry.get('google_maps')
        if google_maps_config and google_maps_config['download_function']:
            log.progress("Downloading Google Maps Timeline...")
            download_confirmed = google_maps_config['download_function']()

            if download_confirmed and google_maps_config['move_function']:
                google_maps_config['move_function']()
                log.success("Google Maps files moved successfully")

        # Process google_maps to generate location data
        log.progress("⚙️  Processing Google Maps data...")
        google_maps_success = full_google_maps_pipeline(auto_process_only=True)

        if not google_maps_success:
            log.error("Google Maps processing failed!")
            return False

        # Process manual_location to generate geocoding file
        log.progress("⚙️  Generating geocoding file from location data...")
        manual_location_success = full_manual_location_pipeline(auto_process_only=True)

        if not manual_location_success:
            log.error("Geocoding file generation failed!")
            return False

        log.milestone("Geocoding data is ready for timezone correction")
        return True

    except Exception as e:
        log.error(f"ERROR during geocoding setup: {str(e)}")
        log.warning("Timezone-dependent sources may not process correctly.")
        return False


def handle_manual_downloads(selected_sources, registry):
    """
    Handle manual downloads individually with detailed instructions for each source.
    Calls each source's download_*_data() function and moves files immediately after confirmation.

    Args:
        selected_sources (list): List of selected source names
        registry (dict): Pipeline registry
    """
    # Filter sources that require manual downloads
    manual_sources = [s for s in selected_sources
                      if s in registry and registry[s]['download_method'] in ['manual_web', 'manual_app', 'email']]

    if not manual_sources:
        log.info("No manual downloads required (API sources only)")
        return

    log.header("MANUAL DOWNLOAD PHASE")
    log.prompt(f"Processing {len(manual_sources)} source(s) requiring manual downloads\n")

    for i, source in enumerate(manual_sources, 1):
        config = registry[source]
        log.source_header(i, len(manual_sources), config['name'])

        if config['download_function']:
            try:
                download_confirmed = config['download_function']()

                if download_confirmed:
                    if config['move_function']:
                        try:
                            log.progress(f"📁 Moving {config['name']} files...")
                            config['move_function']()
                            log.success("Files moved successfully")
                        except Exception as e:
                            log.warning(f"Error moving files: {str(e)}")
                            log.warning("You may need to move files manually to the appropriate export folder")
                    else:
                        log.success("Download confirmed (no file movement needed)")
                else:
                    log.info(f"Skipping {config['name']} - download not confirmed")

            except Exception as e:
                log.error(f"Error during {config['name']} download process: {str(e)}")
                log.info("Continuing with next source...")
        else:
            log.warning(f"No download function configured for {config['name']}")

    log.milestone("Manual download phase complete")


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
    log.header("AUTOMATED PROCESSING PHASE")
    log.prompt(f"Processing {len(selected_topics)} data source(s)...\n")

    results = {
        'success': [],
        'failed': []
    }

    for i, topic in enumerate(selected_topics, 1):
        config = registry[topic]
        log.source_header(i, len(selected_topics), config['name'])

        try:
            t_start = time.time()
            success = config['function'](auto_process_only=True)
            elapsed = time.time() - t_start

            if success:
                results['success'].append(topic)
                log.milestone(f"{config['name']} completed ({elapsed:.1f}s)")
            else:
                results['failed'].append({
                    'topic': topic,
                    'name': config['name'],
                    'error': 'Pipeline returned failure status'
                })
                log.error(f"{config['name']} processing failed")

        except Exception as e:
            results['failed'].append({
                'topic': topic,
                'name': config['name'],
                'error': str(e)
            })
            log.error(f"ERROR processing {config['name']}: {str(e)}")
            log.info("Continuing with next source...")

    return results


def run_impacted_topics(successful_sources, registry):
    """
    Automatically run topic coordinators based on which sources were successfully processed.
    Each topic coordinator runs with option 2 (auto_process_only=True) so it won't prompt
    for downloads - it just merges existing processed source files.

    Args:
        successful_sources (list): List of source names that were successfully processed
        registry (dict): Pipeline registry

    Returns:
        dict: Results with 'success' and 'failed' lists
    """
    # Determine which topics need to be run
    impacted_topics = get_impacted_topics(successful_sources)

    if not impacted_topics:
        log.info("No topic coordinators need to be run")
        return {'success': [], 'failed': []}

    log.header("TOPIC COORDINATION")
    log.prompt(f"Detected {len(impacted_topics)} impacted topic(s) based on processed sources")
    topic_names = ', '.join([registry[t]['name'] for t in impacted_topics])
    log.prompt(f"Impacted topics: {topic_names}\n")

    results = {
        'success': [],
        'failed': []
    }

    for i, topic in enumerate(impacted_topics, 1):
        config = registry[topic]
        log.source_header(i, len(impacted_topics), config['name'])

        try:
            # Run topic coordinator with merge_only=True (sources already processed)
            success = config['function'](merge_only=True)

            if success:
                results['success'].append(topic)
                log.milestone(f"{config['name']} completed successfully")
            else:
                results['failed'].append({
                    'topic': topic,
                    'name': config['name'],
                    'error': 'Topic coordinator returned failure status'
                })
                log.error(f"{config['name']} failed")

        except Exception as e:
            results['failed'].append({
                'topic': topic,
                'name': config['name'],
                'error': str(e)
            })
            log.error(f"ERROR running {config['name']}: {str(e)}")
            log.info("Continuing with next topic...")

    return results


def run_website_maintenance(registry):
    """
    Run website maintenance pipeline at the end of all processing.
    This always runs regardless of what sources/topics were processed.

    Args:
        registry (dict): Pipeline registry

    Returns:
        dict: Results with 'success' and 'failed' lists
    """
    log.subheader("WEBSITE MAINTENANCE")

    config = registry['website_maintenance']
    results = {
        'success': [],
        'failed': []
    }

    try:
        success = config['function'](auto_mode=True)

        if success:
            results['success'].append('website_maintenance')
            log.milestone("Website maintenance completed successfully")
        else:
            results['failed'].append({
                'topic': 'website_maintenance',
                'name': config['name'],
                'error': 'Website maintenance returned failure status'
            })
            log.error("Website maintenance failed")

    except Exception as e:
        results['failed'].append({
            'topic': 'website_maintenance',
            'name': config['name'],
            'error': str(e)
        })
        log.error(f"ERROR running website maintenance: {str(e)}")

    return results


def generate_report(source_results, topic_results, maintenance_results, registry):
    """Generate and display compact final processing report."""
    log.report_table(source_results, topic_results, maintenance_results, registry)


def batch_download_process_upload():
    """
    Main orchestration function for source-centric processing workflow.

    Workflow:
    1. Collect user selections for sources (Y/N for each)
    2. Handle timezone dependency (run location pipeline if needed)
    3. Handle all manual downloads sequentially
    4. Process all selected sources with option 2 (auto_process_only=True)
    5. Auto-detect and run impacted topic coordinators
    6. Run website maintenance (always)
    7. Generate comprehensive report
    """
    try:
        log.header("LIFELOG PROCESSING PIPELINE")

        # Step 1: Collect user choices (source selection)
        selected_sources = collect_user_choices(PIPELINE_REGISTRY)

        if not selected_sources:
            log.warning("No sources selected. Exiting.")
            return

        # Step 2: Handle geocoding / Google Maps (auto or optional)
        handle_geocoding_dependency(selected_sources, PIPELINE_REGISTRY)

        # Step 3: Handle manual downloads (all other downloads)
        handle_manual_downloads(selected_sources, PIPELINE_REGISTRY)

        log.header("USER INPUT COMPLETE — STARTING AUTOMATED PROCESSING")

        # Step 4: Process all selected sources (uninterrupted)
        source_results = run_processing_pipelines(selected_sources, PIPELINE_REGISTRY)

        # Step 5: Auto-detect and run impacted topic coordinators (merge_only)
        successful_sources = source_results['success']
        topic_results = run_impacted_topics(successful_sources, PIPELINE_REGISTRY)

        # Step 6: Run website maintenance (always runs)
        maintenance_results = run_website_maintenance(PIPELINE_REGISTRY)

        # Step 7: Generate compact report
        generate_report(source_results, topic_results, maintenance_results, PIPELINE_REGISTRY)

    except KeyboardInterrupt:
        log.blank()
        log.warning("Process interrupted by user.")
        log.info("Partial processing may have occurred.")
    except Exception as e:
        log.blank()
        log.error(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        log.blank()
        log.summary("Processing session complete.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_verbosity_args():
    """Parse command-line arguments for verbosity control."""
    parser = argparse.ArgumentParser(
        description="LifeLog Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quiet', action='store_true', help='Minimal output: milestones, errors, actionable warnings')
    group.add_argument('--normal', action='store_true', help='Scope and summary per source (default)')
    group.add_argument('--verbose', action='store_true', help='Detailed output: progress steps, success confirmations')
    group.add_argument('--debug', action='store_true', help='Full output: includes DataFrame shapes, sample data')
    return parser.parse_args()


def prompt_verbosity():
    """Interactively ask the user for verbosity level."""
    choice = input("\nOutput verbosity? [Q]uiet / [N]ormal / [V]erbose / [D]ebug (default: N): ").strip().upper()
    if choice in ('Q', 'QUIET'):
        return Verbosity.QUIET
    elif choice in ('V', 'VERBOSE'):
        return Verbosity.VERBOSE
    elif choice in ('D', 'DEBUG'):
        return Verbosity.DEBUG
    else:
        return Verbosity.NORMAL


if __name__ == "__main__":
    # 1. Parse verbosity from command-line flags
    args = parse_verbosity_args()

    if args.quiet:
        log.set_verbosity(Verbosity.QUIET)
    elif args.normal:
        log.set_verbosity(Verbosity.NORMAL)
    elif args.verbose:
        log.set_verbosity(Verbosity.VERBOSE)
    elif args.debug:
        log.set_verbosity(Verbosity.DEBUG)
    else:
        # No flag provided — ask interactively
        log.set_verbosity(prompt_verbosity())

    # 2. Initialize Drive connection (now runs after verbosity is set)
    if not initialize_drive_connection():
        log.error("Cannot proceed without Google Drive connection. Exiting.")
        sys.exit(1)

    # 3. Run the pipeline
    batch_download_process_upload()
