import os
import zipfile
import shutil
import glob
import json
import pandas as pd
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.utils_functions import record_successful_run, enforce_snake_case

load_dotenv()


def download_trakt_data():
    """
    Opens Trakt export page and prompts user to download data.
    Returns True if user confirms download, False otherwise.
    """
    print("🎬 Starting Trakt data download...")
    
    urls = ['https://trakt.tv/settings/data']
    open_web_urls(urls)
    
    print("📝 Instructions:")
    print("   1. Click 'Export your data'")
    print("   2. Wait for the export to be prepared")
    print("   3. Download the ZIP file when ready")
    print("   4. The file will be named like 'YYYY-MM-DDTHH-MM-SSZ-entinval.zip'")
    print("   5. The ZIP will contain multiple history-X.json files in the watched folder")
    
    response = prompt_user_download_status("Trakt")
    return response


def move_trakt_files():
    """
    Moves the downloaded Trakt files from Downloads, unzips the file,
    and extracts all history-X.json files from the watched subfolder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving and extracting Trakt files...")

    downloads_path = "/Users/valen/Downloads"

    # Find the timestamped entinval zip file
    zip_pattern = os.path.join(downloads_path, "*entinval.zip")
    zip_files = glob.glob(zip_pattern)

    if not zip_files:
        print("❌ No entinval.zip file found in Downloads folder")
        return False

    # Use the most recent file if multiple exist
    zip_file_path = max(zip_files, key=os.path.getctime)
    print(f"📦 Found zip file: {os.path.basename(zip_file_path)}")

    # Create export folder if it doesn't exist
    export_folder = "files/exports/trakt_exports"
    os.makedirs(export_folder, exist_ok=True)

    try:
        # Extract the zip file
        print("📦 Extracting zip file...")
        temp_extract_path = os.path.join(downloads_path, "entinval_temp")

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)

        # Look for all history*.json files in the entinval/watched subfolder
        watched_folder = os.path.join(temp_extract_path, "entinval", "watched")

        if not os.path.exists(watched_folder):
            print("❌ entinval/watched subfolder not found")
            shutil.rmtree(temp_extract_path, ignore_errors=True)
            return False

        # Find all history files (history.json, history-1.json, history-2.json, etc.)
        history_files = glob.glob(os.path.join(watched_folder, "history*.json"))

        if not history_files:
            print("❌ No history files found in entinval/watched subfolder")
            shutil.rmtree(temp_extract_path, ignore_errors=True)
            return False

        print(f"📊 Found {len(history_files)} history file(s)")

        # Move all history files to the export folder
        moved_count = 0
        for history_file_path in history_files:
            filename = os.path.basename(history_file_path)
            destination_path = os.path.join(export_folder, filename)
            shutil.move(history_file_path, destination_path)
            print(f"✅ Moved {filename}")
            moved_count += 1

        print(f"✅ Successfully moved {moved_count} history file(s) to {export_folder}")

        # Look for ratings files in the entinval/ratings subfolder
        ratings_folder = os.path.join(temp_extract_path, "entinval", "ratings")

        if os.path.exists(ratings_folder):
            ratings_files = glob.glob(os.path.join(ratings_folder, "ratings*.json"))

            if ratings_files:
                print(f"📊 Found {len(ratings_files)} ratings file(s)")

                # Move all ratings files to the export folder
                ratings_moved = 0
                for ratings_file_path in ratings_files:
                    filename = os.path.basename(ratings_file_path)
                    destination_path = os.path.join(export_folder, filename)
                    shutil.move(ratings_file_path, destination_path)
                    print(f"✅ Moved {filename}")
                    ratings_moved += 1

                print(f"✅ Successfully moved {ratings_moved} ratings file(s) to {export_folder}")
            else:
                print("ℹ️  No ratings files found in ratings subfolder")
        else:
            print("ℹ️  No ratings subfolder found")

        # Clean up temp folder and original zip
        shutil.rmtree(temp_extract_path, ignore_errors=True)
        os.remove(zip_file_path)

        return True

    except Exception as e:
        print(f"❌ Error processing Trakt files: {e}")
        # Clean up temp folder if it exists
        temp_extract_path = os.path.join(downloads_path, "entinval_temp")
        shutil.rmtree(temp_extract_path, ignore_errors=True)
        return False


def load_ratings_data():
    """
    Load ratings data from ratings JSON files.
    Returns dictionaries for episode, season, and show ratings.
    """
    print("⭐ Loading ratings data...")

    export_folder = "files/exports/trakt_exports"

    episode_ratings = {}
    season_ratings = {}
    show_ratings = {}

    # Load episode ratings
    episode_ratings_file = os.path.join(export_folder, "ratings-episodes.json")
    if os.path.exists(episode_ratings_file):
        try:
            with open(episode_ratings_file, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)

            for entry in episode_data:
                episode_trakt_id = entry.get('episode', {}).get('ids', {}).get('trakt')
                rating = entry.get('rating')
                if episode_trakt_id and rating:
                    episode_ratings[episode_trakt_id] = rating

            print(f"   Loaded {len(episode_ratings)} episode ratings")
        except Exception as e:
            print(f"⚠️  Error loading episode ratings: {e}")

    # Load season ratings
    season_ratings_file = os.path.join(export_folder, "ratings-seasons.json")
    if os.path.exists(season_ratings_file):
        try:
            with open(season_ratings_file, 'r', encoding='utf-8') as f:
                season_data = json.load(f)

            for entry in season_data:
                show_trakt_id = entry.get('show', {}).get('ids', {}).get('trakt')
                season_num = entry.get('season', {}).get('number')
                rating = entry.get('rating')
                if show_trakt_id and season_num is not None and rating:
                    # Key format: "show_trakt_id_season_num"
                    season_ratings[f"{show_trakt_id}_{season_num}"] = rating

            print(f"   Loaded {len(season_ratings)} season ratings")
        except Exception as e:
            print(f"⚠️  Error loading season ratings: {e}")

    # Load show ratings
    show_ratings_file = os.path.join(export_folder, "ratings-shows.json")
    if os.path.exists(show_ratings_file):
        try:
            with open(show_ratings_file, 'r', encoding='utf-8') as f:
                show_data = json.load(f)

            for entry in show_data:
                show_trakt_id = entry.get('show', {}).get('ids', {}).get('trakt')
                rating = entry.get('rating')
                if show_trakt_id and rating:
                    show_ratings[show_trakt_id] = rating

            print(f"   Loaded {len(show_ratings)} show ratings")
        except Exception as e:
            print(f"⚠️  Error loading show ratings: {e}")

    return episode_ratings, season_ratings, show_ratings


def load_season_artwork_cache():
    """Load season artwork cache from JSON file"""
    cache_path = 'files/work_files/trakt_work_files/season_artwork_cache.json'

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            print(f"🎨 Loaded {len(cache_data)} season artwork URLs from cache")
            return cache_data
        except Exception as e:
            print(f"⚠️  Error loading artwork cache: {e}")
            return {}
    else:
        print("🎨 No existing season artwork cache found - creating new one")
        return {}


def save_season_artwork_cache(cache_data):
    """Save season artwork cache to JSON file"""
    cache_path = 'files/work_files/trakt_work_files/season_artwork_cache.json'
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"💾 Saved {len(cache_data)} season artwork URLs to cache")
        return True
    except Exception as e:
        print(f"❌ Error saving artwork cache: {e}")
        return False


def get_tmdb_season_artwork(show_title, show_year, season_number, cache_data):
    """
    Fetches season artwork from TMDB API with caching
    Returns: season_poster_url or 'No poster found'
    """
    # Create cache key
    cache_key = f"{show_title}_{show_year}_S{season_number}"
    
    # Check if we already have this in cache
    if cache_key in cache_data:
        return cache_data[cache_key]
    
    # Get TMDB API key
    api_key = os.environ.get('TMDB_Key')
    if not api_key:
        print("⚠️  TMDB_Key not found in environment variables")
        cache_data[cache_key] = 'No API key'
        return 'No API key'
    
    try:
        print(f"🔍 Fetching season artwork for: {show_title} ({show_year}) - Season {season_number}")
        
        # Step 1: Search for TV show
        search_url = f"https://api.themoviedb.org/3/search/tv"
        params = {
            'api_key': api_key,
            'query': show_title,
            'first_air_date_year': show_year
        }
        
        response = requests.get(search_url, params=params)
        time.sleep(0.25)  # Be respectful to API
        
        if response.status_code != 200:
            print(f"❌ TMDB search failed with status {response.status_code}")
            cache_data[cache_key] = 'API error'
            return 'API error'
        
        search_data = response.json()
        
        if not search_data['results']:
            print(f"❌ No TMDB results found for {show_title} ({show_year})")
            cache_data[cache_key] = 'No show found'
            return 'No show found'
        
        # Get the first result (usually most relevant)
        tv_show = search_data['results'][0]
        tv_id = tv_show['id']
        
        # Step 2: Get season details
        season_url = f"https://api.themoviedb.org/3/tv/{tv_id}/season/{season_number}"
        season_params = {'api_key': api_key}
        
        season_response = requests.get(season_url, params=season_params)
        time.sleep(0.25)  # Be respectful to API
        
        if season_response.status_code != 200:
            print(f"❌ Season details failed with status {season_response.status_code}")
            cache_data[cache_key] = 'Season not found'
            return 'Season not found'
        
        season_data = season_response.json()
        
        # Step 3: Extract poster URL
        poster_url = 'No poster found'
        if season_data.get('poster_path'):
            poster_url = f"https://image.tmdb.org/t/p/w500{season_data['poster_path']}"
            print(f"✅ Found season artwork for {show_title} S{season_number}")
        else:
            print(f"❌ No poster found for {show_title} S{season_number}")
        
        # Cache the result
        cache_data[cache_key] = poster_url
        return poster_url
        
    except Exception as e:
        print(f"❌ Error fetching season artwork: {e}")
        cache_data[cache_key] = 'Error'
        return 'Error'


def get_existing_season_artwork(processed_file_path):
    """
    Load existing season artwork from processed CSV file
    Returns dict of {show_title_year_season: poster_url}
    """
    existing_artwork = {}
    
    if not os.path.exists(processed_file_path):
        return existing_artwork
    
    try:
        # Read existing processed file
        df_existing = pd.read_csv(processed_file_path, sep='|', encoding='utf-8')
        
        # Check if season_poster_url column exists
        if 'season_poster_url' in df_existing.columns:
            # Extract unique combinations that have artwork
            for _, row in df_existing.iterrows():
                if pd.notna(row['season_poster_url']) and row['season_poster_url'] != '':
                    cache_key = f"{row['show_title']}_{row['show_year']}_S{row['season']}"
                    existing_artwork[cache_key] = row['season_poster_url']
            
            print(f"📚 Found {len(existing_artwork)} existing season artwork URLs")
        
        return existing_artwork
        
    except Exception as e:
        print(f"⚠️  Error reading existing artwork: {e}")
        return {}


def create_trakt_file():
    """
    Processes all history*.json files and converts them to CSV format.
    Automatically fetches missing season artwork from TMDB.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Trakt history data...")

    # Find all history files in the exports folder
    export_folder = "files/exports/trakt_exports"
    history_pattern = os.path.join(export_folder, "history*.json")
    history_files = glob.glob(history_pattern)

    if not history_files:
        print("❌ No history files found in exports folder")
        return False

    print(f"📊 Found {len(history_files)} history file(s)")

    try:
        # Load and combine data from all history files
        all_history_data = []

        for history_file in sorted(history_files):
            filename = os.path.basename(history_file)
            print(f"📖 Reading {filename}...")

            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

            print(f"   Loaded {len(history_data)} entries")
            all_history_data.extend(history_data)

        print(f"📊 Total combined entries: {len(all_history_data)}")

        # Load ratings data
        episode_ratings, season_ratings, show_ratings = load_ratings_data()

        # Filter for episodes only
        episodes = [entry for entry in all_history_data if entry.get('type') == 'episode']
        print(f"📺 Found {len(episodes)} episode entries")

        # Convert to CSV format
        csv_data = []

        for entry in episodes:
            # Extract IDs for rating lookup
            show_trakt_id = entry.get('show', {}).get('ids', {}).get('trakt', '')
            episode_trakt_id = entry.get('episode', {}).get('ids', {}).get('trakt', '')
            season_num = entry.get('episode', {}).get('season', '')

            # Look up ratings
            episode_rating = episode_ratings.get(episode_trakt_id, '')
            season_rating = season_ratings.get(f"{show_trakt_id}_{season_num}", '')
            show_rating = show_ratings.get(show_trakt_id, '')

            # Extract data with safe defaults
            episode_data = {
                'watch_id': entry.get('id', ''),
                'watched_at': entry.get('watched_at', ''),
                'show_title': entry.get('show', {}).get('title', ''),
                'show_year': entry.get('show', {}).get('year', ''),
                'season': season_num,
                'episode_number': entry.get('episode', {}).get('number', ''),
                'episode_title': entry.get('episode', {}).get('title', ''),
                'progress': entry.get('progress', ''),
                'location': entry.get('location', ''),
                'duration': entry.get('duration', ''),
                'show_trakt_id': show_trakt_id,
                'show_imdb_id': entry.get('show', {}).get('ids', {}).get('imdb', ''),
                'episode_trakt_id': episode_trakt_id,
                'episode_imdb_id': entry.get('episode', {}).get('ids', {}).get('imdb', ''),
                'episode_rating': episode_rating,
                'season_rating': season_rating,
                'show_rating': show_rating
            }

            csv_data.append(episode_data)

        # Create DataFrame
        df = pd.DataFrame(csv_data)

        # Remove duplicates based on watch_id (unique identifier for each watch event)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['watch_id'], keep='first')
        duplicates_removed = initial_count - len(df)

        if duplicates_removed > 0:
            print(f"🔄 Removed {duplicates_removed} duplicate entries")

        # Convert watched_at to datetime
        df['watched_at'] = pd.to_datetime(df['watched_at'])

        # Replace 1970-01-01 timestamps with NaT (null)
        df.loc[df['watched_at'].dt.year == 1970, 'watched_at'] = pd.NaT

        # Sort by watched_at descending (nulls last), then by show_year descending
        df = df.sort_values(['watched_at', 'show_year'], ascending=[False, False], na_position='last')
        
        # Add additional columns for consistency with other processing
        df['Seconds'] = pd.NaT
        df['Source'] = 'Trakt'
        df['Timestamp'] = df['watched_at']
        
        # Create processed files directory
        output_file = 'files/source_processed_files/trakt/trakt_processed.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # ARTWORK FETCHING SECTION
        print("\n🎨 Fetching season artwork...")
        
        # Load existing artwork cache
        artwork_cache = load_season_artwork_cache()
        
        # Load existing artwork from processed file (if it exists)
        existing_artwork = get_existing_season_artwork(output_file)
        
        # Merge existing artwork into cache
        artwork_cache.update(existing_artwork)
        
        # Find unique show+season combinations that need artwork
        unique_seasons = df.groupby(['show_title', 'show_year', 'season']).size().reset_index(name='count')
        
        seasons_needing_artwork = []
        for _, row in unique_seasons.iterrows():
            cache_key = f"{row['show_title']}_{row['show_year']}_S{row['season']}"
            if cache_key not in artwork_cache or artwork_cache[cache_key] in ['', 'No poster found', 'API error', 'Error']:
                seasons_needing_artwork.append({
                    'show_title': row['show_title'],
                    'show_year': row['show_year'],
                    'season': row['season'],
                    'cache_key': cache_key
                })
        
        print(f"📊 Found {len(unique_seasons)} unique seasons")
        print(f"🎨 Have artwork for {len(artwork_cache)} seasons")
        print(f"🔍 Need to fetch artwork for {len(seasons_needing_artwork)} seasons")
        
        # Fetch missing artwork automatically
        if seasons_needing_artwork:
            print("🚀 Fetching season artwork from TMDB...")

            fetch_count = 0
            for season_info in seasons_needing_artwork:
                artwork_url = get_tmdb_season_artwork(
                    season_info['show_title'],
                    season_info['show_year'],
                    season_info['season'],
                    artwork_cache
                )

                fetch_count += 1

                # Save cache every 5 retrievals
                if fetch_count % 5 == 0:
                    save_season_artwork_cache(artwork_cache)
                    print(f"💾 Saved cache progress ({fetch_count}/{len(seasons_needing_artwork)} fetched)")

                # Small delay between requests
                time.sleep(0.5)

            # Save final cache update
            save_season_artwork_cache(artwork_cache)
            print(f"✅ Updated artwork cache with {len(seasons_needing_artwork)} new entries")
        
        # Add season_poster_url column to dataframe
        df['season_poster_url'] = df.apply(
            lambda row: artwork_cache.get(f"{row['show_title']}_{row['show_year']}_S{row['season']}", ''),
            axis=1
        )

        # Enforce snake_case before saving
        df = enforce_snake_case(df, "processed file")

        # Save to CSV with pipe separator (consistent with other processors)
        df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
        
        print(f"✅ Successfully processed {len(df)} episodes")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Date range: {df['watched_at'].min()} to {df['watched_at'].max()}")

        return True

    except Exception as e:
        print(f"❌ Error processing Trakt data: {e}")
        import traceback
        traceback.print_exc()
        return False


def full_trakt_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Trakt SOURCE pipeline with 2 options.

    Options:
    1. Download new data and process
    2. Process existing data

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🎬 TRAKT SOURCE DATA PIPELINE")
    print("="*60)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new data and process")
        print("2. Process existing data")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Download new data and process...")
        download_success = download_trakt_data()

        if download_success:
            move_success = move_trakt_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_trakt_files()

        if move_success:
            process_success = create_trakt_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_trakt_file()

        success = process_success

    elif choice == "2":
        print("\n⚙️  Process existing data...")
        success = create_trakt_file()

    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    print("\n" + "="*60)
    if success:
        print("✅ Trakt source pipeline completed successfully!")
        print("ℹ️  Note: To upload to Drive, run the Shows topic pipeline.")
        record_successful_run('source_trakt', 'active')
    else:
        print("❌ Trakt source pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    print("🎬 Trakt Source Processing Tool")
    print("This tool processes Trakt exports into source data files.")
    print("For website generation and upload, use the Shows topic coordinator.")
    full_trakt_pipeline(auto_full=False)