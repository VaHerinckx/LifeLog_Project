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
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
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
    
    response = prompt_user_download_status("Trakt")
    return response


def move_trakt_files():
    """
    Moves the downloaded Trakt files from Downloads, unzips the file,
    and extracts history.json from the watched subfolder.
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
        
        # Look for history.json in the entinval/watched subfolder
        history_file_path = os.path.join(temp_extract_path, "entinval", "watched", "history.json")
        
        if not os.path.exists(history_file_path):
            print("❌ history.json not found in entinval/watched subfolder")
            # Clean up temp folder
            shutil.rmtree(temp_extract_path, ignore_errors=True)
            return False
        
        # Move history.json to the export folder
        destination_path = os.path.join(export_folder, "history.json")
        shutil.move(history_file_path, destination_path)
        
        print(f"✅ Successfully moved history.json to {export_folder}")
        
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
    Processes the history.json file and converts it to CSV format.
    Automatically fetches missing season artwork from TMDB.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Trakt history data...")
    
    # Check if history.json exists
    history_file = "files/exports/trakt_exports/history.json"
    if not os.path.exists(history_file):
        print("❌ history.json not found in exports folder")
        return False
    
    try:
        # Load JSON data
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        print(f"📊 Loaded {len(history_data)} entries from history.json")
        
        # Filter for episodes only
        episodes = [entry for entry in history_data if entry.get('type') == 'episode']
        print(f"📺 Found {len(episodes)} episode entries")
        
        # Convert to CSV format
        csv_data = []
        
        for entry in episodes:
            # Extract data with safe defaults
            episode_data = {
                'watch_id': entry.get('id', ''),
                'watched_at': entry.get('watched_at', ''),
                'show_title': entry.get('show', {}).get('title', ''),
                'show_year': entry.get('show', {}).get('year', ''),
                'season': entry.get('episode', {}).get('season', ''),
                'episode_number': entry.get('episode', {}).get('number', ''),
                'episode_title': entry.get('episode', {}).get('title', ''),
                'progress': entry.get('progress', ''),
                'location': entry.get('location', ''),
                'duration': entry.get('duration', ''),
                'show_trakt_id': entry.get('show', {}).get('ids', {}).get('trakt', ''),
                'show_imdb_id': entry.get('show', {}).get('ids', {}).get('imdb', ''),
                'episode_trakt_id': entry.get('episode', {}).get('ids', {}).get('trakt', ''),
                'episode_imdb_id': entry.get('episode', {}).get('ids', {}).get('imdb', '')
            }
            
            csv_data.append(episode_data)
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Convert watched_at to datetime and sort by it (most recent first)
        df['watched_at'] = pd.to_datetime(df['watched_at'])
        df = df.sort_values('watched_at', ascending=False)
        
        # Add additional columns for consistency with other processing
        df['Seconds'] = pd.NaT
        df['Source'] = 'Trakt'
        df['Timestamp'] = df['watched_at']
        
        # Create processed files directory
        os.makedirs('files/processed_files/movies', exist_ok=True)
        output_file = 'files/processed_files/movies/trakt_processed.csv'
        
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

            for season_info in seasons_needing_artwork:
                artwork_url = get_tmdb_season_artwork(
                    season_info['show_title'],
                    season_info['show_year'],
                    season_info['season'],
                    artwork_cache
                )

                # Small delay between requests
                time.sleep(0.5)

            # Save updated cache
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

        # Generate website files
        generate_shows_website_page_files(df)

        return True

    except Exception as e:
        print(f"❌ Error processing Trakt data: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_shows_website_page_files(df):
    """
    Generate website-optimized files for the Shows page.

    Args:
        df: Processed dataframe (already in snake_case)

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🌐 Generating website files for Shows page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/shows'
        os.makedirs(website_dir, exist_ok=True)

        # Work with copy to avoid modifying original
        df_web = df.copy()

        # Enforce snake_case before saving
        df_web = enforce_snake_case(df_web, "shows_page_data")

        # Save website file
        website_path = f'{website_dir}/shows_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Website file: {len(df_web):,} records → {website_path}")

        return True

    except Exception as e:
        print(f"❌ Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_trakt_processed_file(fetch_artwork_auto=None):
    """
    Legacy wrapper for backward compatibility.
    Use create_trakt_file() instead.
    Note: fetch_artwork_auto parameter is deprecated and ignored.
    """
    return create_trakt_file()


def upload_trakt_results():
    """Upload the processed Trakt files to Google Drive"""
    print("☁️  Uploading Trakt results to Google Drive...")

    files_to_upload = [
        'files/website_files/shows/shows_page_data.csv'
    ]
    
    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No files found to upload")
        return False
    
    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)
    
    if success:
        print("✅ Trakt results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")
    
    return success


def manual_season_artwork_override():
    """
    Allows manual override of season artwork URLs for specific shows/seasons.
    Updates both cache and processed CSV file.
    """
    print("\n🎨 Manual Season Artwork Override")
    print("="*50)
    
    # Load existing cache
    artwork_cache = load_season_artwork_cache()
    
    # Load current processed file to show available seasons
    processed_file = 'files/processed_files/movies/trakt_processed.csv'
    
    if not os.path.exists(processed_file):
        print("❌ No processed Trakt file found. Please run processing first.")
        return False
    
    try:
        df = pd.read_csv(processed_file, sep='|', encoding='utf-8')
        
        # Get unique show+season combinations
        unique_seasons = df.groupby(['show_title', 'show_year', 'season']).size().reset_index(name='count')
        unique_seasons = unique_seasons.sort_values(['show_title', 'season'])
        
        print(f"\n📺 Available shows and seasons ({len(unique_seasons)} total):")
        print("-" * 60)
        
        for i, (_, row) in enumerate(unique_seasons.iterrows(), 1):
            cache_key = f"{row['show_title']}_{row['show_year']}_S{row['season']}"
            current_url = artwork_cache.get(cache_key, 'No artwork')
            
            print(f"{i:2d}. {row['show_title']} ({row['show_year']}) - Season {row['season']}")
            print(f"    Episodes: {row['count']}")
            print(f"    Current artwork: {current_url}")
            print()
        
        # Get user selection
        while True:
            try:
                choice = input("Select a season to override (number) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("👋 Exiting manual override")
                    return True
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(unique_seasons):
                    selected_season = unique_seasons.iloc[choice_num - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(unique_seasons)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
        
        # Show selected season details
        cache_key = f"{selected_season['show_title']}_{selected_season['show_year']}_S{selected_season['season']}"
        current_url = artwork_cache.get(cache_key, 'No artwork')
        
        print(f"\n🎯 Selected: {selected_season['show_title']} ({selected_season['show_year']}) - Season {selected_season['season']}")
        print(f"Current artwork: {current_url}")
        print(f"Episodes affected: {selected_season['count']}")
        
        # Get new artwork URL
        new_url = input("\nEnter new artwork URL (or 'remove' to clear): ").strip()
        
        if new_url.lower() == 'remove':
            new_url = ''
            print("🗑️  Artwork will be removed")
        elif new_url == '':
            print("❌ No URL provided, operation cancelled")
            return False
        else:
            print(f"🎨 New artwork URL: {new_url}")
        
        # Confirm the change
        confirm = input("\nConfirm this change? (y/N): ").strip().lower()
        
        if confirm != 'y':
            print("❌ Operation cancelled")
            return False
        
        # Update cache
        artwork_cache[cache_key] = new_url
        
        # Save updated cache
        if not save_season_artwork_cache(artwork_cache):
            print("❌ Failed to save cache")
            return False
        
        # Update processed CSV file
        df['season_poster_url'] = df.apply(
            lambda row: artwork_cache.get(f"{row['show_title']}_{row['show_year']}_S{row['season']}", ''),
            axis=1
        )
        
        # Save updated CSV
        df.to_csv(processed_file, sep='|', index=False, encoding='utf-8')
        
        print(f"✅ Successfully updated artwork for {selected_season['show_title']} Season {selected_season['season']}")
        print(f"📊 Updated {selected_season['count']} episode records")
        print(f"💾 Updated cache and CSV file")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during manual override: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_season_artwork_override():
    """
    Allows batch override of multiple season artwork URLs via file input.
    Expected format: CSV with columns: show_title, show_year, season, artwork_url
    """
    print("\n🎨 Batch Season Artwork Override")
    print("="*50)
    
    # Check for batch file
    batch_file = 'files/work_files/trakt_work_files/season_artwork_overrides.csv'
    
    if not os.path.exists(batch_file):
        print(f"📝 Creating template file: {batch_file}")
        
        # Create template file
        os.makedirs(os.path.dirname(batch_file), exist_ok=True)
        
        template_data = {
            'show_title': ['Succession', 'The Last of Us'],
            'show_year': [2018, 2023],
            'season': [1, 2],
            'artwork_url': ['https://example.com/succession-s1.jpg', 'https://example.com/tlou-s2.jpg']
        }
        
        template_df = pd.DataFrame(template_data)
        template_df.to_csv(batch_file, index=False)
        
        print(f"✅ Template created. Edit the file and run this function again.")
        print(f"📄 File location: {batch_file}")
        print("\n📋 Template format:")
        print("   - show_title: Exact show name")
        print("   - show_year: Year the show started")
        print("   - season: Season number")
        print("   - artwork_url: New artwork URL (or empty to remove)")
        
        return True
    
    try:
        # Load batch file
        batch_df = pd.read_csv(batch_file)
        
        required_columns = ['show_title', 'show_year', 'season', 'artwork_url']
        missing_columns = [col for col in required_columns if col not in batch_df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns in batch file: {missing_columns}")
            return False
        
        print(f"📊 Found {len(batch_df)} artwork overrides to process")
        
        # Load existing cache
        artwork_cache = load_season_artwork_cache()
        
        # Process each override
        updates_made = 0
        
        for _, row in batch_df.iterrows():
            cache_key = f"{row['show_title']}_{row['show_year']}_S{row['season']}"
            new_url = str(row['artwork_url']).strip()
            
            if pd.isna(row['artwork_url']) or new_url == '':
                new_url = ''
            
            old_url = artwork_cache.get(cache_key, 'No artwork')
            
            if old_url != new_url:
                artwork_cache[cache_key] = new_url
                updates_made += 1
                print(f"🔄 Updated: {row['show_title']} S{row['season']} - {row['show_year']}")
                print(f"   Old: {old_url}")
                print(f"   New: {new_url}")
            else:
                print(f"⏭️  Skipped: {row['show_title']} S{row['season']} - No change needed")
        
        if updates_made == 0:
            print("✅ No updates needed - all artwork URLs are already current")
            return True
        
        # Save updated cache
        if not save_season_artwork_cache(artwork_cache):
            print("❌ Failed to save cache")
            return False
        
        # Update processed CSV file
        processed_file = 'files/processed_files/movies/trakt_processed.csv'
        
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, sep='|', encoding='utf-8')
            
            # Update season_poster_url column
            df['season_poster_url'] = df.apply(
                lambda row: artwork_cache.get(f"{row['show_title']}_{row['show_year']}_S{row['season']}", ''),
                axis=1
            )
            
            # Save updated CSV
            df.to_csv(processed_file, sep='|', index=False, encoding='utf-8')
            
            print(f"📊 Updated processed CSV file")
        
        print(f"✅ Batch override completed - {updates_made} updates made")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during batch override: {e}")
        import traceback
        traceback.print_exc()
        return False


def full_trakt_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Trakt processing pipeline with 5 standard options.

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🎬 TRAKT PROCESSING PIPELINE")
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
        print("4. Manual season artwork override")
        print("5. Batch season artwork override")

        choice = input("\nEnter your choice (1-5): ").strip()

    success = False

    if choice == "1":
        # Option 1: Download new data, process, and upload to Drive
        print("\n🚀 Running full pipeline...")
        if download_trakt_data():
            if move_trakt_files():
                if create_trakt_file():  # Will prompt for artwork
                    success = upload_trakt_results()
                else:
                    print("❌ Failed to process Trakt data")
            else:
                print("❌ Failed to move Trakt files")
        else:
            print("❌ Download not confirmed")

    elif choice == "2":
        # Option 2: Process existing data and upload to Drive
        print("\n⚙️  Processing existing data and uploading to Drive...")
        if create_trakt_file():  # Will prompt for artwork
            success = upload_trakt_results()
        else:
            print("❌ Failed to process Trakt data")

    elif choice == "3":
        # Option 3: Upload existing processed files to Drive
        print("\n⬆️  Uploading existing processed files to Drive...")
        success = upload_trakt_results()

    elif choice == "4":
        # Option 4: Manual season artwork override
        print("\n🎨 Manual season artwork override...")
        success = manual_season_artwork_override()

    elif choice == "5":
        # Option 5: Batch season artwork override
        print("\n🎨 Batch season artwork override...")
        success = batch_season_artwork_override()

    else:
        print("❌ Invalid choice. Please select 1-5.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Trakt pipeline completed successfully!")
        record_successful_run('movies_trakt', 'active')
    else:
        print("❌ Trakt pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    full_trakt_pipeline()