import os
import requests
import pandas as pd
import time
import json
from dotenv import load_dotenv
from src.utils.file_operations import find_unzip_folder, clean_rename_move_folder
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.utils_functions import record_successful_run, enforce_snake_case

load_dotenv()

# Path to TMDB cache file
TMDB_CACHE_PATH = 'files/work_files/letterboxd_work_files/tmdb_cache.json'


def load_tmdb_cache():
    """Load TMDB cache from JSON file"""
    try:
        if os.path.exists(TMDB_CACHE_PATH):
            with open(TMDB_CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading TMDB cache: {e}")
    return {}


def save_to_tmdb_cache(movie_key, poster_url, genres):
    """Save movie data to TMDB cache"""
    try:
        # Load existing cache
        cache = load_tmdb_cache()

        # Update cache
        cache[movie_key] = {
            'poster_url': poster_url,
            'genres': genres
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(TMDB_CACHE_PATH), exist_ok=True)

        # Save back to file
        with open(TMDB_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"⚠️  Error saving to TMDB cache: {e}")
        return False


def get_genre(title, release_year):
    """Legacy function - now redirects to TMDB for consistency"""
    movie_info = get_tmdb_movie_info(title, release_year)
    return movie_info['genres']


def get_tmdb_movie_info(title, release_year):
    """Retrieves both poster URL and genres from TMDB API (with JSON cache)"""
    # Create cache key
    movie_key = f"{title}_{release_year}"

    # Check JSON cache first (primary cache)
    cache = load_tmdb_cache()
    if movie_key in cache:
        cached_data = cache[movie_key]
        if (cached_data.get('poster_url') != 'No poster found' and
            cached_data.get('genres') != 'Unknown'):
            print(f"📦 Using cached data for {title} ({release_year})")
            return {
                'poster_url': cached_data['poster_url'],
                'genres': cached_data['genres']
            }

    # Get TMDB API key from environment
    api_key = os.environ.get('TMDB_Key')
    if not api_key:
        print("TMDB_API_KEY not found in environment variables")
        return {
            'poster_url': 'No poster found',
            'genres': 'Unknown'
        }

    try:
        print(f"Fetching data for: {title} ({release_year})")

        # Search for the movie on TMDB
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': api_key,
            'query': title,
            'year': release_year
        }

        response = requests.get(search_url, params=params)

        # Add a small delay to be respectful to the API
        time.sleep(0.25)

        if response.status_code == 200:
            data = response.json()

            if data['results'] and len(data['results']) > 0:
                # Get the first result (usually the most relevant)
                movie = data['results'][0]

                # Get poster URL
                poster_url = 'No poster found'
                if movie.get('poster_path'):
                    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"

                # Get genres - need to fetch detailed movie info for genres
                movie_id = movie['id']
                details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                details_params = {'api_key': api_key}

                # Add another small delay
                time.sleep(0.25)

                details_response = requests.get(details_url, params=details_params)
                genres_list = []

                if details_response.status_code == 200:
                    details_data = details_response.json()
                    if details_data.get('genres'):
                        genres_list = [genre['name'] for genre in details_data['genres']]

                genres_string = ', '.join(genres_list) if genres_list else 'Unknown'

                print(f"Found data - Poster: {'✅' if poster_url != 'No poster found' else '❌'}, Genres: {genres_string}")

                # Save to JSON cache for future use
                save_to_tmdb_cache(movie_key, poster_url, genres_string)

                return {
                    'poster_url': poster_url,
                    'genres': genres_string
                }
            else:
                print(f"No TMDB results found for {title} ({release_year})")
                return {
                    'poster_url': 'No poster found',
                    'genres': 'Unknown'
                }
        else:
            print(f"TMDB API error {response.status_code} for {title} ({release_year})")
            return {
                'poster_url': 'No poster found',
                'genres': 'Unknown'
            }

    except Exception as e:
        print(f"Error fetching data for {title} ({release_year}): {str(e)}")
        return {
            'poster_url': 'No poster found',
            'genres': 'Unknown'
        }


def get_watched_rating(path_watched, path_ratings, path_diary):
    """Merges watched, ratings, and diary dfs - using diary for actual watch dates"""
    df_watched = pd.read_csv(path_watched)
    df_ratings = pd.read_csv(path_ratings)
    df_diary = pd.read_csv(path_diary)

    # First merge watched films with ratings
    df_merged = df_watched.merge(df_ratings[['Name', 'Year', 'Rating']], on=['Name', 'Year'], how='left')

    # Then merge with diary to get actual watch dates
    # Only keep the 'Watched Date' column from diary, rename it to 'ActualDate'
    df_diary_dates = df_diary[['Name', 'Year', 'Watched Date']].rename(columns={'Watched Date': 'ActualDate'})

    # Merge with diary dates (left join to keep all watched movies)
    df_final = df_merged.merge(df_diary_dates, on=['Name', 'Year'], how='left')

    # Replace the original Date with ActualDate where available, otherwise leave blank
    df_final['Date'] = df_final['ActualDate'].fillna('')

    # Drop the temporary ActualDate column
    df_final = df_final.drop(columns=['ActualDate'])

    return df_final


def download_letterboxd_data():
    """
    Opens Letterboxd export page and prompts user to download data.
    Returns True if user confirms download, False otherwise.
    """
    print("🎬 Starting Letterboxd data download...")

    urls = ['https://letterboxd.com/settings/data/']
    open_web_urls(urls)

    print("📝 Instructions:")
    print("   1. Click 'Export your data'")
    print("   2. Wait for the export to be prepared")
    print("   3. Download the ZIP file when ready")
    print("   4. The file will be named like 'letterboxd-username-YYYY-MM-DD-HH-MM-utc.zip'")

    response = prompt_user_download_status("Letterboxd")

    return response


def move_letterboxd_files():
    """
    Moves the downloaded Letterboxd files from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving Letterboxd files...")

    # First, try to unzip the letterboxd file
    unzip_success = find_unzip_folder("letterboxd")
    if not unzip_success:
        print("❌ Failed to find or unzip Letterboxd file")
        return False

    # Then move the unzipped folder
    move_success = clean_rename_move_folder(
        export_folder="files/exports",
        download_folder="/Users/valen/Downloads",
        folder_name="letterboxd_export_unzipped",
        new_folder_name="letterboxd_exports"
    )

    if move_success:
        print("✅ Successfully moved Letterboxd files to exports folder")
    else:
        print("❌ Failed to move Letterboxd files")

    return move_success


def create_letterboxd_file():
    """
    Main processing function that adds poster URLs and genres to the letterboxd data.
    Returns True if successful, False otherwise.

    This is the SOURCE-level processor - outputs raw processed data only.
    Website file generation happens in the topic coordinator.
    """
    print("⚙️  Processing Letterboxd source data...")

    path_watched = "files/exports/letterboxd_exports/watched.csv"
    path_ratings = "files/exports/letterboxd_exports/ratings.csv"
    path_diary = "files/exports/letterboxd_exports/diary.csv"
    output_path = 'files/source_processed_files/letterboxd/letterboxd_processed.csv'

    try:
        # Check if input files exist
        if not os.path.exists(path_watched):
            print(f"❌ Watched file not found: {path_watched}")
            return False

        if not os.path.exists(path_ratings):
            print(f"❌ Ratings file not found: {path_ratings}")
            return False

        if not os.path.exists(path_diary):
            print(f"❌ Diary file not found: {path_diary}")
            return False

        # Merge watched, ratings, and diary data
        print("📖 Reading and merging watched, ratings, and diary data...")
        df = get_watched_rating(path_watched, path_ratings, path_diary)

        # Add Genre and Poster URLs from TMDB
        print("🎭 Adding genre and poster information from TMDB...")
        print(f"Processing {len(df)} movies for TMDB data...")

        # Get unique movies to avoid duplicate API calls
        unique_movies = df[['Name', 'Year']].drop_duplicates()
        print(f"Found {len(unique_movies)} unique movies")

        # Create dictionaries to store the data
        poster_dict = {}
        genre_dict = {}

        for idx, row in unique_movies.iterrows():
            movie_name = row['Name']
            movie_year = row['Year']

            # Create a key for the dictionaries
            movie_key = f"{movie_name}_{movie_year}"

            # Get both poster and genre data from TMDB
            movie_info = get_tmdb_movie_info(movie_name, movie_year)
            poster_dict[movie_key] = movie_info['poster_url']
            genre_dict[movie_key] = movie_info['genres']

            # Print progress every 10 movies
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(unique_movies)} unique movies")

        # Map data back to the main dataframe
        df['PosterURL'] = df.apply(lambda x: poster_dict.get(f"{x['Name']}_{x['Year']}", 'No poster found'), axis=1)
        df['Genre'] = df.apply(lambda x: genre_dict.get(f"{x['Name']}_{x['Year']}", 'Unknown'), axis=1)

        # Rename columns to snake_case for consistency
        df = df.rename(columns={
            'Name': 'name',
            'Year': 'year',
            'Rating': 'rating',
            'Date': 'date',
            'PosterURL': 'poster_url',
            'Genre': 'genre'
        })

        # Convert date to datetime for proper sorting
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date (most recent first)
        df = df.sort_values('date', ascending=False)

        # Enforce snake_case before saving
        df = enforce_snake_case(df, "processed file")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        print(f"💾 Saving source processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        print(f"\n✅ Processing complete!")
        print(f"📊 Processed {len(df)} movie entries")
        print(f"🎬 Found posters for {len([url for url in poster_dict.values() if url != 'No poster found'])} movies")
        print(f"🎭 Found genres for {len([genre for genre in genre_dict.values() if genre != 'Unknown'])} movies")

        # Print some sample data for verification
        movies_with_data = df[(df['poster_url'] != 'No poster found') & (df['genre'] != 'Unknown')].head(5)
        if not movies_with_data.empty:
            print("\n🎯 Sample movies with complete data:")
            for _, movie in movies_with_data.iterrows():
                print(f"  • {movie['name']} ({movie['year']})")
                print(f"    Genres: {movie['genre']}")
                print(f"    Poster: {movie['poster_url'][:50]}...")

        return True

    except Exception as e:
        print(f"❌ Error processing Letterboxd data: {e}")
        return False


def full_letterboxd_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Letterboxd SOURCE pipeline with 2 options.

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
    print("🎬 LETTERBOXD SOURCE DATA PIPELINE")
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

        # Step 1: Download
        download_success = download_letterboxd_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_letterboxd_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_letterboxd_files()

        # Step 3: Process (fallback to existing files if no new files)
        if move_success:
            process_success = create_letterboxd_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_letterboxd_file()

        success = process_success

    elif choice == "2":
        print("\n⚙️  Process existing data...")
        success = create_letterboxd_file()

    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Letterboxd source pipeline completed successfully!")
        print("ℹ️  Note: To upload to Drive, run the Movies topic pipeline.")
        # Record successful run
        record_successful_run('source_letterboxd', 'active')
    else:
        print("❌ Letterboxd source pipeline failed")
    print("="*60)

    return success


# Legacy function for backward compatibility
def process_letterboxd_export(upload="Y"):
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use full_letterboxd_pipeline() for source processing or
    the Movies topic coordinator for website generation and upload.
    """
    print("⚠️  process_letterboxd_export() is deprecated.")
    print("   Using new source pipeline...")
    return full_letterboxd_pipeline(auto_process_only=True)


if __name__ == "__main__":
    # Allow running this file directly
    print("🎬 Letterboxd Source Processing Tool")
    print("This tool processes Letterboxd exports into source data files.")
    print("For website generation and upload, use the Movies topic coordinator.")

    # Run the pipeline
    full_letterboxd_pipeline(auto_full=False)
