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


def save_to_tmdb_cache(movie_key, movie_data):
    """Save comprehensive movie data to TMDB cache

    Args:
        movie_key (str): Cache key (e.g., "Movie Title_2024")
        movie_data (dict): Dictionary containing all movie fields to cache
    """
    try:
        # Load existing cache
        cache = load_tmdb_cache()

        # Update cache with comprehensive movie data
        cache[movie_key] = movie_data

        # Ensure directory exists
        os.makedirs(os.path.dirname(TMDB_CACHE_PATH), exist_ok=True)

        # Save back to file
        with open(TMDB_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"⚠️  Error saving to TMDB cache: {e}")
        return False


def extract_director(crew_list):
    """Extract director name from crew array"""
    try:
        if crew_list:
            for person in crew_list:
                if person.get('job') == 'Director':
                    return person.get('name', 'Unknown')
        return 'Unknown'
    except Exception:
        return 'Unknown'


def extract_writers(crew_list):
    """Extract writers from crew array"""
    try:
        if crew_list:
            writers = [person.get('name', '') for person in crew_list
                      if person.get('department') == 'Writing' and person.get('job') in ['Screenplay', 'Writer']]
            return ', '.join(writers[:3]) if writers else 'Unknown'
        return 'Unknown'
    except Exception:
        return 'Unknown'


def extract_top_cast(cast_list, limit=10):
    """Extract top cast members from cast array"""
    try:
        if cast_list:
            # Cast is already sorted by order (0 is first)
            top_actors = [person.get('name', '') for person in cast_list[:limit] if person.get('name')]
            return ', '.join(top_actors) if top_actors else 'Unknown'
        return 'Unknown'
    except Exception:
        return 'Unknown'


def extract_us_certification(release_dates_data):
    """Extract US certification (rating) from release_dates"""
    try:
        if release_dates_data and 'results' in release_dates_data:
            for country in release_dates_data['results']:
                if country.get('iso_3166_1') == 'US':
                    for release in country.get('release_dates', []):
                        cert = release.get('certification', '')
                        if cert:
                            return cert
        return 'Unknown'
    except Exception:
        return 'Unknown'


def extract_trailer_key(videos_data):
    """Extract YouTube trailer key from videos"""
    try:
        if videos_data and 'results' in videos_data:
            # Find first official trailer
            for video in videos_data['results']:
                if (video.get('site') == 'YouTube' and
                    video.get('type') == 'Trailer' and
                    video.get('official', False)):
                    return video.get('key', 'No trailer found')
            # If no official trailer, try any trailer
            for video in videos_data['results']:
                if video.get('site') == 'YouTube' and video.get('type') == 'Trailer':
                    return video.get('key', 'No trailer found')
        return 'No trailer found'
    except Exception:
        return 'No trailer found'


def extract_keywords(keywords_data):
    """Extract keywords from keywords data"""
    try:
        if keywords_data and 'keywords' in keywords_data:
            keywords = [kw.get('name', '') for kw in keywords_data['keywords'][:10]]
            return ', '.join(keywords) if keywords else 'None'
        return 'None'
    except Exception:
        return 'None'


def get_genre(title, release_year):
    """Legacy function - now redirects to TMDB for consistency"""
    movie_info = get_tmdb_movie_info(title, release_year)
    return movie_info.get('genres', 'Unknown')


def get_tmdb_movie_info(title, release_year):
    """Retrieves comprehensive movie data from TMDB API (with JSON cache)

    Returns dictionary with 30+ fields including:
    - Basic info: title, original_title, tagline, overview, release_date, runtime, status
    - Ratings: vote_average, vote_count, popularity
    - Financial: budget, revenue
    - Images: poster_url, backdrop_path
    - Credits: director, cast, writer
    - Production: production_companies, production_countries
    - Classification: certification_us, keywords
    - Media: trailer_key
    - External IDs: tmdb_id, imdb_id, wikidata_id
    """
    # Create cache key
    movie_key = f"{title}_{release_year}"

    # Check JSON cache first (primary cache)
    cache = load_tmdb_cache()
    if movie_key in cache:
        cached_data = cache[movie_key]
        # Validate cache has essential fields
        if (cached_data.get('poster_url') != 'No poster found' and
            cached_data.get('genres') != 'Unknown'):
            print(f"📦 Using cached data for {title} ({release_year})")
            return cached_data

    # Get TMDB API key from environment
    api_key = os.environ.get('TMDB_Key')
    if not api_key:
        print("TMDB_API_KEY not found in environment variables")
        return _get_default_movie_data()

    try:
        print(f"🎬 Fetching comprehensive data for: {title} ({release_year})")

        # Step 1: Search for the movie on TMDB
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': api_key,
            'query': title,
            'year': release_year
        }

        response = requests.get(search_url, params=params)
        time.sleep(0.25)  # Rate limiting

        if response.status_code != 200:
            print(f"❌ TMDB search API error {response.status_code} for {title} ({release_year})")
            return _get_default_movie_data()

        data = response.json()

        if not data['results'] or len(data['results']) == 0:
            print(f"❌ No TMDB results found for {title} ({release_year})")
            return _get_default_movie_data()

        # Get the first result (usually the most relevant)
        movie = data['results'][0]
        movie_id = movie['id']

        # Step 2: Fetch detailed movie info with comprehensive append_to_response
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        details_params = {
            'api_key': api_key,
            'append_to_response': 'credits,external_ids,videos,keywords,release_dates'
        }

        time.sleep(0.25)  # Rate limiting
        details_response = requests.get(details_url, params=details_params)

        if details_response.status_code != 200:
            print(f"❌ TMDB details API error {details_response.status_code} for {title} ({release_year})")
            return _get_default_movie_data()

        details_data = details_response.json()

        # Extract all fields from the comprehensive response
        movie_data = _extract_all_movie_fields(details_data)

        print(f"✅ Found comprehensive data for {title} ({release_year})")
        print(f"   Director: {movie_data.get('director', 'Unknown')}, Runtime: {movie_data.get('runtime', 'N/A')} min, Rating: {movie_data.get('vote_average', 'N/A')}")

        # Save to JSON cache for future use
        save_to_tmdb_cache(movie_key, movie_data)

        return movie_data

    except Exception as e:
        print(f"❌ Error fetching data for {title} ({release_year}): {str(e)}")
        return _get_default_movie_data()


def _get_default_movie_data():
    """Returns default movie data structure with 'Unknown' or 'Not found' values"""
    return {
        # Basic Info
        'tmdb_id': None,
        'imdb_id': 'Unknown',
        'title': 'Unknown',
        'original_title': 'Unknown',
        'tagline': 'None',
        'overview': 'No overview available',
        'release_date': 'Unknown',
        'runtime': None,
        'status': 'Unknown',
        'original_language': 'Unknown',
        'adult': False,
        'belongs_to_collection': None,

        # Ratings & Popularity
        'vote_average': None,
        'vote_count': None,
        'popularity': None,

        # Financial
        'budget': None,
        'revenue': None,

        # Images
        'poster_url': 'No poster found',
        'backdrop_path': 'No backdrop found',

        # Credits
        'director': 'Unknown',
        'cast': 'Unknown',
        'writer': 'Unknown',

        # Production
        'production_companies': 'Unknown',
        'production_countries': 'Unknown',

        # Classification
        'genres': 'Unknown',
        'certification_us': 'Unknown',
        'keywords': 'None',

        # Media
        'trailer_key': 'No trailer found',

        # External IDs
        'wikidata_id': 'Unknown',

        # Cache metadata
        'cached_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }


def _extract_all_movie_fields(details_data):
    """Extract all useful fields from TMDB details response"""

    # Basic info
    tmdb_id = details_data.get('id')
    imdb_id = details_data.get('imdb_id', 'Unknown')
    title = details_data.get('title', 'Unknown')
    original_title = details_data.get('original_title', 'Unknown')
    tagline = details_data.get('tagline', 'None')
    overview = details_data.get('overview', 'No overview available')
    release_date = details_data.get('release_date', 'Unknown')
    runtime = details_data.get('runtime')
    status = details_data.get('status', 'Unknown')
    original_language = details_data.get('original_language', 'Unknown')
    adult = details_data.get('adult', False)

    # Collection
    belongs_to_collection = None
    if details_data.get('belongs_to_collection'):
        belongs_to_collection = details_data['belongs_to_collection'].get('name')

    # Ratings & Popularity
    vote_average = details_data.get('vote_average')
    vote_count = details_data.get('vote_count')
    popularity = details_data.get('popularity')

    # Financial
    budget = details_data.get('budget')
    revenue = details_data.get('revenue')

    # Images
    poster_url = 'No poster found'
    if details_data.get('poster_path'):
        poster_url = f"https://image.tmdb.org/t/p/w500{details_data['poster_path']}"

    backdrop_path = 'No backdrop found'
    if details_data.get('backdrop_path'):
        backdrop_path = f"https://image.tmdb.org/t/p/w1280{details_data['backdrop_path']}"

    # Genres
    genres = 'Unknown'
    if details_data.get('genres'):
        genres_list = [genre['name'] for genre in details_data['genres']]
        genres = ', '.join(genres_list) if genres_list else 'Unknown'

    # Production
    production_companies = 'Unknown'
    if details_data.get('production_companies'):
        companies = [company['name'] for company in details_data['production_companies'][:3]]
        production_companies = ', '.join(companies) if companies else 'Unknown'

    production_countries = 'Unknown'
    if details_data.get('production_countries'):
        countries = [country['iso_3166_1'] for country in details_data['production_countries']]
        production_countries = ', '.join(countries) if countries else 'Unknown'

    # Credits (from append_to_response)
    credits = details_data.get('credits', {})
    director = extract_director(credits.get('crew', []))
    cast = extract_top_cast(credits.get('cast', []), limit=10)
    writer = extract_writers(credits.get('crew', []))

    # External IDs (from append_to_response)
    external_ids = details_data.get('external_ids', {})
    wikidata_id = external_ids.get('wikidata_id', 'Unknown')
    # Use external_ids imdb_id if base imdb_id is missing
    if imdb_id == 'Unknown' and external_ids.get('imdb_id'):
        imdb_id = external_ids.get('imdb_id')

    # Certification (from append_to_response)
    certification_us = extract_us_certification(details_data.get('release_dates'))

    # Trailer (from append_to_response)
    trailer_key = extract_trailer_key(details_data.get('videos'))

    # Keywords (from append_to_response)
    keywords = extract_keywords(details_data.get('keywords'))

    return {
        # Basic Info
        'tmdb_id': tmdb_id,
        'imdb_id': imdb_id,
        'title': title,
        'original_title': original_title,
        'tagline': tagline,
        'overview': overview,
        'release_date': release_date,
        'runtime': runtime,
        'status': status,
        'original_language': original_language,
        'adult': adult,
        'belongs_to_collection': belongs_to_collection,

        # Ratings & Popularity
        'vote_average': vote_average,
        'vote_count': vote_count,
        'popularity': popularity,

        # Financial
        'budget': budget,
        'revenue': revenue,

        # Images
        'poster_url': poster_url,
        'backdrop_path': backdrop_path,

        # Credits
        'director': director,
        'cast': cast,
        'writer': writer,

        # Production
        'production_companies': production_companies,
        'production_countries': production_countries,

        # Classification
        'genres': genres,
        'certification_us': certification_us,
        'keywords': keywords,

        # Media
        'trailer_key': trailer_key,

        # External IDs
        'wikidata_id': wikidata_id,

        # Cache metadata
        'cached_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
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

        # Add comprehensive movie data from TMDB
        print("🎬 Adding comprehensive movie information from TMDB...")
        print(f"Processing {len(df)} movies for TMDB data...")

        # Get unique movies to avoid duplicate API calls
        unique_movies = df[['Name', 'Year']].drop_duplicates()
        print(f"Found {len(unique_movies)} unique movies")

        # Dictionary to store all movie data
        movie_data_dict = {}

        for idx, row in unique_movies.iterrows():
            movie_name = row['Name']
            movie_year = row['Year']

            # Create a key for the dictionary
            movie_key = f"{movie_name}_{movie_year}"

            # Get comprehensive movie data from TMDB
            movie_info = get_tmdb_movie_info(movie_name, movie_year)
            movie_data_dict[movie_key] = movie_info

            # Print progress every 10 movies
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(unique_movies)} unique movies")

        # Map all data fields back to the main dataframe
        # Core fields for website visualization
        df['PosterURL'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('poster_url', 'No poster found'), axis=1)
        df['BackdropPath'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('backdrop_path', 'No backdrop found'), axis=1)
        df['Genre'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('genres', 'Unknown'), axis=1)
        df['Director'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('director', 'Unknown'), axis=1)
        df['Cast'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('cast', 'Unknown'), axis=1)
        df['Writer'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('writer', 'Unknown'), axis=1)
        df['Runtime'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('runtime'), axis=1)
        df['Certification'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('certification_us', 'Unknown'), axis=1)

        # Additional metadata fields
        df['ImdbId'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('imdb_id', 'Unknown'), axis=1)
        df['TmdbId'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('tmdb_id'), axis=1)
        df['OriginalTitle'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('original_title', 'Unknown'), axis=1)
        df['Tagline'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('tagline', 'None'), axis=1)
        df['Overview'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('overview', 'No overview available'), axis=1)
        df['VoteAverage'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('vote_average'), axis=1)
        df['VoteCount'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('vote_count'), axis=1)
        df['Popularity'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('popularity'), axis=1)
        df['Budget'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('budget'), axis=1)
        df['Revenue'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('revenue'), axis=1)
        df['ProductionCompanies'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('production_companies', 'Unknown'), axis=1)
        df['ProductionCountries'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('production_countries', 'Unknown'), axis=1)
        df['OriginalLanguage'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('original_language', 'Unknown'), axis=1)
        df['TrailerKey'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('trailer_key', 'No trailer found'), axis=1)
        df['Keywords'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('keywords', 'None'), axis=1)
        df['Collection'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('belongs_to_collection'), axis=1)
        df['Status'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('status', 'Unknown'), axis=1)
        df['WikidataId'] = df.apply(lambda x: movie_data_dict.get(f"{x['Name']}_{x['Year']}", {}).get('wikidata_id', 'Unknown'), axis=1)

        # Rename columns to snake_case for consistency
        df = df.rename(columns={
            'Name': 'name',
            'Year': 'year',
            'Rating': 'rating',
            'Date': 'date',
            'PosterURL': 'poster_url',
            'BackdropPath': 'backdrop_path',
            'Genre': 'genre',
            'Director': 'director',
            'Cast': 'cast',
            'Writer': 'writer',
            'Runtime': 'runtime',
            'Certification': 'certification',
            'ImdbId': 'imdb_id',
            'TmdbId': 'tmdb_id',
            'OriginalTitle': 'original_title',
            'Tagline': 'tagline',
            'Overview': 'overview',
            'VoteAverage': 'vote_average',
            'VoteCount': 'vote_count',
            'Popularity': 'popularity',
            'Budget': 'budget',
            'Revenue': 'revenue',
            'ProductionCompanies': 'production_companies',
            'ProductionCountries': 'production_countries',
            'OriginalLanguage': 'original_language',
            'TrailerKey': 'trailer_key',
            'Keywords': 'keywords',
            'Collection': 'collection',
            'Status': 'status',
            'WikidataId': 'wikidata_id'
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

        # Count successful data retrieval
        movies_with_posters = len([m for m in movie_data_dict.values() if m.get('poster_url') != 'No poster found'])
        movies_with_directors = len([m for m in movie_data_dict.values() if m.get('director') != 'Unknown'])
        movies_with_runtime = len([m for m in movie_data_dict.values() if m.get('runtime')])
        movies_with_trailer = len([m for m in movie_data_dict.values() if m.get('trailer_key') != 'No trailer found'])

        print(f"🎬 Found posters for {movies_with_posters}/{len(movie_data_dict)} unique movies")
        print(f"🎭 Found directors for {movies_with_directors}/{len(movie_data_dict)} unique movies")
        print(f"⏱️  Found runtime for {movies_with_runtime}/{len(movie_data_dict)} unique movies")
        print(f"🎥 Found trailers for {movies_with_trailer}/{len(movie_data_dict)} unique movies")

        # Print some sample data for verification
        movies_with_data = df[(df['poster_url'] != 'No poster found') & (df['director'] != 'Unknown')].head(3)
        if not movies_with_data.empty:
            print("\n🎯 Sample movies with comprehensive data:")
            for _, movie in movies_with_data.iterrows():
                print(f"  • {movie['name']} ({movie['year']})")
                print(f"    Director: {movie['director']}")
                print(f"    Runtime: {movie['runtime']} min" if movie['runtime'] else "    Runtime: N/A")
                print(f"    Cast: {movie['cast'][:80]}..." if len(str(movie['cast'])) > 80 else f"    Cast: {movie['cast']}")
                print(f"    Certification: {movie['certification']}")
                print(f"    IMDb ID: {movie['imdb_id']}")

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
