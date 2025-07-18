import os
import requests
import pandas as pd
import time
import re
from dotenv import load_dotenv
from src.utils.file_operations import find_unzip_folder, clean_rename_move_folder, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection

load_dotenv()


def get_genre(title, release_year):
    """Legacy function - now redirects to TMDB for consistency"""
    movie_info = get_tmdb_movie_info(title, release_year)
    return movie_info['genres']


def get_tmdb_movie_info(title, release_year):
    """Retrieves both poster URL and genres from TMDB API"""
    # Check if we already have this data cached in the processed file
    try:
        df_processed = pd.read_csv('files/processed_files/movies/letterboxd_processed.csv', sep='|')
        if 'PosterURL' in df_processed.columns and 'Genre' in df_processed.columns:
            df_processed["Key"] = df_processed["Name"].astype(str) + df_processed["Year"].astype(str)
            key_input = str(title) + str(release_year)
            if key_input in list(df_processed["Key"].unique()):
                existing_row = df_processed[df_processed["Key"] == key_input].iloc[0]
                existing_poster = existing_row['PosterURL']
                existing_genre = existing_row['Genre']
                if pd.notna(existing_poster) and existing_poster != 'No poster found' and pd.notna(existing_genre) and existing_genre != 'Unknown':
                    print(f"Using cached data for {title} ({release_year})")
                    return {
                        'poster_url': existing_poster,
                        'genres': existing_genre
                    }
    except (FileNotFoundError, KeyError, IndexError):
        # File doesn't exist yet or doesn't have the columns we need
        pass

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


def get_watched_rating(path_watched, path_ratings):
    """Merges the watched & ratings dfs"""
    df_watched = pd.read_csv(path_watched)
    df_ratings = pd.read_csv(path_ratings)
    return df_watched.merge(df_ratings[['Name', 'Year', 'Rating']], on=['Name', 'Year'], how='left')


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
    Main processing function that adds poster URLs to the letterboxd data.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Letterboxd data...")

    path_watched = "files/exports/letterboxd_exports/watched.csv"
    path_ratings = "files/exports/letterboxd_exports/ratings.csv"
    output_path = 'files/processed_files/movies/letterboxd_processed.csv'

    try:
        # Check if input files exist
        if not os.path.exists(path_watched):
            print(f"❌ Watched file not found: {path_watched}")
            return False

        if not os.path.exists(path_ratings):
            print(f"❌ Ratings file not found: {path_ratings}")
            return False

        # Merge watched and ratings data
        print("📖 Reading and merging watched and ratings data...")
        df = get_watched_rating(path_watched, path_ratings)

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

        # Convert Date to datetime for proper sorting
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by date (most recent first)
        df = df.sort_values('Date', ascending=False)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        print(f"💾 Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False)

        print(f"\n✅ Processing complete!")
        print(f"📊 Processed {len(df)} movie entries")
        print(f"🎬 Found posters for {len([url for url in poster_dict.values() if url != 'No poster found'])} movies")
        print(f"🎭 Found genres for {len([genre for genre in genre_dict.values() if genre != 'Unknown'])} movies")

        # Print some sample data for verification
        movies_with_data = df[(df['PosterURL'] != 'No poster found') & (df['Genre'] != 'Unknown')].head(5)
        if not movies_with_data.empty:
            print("\n🎯 Sample movies with complete data:")
            for _, movie in movies_with_data.iterrows():
                print(f"  • {movie['Name']} ({movie['Year']})")
                print(f"    Genres: {movie['Genre']}")
                print(f"    Poster: {movie['PosterURL'][:50]}...")

        return True

    except Exception as e:
        print(f"❌ Error processing Letterboxd data: {e}")
        return False


def upload_letterboxd_results():
    """
    Uploads the processed Letterboxd files to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading Letterboxd results to Google Drive...")

    files_to_upload = ['files/processed_files/movies/letterboxd_processed.csv']

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Letterboxd results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


def process_letterboxd_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_letterboxd_pipeline(auto_full=True)
    else:
        return create_letterboxd_file()


def manual_poster_update():
    """
    Interactive function to manually update poster URLs for specific movies
    """
    # Load the processed file
    try:
        df = pd.read_csv('files/processed_files/movies/letterboxd_processed.csv', sep='|')
    except FileNotFoundError:
        print("Error: letterboxd_processed.csv not found. Please run process_letterboxd_export() first.")
        return

    if 'PosterURL' not in df.columns:
        print("Error: PosterURL column not found. Please run process_letterboxd_export() first.")
        return

    print("\n" + "="*60)
    print("MANUAL POSTER URL UPDATE")
    print("="*60)
    print("This function allows you to manually update poster URLs for movies.")
    print("Type 'quit' at any time to exit.\n")

    while True:
        # Ask for movie search term
        search_term = input("Enter part of the movie title to search for (or 'quit' to exit): ").strip()

        if search_term.lower() == 'quit':
            print("Exiting manual poster update.")
            break

        if not search_term:
            print("Please enter a search term.")
            continue

        # Find matching movies (case-insensitive)
        matching_movies = df[df['Name'].str.contains(search_term, case=False, na=False)]

        if matching_movies.empty:
            print(f"No movies found containing '{search_term}'. Try a different search term.")
            continue

        # Get unique movies (same movie might appear multiple times if watched multiple times)
        unique_movies = matching_movies[['Name', 'Year', 'PosterURL']].drop_duplicates()
        unique_movies = unique_movies.sort_values(['Name', 'Year'])

        print(f"\nFound {len(unique_movies)} matching movie(s):")
        print("-" * 80)

        # Display options with current poster URLs
        for idx, (_, movie) in enumerate(unique_movies.iterrows(), 1):
            current_poster = movie['PosterURL']
            if current_poster == 'No poster found':
                poster_status = "❌ No poster"
            elif current_poster.startswith('http'):
                poster_status = "✅ Has poster"
            else:
                poster_status = "⚠️  Unknown status"

            print(f"{idx:2d}. {movie['Name']} ({movie['Year']}) - {poster_status}")
            if current_poster != 'No poster found' and current_poster.startswith('http'):
                print(f"     Current URL: {current_poster}")
            print()

        # Ask user to select a movie
        try:
            choice = input(f"Select a movie (1-{len(unique_movies)}) or 'back' to search again: ").strip()

            if choice.lower() == 'back':
                continue
            elif choice.lower() == 'quit':
                print("Exiting manual poster update.")
                break

            choice_idx = int(choice) - 1

            if choice_idx < 0 or choice_idx >= len(unique_movies):
                print("Invalid selection. Please try again.")
                continue

        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        # Get the selected movie
        selected_movie = unique_movies.iloc[choice_idx]
        movie_name = selected_movie['Name']
        movie_year = selected_movie['Year']
        current_url = selected_movie['PosterURL']

        print(f"\nSelected: {movie_name} ({movie_year})")
        if current_url != 'No poster found':
            print(f"Current poster URL: {current_url}")

        # Ask for new poster URL
        print("\nPlease provide the new poster URL.")
        print("Tips:")
        print("- You can get poster URLs from TMDB, IMDb, or other movie databases")
        print("- Make sure the URL points directly to an image file")
        print("- Recommended size: at least 500px width")
        print()

        new_url = input("Enter the new poster URL (or 'cancel' to go back): ").strip()

        if new_url.lower() == 'cancel':
            continue
        elif new_url.lower() == 'quit':
            print("Exiting manual poster update.")
            break

        if not new_url:
            print("No URL provided. Going back to movie selection.")
            continue

        # Validate URL format (basic check)
        if not new_url.startswith(('http://', 'https://')):
            print("Warning: URL should start with 'http://' or 'https://'")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue

        # Confirm the update
        print(f"\nConfirm update:")
        print(f"Movie: {movie_name} ({movie_year})")
        print(f"Old URL: {current_url}")
        print(f"New URL: {new_url}")

        confirm = input("\nProceed with update? (y/n): ").strip().lower()

        if confirm == 'y':
            # Update all rows for this movie (in case of multiple watches)
            mask = (df['Name'] == movie_name) & (df['Year'] == movie_year)
            rows_updated = mask.sum()

            df.loc[mask, 'PosterURL'] = new_url

            # Save the updated dataframe
            df.to_csv('files/processed_files/movies/letterboxd_processed.csv', sep='|', index=False)

            print(f"✅ Successfully updated poster URL for {movie_name} ({movie_year})")
            print(f"   Updated {rows_updated} row(s) in the dataset")
            print()

        else:
            print("Update cancelled.")

        # Ask if user wants to continue
        continue_choice = input("Update another movie poster? (y/n): ").strip().lower()
        if continue_choice != 'y':
            break

    print("\nManual poster update session complete!")
    return True

def full_letterboxd_pipeline(auto_full=False):
    """
    Complete Letterboxd pipeline with 4 options.

    Options:
    1. Full pipeline (download → move → process → upload)
    2. Download data only (open web page + move files)
    3. Process existing file only (just processing)
    4. Process existing file and upload (process → upload)

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🎬 LETTERBOXD DATA PIPELINE")
    print("="*60)

    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → move → process → upload)")
        print("2. Download data only (open web page + move files)")
        print("3. Process existing file only")
        print("4. Process existing file and upload to Drive")
        print("5. Manually update a poster then upload to Drive")

        choice = input("\nEnter your choice (1-5): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full Letterboxd pipeline...")

        # Step 1: Download
        download_success = download_letterboxd_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_letterboxd_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_letterboxd_files()

        # Step 3: Process (fallback to option 3 if no new files)
        if move_success:
            process_success = create_letterboxd_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_letterboxd_file()

        # Step 4: Upload
        if process_success:
            upload_success = upload_letterboxd_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n📥 Download Letterboxd data only...")
        download_success = download_letterboxd_data()
        if download_success:
            success = move_letterboxd_files()
        else:
            success = False

    elif choice == "3":
        print("\n⚙️  Processing existing Letterboxd file only...")
        success = create_letterboxd_file()

    elif choice == "4":
        print("\n⚙️  Processing existing file and uploading...")
        process_success = create_letterboxd_file()
        if process_success:
            success = upload_letterboxd_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False
    elif choice == "5":
        print("\nManually updating a poster...")
        update_success = manual_poster_update()
        upload_success = upload_letterboxd_results()
        if update_success & upload_success:
            success = True
        else:
            success = False

    else:
        print("❌ Invalid choice. Please select 1-4.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Letterboxd pipeline completed successfully!")
    else:
        print("❌ Letterboxd pipeline failed")
    print("="*60)

    return success





if __name__ == "__main__":
    # Allow running this file directly
    print("🎬 Letterboxd Processing Tool")
    print("This tool helps you download, process, and upload Letterboxd data.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_letterboxd_pipeline(auto_full=False)
