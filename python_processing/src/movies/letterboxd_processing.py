import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv
load_dotenv()

def get_genre(title, release_year):
    """Legacy function - now redirects to TMDB for consistency"""
    movie_info = get_tmdb_movie_info(title, release_year)
    return movie_info['genres']

def get_tmdb_movie_info(title, release_year):
    """Retrieves both poster URL and genres from TMDB API"""
    # Check if we already have this data cached in the processed file
    try:
        df_processed = pd.read_csv('files/processed_files/letterboxd_processed.csv', sep='|')
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
    api_key = os.environ.get('TMDB_API_KEY')
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
    return df_watched.merge(df_ratings[['Name', 'Year', 'Rating']], on = ['Name', 'Year'], how = 'left')

def manual_poster_update():
    """
    Interactive function to manually update poster URLs for specific movies
    """
    # Load the processed file
    try:
        df = pd.read_csv('files/processed_files/letterboxd_processed.csv', sep='|')
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
            df.to_csv('files/processed_files/letterboxd_processed.csv', sep='|', index=False)

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

def process_letterboxd_export():
    """Main processing function that adds poster URLs to the letterboxd data"""
    path_watched = "files/exports/letterboxd_exports/watched.csv"
    path_ratings = "files/exports/letterboxd_exports/ratings.csv"

    # Merge watched and ratings data
    df = get_watched_rating(path_watched, path_ratings)

    # Add Genre and Poster URLs from TMDB
    print("Adding genre and poster information from TMDB...")
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

    # Save to CSV
    output_path = 'files/processed_files/letterboxd_processed.csv'
    df.to_csv(output_path, sep='|', index=False)

    print(f"\nProcessing complete!")
    print(f"Processed {len(df)} movie entries")
    print(f"Found posters for {len([url for url in poster_dict.values() if url != 'No poster found'])} movies")
    print(f"Found genres for {len([genre for genre in genre_dict.values() if genre != 'Unknown'])} movies")
    print(f"Saved to: {output_path}")

    # Print some sample data for verification
    movies_with_data = df[(df['PosterURL'] != 'No poster found') & (df['Genre'] != 'Unknown')].head(5)
    if not movies_with_data.empty:
        print("\nSample movies with complete data:")
        for _, movie in movies_with_data.iterrows():
            print(f"  {movie['Name']} ({movie['Year']})")
            print(f"    Genres: {movie['Genre']}")
            print(f"    Poster: {movie['PosterURL'][:50]}...")
            print()

def manual_poster_update():
    """Function to manually update poster URLs for specific movies"""
    # Load the processed CSV
    try:
        df = pd.read_csv('files/processed_files/letterboxd_processed.csv', sep='|')
    except FileNotFoundError:
        print("Error: letterboxd_processed.csv not found. Please run process_letterboxd_export() first.")
        return

    print("\n" + "="*50)
    print("MANUAL POSTER UPDATE")
    print("="*50)

    # Ask user for movie title input
    search_title = input("Enter the movie title you want to update: ").strip()

    if not search_title:
        print("No title entered. Exiting.")
        return

    # Find movies that contain the search term (case-insensitive)
    matching_movies = df[df['Name'].str.contains(search_title, case=False, na=False)]

    if matching_movies.empty:
        print(f"No movies found containing '{search_title}'.")
        return

    # Get unique movies (remove duplicates for rewatches)
    unique_movies = matching_movies[['Name', 'Year', 'PosterURL']].drop_duplicates()
    unique_movies = unique_movies.sort_values(['Name', 'Year'])

    print(f"\nFound {len(unique_movies)} movie(s) containing '{search_title}':")
    print("-" * 60)

    # Display options
    for idx, (_, movie) in enumerate(unique_movies.iterrows(), 1):
        current_poster = movie['PosterURL']
        poster_status = "✅" if current_poster != 'No poster found' else "❌"
        print(f"Option {idx}: {movie['Name']} ({movie['Year']}) {poster_status}")
        if current_poster != 'No poster found':
            print(f"           Current URL: {current_poster}")

    # Ask user to select option
    try:
        choice = int(input(f"\nSelect option (1-{len(unique_movies)}): "))

        if choice < 1 or choice > len(unique_movies):
            print("Invalid selection.")
            return

    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    # Get selected movie
    selected_movie = unique_movies.iloc[choice - 1]
    movie_name = selected_movie['Name']
    movie_year = selected_movie['Year']
    current_url = selected_movie['PosterURL']

    print(f"\nSelected: {movie_name} ({movie_year})")
    print(f"Current poster URL: {current_url}")

    # Ask for new URL
    new_url = input("\nPaste the new poster URL: ").strip()

    if not new_url:
        print("No URL provided. Exiting.")
        return

    # Update all rows for this movie (handles rewatches)
    mask = (df['Name'] == movie_name) & (df['Year'] == movie_year)
    rows_updated = mask.sum()

    df.loc[mask, 'PosterURL'] = new_url

    # Save the updated dataframe
    df.to_csv('files/processed_files/letterboxd_processed.csv', sep='|', index=False)

    print(f"\n✅ Successfully updated poster URL for '{movie_name} ({movie_year})'")
    print(f"Updated {rows_updated} row(s) in the dataset")
    print(f"New URL: {new_url}")
