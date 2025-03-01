# src/book_covers.py
import requests
import json
import os
import time
import pandas as pd
import re
from pathlib import Path

def get_optimized_book_cover(title, author, max_results=10):
    """
    Retrieves the best book cover from Google Books API, prioritizing:
    1. English editions when available
    2. Popular/canonical editions based on metadata quality

    Args:
        title (str): The book title
        author (str): The book author
        max_results (int): Maximum number of results to consider

    Returns:
        dict: A dictionary with the cover URL and other information
    """
    cover_data = {}
    candidates = []

    try:
        # Clean and encode the query
        title_clean = re.sub(r'[^\w\s]', '', title) if title else ""
        author_clean = re.sub(r'[^\w\s]', '', author) if author else ""
        query = f"intitle:{title_clean} inauthor:{author_clean}"

        # Make the API request with multiple results
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            if 'items' not in data or len(data['items']) == 0:
                print(f"No results found for {title} by {author}")
                return cover_data

            # Prepare all candidates with scoring information
            for item in data['items']:
                if 'volumeInfo' not in item:
                    continue

                volume_info = item['volumeInfo']

                # Skip items without any image
                if 'imageLinks' not in volume_info:
                    continue

                candidate = {
                    'title': title,
                    'author': author,
                    'google_books_id': item['id'],
                    'score': 0  # Base score starts at 0
                }

                # Add all available metadata
                for field in ['description', 'publishedDate', 'pageCount', 'publisher',
                              'language', 'categories', 'averageRating', 'ratingsCount']:
                    if field in volume_info:
                        candidate[field] = volume_info[field]

                # Find best available image
                for size in ['extraLarge', 'large', 'medium', 'thumbnail', 'smallThumbnail']:
                    if size in volume_info['imageLinks']:
                        image_url = volume_info['imageLinks'][size]
                        # Convert HTTP to HTTPS for security
                        image_url = image_url.replace('http://', 'https://')
                        # Remove zoom parameters for best quality
                        image_url = image_url.split('&zoom=')[0]
                        candidate['cover_url'] = image_url
                        candidate['image_size'] = size
                        break

                # Score the candidate based on various factors

                # 1. Major boost for English editions
                if candidate.get('language') == 'en':
                    candidate['score'] += 50

                # 2. Boost for high ratings
                if 'averageRating' in candidate:
                    candidate['score'] += min(candidate['averageRating'] * 5, 20)

                # 3. Boost for number of ratings (popularity)
                if 'ratingsCount' in candidate:
                    # Log scale to prevent extremely popular books from dominating
                    import math
                    rating_boost = min(math.log(candidate['ratingsCount'] + 1) * 3, 20)
                    candidate['score'] += rating_boost

                # 4. Boost for complete metadata
                metadata_completeness = 0
                for field in ['description', 'publishedDate', 'pageCount', 'publisher', 'categories']:
                    if field in candidate:
                        metadata_completeness += 1
                candidate['score'] += metadata_completeness * 2

                # 5. Boost for better image quality
                image_quality = {'extraLarge': 10, 'large': 8, 'medium': 5,
                                'thumbnail': 3, 'smallThumbnail': 1}
                if 'image_size' in candidate:
                    candidate['score'] += image_quality.get(candidate['image_size'], 0)

                # 6. Boost for major publishers
                major_publishers = ['Penguin', 'Random House', 'HarperCollins', 'Simon & Schuster',
                                   'Hachette', 'Oxford', 'Cambridge', 'Scholastic', 'Houghton Mifflin']
                if 'publisher' in candidate:
                    for publisher in major_publishers:
                        if publisher.lower() in candidate['publisher'].lower():
                            candidate['score'] += 10
                            break

                # Add to candidates list
                candidates.append(candidate)

            # If we have candidates, select the best one
            if candidates:
                # First try: Look for English editions
                english_candidates = [c for c in candidates if c.get('language') == 'en']

                if english_candidates:
                    # Sort English editions by score
                    best_candidate = sorted(english_candidates, key=lambda x: x['score'], reverse=True)[0]
                else:
                    # If no English editions, use the highest scoring edition of any language
                    best_candidate = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]

                # Return the best candidate's data
                return best_candidate

    except Exception as e:
        print(f"Error processing {title} by {author}: {str(e)}")

    return cover_data

def get_open_library_cover(title, author):
    """Get book cover from Open Library as a fallback"""
    try:
        query = f"{title} {author}".replace(' ', '+')
        url = f"https://openlibrary.org/search.json?q={query}&limit=1"
        response = requests.get(url)
        data = response.json()

        if data.get('docs') and len(data['docs']) > 0 and 'cover_i' in data['docs'][0]:
            cover_id = data['docs'][0]['cover_i']
            return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    except Exception as e:
        print(f"Open Library error for {title}: {str(e)}")
    return None

def get_book_covers(df, title_col='Title', author_col='Author'):
    """
    Retrieves book covers for an entire DataFrame of books,
    with caching to avoid redundant API calls

    Args:
        df (DataFrame): DataFrame containing book information
        title_col (str): Column name for book titles
        author_col (str): Column name for book authors

    Returns:
        DataFrame: Original DataFrame with added cover_url column
    """
    # Create a copy of the input DataFrame
    result_df = df.copy()

    # Add empty cover_url column
    result_df['cover_url'] = None

    # Create directories if they don't exist
    covers_dir = Path('files/work_files/book_covers')
    covers_dir.mkdir(parents=True, exist_ok=True)

    # Load existing cover cache if available
    cover_dict_path = covers_dir / 'book_covers_cache.json'

    if cover_dict_path.exists():
        with open(cover_dict_path, 'r') as f:
            cover_dict = json.load(f)
    else:
        cover_dict = {}

    # Get list of books we need to process
    unique_books = df[[title_col, author_col]].drop_duplicates().dropna()
    total_books = len(unique_books)

    print(f"Processing {total_books} unique books for cover images...")
    new_covers_count = 0

    for i, (_, row) in enumerate(unique_books.iterrows()):
        title = str(row[title_col]).strip() if row[title_col] == row[title_col] else ""
        author = str(row[author_col]).strip() if row[author_col] == row[author_col] else ""

        if not title or title == "nan":
            continue

        # Create a key for dictionary lookup
        book_key = f"{title}||{author}"

        # Skip if we already have this book cached
        if book_key in cover_dict and 'cover_url' in cover_dict[book_key]:
            continue

        # Process in batches of 10 with status updates
        if (i+1) % 10 == 0:
            print(f"Processing book {i+1}/{total_books}...")

        # First try Google Books API
        cover_data = get_optimized_book_cover(title, author)

        # If no cover found or problematic, try Open Library as fallback
        if not cover_data or 'cover_url' not in cover_data:
            open_library_url = get_open_library_cover(title, author)
            if open_library_url:
                cover_data = {
                    'title': title,
                    'author': author,
                    'cover_url': open_library_url,
                    'source': 'Open Library'
                }

        # Store in cache if we found a cover
        if cover_data and 'cover_url' in cover_data:
            cover_dict[book_key] = {
                'cover_url': cover_data['cover_url'],
                'language': cover_data.get('language', 'Unknown'),
                'publisher': cover_data.get('publisher', 'Unknown'),
                'source': cover_data.get('source', 'Google Books')
            }
            new_covers_count += 1

        # Sleep to avoid hitting API rate limits
        time.sleep(1)

    # Save the updated cache
    if new_covers_count > 0:
        print(f"Added {new_covers_count} new book covers to cache")
        with open(cover_dict_path, 'w') as f:
            json.dump(cover_dict, f, indent=2)

    # Update the DataFrame with cover URLs
    def get_cover_url(row):
        if row[title_col] != row[title_col]:  # Check for NaN
            return None

        title = str(row[title_col]).strip()
        author = str(row[author_col]).strip() if row[author_col] == row[author_col] else ""
        book_key = f"{title}||{author}"

        if book_key in cover_dict and 'cover_url' in cover_dict[book_key]:
            return cover_dict[book_key]['cover_url']
        return None

    result_df['cover_url'] = result_df.apply(get_cover_url, axis=1)

    return result_df
