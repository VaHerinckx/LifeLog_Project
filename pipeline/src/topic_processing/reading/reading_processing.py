import pandas as pd
import numpy as np
import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# Import the pipeline functions from source processors
from src.sources_processing.goodreads.goodreads_processing import full_goodreads_pipeline, download_goodreads_data, move_goodreads_files, clean_isbn
from src.sources_processing.kindle.kindle_processing import full_kindle_pipeline, download_kindle_data, move_kindle_files
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.topic_processing.website_maintenance.website_maintenance_processing import full_website_maintenance_pipeline

# Book enrichment constants
ENRICHMENT_CACHE_PATH = 'files/work_files/book_enrichment_cache.json'
GOOGLE_BOOKS_API = 'https://www.googleapis.com/books/v1/volumes'
OPEN_LIBRARY_AUTHOR_API = 'https://openlibrary.org/search/authors.json'
OPEN_LIBRARY_AUTHOR_PHOTO_URL = 'https://covers.openlibrary.org/a/olid/{}-L.jpg'
API_DELAY_SECONDS = 1.0
from src.utils.logger import log


def load_enrichment_cache():
    """Load the book enrichment cache from disk."""
    if not os.path.exists(ENRICHMENT_CACHE_PATH):
        return {}
    try:
        with open(ENRICHMENT_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        log.warning(f"Warning: Could not load enrichment cache: {e}")
        return {}


def save_enrichment_cache(cache):
    """Save the book enrichment cache to disk."""
    os.makedirs(os.path.dirname(ENRICHMENT_CACHE_PATH), exist_ok=True)
    try:
        with open(ENRICHMENT_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        log.error(f"Error saving enrichment cache: {e}")


def fetch_google_books_synopsis(identifier, search_type='isbn'):
    """Fetch book synopsis from Google Books API."""
    try:
        if search_type in ('isbn', 'isbn10'):
            query = f'isbn:{identifier}'
        else:
            query = identifier

        params = {'q': query, 'maxResults': 1}
        response = requests.get(GOOGLE_BOOKS_API, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if data.get('totalItems', 0) == 0:
            return None

        items = data.get('items', [])
        if not items:
            return None

        return items[0].get('volumeInfo', {}).get('description')

    except requests.RequestException as e:
        log.warning(f"Google Books API error: {e}")
        return None
    except Exception as e:
        log.warning(f"Error fetching Google Books data: {e}")
        return None


def fetch_open_library_author_photo(author_name):
    """Fetch author photo URL from Open Library API."""
    if not author_name:
        return None

    try:
        # Step 1: Search for author
        params = {'q': author_name}
        response = requests.get(OPEN_LIBRARY_AUTHOR_API, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        docs = data.get('docs', [])
        if not docs:
            return None

        author_key = docs[0].get('key')
        if not author_key:
            return None

        olid = author_key.split('/')[-1] if '/' in author_key else author_key

        # Step 2: Fetch full author record to check if they have photos
        author_url = f'https://openlibrary.org/authors/{olid}.json'
        author_response = requests.get(author_url, timeout=10)
        author_response.raise_for_status()

        author_data = author_response.json()
        photos = author_data.get('photos', [])

        # Only return URL if author has actual photos listed
        if not photos:
            return None

        # Use the first photo ID to construct the URL (more reliable than OLID)
        photo_id = photos[0]
        if photo_id and photo_id > 0:  # Valid photo IDs are positive integers
            return f'https://covers.openlibrary.org/a/id/{photo_id}-L.jpg'

        return None

    except requests.RequestException as e:
        log.warning(f"Open Library API error for '{author_name}': {e}")
        return None
    except Exception as e:
        log.warning(f"Error fetching author photo: {e}")
        return None


def enrich_book(isbn13=None, isbn10=None, title=None, author=None, cache=None):
    """Enrich a book with synopsis and author photo."""
    cache_key = isbn13 or isbn10 or f"{title}|{author}"

    if cache is None:
        cache = load_enrichment_cache()

    if cache_key in cache:
        cached = cache[cache_key]
        return {
            'synopsis': cached.get('synopsis'),
            'author_photo_url': cached.get('author_photo_url'),
            'from_cache': True
        }

    synopsis = None
    source = None

    if isbn13:
        synopsis = fetch_google_books_synopsis(isbn13, search_type='isbn')
        if synopsis:
            source = 'isbn13'

    if not synopsis and isbn10:
        time.sleep(API_DELAY_SECONDS)
        synopsis = fetch_google_books_synopsis(isbn10, search_type='isbn10')
        if synopsis:
            source = 'isbn10'

    if not synopsis and title and author:
        time.sleep(API_DELAY_SECONDS)
        fuzzy_query = f'intitle:{title} inauthor:{author}'
        synopsis = fetch_google_books_synopsis(fuzzy_query, search_type='fuzzy')
        if synopsis:
            source = 'fuzzy'
            log.progress(f"Used fuzzy match for '{title}'by {author}")

    time.sleep(API_DELAY_SECONDS)
    author_photo_url = fetch_open_library_author_photo(author)

    result = {
        'synopsis': synopsis,
        'author_photo_url': author_photo_url,
        'enriched_at': datetime.now().isoformat(),
        'source': source
    }

    cache[cache_key] = result
    save_enrichment_cache(cache)

    return {
        'synopsis': synopsis,
        'author_photo_url': author_photo_url,
        'from_cache': False
    }


def enrich_books_dataframe(df, isbn13_col='isbn13', isbn10_col='isbn', title_col='title', author_col='author'):
    """Enrich a DataFrame of books with synopsis and author photos."""
    log.progress("Enriching books with API data...")

    cache = load_enrichment_cache()
    initial_cache_size = len(cache)

    if 'book_id' in df.columns:
        unique_books = df.drop_duplicates(subset=['book_id']).copy()
    elif 'Book Id' in df.columns:
        unique_books = df.drop_duplicates(subset=['Book Id']).copy()
    else:
        unique_books = df.drop_duplicates(subset=[title_col, author_col]).copy()

    log.progress(f"Processing {len(unique_books)} unique books...")

    enriched_count = 0
    cached_count = 0
    failed_count = 0

    synopsis_results = {}
    author_photo_results = {}

    for idx, row in unique_books.iterrows():
        isbn13 = clean_isbn(row.get(isbn13_col)) if isbn13_col in row.index else None
        isbn10 = clean_isbn(row.get(isbn10_col)) if isbn10_col in row.index else None
        title = row.get(title_col)
        author = row.get(author_col)

        if 'book_id' in row.index:
            book_key = row['book_id']
        elif 'Book Id' in row.index:
            book_key = row['Book Id']
        else:
            book_key = f"{title}|{author}"

        result = enrich_book(isbn13=isbn13, isbn10=isbn10, title=title, author=author, cache=cache)

        synopsis_results[book_key] = result['synopsis']
        author_photo_results[book_key] = result['author_photo_url']

        if result.get('from_cache'):
            cached_count += 1
        elif result['synopsis'] or result['author_photo_url']:
            enriched_count += 1
        else:
            failed_count += 1

        processed = cached_count + enriched_count + failed_count
        if processed % 10 == 0:
            log.progress(f"Processed {processed}/{len(unique_books)} books...")

    df = df.copy()

    def get_synopsis(row):
        if 'book_id' in row.index:
            return synopsis_results.get(row['book_id'])
        elif 'Book Id' in row.index:
            return synopsis_results.get(row['Book Id'])
        else:
            return synopsis_results.get(f"{row.get(title_col)}|{row.get(author_col)}")

    def get_author_photo(row):
        if 'book_id' in row.index:
            return author_photo_results.get(row['book_id'])
        elif 'Book Id' in row.index:
            return author_photo_results.get(row['Book Id'])
        else:
            return author_photo_results.get(f"{row.get(title_col)}|{row.get(author_col)}")

    df['synopsis'] = df.apply(get_synopsis, axis=1)
    df['author_photo_url'] = df.apply(get_author_photo, axis=1)

    new_cache_entries = len(cache) - initial_cache_size
    synopsis_found = df['synopsis'].notna().sum()
    photos_found = df['author_photo_url'].notna().sum()

    log.success(f"\n Book enrichment complete:")
    log.progress(f"Synopsis found: {synopsis_found}/{len(unique_books)} books")
    log.info(f" Author photos found: {photos_found}/{len(unique_books)} books")
    log.success(f"From cache: {cached_count} | New API calls: {enriched_count} | No data: {failed_count}")
    if new_cache_entries > 0:
        log.info(f"Added {new_cache_entries} new entries to cache")

    return df


def check_prerequisite_files():
    """
    Check if the required processed files from Goodreads and Kindle exist.

    Returns:
        dict: Status of each required file
    """
    required_files = {
        'goodreads': 'files/source_processed_files/goodreads/goodreads_processed.csv',
        'kindle': 'files/source_processed_files/kindle/kindle_processed.csv'
    }

    file_status = {}

    for source, file_path in required_files.items():
        exists = os.path.exists(file_path)
        file_status[source] = {
            'path': file_path,
            'exists': exists,
            'last_modified': None
        }

        if exists:
            try:
                file_status[source]['last_modified'] = datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).isoformat()
            except Exception as e:
                log.warning(f"Could not get modification time for {file_path}: {e}")

    return file_status


def ensure_prerequisite_files(file_status):
    """
    Ensure that both Goodreads and Kindle processed files exist.
    If not, run the respective pipelines to create them.

    Args:
        file_status (dict): Status of prerequisite files

    Returns:
        bool: True if all files are available, False otherwise
    """
    log.progress("Checking prerequisite files...")

    missing_files = [source for source, info in file_status.items() if not info['exists']]

    if not missing_files:
        log.success("All prerequisite files exist")
        for source, info in file_status.items():
            log.progress(f"{source.title()}: {info['last_modified']}")
        return True

    log.warning(f"Missing files for: {', '.join(missing_files)}")

    # Process missing files
    success = True

    if 'goodreads' in missing_files:
        log.progress("\n Running Goodreads pipeline to create missing file...")
        try:
            gr_success = full_goodreads_pipeline(auto_full=True)
            if not gr_success:
                log.error("Goodreads pipeline failed")
                success = False
        except Exception as e:
            log.error(f"Error running Goodreads pipeline: {e}")
            success = False

    if 'kindle' in missing_files and success:
        log.progress("\n Running Kindle pipeline to create missing file...")
        try:
            kindle_success = full_kindle_pipeline(auto_full=True)
            if not kindle_success:
                log.error("Kindle pipeline failed")
                success = False
        except Exception as e:
            log.error(f"Error running Kindle pipeline: {e}")
            success = False

    return success


def load_goodreads_reading_dates():
    """
    Load the Goodreads reading dates JSON to get official reading completion dates.

    Returns:
        pd.DataFrame: DataFrame with Book ID, Title, and Date ended columns
    """
    json_path = 'files/work_files/gr_work_files/reading_dates.json'

    if not os.path.exists(json_path):
        log.warning("No Goodreads reading dates JSON found - skipping accidental clicks removal")
        return pd.DataFrame()

    try:
        with open(json_path, 'r') as f:
            dates_data = json.load(f)

        # Convert JSON data to DataFrame format similar to original Excel approach
        date_records = []
        for book_id, book_info in dates_data.items():
            if book_info.get('date_ended'):
                try:
                    date_ended = pd.to_datetime(book_info['date_ended'], errors='coerce')
                    if pd.notna(date_ended):
                        date_records.append({
                            'Book Id': book_id,
                            'Title': book_info.get('title', ''),
                            'Date ended': date_ended
                        })
                except Exception as e:
                    log.warning(f"Error parsing date for book {book_id}: {e}")
                    continue

        gr_date_df = pd.DataFrame(date_records)
        log.info(f"📅 Loaded official reading end dates for {len(gr_date_df)} books")
        return gr_date_df

    except Exception as e:
        log.error(f"Error loading Goodreads reading dates: {e}")
        return pd.DataFrame()


def flag_clicks(row, gr_date_df):
    """
    Add a flag to the rows where the date a book has been read is higher than
    what was officially recorded in Goodreads (indicating accidental clicks).
    Uses Book ID for precise matching to avoid title collisions.

    Args:
        row: DataFrame row from combined dataset
        gr_date_df: DataFrame with official Goodreads reading end dates

    Returns:
        int: 1 if this is an accidental click, 0 if legitimate reading
    """
    if gr_date_df.empty:
        return 0

    # Only check records that have valid timestamps
    if pd.isna(row.get('Timestamp')):
        return 0

    # Filter to records that have Date ended information
    gr_date_df_valid = gr_date_df[gr_date_df["Date ended"].notna()]

    if gr_date_df_valid.empty:
        return 0

    # Look for this book's official end date by Book ID
    book_id = row.get("Book Id", "")
    if not book_id:
        return 0

    # Find matching Book ID in Goodreads data
    matching_books = gr_date_df_valid[gr_date_df_valid['Book Id'] == book_id]

    if matching_books.empty:
        return 0

    # Get the official reading end date
    official_end_date = matching_books['Date ended'].iloc[0]

    try:
        # Convert row timestamp to date for comparison
        row_timestamp = pd.to_datetime(row['Timestamp'])
        if pd.isna(row_timestamp):
            return 0

        row_date = row_timestamp.date()
        official_end_date_only = official_end_date.date()

        # Flag as accidental click if reading session is after official completion
        if row_date > official_end_date_only:
            return 1
        else:
            return 0

    except Exception as e:
        log.warning(f"Error comparing dates for Book ID '{book_id}': {e}")
        return 0


def remove_accidental_clicks(df):
    """
    Remove data points where a book was accidentally clicked in the Kindle after
    official completion, to maintain coherent reading dates with Goodreads data.

    Args:
        df (pd.DataFrame): Combined dataset with Kindle and Goodreads data

    Returns:
        pd.DataFrame: Dataset with accidental clicks removed
    """
    log.progress("Removing accidental clicks...")

    # Load official Goodreads reading dates
    gr_date_df = load_goodreads_reading_dates()

    if gr_date_df.empty:
        log.warning("No Goodreads reading dates available, skipping accidental clicks removal")
        return df

    # Apply the flagging function to identify accidental clicks
    log.progress("Identifying accidental clicks...")
    df = df.copy()
    df['Flag_remove'] = df.apply(lambda x: flag_clicks(x, gr_date_df), axis=1)

    # Count flagged records
    flagged_count = df['Flag_remove'].sum()
    total_count = len(df)

    log.info(f"Found {flagged_count} accidental clicks out of {total_count} total records")

    if flagged_count > 0:
        # Show some examples of what's being removed
        flagged_examples = df[df['Flag_remove'] == 1][['Title', 'Timestamp', 'Source']].head(5)
        if not flagged_examples.empty:
            log.info("Examples of accidental clicks being removed:")
            for _, example in flagged_examples.iterrows():
                log.info(f"• '{example['Title']}' on {example['Timestamp']} ({example['Source']})")

    # Remove flagged records
    cleaned_df = df[df['Flag_remove'] == 0].drop('Flag_remove', axis=1)

    removed_count = total_count - len(cleaned_df)
    log.success(f"Removed {removed_count} accidental click records")

    return cleaned_df


def identify_goodreads_only_books(df):
    """
    Identifies books that exist only in Goodreads data (not in Kindle).

    Args:
        df (pd.DataFrame): Combined dataset

    Returns:
        list: List of Book IDs that are Goodreads-only
    """
    log.progress("Identifying Goodreads-only books...")

    # Group by Book ID and check sources
    gr_only_books = []

    for book_id, group in df.groupby('Book Id'):
        sources = set(group['Source'].dropna())
        if sources == {'GoodReads'}:  # Only Goodreads, no Kindle data
            gr_only_books.append(book_id)

    log.progress(f"Found {len(gr_only_books)} books that are Goodreads-only")
    return gr_only_books


def calculate_reading_duration_per_book(df):
    """
    Calculate reading duration once per book title for easier analysis.
    This replicates the logic from the original duration() function.

    Args:
        df (pd.DataFrame): Dataset with reading records

    Returns:
        pd.DataFrame: Dataset with reading_duration_final column
    """
    log.info("⏱️  Calculating reading durations per book...")

    df = df.copy()
    df['reading_duration_final'] = np.nan

    # For each title, find the latest timestamp and assign duration only to that record
    for title in df['Title'].unique():
        if pd.isna(title):
            continue

        title_data = df[df['Title'] == title]

        if len(title_data) == 0:
            continue

        # Find the maximum timestamp for this title
        max_timestamp = title_data['Timestamp'].max()

        if pd.isna(max_timestamp):
            continue

        # Find the record(s) with max timestamp and minimum row number (in case of ties)
        max_timestamp_records = title_data[title_data['Timestamp'] == max_timestamp]

        # Get the reading duration from any record of this title (should be the same)
        reading_duration = title_data['reading_duration'].dropna()

        if len(reading_duration) > 0:
            duration_value = reading_duration.iloc[0]

            # Assign duration only to the latest record
            latest_record_index = max_timestamp_records.index[0]
            df.loc[latest_record_index, 'reading_duration_final'] = duration_value

    books_with_duration = df['reading_duration_final'].notna().sum()
    log.progress(f"Assigned reading duration to {books_with_duration} book records")

    return df


def remove_duplicate_records(df):
    """
    Remove duplicate records from the combined dataset.

    Args:
        df (pd.DataFrame): Combined dataset

    Returns:
        pd.DataFrame: Deduplicated dataset
    """
    log.progress("Removing duplicate records...")

    original_count = len(df)

    # Define columns to consider for duplicate detection
    # We exclude 'reading_duration_final' as it's intentionally sparse
    duplicate_cols = ['Book Id', 'Title', 'Author', 'Timestamp', 'Source', 'page_split']

    # Keep only columns that actually exist in the dataframe
    existing_duplicate_cols = [col for col in duplicate_cols if col in df.columns]

    if existing_duplicate_cols:
        df_deduplicated = df.drop_duplicates(subset=existing_duplicate_cols, keep='first')
    else:
        log.warning("No suitable columns for duplicate detection, keeping all records")
        df_deduplicated = df

    removed_count = original_count - len(df_deduplicated)

    if removed_count > 0:
        log.info(f"🗑️  Removed {removed_count} duplicate records")
    else:
        log.success("No duplicates found")

    return df_deduplicated


def create_books_file():
    """
    Main function to merge Goodreads and Kindle processed files into a unified books dataset.

    Returns:
        bool: True if successful, False otherwise
    """
    log.progress("Creating unified books file...")

    try:
        # Check prerequisite files
        file_status = check_prerequisite_files()

        if not ensure_prerequisite_files(file_status):
            log.error("Could not ensure all prerequisite files exist")
            return False

        # Load processed files from NEW locations
        log.progress("\n Loading processed data files...")

        # Load Goodreads data from NEW location
        gr_path = 'files/source_processed_files/goodreads/goodreads_processed.csv'
        if not os.path.exists(gr_path):
            log.error(f"Goodreads file not found: {gr_path}")
            return False

        df_gr = pd.read_csv(gr_path, sep='|', encoding='utf-8')
        log.success(f"Loaded {len(df_gr)} Goodreads records")

        # Load Kindle data from NEW location
        kindle_path = 'files/source_processed_files/kindle/kindle_processed.csv'
        if not os.path.exists(kindle_path):
            log.error(f"Kindle file not found: {kindle_path}")
            return False

        df_kindle = pd.read_csv(kindle_path, sep='|', encoding='utf-8')
        log.success(f"Loaded {len(df_kindle)} Kindle records")

        # Enhance Kindle data with Goodreads metadata
        log.info("\n🔗 Enhancing Kindle data with Goodreads metadata...")

        # Prepare Goodreads metadata for merging (including cover_url and ISBN)
        gr_metadata_columns = ['Book Id', 'Title', 'Author', 'Original Publication Year',
                              'My Rating', 'Average Rating', 'Genre', 'Fiction_yn',
                              'Number of Pages', 'reading_duration', 'cover_url', 'isbn', 'isbn13']

        # Keep only existing columns
        existing_gr_columns = [col for col in gr_metadata_columns if col in df_gr.columns]

        gr_metadata = df_gr[existing_gr_columns].drop_duplicates(subset=['Book Id'])

        # Merge Kindle data with Goodreads metadata
        df_kindle_enhanced = pd.merge(
            df_kindle,
            gr_metadata,
            on='Book Id',
            how='left',
            suffixes=('', '_gr')
        )

        # Handle column conflicts (prefer Goodreads data for metadata)
        for col in ['Title', 'Author', 'My Rating', 'Average Rating', 'Genre', 'Fiction_yn', 'Number of Pages', 'reading_duration', 'cover_url']:
            if f'{col}_gr' in df_kindle_enhanced.columns:
                df_kindle_enhanced[col] = df_kindle_enhanced[f'{col}_gr'].fillna(df_kindle_enhanced.get(col, ''))
                df_kindle_enhanced.drop(columns=[f'{col}_gr'], inplace=True)

        log.progress(f"Enhanced {len(df_kindle_enhanced)} Kindle records with Goodreads metadata")

        # Combine datasets
        log.progress("\n Combining Goodreads and Kindle datasets...")

        # Ensure both dataframes have the same columns
        all_columns = set(df_gr.columns) | set(df_kindle_enhanced.columns)

        for col in all_columns:
            if col not in df_gr.columns:
                df_gr[col] = np.nan
            if col not in df_kindle_enhanced.columns:
                df_kindle_enhanced[col] = np.nan

        # Reorder columns to match
        df_gr = df_gr[sorted(all_columns)]
        df_kindle_enhanced = df_kindle_enhanced[sorted(all_columns)]

        # Combine the datasets
        combined_df = pd.concat([df_gr, df_kindle_enhanced], ignore_index=True)
        log.progress(f"Combined dataset has {len(combined_df)} total records")

        # Remove accidental clicks BEFORE further processing
        combined_df = remove_accidental_clicks(combined_df)

        # Identify Goodreads-only books
        gr_only_books = identify_goodreads_only_books(combined_df)

        # Filter to keep only Kindle data + Goodreads-only books
        log.success("\n Filtering to final dataset...")

        kindle_data = combined_df[combined_df['Source'] == 'Kindle']
        gr_only_data = combined_df[
            (combined_df['Source'] == 'GoodReads') &
            (combined_df['Book Id'].isin(gr_only_books))
        ]

        final_df = pd.concat([kindle_data, gr_only_data], ignore_index=True)
        log.progress(f"Final dataset: {len(kindle_data)} Kindle + {len(gr_only_data)} Goodreads-only = {len(final_df)} records")

        # Process timestamps
        log.info("\n⏰ Processing timestamps...")

        # First convert to datetime
        final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], errors='coerce')

        # Force conversion to timezone-naive for all timestamps
        log.progress("Converting all timestamps to timezone-naive...")

        def force_timezone_naive(timestamp):
            """Force any timestamp to be timezone-naive"""
            if pd.isna(timestamp):
                return timestamp

            # Convert pandas Timestamp to python datetime if needed
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()

            # If it has timezone info, remove it
            if hasattr(timestamp, 'replace') and hasattr(timestamp, 'tzinfo'):
                if timestamp.tzinfo is not None:
                    return timestamp.replace(tzinfo=None)

            return timestamp

        # Apply the conversion to all timestamps
        final_df['Timestamp'] = final_df['Timestamp'].apply(force_timezone_naive)

        # Convert back to pandas datetime (timezone-naive)
        final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], errors='coerce')

        # Remove duplicate records
        final_df = remove_duplicate_records(final_df)

        # Calculate reading duration per book
        final_df = calculate_reading_duration_per_book(final_df)

        # Enrich books with API data (synopsis and author photos)
        log.progress("\n Enriching books with external API data...")
        try:
            final_df = enrich_books_dataframe(
                final_df,
                isbn13_col='isbn13',
                isbn10_col='isbn',
                title_col='Title',
                author_col='Author'
            )
        except Exception as e:
            log.warning(f"Book enrichment failed (continuing without enrichment): {e}")

        # Sort by timestamp (most recent first)
        final_df = final_df.sort_values('Timestamp', ascending=False)

        # Clean up and finalize columns
        log.progress("\n Finalizing dataset...")

        # Ensure required columns exist
        required_columns = ['Book Id', 'Title', 'Author', 'Timestamp', 'Source', 'page_split', 'cover_url']
        for col in required_columns:
            if col not in final_df.columns:
                final_df[col] = np.nan if col != 'cover_url' else ''

        # Reset index
        final_df = final_df.reset_index(drop=True)

        # Enforce snake_case before saving
        final_df = enforce_snake_case(final_df, "processed file")

        # Save the final processed file to NEW location
        output_path = 'files/topic_processed_files/reading/reading_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        final_df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        log.success(f"\n Successfully created unified books file!")
        log.info(f"📁 Saved to: {output_path}")
        log.progress(f"Final statistics:")
        log.progress(f"Total records: {len(final_df):,}")
        log.progress(f"Unique books: {final_df['book_id'].nunique():,}")
        log.progress(f"Kindle records: {len(final_df[final_df['source'] == 'Kindle']):,}")
        log.progress(f"Goodreads-only records: {len(final_df[final_df['source'] == 'GoodReads']):,}")

        # Check cover URL availability
        if 'cover_url' in final_df.columns:
            covers_count = (final_df['cover_url'].notna() & (final_df['cover_url'] != '')).sum()
            log.info(f"🖼️  Records with covers: {covers_count:,}")

        if len(final_df) > 0:
            log.info(f"📅 Date range: {final_df['timestamp'].min().date()} to {final_df['timestamp'].max().date()}")

        # Generate website files
        log.progress("\n Generating website-optimized files...")
        website_success = generate_reading_website_page_files(final_df)

        if not website_success:
            log.warning("Warning: Website files generation failed, but processed file was saved")

        return True

    except Exception as e:
        log.error(f"Error creating books file: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_reading_website_page_files(df):
    """
    Generate website-optimized files for the Reading page.
    Creates dual outputs: sessions (all records) + aggregated books (unique books).

    Args:
        df: Processed dataframe (already in snake_case)

    Returns:
        bool: True if successful, False otherwise
    """
    log.progress("\n Generating website files for Reading page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/reading'
        os.makedirs(website_dir, exist_ok=True)

        # Work with copy to avoid modifying original
        df_web = df[df["title"] != "Unknown Title"].copy()

        # Ensure timestamp is datetime
        df_web['timestamp'] = pd.to_datetime(df_web['timestamp'], errors='coerce')

        # Add derived date columns
        log.info("📅 Adding derived date columns...")
        df_web['reading_year'] = df_web['timestamp'].dt.year.astype('Int64').astype(str)
        df_web['reading_month'] = df_web['timestamp'].dt.month.astype('Int64').astype(str)
        df_web['reading_quarter'] = df_web['timestamp'].dt.quarter.astype('Int64').astype(str)

        # Standardize fiction_yn field to consistent casing
        log.progress("Standardizing fiction_yn field...")
        if 'fiction_yn' in df_web.columns:
            df_web['fiction_yn'] = df_web['fiction_yn'].apply(
                lambda x: 'Fiction' if str(x).lower() == 'fiction' else 'Non-Fiction'
            )

        # Sort by timestamp descending (most recent first)
        df_web = df_web.sort_values('timestamp', ascending=False)

        # Add reading_format column based on source
        log.progress("Adding reading format column...")
        df_web['reading_format'] = df_web['source'].apply(
            lambda x: 'Kindle' if x == 'Kindle' else 'Physical'
        )

        # FILE 1: Sessions (all reading sessions)
        sessions_columns = [
            'book_id', 'title', 'author', 'timestamp', 'source', 'genre',
            'page_split', 'my_rating', 'reading_format', 'reading_year', 'reading_month', 'reading_quarter'
        ]

        sessions_df = df_web[sessions_columns].copy()

        # Enforce snake_case before saving
        sessions_df = enforce_snake_case(sessions_df, "reading_page_sessions")

        sessions_path = f'{website_dir}/reading_page_sessions.csv'
        sessions_df.sort_values('timestamp', ascending=False).to_csv(sessions_path, sep='|', index=False, encoding='utf-8')
        log.success(f"Sessions file: {len(sessions_df):,} records  {sessions_path}")

        # FILE 2: Books (aggregated unique books)
        log.progress("Aggregating unique books...")

        # Add computed column
        df_web["pages_per_day"] = df_web["number_of_pages"] / df_web["reading_duration_final"]

        # Determine reading format per book (Kindle if any session was Kindle)
        book_format = df_web.groupby(['book_id', 'title', 'author'])['source'].apply(
            lambda sources: 'Kindle' if 'Kindle' in sources.values else 'Physical'
        ).reset_index(name='reading_format_agg')

        # Group by book_id + title + author, take most recent record (already sorted above)
        books_df = df_web.groupby(['book_id', 'title', 'author'], as_index=False).first()

        # Merge the aggregated reading format
        books_df = books_df.merge(book_format[['book_id', 'title', 'author', 'reading_format_agg']],
                                   on=['book_id', 'title', 'author'], how='left')
        books_df['reading_format'] = books_df['reading_format_agg']
        books_df = books_df.drop(columns=['reading_format_agg'])

        books_columns = [
            'book_id', 'title', 'author', 'original_publication_year',
            'my_rating', 'average_rating', 'genre', 'fiction_yn',
            'number_of_pages', 'pages_per_day', 'reading_duration_final', 'cover_url',
            'synopsis', 'author_photo_url',
            'reading_format', 'timestamp', 'reading_year', 'reading_month', 'reading_quarter'
        ]

        # Only include columns that exist (synopsis/author_photo_url may not exist if enrichment failed)
        existing_books_columns = [col for col in books_columns if col in books_df.columns]
        books_df = books_df[existing_books_columns].copy()

        # Enforce snake_case before saving
        books_df = enforce_snake_case(books_df, "reading_page_books")

        books_path = f'{website_dir}/reading_page_books.csv'
        books_df.sort_values('timestamp', ascending=False).to_csv(books_path, sep='|', index=False, encoding='utf-8')
        log.success(f"Books file: {len(books_df):,} unique books  {books_path}")

        return True

    except Exception as e:
        log.error(f"Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_books_results():
    """
    Upload the website-ready books files to Google Drive.
    Only uploads website files (not processed files).

    Returns:
        bool: True if successful, False otherwise
    """
    log.progress("Uploading Reading page website files to Google Drive...")

    # Only upload website files
    files_to_upload = [
        'files/website_files/reading/reading_page_sessions.csv',
        'files/website_files/reading/reading_page_books.csv'
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        log.error("No website files found to upload")
        log.info("Make sure generate_reading_website_page_files() ran successfully")
        return False

    log.info(f"📤 Uploading {len(existing_files)} website files...")
    success = upload_multiple_files(existing_files)

    if success:
        log.success("Website files uploaded successfully!")
    else:
        log.error("Some website files failed to upload")

    return success


def update_cover_url():
    """
    Update book cover URLs in the reading_dates.json source file.
    This ensures manual updates persist through all future pipeline runs.
    """
    log.info("\n" + "="*50)
    log.progress("UPDATE BOOK COVER URLs")
    log.info("="*50)

    # Load the reading_dates.json file (source of truth)
    json_path = 'files/work_files/gr_work_files/reading_dates.json'
    try:
        with open(json_path, 'r') as f:
            dates_data = json.load(f)
        log.success(f"Loaded {len(dates_data)} books from reading_dates.json")
    except FileNotFoundError:
        log.error(f"File not found: {json_path}")
        log.info("Run the Goodreads pipeline first to create the reading dates file.")
        return False
    except Exception as e:
        log.error(f"Error loading JSON file: {e}")
        return False

    # Convert JSON to list for searching
    books_list = []
    for book_id, book_info in dates_data.items():
        title = book_info.get('title', '')
        # Ensure title is a string (handle cases where it might be int or other types)
        title = str(title) if title else ''
        books_list.append({
            'book_id': book_id,
            'title': title,
            'cover_url': book_info.get('cover_url', '')
        })

    updated_books = []

    # Loop to allow multiple cover updates
    while True:
        # Get search term from user
        search_term = input("\nEnter part of book title to search (or 'quit' to finish): ").strip()
        if not search_term or search_term.lower() == 'quit':
            break

        # Find matching books (case-insensitive)
        matches = [b for b in books_list if b['title'] and search_term.lower() in b['title'].lower()]

        if not matches:
            log.error(f"No books found containing '{search_term}'")
            continue

        log.progress(f"\n Found {len(matches)} books:")

        # Display matches with numbers
        for i, book in enumerate(matches, 1):
            current_url = book['cover_url'] if book['cover_url'] else 'No cover URL'
            if len(str(current_url)) > 60:
                current_url = str(current_url)[:60] + "..."
            log.info(f"{i}. {book['title']}")
            log.info(f"Current cover: {current_url}")
            log.blank()

        # Get user selection
        try:
            choice = int(input(f"Select book (1-{len(matches)}): "))
            if choice < 1 or choice > len(matches):
                log.error(f"Invalid choice. Please select 1-{len(matches)}")
                continue
        except ValueError:
            log.error("Invalid input. Please enter a number.")
            continue

        # Get the selected book
        selected_book = matches[choice - 1]
        selected_book_id = selected_book['book_id']
        selected_title = selected_book['title']

        log.success(f"\n Selected: {selected_title}")
        log.info(f"Current cover URL: {selected_book['cover_url'] or 'None'}")

        # Get new URL
        new_url = input("\nEnter new cover URL: ").strip()
        if not new_url:
            log.error("No URL provided, skipping this book")
            continue

        # Update the cover URL in the JSON data
        dates_data[selected_book_id]['cover_url'] = new_url
        log.success(f"Updated cover URL for '{selected_title}'")
        updated_books.append(selected_title)

        # Ask if user wants to continue
        continue_choice = input("\nUpdate another book cover? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break

    # If no books were updated, exit
    if not updated_books:
        log.error("No books were updated")
        return False

    # Step 1: Save the updated JSON file
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(dates_data, f, indent=2, default=str)
        log.success(f"\n Saved {len(updated_books)} cover URL updates to reading_dates.json")
    except Exception as e:
        log.error(f"Error saving JSON file: {e}")
        return False

    # Step 2: Regenerate Goodreads processed file (loads updated JSON with new cover URLs)
    try:
        log.progress("\n Regenerating Goodreads processed file with updated cover URLs...")
        from src.sources_processing.goodreads.goodreads_processing import create_goodreads_file
        gr_success = create_goodreads_file()
        if not gr_success:
            log.error("Failed to regenerate Goodreads processed file")
            return False
    except Exception as e:
        log.error(f"Error regenerating Goodreads file: {e}")
        return False

    # Step 3: Regenerate the unified books file (merges updated Goodreads data with Kindle)
    try:
        log.progress("\n Regenerating unified books file...")
        create_success = create_books_file()
        if not create_success:
            log.error("Failed to regenerate books file")
            return False
    except Exception as e:
        log.error(f"Error regenerating books file: {e}")
        return False

    # Step 4: Upload website files to Google Drive
    try:
        log.info("\n📤 Uploading website files to Google Drive...")
        upload_success = upload_books_results()
        if upload_success:
            log.success("Successfully uploaded website files to Google Drive!")
            log.success(f"Cover URLs updated for: {', '.join(updated_books)}")
            log.info(f"Updates are now permanent - they will persist through future pipeline runs")
            return True
        else:
            log.warning("Website files updated locally but failed to upload to Google Drive")
            return False
    except Exception as e:
        log.error(f"Error uploading to Google Drive: {e}")
        return False


def full_books_pipeline(auto_full=False, auto_process_only=False, merge_only=False):
    """
    Complete books TOPIC COORDINATOR pipeline that orchestrates Goodreads and Kindle processing.

    Options:
    1. Full pipeline (download both → process both → merge → upload)
    2. Process existing files (process exports → merge → upload)
    3. Upload only (upload existing processed/website files to Drive)
    4. Update book cover URL (interactive cover URL updates)

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    log.normal("Processing Reading...")
    log.info("\n" + "="*60)
    log.progress("READING TOPIC COORDINATOR")
    log.info("="*60)
    log.info("This pipeline combines Goodreads and Kindle data into a unified books dataset")

    if merge_only:
        # Called from main orchestrator — sources already processed, just merge + upload
        log.info("🔗 Merge mode: Merging existing source data and uploading...")
        merge_success = create_books_file()
        if not merge_success:
            log.error("Books merge failed")
            return False
        success = upload_books_results()
    elif auto_process_only:
        log.info("🤖 Auto process mode: Processing existing data and uploading...")
        choice = "2"
    elif auto_full:
        log.info("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        log.prompt("\nSelect an option:")
        log.prompt("1. Full pipeline (download both  process both  merge  upload)")
        log.prompt("2. Process existing files (process exports  merge  upload)")
        log.prompt("3. Upload only (upload existing processed/website files to Drive)")
        log.prompt("4. Update book cover URL (interactive cover URL updates)")

        choice = input("\nEnter your choice (1-4): ").strip()

    success = False

    if choice == "1":
        log.milestone("\n Starting full books pipeline...")

        # ============================================================
        # PHASE 1: DOWNLOAD - All user interaction upfront
        # ============================================================
        log.info("\n" + "="*60)
        log.info("📥 DOWNLOAD PHASE - User interaction required")
        log.info("="*60)
        log.info("\nYou will be prompted to download each data source.")
        log.info("After all downloads are complete, processing will run automatically.\n")

        download_status = {}

        # Download 1: Goodreads
        log.progress("\n Download 1/2: Goodreads...")
        download_confirmed = download_goodreads_data()
        if download_confirmed:
            move_success = move_goodreads_files()
            download_status['goodreads'] = move_success
            if move_success:
                log.success("Goodreads files ready")
            else:
                log.warning("Goodreads file move failed")
        else:
            download_status['goodreads'] = False
            log.info("⏭️  Skipping Goodreads download")

        # Download 2: Kindle
        log.progress("\n Download 2/2: Kindle...")
        download_confirmed = download_kindle_data()
        if download_confirmed:
            move_success = move_kindle_files()
            download_status['kindle'] = move_success
            if move_success:
                log.success("Kindle files ready")
            else:
                log.warning("Kindle file move failed")
        else:
            download_status['kindle'] = False
            log.info("⏭️  Skipping Kindle download")

        # Summary of downloads
        log.info("\n" + "="*60)
        log.progress("DOWNLOAD SUMMARY")
        log.info("="*60)
        for source, status in download_status.items():
            status_icon = "✅" if status else "⏭️"
            log.info(f"{status_icon} {source.title()}: {'Ready' if status else 'Skipped'}")

        # ============================================================
        # PHASE 2: PROCESSING - Automated, no user interaction
        # ============================================================
        log.info("\n" + "="*60)
        log.info("⚙️  AUTOMATED PROCESSING PHASE - No more user input needed")
        log.info("="*60)

        # Step 1: Process Goodreads
        log.progress("\n Step 1/4: Processing Goodreads data...")
        try:
            gr_success = full_goodreads_pipeline(auto_process_only=True)
            if not gr_success:
                log.error("Goodreads pipeline failed, stopping books pipeline")
                return False
        except Exception as e:
            log.error(f"Error in Goodreads pipeline: {e}")
            return False

        # Step 2: Process Kindle
        log.progress("\n Step 2/4: Processing Kindle data...")
        try:
            kindle_success = full_kindle_pipeline(auto_process_only=True)
            if not kindle_success:
                log.error("Kindle pipeline failed, stopping books pipeline")
                return False
        except Exception as e:
            log.error(f"Error in Kindle pipeline: {e}")
            return False

        # Step 3: Merge the data
        log.info("\n🔗 Step 3/4: Merging Goodreads and Kindle data...")
        merge_success = create_books_file()
        if not merge_success:
            log.error("Books merge failed, stopping pipeline")
            return False

        # Step 4: Upload results
        log.progress("\n Step 4/4: Uploading results...")
        success = upload_books_results()

    elif choice == "2":
        log.info("\n⚙️  Processing existing export files and uploading...")

        # Step 1: Process Goodreads data from existing exports
        log.progress("\n Step 1: Processing Goodreads data from existing exports...")
        try:
            gr_success = full_goodreads_pipeline(auto_process_only=True)
            if not gr_success:
                log.error("Goodreads processing failed, stopping books pipeline")
                return False
        except Exception as e:
            log.error(f"Error in Goodreads processing: {e}")
            return False

        # Step 2: Process Kindle data from existing exports
        log.progress("\n Step 2: Processing Kindle data from existing exports...")
        try:
            kindle_success = full_kindle_pipeline(auto_process_only=True)
            if not kindle_success:
                log.error("Kindle processing failed, stopping books pipeline")
                return False
        except Exception as e:
            log.error(f"Error in Kindle processing: {e}")
            return False

        # Step 3: Merge the processed data
        log.info("\n🔗 Step 3: Merging Goodreads and Kindle data...")
        merge_success = create_books_file()
        if not merge_success:
            log.error("Books merge failed, stopping pipeline")
            return False

        # Step 4: Upload results
        log.progress("\n Step 4: Uploading results...")
        success = upload_books_results()

    elif choice == "3":
        log.progress("\n Upload only: Uploading existing files to Drive...")
        success = upload_books_results()

    elif choice == "4":
        log.progress("\n Update cover URL...")
        success = update_cover_url()

    else:
        log.error("Invalid choice. Please select 1-4.")
        return False

    # Final status
    log.info("\n" + "="*60)
    if success:
        log.success("Reading topic coordinator completed successfully!")
        log.normal_success("Reading processing completed successfully")
        log.progress("Your unified books dataset is ready for analysis!")
        # Record successful run with new tracking name
        record_successful_run('topic_reading', 'active')
        # Update website tracking file
        full_website_maintenance_pipeline(auto_mode=True, quiet=True)
    else:
        log.error("Reading topic coordinator failed")
        log.normal("Reading processing failed")
    log.info("="*60)

    return success


# Legacy function for backward compatibility
def process_book_exports(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_books_pipeline(auto_full=True)
    else:
        return create_books_file()


if __name__ == "__main__":
    # Allow running this file directly
    log.progress("Reading Topic Coordinator")
    log.info("This tool combines Goodreads and Kindle data into a unified dataset.")

    # Test drive connection first
    if not verify_drive_connection():
        log.warning("Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_books_pipeline(auto_full=False)
