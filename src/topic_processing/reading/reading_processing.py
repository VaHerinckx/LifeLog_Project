import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Import the pipeline functions from source processors
from src.sources_processing.goodreads.goodreads_processing import full_goodreads_pipeline
from src.sources_processing.kindle.kindle_processing import full_kindle_pipeline
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case


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
                print(f"⚠️  Could not get modification time for {file_path}: {e}")

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
    print("🔍 Checking prerequisite files...")

    missing_files = [source for source, info in file_status.items() if not info['exists']]

    if not missing_files:
        print("✅ All prerequisite files exist")
        for source, info in file_status.items():
            print(f"  📚 {source.title()}: {info['last_modified']}")
        return True

    print(f"⚠️  Missing files for: {', '.join(missing_files)}")

    # Process missing files
    success = True

    if 'goodreads' in missing_files:
        print("\n📚 Running Goodreads pipeline to create missing file...")
        try:
            gr_success = full_goodreads_pipeline(auto_full=True)
            if not gr_success:
                print("❌ Goodreads pipeline failed")
                success = False
        except Exception as e:
            print(f"❌ Error running Goodreads pipeline: {e}")
            success = False

    if 'kindle' in missing_files and success:
        print("\n📱 Running Kindle pipeline to create missing file...")
        try:
            kindle_success = full_kindle_pipeline(auto_full=True)
            if not kindle_success:
                print("❌ Kindle pipeline failed")
                success = False
        except Exception as e:
            print(f"❌ Error running Kindle pipeline: {e}")
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
        print("⚠️  No Goodreads reading dates JSON found - skipping accidental clicks removal")
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
                    print(f"⚠️  Error parsing date for book {book_id}: {e}")
                    continue

        gr_date_df = pd.DataFrame(date_records)
        print(f"📅 Loaded official reading end dates for {len(gr_date_df)} books")
        return gr_date_df

    except Exception as e:
        print(f"❌ Error loading Goodreads reading dates: {e}")
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
        print(f"⚠️  Error comparing dates for Book ID '{book_id}': {e}")
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
    print("🧹 Removing accidental clicks...")

    # Load official Goodreads reading dates
    gr_date_df = load_goodreads_reading_dates()

    if gr_date_df.empty:
        print("⚠️  No Goodreads reading dates available, skipping accidental clicks removal")
        return df

    # Apply the flagging function to identify accidental clicks
    print("🔍 Identifying accidental clicks...")
    df = df.copy()
    df['Flag_remove'] = df.apply(lambda x: flag_clicks(x, gr_date_df), axis=1)

    # Count flagged records
    flagged_count = df['Flag_remove'].sum()
    total_count = len(df)

    print(f"🚫 Found {flagged_count} accidental clicks out of {total_count} total records")

    if flagged_count > 0:
        # Show some examples of what's being removed
        flagged_examples = df[df['Flag_remove'] == 1][['Title', 'Timestamp', 'Source']].head(5)
        if not flagged_examples.empty:
            print("📝 Examples of accidental clicks being removed:")
            for _, example in flagged_examples.iterrows():
                print(f"   • '{example['Title']}' on {example['Timestamp']} ({example['Source']})")

    # Remove flagged records
    cleaned_df = df[df['Flag_remove'] == 0].drop('Flag_remove', axis=1)

    removed_count = total_count - len(cleaned_df)
    print(f"✅ Removed {removed_count} accidental click records")

    return cleaned_df


def identify_goodreads_only_books(df):
    """
    Identifies books that exist only in Goodreads data (not in Kindle).

    Args:
        df (pd.DataFrame): Combined dataset

    Returns:
        list: List of Book IDs that are Goodreads-only
    """
    print("🔍 Identifying Goodreads-only books...")

    # Group by Book ID and check sources
    gr_only_books = []

    for book_id, group in df.groupby('Book Id'):
        sources = set(group['Source'].dropna())
        if sources == {'GoodReads'}:  # Only Goodreads, no Kindle data
            gr_only_books.append(book_id)

    print(f"📚 Found {len(gr_only_books)} books that are Goodreads-only")
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
    print("⏱️  Calculating reading durations per book...")

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
    print(f"📊 Assigned reading duration to {books_with_duration} book records")

    return df


def remove_duplicate_records(df):
    """
    Remove duplicate records from the combined dataset.

    Args:
        df (pd.DataFrame): Combined dataset

    Returns:
        pd.DataFrame: Deduplicated dataset
    """
    print("🔄 Removing duplicate records...")

    original_count = len(df)

    # Define columns to consider for duplicate detection
    # We exclude 'reading_duration_final' as it's intentionally sparse
    duplicate_cols = ['Book Id', 'Title', 'Author', 'Timestamp', 'Source', 'page_split']

    # Keep only columns that actually exist in the dataframe
    existing_duplicate_cols = [col for col in duplicate_cols if col in df.columns]

    if existing_duplicate_cols:
        df_deduplicated = df.drop_duplicates(subset=existing_duplicate_cols, keep='first')
    else:
        print("⚠️  No suitable columns for duplicate detection, keeping all records")
        df_deduplicated = df

    removed_count = original_count - len(df_deduplicated)

    if removed_count > 0:
        print(f"🗑️  Removed {removed_count} duplicate records")
    else:
        print("✅ No duplicates found")

    return df_deduplicated


def create_books_file():
    """
    Main function to merge Goodreads and Kindle processed files into a unified books dataset.

    Returns:
        bool: True if successful, False otherwise
    """
    print("📚 Creating unified books file...")

    try:
        # Check prerequisite files
        file_status = check_prerequisite_files()

        if not ensure_prerequisite_files(file_status):
            print("❌ Could not ensure all prerequisite files exist")
            return False

        # Load processed files from NEW locations
        print("\n📖 Loading processed data files...")

        # Load Goodreads data from NEW location
        gr_path = 'files/source_processed_files/goodreads/goodreads_processed.csv'
        if not os.path.exists(gr_path):
            print(f"❌ Goodreads file not found: {gr_path}")
            return False

        df_gr = pd.read_csv(gr_path, sep='|', encoding='utf-8')
        print(f"✅ Loaded {len(df_gr)} Goodreads records")

        # Load Kindle data from NEW location
        kindle_path = 'files/source_processed_files/kindle/kindle_processed.csv'
        if not os.path.exists(kindle_path):
            print(f"❌ Kindle file not found: {kindle_path}")
            return False

        df_kindle = pd.read_csv(kindle_path, sep='|', encoding='utf-8')
        print(f"✅ Loaded {len(df_kindle)} Kindle records")

        # Enhance Kindle data with Goodreads metadata
        print("\n🔗 Enhancing Kindle data with Goodreads metadata...")

        # Prepare Goodreads metadata for merging (including cover_url)
        gr_metadata_columns = ['Book Id', 'Title', 'Author', 'Original Publication Year',
                              'My Rating', 'Average Rating', 'Genre', 'Fiction_yn',
                              'Number of Pages', 'reading_duration', 'cover_url']

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

        print(f"📱 Enhanced {len(df_kindle_enhanced)} Kindle records with Goodreads metadata")

        # Combine datasets
        print("\n🔄 Combining Goodreads and Kindle datasets...")

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
        print(f"📊 Combined dataset has {len(combined_df)} total records")

        # Remove accidental clicks BEFORE further processing
        combined_df = remove_accidental_clicks(combined_df)

        # Identify Goodreads-only books
        gr_only_books = identify_goodreads_only_books(combined_df)

        # Filter to keep only Kindle data + Goodreads-only books
        print("\n🎯 Filtering to final dataset...")

        kindle_data = combined_df[combined_df['Source'] == 'Kindle']
        gr_only_data = combined_df[
            (combined_df['Source'] == 'GoodReads') &
            (combined_df['Book Id'].isin(gr_only_books))
        ]

        final_df = pd.concat([kindle_data, gr_only_data], ignore_index=True)
        print(f"📈 Final dataset: {len(kindle_data)} Kindle + {len(gr_only_data)} Goodreads-only = {len(final_df)} records")

        # Process timestamps
        print("\n⏰ Processing timestamps...")

        # First convert to datetime
        final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], errors='coerce')

        # Force conversion to timezone-naive for all timestamps
        print("🌍 Converting all timestamps to timezone-naive...")

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

        # Sort by timestamp (most recent first)
        final_df = final_df.sort_values('Timestamp', ascending=False)

        # Clean up and finalize columns
        print("\n🧹 Finalizing dataset...")

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

        print(f"\n✅ Successfully created unified books file!")
        print(f"📁 Saved to: {output_path}")
        print(f"📊 Final statistics:")
        print(f"   📚 Total records: {len(final_df):,}")
        print(f"   📖 Unique books: {final_df['book_id'].nunique():,}")
        print(f"   📱 Kindle records: {len(final_df[final_df['source'] == 'Kindle']):,}")
        print(f"   📚 Goodreads-only records: {len(final_df[final_df['source'] == 'GoodReads']):,}")

        # Check cover URL availability
        if 'cover_url' in final_df.columns:
            covers_count = (final_df['cover_url'].notna() & (final_df['cover_url'] != '')).sum()
            print(f"   🖼️  Records with covers: {covers_count:,}")

        if len(final_df) > 0:
            print(f"   📅 Date range: {final_df['timestamp'].min().date()} to {final_df['timestamp'].max().date()}")

        # Generate website files
        print("\n🌐 Generating website-optimized files...")
        website_success = generate_reading_website_page_files(final_df)

        if not website_success:
            print("⚠️  Warning: Website files generation failed, but processed file was saved")

        return True

    except Exception as e:
        print(f"❌ Error creating books file: {e}")
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
    print("\n🌐 Generating website files for Reading page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/reading'
        os.makedirs(website_dir, exist_ok=True)

        # Work with copy to avoid modifying original
        df_web = df[df["title"] != "Unknown Title"].copy()

        # Ensure timestamp is datetime
        df_web['timestamp'] = pd.to_datetime(df_web['timestamp'], errors='coerce')

        # Add derived date columns
        print("📅 Adding derived date columns...")
        df_web['reading_year'] = df_web['timestamp'].dt.year.astype('Int64').astype(str)
        df_web['reading_month'] = df_web['timestamp'].dt.month.astype('Int64').astype(str)
        df_web['reading_quarter'] = df_web['timestamp'].dt.quarter.astype('Int64').astype(str)

        # Standardize fiction_yn field to consistent casing
        print("🔧 Standardizing fiction_yn field...")
        if 'fiction_yn' in df_web.columns:
            df_web['fiction_yn'] = df_web['fiction_yn'].apply(
                lambda x: 'Fiction' if str(x).lower() == 'fiction' else 'Non-Fiction'
            )

        # Sort by timestamp descending (most recent first)
        df_web = df_web.sort_values('timestamp', ascending=False)

        # FILE 1: Sessions (all reading sessions)
        sessions_columns = [
            'book_id', 'title', 'author', 'timestamp', 'source', 'genre',
            'page_split', 'my_rating', 'reading_year', 'reading_month', 'reading_quarter'
        ]

        sessions_df = df_web[sessions_columns].copy()

        # Enforce snake_case before saving
        sessions_df = enforce_snake_case(sessions_df, "reading_page_sessions")

        sessions_path = f'{website_dir}/reading_page_sessions.csv'
        sessions_df.sort_values('timestamp', ascending=False).to_csv(sessions_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Sessions file: {len(sessions_df):,} records → {sessions_path}")

        # FILE 2: Books (aggregated unique books)
        print("📚 Aggregating unique books...")

        # Add computed column
        df_web["pages_per_day"] = df_web["number_of_pages"] / df_web["reading_duration_final"]

        # Group by book_id + title + author, take most recent record (already sorted above)
        books_df = df_web.groupby(['book_id', 'title', 'author'], as_index=False).first()

        books_columns = [
            'book_id', 'title', 'author', 'original_publication_year',
            'my_rating', 'average_rating', 'genre', 'fiction_yn',
            'number_of_pages', 'pages_per_day', 'reading_duration_final', 'cover_url',
            'timestamp', 'reading_year', 'reading_month', 'reading_quarter'
        ]

        books_df = books_df[books_columns].copy()

        # Enforce snake_case before saving
        books_df = enforce_snake_case(books_df, "reading_page_books")

        books_path = f'{website_dir}/reading_page_books.csv'
        books_df.sort_values('timestamp', ascending=False).to_csv(books_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Books file: {len(books_df):,} unique books → {books_path}")

        return True

    except Exception as e:
        print(f"❌ Error generating website files: {e}")
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
    print("☁️  Uploading Reading page website files to Google Drive...")

    # Only upload website files
    files_to_upload = [
        'files/website_files/reading/reading_page_sessions.csv',
        'files/website_files/reading/reading_page_books.csv'
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No website files found to upload")
        print("💡 Make sure generate_reading_website_page_files() ran successfully")
        return False

    print(f"📤 Uploading {len(existing_files)} website files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Website files uploaded successfully!")
    else:
        print("❌ Some website files failed to upload")

    return success


def update_cover_url():
    """
    Update book cover URLs in the reading_dates.json source file.
    This ensures manual updates persist through all future pipeline runs.
    """
    print("\n" + "="*50)
    print("📖 UPDATE BOOK COVER URLs")
    print("="*50)

    # Load the reading_dates.json file (source of truth)
    json_path = 'files/work_files/gr_work_files/reading_dates.json'
    try:
        with open(json_path, 'r') as f:
            dates_data = json.load(f)
        print(f"✅ Loaded {len(dates_data)} books from reading_dates.json")
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        print("Run the Goodreads pipeline first to create the reading dates file.")
        return False
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
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
            print(f"❌ No books found containing '{search_term}'")
            continue

        print(f"\n📚 Found {len(matches)} books:")
        print("-" * 50)

        # Display matches with numbers
        for i, book in enumerate(matches, 1):
            current_url = book['cover_url'] if book['cover_url'] else 'No cover URL'
            if len(str(current_url)) > 60:
                current_url = str(current_url)[:60] + "..."
            print(f"{i}. {book['title']}")
            print(f"   Current cover: {current_url}")
            print()

        # Get user selection
        try:
            choice = int(input(f"Select book (1-{len(matches)}): "))
            if choice < 1 or choice > len(matches):
                print(f"❌ Invalid choice. Please select 1-{len(matches)}")
                continue
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
            continue

        # Get the selected book
        selected_book = matches[choice - 1]
        selected_book_id = selected_book['book_id']
        selected_title = selected_book['title']

        print(f"\n✅ Selected: {selected_title}")
        print(f"Current cover URL: {selected_book['cover_url'] or 'None'}")

        # Get new URL
        new_url = input("\nEnter new cover URL: ").strip()
        if not new_url:
            print("❌ No URL provided, skipping this book")
            continue

        # Update the cover URL in the JSON data
        dates_data[selected_book_id]['cover_url'] = new_url
        print(f"✅ Updated cover URL for '{selected_title}'")
        updated_books.append(selected_title)

        # Ask if user wants to continue
        continue_choice = input("\nUpdate another book cover? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break

    # If no books were updated, exit
    if not updated_books:
        print("❌ No books were updated")
        return False

    # Step 1: Save the updated JSON file
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(dates_data, f, indent=2, default=str)
        print(f"\n💾 Saved {len(updated_books)} cover URL updates to reading_dates.json")
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")
        return False

    # Step 2: Regenerate Goodreads processed file (loads updated JSON with new cover URLs)
    try:
        print("\n🔄 Regenerating Goodreads processed file with updated cover URLs...")
        from src.sources_processing.goodreads.goodreads_processing import create_goodreads_file
        gr_success = create_goodreads_file()
        if not gr_success:
            print("❌ Failed to regenerate Goodreads processed file")
            return False
    except Exception as e:
        print(f"❌ Error regenerating Goodreads file: {e}")
        return False

    # Step 3: Regenerate the unified books file (merges updated Goodreads data with Kindle)
    try:
        print("\n🔄 Regenerating unified books file...")
        create_success = create_books_file()
        if not create_success:
            print("❌ Failed to regenerate books file")
            return False
    except Exception as e:
        print(f"❌ Error regenerating books file: {e}")
        return False

    # Step 4: Upload website files to Google Drive
    try:
        print("\n📤 Uploading website files to Google Drive...")
        upload_success = upload_books_results()
        if upload_success:
            print("✅ Successfully uploaded website files to Google Drive!")
            print(f"🎉 Cover URLs updated for: {', '.join(updated_books)}")
            print(f"💡 Updates are now permanent - they will persist through future pipeline runs")
            return True
        else:
            print("⚠️  Website files updated locally but failed to upload to Google Drive")
            return False
    except Exception as e:
        print(f"❌ Error uploading to Google Drive: {e}")
        return False


def full_books_pipeline(auto_full=False, auto_process_only=False):
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
    print("\n" + "="*60)
    print("📚 READING TOPIC COORDINATOR")
    print("="*60)
    print("This pipeline combines Goodreads and Kindle data into a unified books dataset")

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data and uploading...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download both → process both → merge → upload)")
        print("2. Process existing files (process exports → merge → upload)")
        print("3. Upload only (upload existing processed/website files to Drive)")
        print("4. Update book cover URL (interactive cover URL updates)")

        choice = input("\nEnter your choice (1-4): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full books pipeline...")

        # Step 1: Run Goodreads pipeline
        print("\n📚 Step 1: Processing Goodreads data...")
        try:
            gr_success = full_goodreads_pipeline(auto_full=True)
            if not gr_success:
                print("❌ Goodreads pipeline failed, stopping books pipeline")
                return False
        except Exception as e:
            print(f"❌ Error in Goodreads pipeline: {e}")
            return False

        # Step 2: Run Kindle pipeline
        print("\n📱 Step 2: Processing Kindle data...")
        try:
            kindle_success = full_kindle_pipeline(auto_full=True)
            if not kindle_success:
                print("❌ Kindle pipeline failed, stopping books pipeline")
                return False
        except Exception as e:
            print(f"❌ Error in Kindle pipeline: {e}")
            return False

        # Step 3: Merge the data
        print("\n🔗 Step 3: Merging Goodreads and Kindle data...")
        merge_success = create_books_file()
        if not merge_success:
            print("❌ Books merge failed, stopping pipeline")
            return False

        # Step 4: Upload results
        print("\n☁️  Step 4: Uploading results...")
        success = upload_books_results()

    elif choice == "2":
        print("\n⚙️  Processing existing export files and uploading...")

        # Step 1: Process Goodreads data from existing exports
        print("\n📚 Step 1: Processing Goodreads data from existing exports...")
        try:
            gr_success = full_goodreads_pipeline(auto_process_only=True)
            if not gr_success:
                print("❌ Goodreads processing failed, stopping books pipeline")
                return False
        except Exception as e:
            print(f"❌ Error in Goodreads processing: {e}")
            return False

        # Step 2: Process Kindle data from existing exports
        print("\n📱 Step 2: Processing Kindle data from existing exports...")
        try:
            kindle_success = full_kindle_pipeline(auto_process_only=True)
            if not kindle_success:
                print("❌ Kindle processing failed, stopping books pipeline")
                return False
        except Exception as e:
            print(f"❌ Error in Kindle processing: {e}")
            return False

        # Step 3: Merge the processed data
        print("\n🔗 Step 3: Merging Goodreads and Kindle data...")
        merge_success = create_books_file()
        if not merge_success:
            print("❌ Books merge failed, stopping pipeline")
            return False

        # Step 4: Upload results
        print("\n☁️  Step 4: Uploading results...")
        success = upload_books_results()

    elif choice == "3":
        print("\n☁️  Upload only: Uploading existing files to Drive...")
        success = upload_books_results()

    elif choice == "4":
        print("\n📖 Update cover URL...")
        success = update_cover_url()

    else:
        print("❌ Invalid choice. Please select 1-4.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Reading topic coordinator completed successfully!")
        print("📊 Your unified books dataset is ready for analysis!")
        # Record successful run with new tracking name
        record_successful_run('topic_reading', 'active')
    else:
        print("❌ Reading topic coordinator failed")
    print("="*60)

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
    print("📚 Reading Topic Coordinator")
    print("This tool combines Goodreads and Kindle data into a unified dataset.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_books_pipeline(auto_full=False)
