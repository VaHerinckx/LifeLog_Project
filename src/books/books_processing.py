import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Import the pipeline functions from other modules
from src.books.goodreads_processing import full_goodreads_pipeline
from src.books.kindle_processing import full_kindle_pipeline
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection


def check_prerequisite_files():
    """
    Check if the required processed files from Goodreads and Kindle exist.

    Returns:
        dict: Status of each required file
    """
    required_files = {
        'goodreads': 'files/processed_files/books/gr_processed.csv',
        'kindle': 'files/processed_files/books/kindle_processed.csv'
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

    # Look for this book's official end date by Title
    book_title = row.get("Title", "")
    if not book_title:
        return 0

    # Find matching title in Goodreads data
    matching_books = gr_date_df_valid[gr_date_df_valid['Title'] == book_title]

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
        print(f"⚠️  Error comparing dates for book '{book_title}': {e}")
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

        # Load processed files
        print("\n📖 Loading processed data files...")

        # Load Goodreads data
        gr_path = 'files/processed_files/books/gr_processed.csv'
        if not os.path.exists(gr_path):
            print(f"❌ Goodreads file not found: {gr_path}")
            return False

        df_gr = pd.read_csv(gr_path, sep='|')
        print(f"✅ Loaded {len(df_gr)} Goodreads records")

        # Load Kindle data
        kindle_path = 'files/processed_files/books/kindle_processed.csv'
        if not os.path.exists(kindle_path):
            print(f"❌ Kindle file not found: {kindle_path}")
            return False

        df_kindle = pd.read_csv(kindle_path, sep='|')
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

        # Save the final processed file
        output_path = 'files/processed_files/books/kindle_gr_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        final_df.to_csv(output_path, sep='|', index=False)

        print(f"\n✅ Successfully created unified books file!")
        print(f"📁 Saved to: {output_path}")
        print(f"📊 Final statistics:")
        print(f"   📚 Total records: {len(final_df):,}")
        print(f"   📖 Unique books: {final_df['Book Id'].nunique():,}")
        print(f"   📱 Kindle records: {len(final_df[final_df['Source'] == 'Kindle']):,}")
        print(f"   📚 Goodreads-only records: {len(final_df[final_df['Source'] == 'GoodReads']):,}")

        # Check cover URL availability
        if 'cover_url' in final_df.columns:
            covers_count = (final_df['cover_url'].notna() & (final_df['cover_url'] != '')).sum()
            print(f"   🖼️  Records with covers: {covers_count:,}")

        if len(final_df) > 0:
            print(f"   📅 Date range: {final_df['Timestamp'].min().date()} to {final_df['Timestamp'].max().date()}")

        return True

    except Exception as e:
        print(f"❌ Error creating books file: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_books_results():
    """
    Upload the processed books files to Google Drive.

    Returns:
        bool: True if successful, False otherwise
    """
    print("☁️  Uploading books results to Google Drive...")

    files_to_upload = [
        'files/processed_files/books/kindle_gr_processed.csv'
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No books files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Books results uploaded successfully!")
    else:
        print("❌ Some books files failed to upload")

    return success


def full_books_pipeline(auto_full=False):
    """
    Complete books processing pipeline that orchestrates Goodreads and Kindle processing.

    Options:
    1. Full pipeline (download both → process both → merge → upload)
    2. Process with existing files only (merge existing processed files)
    3. Force refresh all data (re-run both pipelines + merge)
    4. Merge only (create books file from existing processed files)

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("📚 BOOKS DATA PIPELINE")
    print("="*60)
    print("This pipeline combines Goodreads and Kindle data into a unified books dataset")

    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download both → process both → merge → upload)")
        print("2. Process with existing files only (merge existing processed files)")
        print("3. Force refresh all data (re-run both pipelines + merge)")
        print("4. Merge only (create books file from existing processed files)")

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
        print("\n⚙️  Processing with existing files...")

        # Check if prerequisite files exist
        file_status = check_prerequisite_files()
        missing_files = [source for source, info in file_status.items() if not info['exists']]

        if missing_files:
            print(f"⚠️  Missing prerequisite files: {', '.join(missing_files)}")
            print("💡 Tip: Run option 1 or 3 to download and process the missing data first")
            return False

        # Merge existing files
        merge_success = create_books_file()
        if merge_success:
            success = upload_books_results()
        else:
            success = False

    elif choice == "3":
        print("\n🔄 Force refresh: Re-running all pipelines...")

        # Force re-run Goodreads pipeline
        print("\n📚 Force refresh: Goodreads pipeline...")
        try:
            gr_success = full_goodreads_pipeline(auto_full=True)
            if not gr_success:
                print("❌ Goodreads pipeline failed")
                return False
        except Exception as e:
            print(f"❌ Error in Goodreads pipeline: {e}")
            return False

        # Force re-run Kindle pipeline
        print("\n📱 Force refresh: Kindle pipeline...")
        try:
            kindle_success = full_kindle_pipeline(auto_full=True)
            if not kindle_success:
                print("❌ Kindle pipeline failed")
                return False
        except Exception as e:
            print(f"❌ Error in Kindle pipeline: {e}")
            return False

        # Merge and upload
        print("\n🔗 Merging refreshed data...")
        merge_success = create_books_file()
        if merge_success:
            success = upload_books_results()
        else:
            success = False

    elif choice == "4":
        print("\n🔗 Merge only: Creating books file from existing data...")
        success = create_books_file()

    else:
        print("❌ Invalid choice. Please select 1-4.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Books pipeline completed successfully!")
        print("📊 Your unified books dataset is ready for analysis!")
    else:
        print("❌ Books pipeline failed")
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
    print("📚 Books Data Processing Pipeline")
    print("This tool combines Goodreads and Kindle data into a unified dataset.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_books_pipeline(auto_full=False)
