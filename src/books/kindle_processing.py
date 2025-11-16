import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path

from src.utils.file_operations import find_unzip_folder, clean_rename_move_folder, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run


def load_existing_mapping(json_file_path: str) -> Dict:
    """
    Load existing ASIN to Book ID mapping from JSON file.

    Args:
        json_file_path: Path to the JSON mapping file

    Returns:
        Dictionary with existing mappings or empty dict if file doesn't exist
    """
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📂 Loaded existing mapping with {len(data.get('successful_mappings', {}))} entries")
            return data
        except Exception as e:
            print(f"⚠️  Error loading existing mapping: {e}")
            return {}
    else:
        print("📂 No existing mapping file found - creating new one")
        return {}


def load_asin_blacklist(blacklist_file_path: str) -> set:
    """
    Load ASIN blacklist from JSON file.

    Args:
        blacklist_file_path: Path to the blacklist JSON file

    Returns:
        Set of blacklisted ASINs
    """
    if os.path.exists(blacklist_file_path):
        try:
            with open(blacklist_file_path, 'r', encoding='utf-8') as f:
                blacklist_data = json.load(f)

            blacklisted_asins = set(blacklist_data.get('blacklisted_asins', []))

            if blacklisted_asins:
                print(f"🚫 Loaded {len(blacklisted_asins)} blacklisted ASINs")

                # Show reasons if available
                reasons = blacklist_data.get('blacklist_reasons', {})
                for asin in list(blacklisted_asins)[:5]:  # Show first 5
                    reason = reasons.get(asin, "No reason specified")
                    print(f"   • {asin}: {reason}")
                if len(blacklisted_asins) > 5:
                    print(f"   ... and {len(blacklisted_asins) - 5} more")
            else:
                print("📝 Blacklist file found but no ASINs listed")

            return blacklisted_asins

        except Exception as e:
            print(f"⚠️  Error loading blacklist file: {e}")
            return set()
    else:
        print("📝 No blacklist file found - processing all ASINs")
        return set()


def extract_kindle_reading_periods(kindle_df: pd.DataFrame, blacklisted_asins: set = None) -> Dict[str, List[Tuple[datetime, datetime]]]:
    """
    Extract reading periods for each ASIN from Kindle data, excluding blacklisted ASINs.

    Args:
        kindle_df: DataFrame with Kindle reading sessions
        blacklisted_asins: Set of ASINs to exclude from processing

    Returns:
        Dictionary mapping ASIN to list of (start_time, end_time) tuples
    """
    print("📱 Processing Kindle reading sessions...")

    if blacklisted_asins is None:
        blacklisted_asins = set()

    # First, let's examine the data to understand what we're working with
    print(f"📊 Total rows: {len(kindle_df)}")
    print(f"📊 Columns: {list(kindle_df.columns)}")

    # Clean the timestamp columns by replacing problematic values
    timestamp_cols = ['start_timestamp', 'end_timestamp']

    for col in timestamp_cols:
        if col in kindle_df.columns:
            # Count problematic values
            problematic_count = kindle_df[col].isin(['Not Available', 'N/A', '', 'null', 'NULL']).sum()
            null_count = kindle_df[col].isnull().sum()

            print(f"📊 {col}: {problematic_count} problematic values, {null_count} null values")

            # Replace problematic string values with NaN
            kindle_df[col] = kindle_df[col].replace(['Not Available', 'N/A', '', 'null', 'NULL'], pd.NaT)

    # Convert timestamp columns to datetime with error handling
    for col in timestamp_cols:
        if col in kindle_df.columns:
            try:
                # Convert to datetime and then remove timezone info to make them naive
                kindle_df[col] = pd.to_datetime(kindle_df[col], errors='coerce')

                # Remove timezone information if present
                if kindle_df[col].dtype.name.startswith('datetime64[ns,'):
                    kindle_df[col] = kindle_df[col].dt.tz_localize(None)
                elif hasattr(kindle_df[col].iloc[0], 'tz') and kindle_df[col].iloc[0] is not pd.NaT:
                    # Handle pandas Timestamp with timezone
                    kindle_df[col] = kindle_df[col].apply(
                        lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tz') and x.tz else x
                    )

                print(f"✅ Successfully converted {col} to datetime")
            except Exception as e:
                print(f"⚠️  Error converting {col}: {e}")
                kindle_df[col] = pd.NaT

    # Filter out blacklisted ASINs BEFORE processing
    if blacklisted_asins:
        original_count = len(kindle_df)
        kindle_df = kindle_df[~kindle_df['ASIN'].isin(blacklisted_asins)]
        filtered_count = original_count - len(kindle_df)
        if filtered_count > 0:
            print(f"🚫 Filtered out {filtered_count} rows with blacklisted ASINs")

    # Count valid sessions after cleaning
    valid_sessions = kindle_df.dropna(subset=['start_timestamp', 'end_timestamp'])
    print(f"📊 Valid sessions after cleaning: {len(valid_sessions)} out of {len(kindle_df)}")

    if len(valid_sessions) == 0:
        print("❌ No valid timestamp data found in Kindle file")
        return {}

    # Group by ASIN and collect all reading periods
    asin_periods = defaultdict(list)
    blacklisted_found = set()

    for _, row in valid_sessions.iterrows():
        asin = row.get('ASIN', row.get('asin', ''))  # Handle different column name cases

        # Double-check blacklist (shouldn't be needed after filtering above, but safe)
        if asin in blacklisted_asins:
            blacklisted_found.add(asin)
            continue

        start_time = row['start_timestamp']
        end_time = row['end_timestamp']

        # Convert pandas Timestamp to datetime if needed
        if hasattr(start_time, 'to_pydatetime'):
            start_time = start_time.to_pydatetime()
        if hasattr(end_time, 'to_pydatetime'):
            end_time = end_time.to_pydatetime()

        # Only include valid reading sessions
        if (pd.notna(start_time) and pd.notna(end_time) and
            start_time <= end_time and asin and asin != ''):
            asin_periods[asin].append((start_time, end_time))

    # Report any blacklisted ASINs that were found
    if blacklisted_found:
        print(f"🚫 Skipped {len(blacklisted_found)} blacklisted ASINs: {', '.join(list(blacklisted_found)[:5])}")

    # Sort periods by start time for each ASIN
    for asin in asin_periods:
        asin_periods[asin].sort(key=lambda x: x[0])

    print(f"📊 Found reading periods for {len(asin_periods)} unique ASINs (after blacklist filtering)")

    return dict(asin_periods)


def extract_goodreads_reading_periods(gr_df: pd.DataFrame) -> Dict[str, Tuple[datetime, datetime, str]]:
    """
    Extract reading periods for each Book ID from Goodreads data.

    Args:
        gr_df: DataFrame with Goodreads processed data

    Returns:
        Dictionary mapping Book ID to (start_date, end_date, title) tuple
    """
    print("📚 Processing Goodreads reading periods...")

    # Convert Timestamp to datetime and make timezone-naive
    try:
        gr_df['Timestamp'] = pd.to_datetime(gr_df['Timestamp'], errors='coerce')

        # Remove timezone information if present
        if gr_df['Timestamp'].dtype.name.startswith('datetime64[ns,'):
            gr_df['Timestamp'] = gr_df['Timestamp'].dt.tz_localize(None)
        elif hasattr(gr_df['Timestamp'].iloc[0], 'tz') and gr_df['Timestamp'].iloc[0] is not pd.NaT:
            # Handle pandas Timestamp with timezone
            gr_df['Timestamp'] = gr_df['Timestamp'].apply(
                lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tz') and x.tz else x
            )

        print("✅ Successfully converted Goodreads timestamps")
    except Exception as e:
        print(f"⚠️  Error converting Goodreads timestamps: {e}")

    # Group by Book ID and find date ranges
    book_periods = {}

    for book_id, group in gr_df.groupby('Book Id'):
        if pd.isna(book_id) or book_id == '':
            continue

        # Get date range for this book
        timestamps = group['Timestamp'].dropna()
        if len(timestamps) > 0:
            start_date = timestamps.min()
            end_date = timestamps.max()

            # Convert pandas Timestamp to datetime if needed
            if hasattr(start_date, 'to_pydatetime'):
                start_date = start_date.to_pydatetime()
            if hasattr(end_date, 'to_pydatetime'):
                end_date = end_date.to_pydatetime()

            # Get book title (should be the same for all rows of this book)
            title = group['Title'].iloc[0] if 'Title' in group.columns else 'Unknown Title'

            book_periods[str(book_id)] = (start_date, end_date, title)

    print(f"📊 Found reading periods for {len(book_periods)} unique Book IDs")

    return book_periods


def calculate_overlap_score(kindle_periods: List[Tuple[datetime, datetime]],
                          gr_start: datetime, gr_end: datetime) -> float:
    """
    Calculate overlap score between Kindle reading sessions and Goodreads reading period.

    Args:
        kindle_periods: List of (start, end) tuples from Kindle
        gr_start: Goodreads reading start date
        gr_end: Goodreads reading end date

    Returns:
        Overlap score (0-1, where 1 is perfect overlap)
    """
    total_kindle_duration = timedelta(0)
    total_overlap_duration = timedelta(0)

    for k_start, k_end in kindle_periods:
        # Calculate duration of this Kindle session
        kindle_duration = k_end - k_start
        total_kindle_duration += kindle_duration

        # Calculate overlap with Goodreads period
        overlap_start = max(k_start.date(), gr_start.date())
        overlap_end = min(k_end.date(), gr_end.date())

        if overlap_start <= overlap_end:
            # Convert back to datetime for calculation (use start of day)
            overlap_start_dt = datetime.combine(overlap_start, datetime.min.time())
            overlap_end_dt = datetime.combine(overlap_end, datetime.max.time())

            # Calculate actual overlap duration
            actual_overlap_start = max(k_start, overlap_start_dt)
            actual_overlap_end = min(k_end, overlap_end_dt)

            if actual_overlap_start <= actual_overlap_end:
                overlap_duration = actual_overlap_end - actual_overlap_start
                total_overlap_duration += overlap_duration

    # Calculate overlap ratio
    if total_kindle_duration.total_seconds() > 0:
        overlap_ratio = total_overlap_duration.total_seconds() / total_kindle_duration.total_seconds()
        return min(overlap_ratio, 1.0)  # Cap at 1.0
    else:
        return 0.0


def find_best_matches(asin_periods: Dict[str, List[Tuple[datetime, datetime]]],
                     book_periods: Dict[str, Tuple[datetime, datetime, str]],
                     min_overlap_threshold: float = 0.3) -> Dict[str, Dict]:
    """
    Find the best Book ID matches for each ASIN based on timestamp overlap.

    Args:
        asin_periods: ASIN to reading periods mapping
        book_periods: Book ID to reading period mapping
        min_overlap_threshold: Minimum overlap score to consider a match

    Returns:
        Dictionary mapping ASIN to best match info
    """
    print(f"🔍 Finding matches with minimum overlap threshold: {min_overlap_threshold}")

    matches = {}

    for asin, kindle_sessions in asin_periods.items():
        print(f"\n🔍 Processing ASIN: {asin}")

        # Calculate Kindle reading period bounds
        kindle_start = min(session[0] for session in kindle_sessions)
        kindle_end = max(session[1] for session in kindle_sessions)
        kindle_days = (kindle_end.date() - kindle_start.date()).days + 1

        print(f"  📱 Kindle period: {kindle_start.date()} to {kindle_end.date()} ({kindle_days} days)")

        best_match = None
        best_score = 0.0

        # Try to match with each Goodreads book
        for book_id, (gr_start, gr_end, title) in book_periods.items():
            overlap_score = calculate_overlap_score(kindle_sessions, gr_start, gr_end)

            if overlap_score >= min_overlap_threshold and overlap_score > best_score:
                best_score = overlap_score
                best_match = {
                    'book_id': book_id,
                    'title': title,
                    'gr_start': gr_start.date().isoformat(),
                    'gr_end': gr_end.date().isoformat(),
                    'overlap_score': round(overlap_score, 3),
                    'kindle_start': kindle_start.date().isoformat(),
                    'kindle_end': kindle_end.date().isoformat(),
                    'kindle_sessions_count': len(kindle_sessions)
                }

        if best_match:
            matches[asin] = best_match
            print(f"  ✅ Match found: '{best_match['title']}' (score: {best_match['overlap_score']:.3f})")
        else:
            print(f"  ❌ No suitable match found (best score: {best_score:.3f})")

    return matches


def download_kindle_data():
    """
    Opens Amazon account page and prompts user to request Kindle data.
    Returns True if user confirms download, False otherwise.
    """
    print("📱 Starting Kindle data download...")

    urls = ['https://www.amazon.com/hz/privacy-central/data-requests/preview.html']
    open_web_urls(urls)

    print("📝 Instructions:")
    print("   1. Sign in to your Amazon account")
    print("   2. Find 'Digital content and services' section")
    print("   3. Select 'Kindle' and request data")
    print("   4. Wait for Amazon to prepare your data (can take several days)")
    print("   5. Download the ZIP file when you receive the email notification")

    response = prompt_user_download_status("Kindle")
    return response


def move_kindle_files():
    """
    Moves the downloaded Kindle files from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving Kindle files...")

    # First, try to unzip the kindle file
    unzip_success = find_unzip_folder("kindle")
    if not unzip_success:
        print("❌ Failed to find or unzip Kindle file")
        return False

    # Then move the unzipped folder
    move_success = clean_rename_move_folder(
        export_folder="files/exports",
        download_folder="/Users/valen/Downloads",
        folder_name="kindle_export_unzipped",
        new_folder_name="kindle_exports"
    )

    if move_success:
        print("✅ Successfully moved Kindle files to exports folder")
    else:
        print("❌ Failed to move Kindle files")

    return move_success


def create_asin_bookid_mapping(min_overlap_threshold: float = 0.3) -> Dict:
    """
    Create ASIN to Book ID mapping based on timestamp overlap analysis.

    Args:
        min_overlap_threshold: Minimum overlap score to consider a match (0-1)

    Returns:
        Dictionary with the complete mapping results
    """
    print("🔄 Starting ASIN to Book ID mapping based on timestamp overlap...")

    # Define file paths
    kindle_csv_path = "files/exports/kindle_exports/Kindle.Devices.ReadingSession/Kindle.Devices.ReadingSession.csv"
    goodreads_csv_path = "files/processed_files/books/gr_processed.csv"
    output_json_path = "files/work_files/kindle_work_files/asin_bookid_mapping.json"
    blacklist_file_path = "files/work_files/kindle_work_files/asin_blacklist.json"

    print(f"📱 Kindle file: {kindle_csv_path}")
    print(f"📚 Goodreads file: {goodreads_csv_path}")
    print(f"💾 Output file: {output_json_path}")
    print(f"🚫 Blacklist file: {blacklist_file_path}")

    # Load ASIN blacklist
    blacklisted_asins = load_asin_blacklist(blacklist_file_path)

    # Load existing mapping
    existing_mapping = load_existing_mapping(output_json_path)
    existing_successful = existing_mapping.get('successful_mappings', {})

    # Step 1: Read Kindle data
    try:
        print(f"\n📱 Reading Kindle file...")
        if not os.path.exists(kindle_csv_path):
            print(f"❌ Kindle file not found: {kindle_csv_path}")
            return existing_mapping

        kindle_df = pd.read_csv(kindle_csv_path)
        print(f"✅ Loaded {len(kindle_df)} Kindle reading sessions")

        # Check for required columns
        required_cols = ['start_timestamp', 'end_timestamp']
        possible_asin_cols = ['ASIN', 'asin', 'Asin']

        # Find ASIN column
        asin_col = None
        for col in possible_asin_cols:
            if col in kindle_df.columns:
                asin_col = col
                break

        if asin_col is None:
            print(f"❌ No ASIN column found. Available columns: {list(kindle_df.columns)}")
            return existing_mapping

        # Standardize ASIN column name
        if asin_col != 'ASIN':
            kindle_df['ASIN'] = kindle_df[asin_col]

        # Check for timestamp columns
        missing_cols = [col for col in required_cols if col not in kindle_df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return existing_mapping

    except Exception as e:
        print(f"❌ Error reading Kindle CSV: {e}")
        return existing_mapping

    # Step 2: Read Goodreads data
    try:
        print(f"\n📚 Reading Goodreads file...")
        if not os.path.exists(goodreads_csv_path):
            print(f"❌ Goodreads file not found: {goodreads_csv_path}")
            return existing_mapping

        gr_df = pd.read_csv(goodreads_csv_path, sep='|', encoding='utf-8')
        print(f"✅ Loaded {len(gr_df)} Goodreads reading records")

        # Check for required columns
        required_gr_cols = ['Book Id', 'Timestamp']
        missing_gr_cols = [col for col in required_gr_cols if col not in gr_df.columns]
        if missing_gr_cols:
            print(f"❌ Missing required Goodreads columns: {missing_gr_cols}")
            return existing_mapping

    except Exception as e:
        print(f"❌ Error reading Goodreads CSV: {e}")
        return existing_mapping

    # Step 3: Extract reading periods
    try:
        asin_periods = extract_kindle_reading_periods(kindle_df, blacklisted_asins)
        book_periods = extract_goodreads_reading_periods(gr_df)
    except Exception as e:
        print(f"❌ Error extracting reading periods: {e}")
        return existing_mapping

    if not asin_periods:
        print("❌ No valid ASIN periods found in Kindle data")
        return existing_mapping

    if not book_periods:
        print("❌ No valid book periods found in Goodreads data")
        return existing_mapping

    # Step 4: Only process ASINs that aren't already mapped and aren't blacklisted
    new_asins = {asin: periods for asin, periods in asin_periods.items()
                 if asin not in existing_successful and asin not in blacklisted_asins}

    if new_asins:
        print(f"\n🆕 Processing {len(new_asins)} new ASINs (skipping {len(existing_successful)} already mapped)")
        try:
            new_matches = find_best_matches(new_asins, book_periods, min_overlap_threshold)
        except Exception as e:
            print(f"❌ Error finding matches: {e}")
            new_matches = {}
    else:
        all_mapped_count = len([asin for asin in asin_periods.keys() if asin in existing_successful])
        blacklisted_count = len([asin for asin in asin_periods.keys() if asin in blacklisted_asins])
        print(f"\n✅ All valid ASINs processed: {all_mapped_count} already mapped, {blacklisted_count} blacklisted")
        new_matches = {}

    # Step 5: Combine with existing mappings
    all_successful_mappings = {**existing_successful, **new_matches}

    # Step 6: Identify failed ASINs (excluding blacklisted ones)
    all_asins = set(asin_periods.keys())
    successful_asins = set(all_successful_mappings.keys())
    failed_asins = list(all_asins - successful_asins - blacklisted_asins)

    # Step 7: Prepare final results
    final_results = {
        'successful_mappings': all_successful_mappings,
        'failed_asins': failed_asins,
        'blacklisted_asins': list(blacklisted_asins) if blacklisted_asins else [],
        'summary': {
            'total_asins': len(all_asins),
            'successful': len(all_successful_mappings),
            'failed': len(failed_asins),
            'blacklisted': len(blacklisted_asins),
            'success_rate': round((len(all_successful_mappings) / len(all_asins) * 100), 1) if all_asins else 0,
            'new_mappings_added': len(new_matches),
            'min_overlap_threshold': min_overlap_threshold,
            'last_updated': datetime.now().isoformat()
        }
    }

    # Step 8: Save results
    print(f"\n💾 Saving results to: {output_json_path}")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_json_path) if os.path.dirname(output_json_path) else '.', exist_ok=True)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print("✅ JSON file saved successfully!")
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")
        return existing_mapping

    # Step 9: Print summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"📱 Total ASINs processed: {final_results['summary']['total_asins']}")
    print(f"✅ Successfully mapped: {final_results['summary']['successful']}")
    print(f"🆕 New mappings added: {final_results['summary']['new_mappings_added']}")
    print(f"❌ Failed to map: {final_results['summary']['failed']}")
    print(f"🚫 Blacklisted (excluded): {final_results['summary']['blacklisted']}")
    print(f"📈 Overall success rate: {final_results['summary']['success_rate']}%")

    return final_results


def manual_asin_mapping() -> bool:
    """
    Interactive function to manually map failed ASINs to Book IDs.

    Returns:
        True if any manual mappings were added, False otherwise
    """
    mapping_file_path = "files/work_files/kindle_work_files/asin_bookid_mapping.json"
    goodreads_csv_path = "files/processed_files/books/gr_processed.csv"

    try:
        # Load existing mapping
        if not os.path.exists(mapping_file_path):
            print("❌ No mapping file found. Run create_asin_bookid_mapping() first.")
            return False

        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        failed_asins = mapping_data.get('failed_asins', [])
        if not failed_asins:
            print("✅ No failed ASINs to map manually")
            return False

        # Load Goodreads data for book lookup
        if not os.path.exists(goodreads_csv_path):
            print(f"❌ Goodreads file not found: {goodreads_csv_path}")
            return False

        gr_df = pd.read_csv(goodreads_csv_path, sep='|', encoding='utf-8')
        available_books = gr_df[['Book Id', 'Title']].drop_duplicates()

        print(f"\n🔧 MANUAL ASIN MAPPING")
        print(f"Found {len(failed_asins)} ASINs that need manual mapping")
        print("Commands: 'skip' to skip, 'list' to see available books, 'quit' to exit")

        mappings_added = 0

        for i, asin in enumerate(failed_asins, 1):
            print(f"\n📖 ASIN {i}/{len(failed_asins)}: {asin}")

            while True:
                user_input = input("Enter Book ID (or command): ").strip()

                if user_input.lower() == 'quit':
                    print("🛑 Stopping manual mapping")
                    break
                elif user_input.lower() == 'skip':
                    print("⏭️  Skipping this ASIN")
                    break
                elif user_input.lower() == 'list':
                    print("\n📚 Available books (showing first 10):")
                    for _, row in available_books.head(10).iterrows():
                        print(f"  • ID: {row['Book Id']} | Title: {row['Title']}")
                    continue
                elif user_input.isdigit() or user_input in available_books['Book Id'].astype(str).values:
                    # Valid Book ID provided
                    book_info = available_books[available_books['Book Id'].astype(str) == user_input]
                    if not book_info.empty:
                        title = book_info.iloc[0]['Title']

                        # Add to successful mappings
                        mapping_data['successful_mappings'][asin] = {
                            'book_id': user_input,
                            'title': title,
                            'manually_mapped': True,
                            'mapped_at': datetime.now().isoformat()
                        }

                        # Remove from failed ASINs
                        mapping_data['failed_asins'].remove(asin)

                        # Update summary
                        mapping_data['summary']['successful'] += 1
                        mapping_data['summary']['failed'] -= 1
                        mapping_data['summary']['success_rate'] = round(
                            (mapping_data['summary']['successful'] / mapping_data['summary']['total_asins'] * 100), 1
                        )

                        mappings_added += 1
                        print(f"✅ Mapped {asin} → '{title}'")
                        break
                    else:
                        print(f"❌ Book ID '{user_input}' not found")
                else:
                    print("❌ Invalid input. Please enter a valid Book ID or command")

            if user_input.lower() == 'quit':
                break

        # Save updated mapping
        if mappings_added > 0:
            with open(mapping_file_path, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Added {mappings_added} manual mappings and saved to file")

        return mappings_added > 0

    except Exception as e:
        print(f"❌ Error in manual mapping: {e}")
        return False


def create_kindle_file():
    """
    Process the Kindle export and create final processed file.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Kindle data...")

    # Define file paths
    kindle_csv_path = "files/exports/kindle_exports/Kindle.Devices.ReadingSession/Kindle.Devices.ReadingSession.csv"
    mapping_file_path = "files/work_files/kindle_work_files/asin_bookid_mapping.json"
    goodreads_csv_path = "files/processed_files/books/gr_processed.csv"
    output_path = 'files/processed_files/books/kindle_processed.csv'

    try:
        # First, ensure we have the ASIN to Book ID mapping
        if not os.path.exists(mapping_file_path):
            print("📱 No ASIN mapping found, creating mapping first...")
            mapping_result = create_asin_bookid_mapping()
            if not mapping_result or not mapping_result.get('successful_mappings'):
                print("❌ Failed to create ASIN mapping")
                return False

        # Load the ASIN to Book ID mapping
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        successful_mappings = mapping_data.get('successful_mappings', {})
        if not successful_mappings:
            print("❌ No successful ASIN mappings found")
            return False

        print(f"✅ Loaded {len(successful_mappings)} ASIN to Book ID mappings")

        # Load Kindle reading sessions
        if not os.path.exists(kindle_csv_path):
            print(f"❌ Kindle file not found: {kindle_csv_path}")
            return False

        print("📱 Loading Kindle reading sessions...")
        kindle_df = pd.read_csv(kindle_csv_path)
        print(f"✅ Loaded {len(kindle_df)} Kindle reading sessions")

        # Clean and prepare timestamp columns
        timestamp_cols = ['start_timestamp', 'end_timestamp']
        for col in timestamp_cols:
            if col in kindle_df.columns:
                # Replace problematic values
                kindle_df[col] = kindle_df[col].replace(['Not Available', 'N/A', '', 'null', 'NULL'], pd.NaT)
                # Convert to datetime
                kindle_df[col] = pd.to_datetime(kindle_df[col], errors='coerce')

        # Filter to only sessions with valid timestamps and mapped ASINs
        valid_sessions = kindle_df.dropna(subset=['start_timestamp', 'end_timestamp'])
        mapped_sessions = valid_sessions[valid_sessions['ASIN'].isin(successful_mappings.keys())]

        print(f"📊 Valid sessions: {len(valid_sessions)}")
        print(f"📊 Mapped sessions: {len(mapped_sessions)}")

        if len(mapped_sessions) == 0:
            print("❌ No valid mapped sessions found")
            return False

        # Load Goodreads data to get book information
        if not os.path.exists(goodreads_csv_path):
            print(f"❌ Goodreads file not found: {goodreads_csv_path}")
            return False

        gr_df = pd.read_csv(goodreads_csv_path, sep='|', encoding='utf-8')

        # Create a Book ID to book info mapping
        book_info = {}
        for _, row in gr_df.groupby('Book Id').first().iterrows():
            book_id = str(row.name)
            book_info[book_id] = {
                'Title': row.get('Title', 'Unknown Title'),
                'Author': row.get('Author', 'Unknown Author'),
                'Number of Pages': row.get('Number of Pages', 0),
                'My Rating': row.get('My Rating', 0),
                'Average Rating': row.get('Average Rating', 0),
                'Genre': row.get('Genre', 'Unknown'),
                'Fiction_yn': row.get('Fiction_yn', 'unknown'),
                'reading_duration': row.get('reading_duration', 0)
            }

        # Process each reading session and expand to minute-by-minute records
        print("🔄 Expanding reading sessions to minute-by-minute records...")
        expanded_records = []

        for _, session in mapped_sessions.iterrows():
            asin = session['ASIN']
            mapping_info = successful_mappings[asin]
            book_id = mapping_info['book_id']

            # Get book information
            book_data = book_info.get(book_id, {})

            start_time = session['start_timestamp']
            end_time = session['end_timestamp']

            # Round start and end times down to the minute (ignore seconds)
            start_minute = start_time.replace(second=0, microsecond=0)
            end_minute = end_time.replace(second=0, microsecond=0)

            # If start and end are in the same minute, move end to next minute
            if start_minute == end_minute:
                end_minute = end_minute + timedelta(minutes=1)

            # Calculate total minutes in this session
            total_minutes = int((end_minute - start_minute).total_seconds() / 60)

            if total_minutes <= 0:
                continue

            # Get page flips for this session (this represents actual reading progress)
            page_flips = session.get('number_of_page_flips', 0)

            # If no page flips recorded, estimate based on time (very conservative)
            if page_flips <= 0:
                # Assume 1 page per 2 minutes as fallback
                page_flips = max(1, total_minutes / 2)

            # Calculate pages per minute - distribute evenly across all minutes
            pages_per_minute = page_flips / total_minutes

            # Create one record for each minute in the session
            current_minute = start_minute

            for minute_num in range(total_minutes):
                record = {
                    'ASIN': asin,
                    'Book Id': book_id,
                    'Title': book_data.get('Title', 'Unknown Title'),
                    'Author': book_data.get('Author', 'Unknown Author'),
                    'Timestamp': current_minute,
                    'Seconds': 60.0,  # Always 60 seconds per minute record
                    'page_split': pages_per_minute,  # Uniform distribution
                    'Number of Pages': book_data.get('Number of Pages', 0),
                    'My Rating': book_data.get('My Rating', 0),
                    'Average Rating': book_data.get('Average Rating', 0),
                    'Genre': book_data.get('Genre', 'Unknown'),
                    'Fiction_yn': book_data.get('Fiction_yn', 'unknown'),
                    'reading_duration': book_data.get('reading_duration', 0),
                    'Source': 'Kindle'
                }

                expanded_records.append(record)
                current_minute += timedelta(minutes=1)

        if not expanded_records:
            print("❌ No expanded records created")
            return False

        # Create final DataFrame
        final_df = pd.DataFrame(expanded_records)

        # Sort by timestamp (most recent first)
        final_df = final_df.sort_values('Timestamp', ascending=False)

        print(f"📊 Created {len(final_df)} minute-by-minute reading records")
        print(f"📚 Covering {final_df['Book Id'].nunique()} unique books")

        # Save processed file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Saved processed Kindle data to: {output_path}")

        # Print summary statistics
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"📱 Total minute records: {len(final_df):,}")
        print(f"📚 Books covered: {final_df['Book Id'].nunique()}")
        print(f"📅 Date range: {final_df['Timestamp'].min().date()} to {final_df['Timestamp'].max().date()}")
        print(f"⏱️  Total reading time: {final_df['Seconds'].sum() / 3600:.1f} hours")
        print(f"📖 Total pages: {final_df['page_split'].sum():.0f}")

        return True

    except Exception as e:
        print(f"❌ Error processing Kindle data: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_kindle_results():
    """
    Uploads the processed Kindle files to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading Kindle results to Google Drive...")

    files_to_upload = [
        'files/processed_files/books/kindle_processed.csv',
        'files/work_files/kindle_work_files/asin_bookid_mapping.json'
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Kindle results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


def process_kindle_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_kindle_pipeline(auto_full=True)
    else:
        return create_kindle_file()


def full_kindle_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Kindle pipeline with multiple options.

    Options:
    1. Full pipeline (download → move → map → process → upload)
    2. Process existing data and upload to Drive
    3. Upload existing processed files to Drive
    4. Create ASIN mapping only (timestamp-based mapping)
    5. Manual ASIN mapping (interactive mapping of failed ASINs)

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("📱 KINDLE DATA PIPELINE")
    print("="*60)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data and uploading...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → move → map → process → upload)")
        print("2. Process existing data and upload to Drive")
        print("3. Upload existing processed files to Drive")
        print("4. Create ASIN mapping only (timestamp-based mapping)")
        print("5. Manual ASIN mapping (interactive mapping of failed ASINs)")

        choice = input("\nEnter your choice (1-5): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full Kindle pipeline...")

        # Step 1: Download
        download_success = download_kindle_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_kindle_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_kindle_files()

        # Step 3: Create ASIN mapping
        if move_success or os.path.exists("files/exports/kindle_exports/Kindle.Devices.ReadingSession/Kindle.Devices.ReadingSession.csv"):
            mapping_success = create_asin_bookid_mapping()
            if not mapping_success:
                print("❌ ASIN mapping failed, stopping pipeline")
                return False
        else:
            print("❌ No Kindle files found, stopping pipeline")
            return False

        # Step 4: Process data
        process_success = create_kindle_file()
        if not process_success:
            print("❌ Processing failed, stopping pipeline")
            return False

        # Step 5: Upload
        upload_success = upload_kindle_results()
        success = upload_success

    elif choice == "2":
        print("\n⚙️  Processing existing data and uploading to Drive...")
        process_success = create_kindle_file()
        if process_success:
            success = upload_kindle_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "3":
        print("\n⬆️  Uploading existing processed files to Drive...")
        success = upload_kindle_results()

    elif choice == "4":
        print("\n🔗 Creating ASIN to Book ID mapping...")
        mapping_result = create_asin_bookid_mapping()
        success = bool(mapping_result and mapping_result.get('successful_mappings'))

    elif choice == "5":
        print("\n🔧 Manual ASIN mapping...")
        success = manual_asin_mapping()

    else:
        print("❌ Invalid choice. Please select 1-5.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Kindle pipeline completed successfully!")
        # Record successful run
        record_successful_run('books_kindle', 'active')
    else:
        print("❌ Kindle pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("📱 Kindle Processing Tool")
    print("This tool helps you download, process, and upload Kindle reading data.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_kindle_pipeline(auto_full=False)
