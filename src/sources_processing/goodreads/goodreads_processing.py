import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
# Drive operations not needed - source processor doesn't upload
from src.utils.utils_functions import record_successful_run

# Fiction genres classification
fiction_genres = ['drama', 'horror', 'thriller', 'classics', 'science-fiction']


def load_reading_dates_json():
    """Load reading dates from JSON file"""
    json_path = 'files/work_files/gr_work_files/reading_dates.json'

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                dates_data = json.load(f)
            print(f"📚 Loaded {len(dates_data)} book dates from JSON")
            return dates_data
        except Exception as e:
            print(f"⚠️  Error loading JSON file: {e}")
            return {}
    else:
        print("📚 No existing reading dates JSON found - creating new one")
        return {}


def save_reading_dates_json(dates_data):
    """Save reading dates to JSON file"""
    json_path = 'files/work_files/gr_work_files/reading_dates.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    try:
        with open(json_path, 'w') as f:
            json.dump(dates_data, f, indent=2, default=str)
        print(f"💾 Saved {len(dates_data)} book dates to JSON")
        return True
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")
        return False


def migrate_excel_to_json():
    """One-time migration from Excel file to JSON format"""
    print("🔄 Checking for Excel to JSON migration...")

    excel_path = 'files/work_files/gr_work_files/gr_dates_input.xlsx'
    json_path = 'files/work_files/gr_work_files/reading_dates.json'

    # If JSON already exists, skip migration
    if os.path.exists(json_path):
        print("✅ JSON file already exists, skipping migration")
        return True

    # If Excel doesn't exist, nothing to migrate
    if not os.path.exists(excel_path):
        print("ℹ️  No Excel file found to migrate")
        return True

    try:
        print("📊 Migrating Excel data to JSON...")
        df_excel = pd.read_excel(excel_path)

        dates_data = {}
        migrated_count = 0

        for _, row in df_excel.iterrows():
            book_id = str(row.get('Book Id', ''))
            title = row.get('Title', '')

            if book_id and book_id != 'nan' and book_id != '0':
                dates_data[book_id] = {
                    'title': title,
                    'date_started': row.get('Date started', ''),
                    'date_ended': row.get('Date ended', ''),
                    'cover_url': row.get('cover_url', ''),
                    'check_status': row.get('Check', ''),
                    'migrated_from_excel': True
                }
                migrated_count += 1

        if save_reading_dates_json(dates_data):
            print(f"✅ Successfully migrated {migrated_count} books from Excel to JSON")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False


def download_goodreads_data():
    """Opens Goodreads export page and prompts user to download data"""
    print("📚 Starting Goodreads data download...")

    urls = ['https://www.goodreads.com/review/import']
    open_web_urls(urls)

    print("📝 Instructions:")
    print("   1. Click 'Export Library'")
    print("   2. Wait for the export to be prepared")
    print("   3. Download the CSV file when ready")
    print("   4. The file will be named 'goodreads_library_export.csv'")

    response = prompt_user_download_status("Goodreads")
    return response


def move_goodreads_files():
    """Moves the downloaded Goodreads file from Downloads to the correct export folder"""
    print("📁 Moving Goodreads files...")

    success = clean_rename_move_file(
        export_folder="files/exports/goodreads_exports",
        download_folder="/Users/valen/Downloads",
        file_name="goodreads_library_export.csv",
        new_file_name="gr_export.csv"
    )

    if success:
        print("✅ Successfully moved Goodreads export to exports folder")
    else:
        print("❌ Failed to move Goodreads files")

    return success


def extract_detailed_reading_dates_and_cover(driver, view_link, book_title):
    """Click on a view link and extract Start/End reading dates AND book cover URL"""
    try:
        print(f"    🔍 Extracting data for '{book_title}'...")

        # Handle potential overlays/banners that might intercept clicks
        try:
            overlay_selectors = [
                ".siteHeader__topFullImage",
                "[class*='banner']",
                "[class*='overlay']",
                "[class*='modal']"
            ]
            for selector in overlay_selectors:
                try:
                    overlay = driver.find_element(By.CSS_SELECTOR, selector)
                    if overlay.is_displayed():
                        driver.execute_script("arguments[0].style.display = 'none';", overlay)
                except:
                    pass
        except:
            pass

        # Try multiple click methods
        click_successful = False

        # Method 1: Scroll and regular click
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", view_link)
            time.sleep(1)
            view_link.click()
            click_successful = True
        except Exception as e1:
            # Method 2: JavaScript click
            try:
                driver.execute_script("arguments[0].click();", view_link)
                click_successful = True
            except Exception as e2:
                # Method 3: Navigate directly to the URL
                try:
                    view_url = view_link.get_attribute('href')
                    if view_url:
                        driver.get(view_url)
                        click_successful = True
                except Exception as e3:
                    pass

        if not click_successful:
            print(f"    ❌ Could not access detailed view for '{book_title}'")
            return {}

        # Wait for the detailed view to load
        time.sleep(3)

        result = {}

        # Extract reading dates
        try:
            # Look for reading progress elements
            progress_selectors = [
                "//h3[contains(text(), 'READING PROGRESS')]/following-sibling::ul",
                "//h3[contains(text(), 'READING PROGRESS')]/following-sibling::div",
                "//*[contains(text(), 'READING PROGRESS')]/following-sibling::*",
                "//div[contains(@class, 'readingProgress')]",
                "//*[contains(text(), 'Started Reading') or contains(text(), 'Finished Reading')]"
            ]

            progress_items = []
            for selector in progress_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        for element in elements:
                            progress_items.extend(element.find_elements(By.XPATH, ".//*"))
                        break
                except:
                    continue

            # If we couldn't find a section, search the whole page
            if not progress_items:
                progress_items = driver.find_elements(By.XPATH, "//*[contains(text(), 'Started Reading') or contains(text(), 'Finished Reading')]")

            # Parse the progress items for dates
            for item in progress_items:
                try:
                    text = item.text.strip()

                    if "Started Reading" in text:
                        date_part = text.replace("Started Reading", "").replace("–", "").strip()
                        if date_part:
                            result['date_started'] = date_part
                            print(f"    📅 Found start date: {date_part}")

                    elif "Finished Reading" in text:
                        date_part = text.replace("Finished Reading", "").replace("–", "").strip()
                        if date_part:
                            result['date_ended'] = date_part
                            print(f"    📅 Found end date: {date_part}")

                except Exception:
                    continue

            # Fallback: search entire page text for dates
            if not result:
                try:
                    all_text = driver.find_element(By.TAG_NAME, "body").text
                    lines = all_text.split('\n')

                    for line in lines:
                        line = line.strip()
                        if "Started Reading" in line:
                            date_part = line.replace("Started Reading", "").replace("–", "").strip()
                            if date_part:
                                result['date_started'] = date_part
                        elif "Finished Reading" in line:
                            date_part = line.replace("Finished Reading", "").replace("–", "").strip()
                            if date_part:
                                result['date_ended'] = date_part
                except:
                    pass

        except Exception as e:
            print(f"    ⚠️  Error extracting dates: {e}")

        # Extract book cover URL
        try:
            cover_selectors = [
                "img[id*='coverImage']",
                ".BookCover img",
                ".bookCover img",
                ".leftContainer img",
                "img[src*='book']",
                ".editionCover img"
            ]

            cover_url = None
            for selector in cover_selectors:
                try:
                    cover_element = driver.find_element(By.CSS_SELECTOR, selector)
                    cover_url = cover_element.get_attribute('src')

                    # Validate that it's actually a book cover URL
                    if cover_url and ('book' in cover_url.lower() or 'cover' in cover_url.lower()):
                        # Try to get higher resolution version
                        if '_SX' in cover_url or '_SY' in cover_url:
                            cover_url = cover_url.replace('_SX50_', '_SX500_').replace('_SY75_', '_SY500_')

                        result['cover_url'] = cover_url
                        print(f"    🖼️  Found cover URL")
                        break
                except:
                    continue

        except Exception as e:
            print(f"    ⚠️  Error extracting cover: {e}")

        return result

    except Exception as e:
        print(f"    ❌ Error in detailed extraction: {e}")
        return {}


def scrape_missing_reading_dates(books_needing_dates):
    """Scrape reading dates for books that don't have them in the JSON file"""
    if not books_needing_dates:
        print("✅ No books need date scraping")
        return {}

    print(f"🤖 Starting scraping for {len(books_needing_dates)} books...")

    # Setup Chrome with persistent session
    chrome_options = Options()
    chrome_options.add_argument("--user-data-dir=/Users/valen/chrome_goodreads_profile")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Add headless mode option - comment out if you want to see the browser
    # chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options)
    scraped_data = {}

    try:
        # Navigate to reading list
        print("🔍 Opening Goodreads reading list...")
        driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")
        time.sleep(5)

        # Check if we need to log in
        if "Sign in" in driver.title or "sign_in" in driver.current_url:
            print("🔐 Please log into Goodreads...")
            print("⏳ Waiting for login... (you have 60 seconds)")

            start_time = time.time()
            while time.time() - start_time < 60:
                if "Sign in" not in driver.title and "sign_in" not in driver.current_url:
                    print("✅ Login detected, continuing...")
                    break
                time.sleep(2)
            else:
                print("❌ Login timeout - continuing anyway")

        books_found = 0
        processed_books = set()  # Track which books we've already processed to avoid duplicates

        while books_found < len(books_needing_dates):
            # Find all book rows on current page
            book_rows = driver.find_elements(By.CSS_SELECTOR, "tr[id*='review_'], tr.bookalike")
            if not book_rows:
                book_rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
                book_rows = [row for row in book_rows if row.find_elements(By.CSS_SELECTOR, "td")]

            print(f"Found {len(book_rows)} total book rows on page")

            # Debug: Show what we're looking for
            if books_found < len(books_needing_dates):
                print(f"🔍 Looking for: {list(books_needing_dates.values())[0]['title']}")

            # Collect all titles on the page for debugging
            page_titles = []

            books_found_this_pass = 0
            for i, row in enumerate(book_rows):
                try:
                    # Extract book title and find matching book
                    title_element = None
                    title_selectors = ["a[class*='bookTitle']", ".title a", "td.title a", "a[href*='/book/show/']"]

                    for selector in title_selectors:
                        try:
                            title_element = row.find_element(By.CSS_SELECTOR, selector)
                            break
                        except:
                            continue

                    if not title_element:
                        continue

                    title = title_element.text.strip()
                    page_titles.append(title)

                    # Check if this is one of the books we need to scrape
                    matching_book_id = None
                    for book_id, book_info in books_needing_dates.items():
                        expected_title = book_info['title'].lower().strip()
                        found_title = title.lower().strip()

                        # Try exact match first
                        if expected_title == found_title:
                            matching_book_id = book_id
                            break

                        # Try fuzzy match: check if one title contains the main part of the other
                        # Remove common suffixes that might differ
                        expected_clean = expected_title.split('(')[0].strip()
                        found_clean = found_title.split('(')[0].strip()

                        if expected_clean in found_title or found_clean in expected_title:
                            print(f"   🔍 Fuzzy match: '{found_title}' ≈ '{expected_title}'")
                            matching_book_id = book_id
                            break

                    if not matching_book_id or matching_book_id in processed_books:
                        continue

                    print(f"📖 Found book needing dates: '{title}' (ID: {matching_book_id})")

                    # Look for "view" link in this row
                    view_link = None
                    try:
                        view_link = row.find_element(By.CSS_SELECTOR, "a[href*='#review_']")
                    except:
                        try:
                            view_link = row.find_element(By.XPATH, ".//a[contains(text(), 'view')]")
                        except:
                            pass

                    if view_link:
                        # Extract detailed data
                        scraped_info = extract_detailed_reading_dates_and_cover(driver, view_link, title)

                        if scraped_info:
                            scraped_data[matching_book_id] = {
                                'title': title,
                                'date_started': scraped_info.get('date_started', ''),
                                'date_ended': scraped_info.get('date_ended', ''),
                                'cover_url': scraped_info.get('cover_url', ''),
                                'scraped_at': datetime.now().isoformat()
                            }
                            books_found += 1
                            books_found_this_pass += 1
                            processed_books.add(matching_book_id)
                            print(f"  ✅ Scraped data for '{title}'")
                        else:
                            print(f"  ❌ Could not extract data for '{title}'")
                            processed_books.add(matching_book_id)  # Mark as processed even if failed

                        # Return to main list
                        time.sleep(2)
                        driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")
                        time.sleep(2)

                        # Break from current row loop to re-find book rows
                        break

                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    continue

            # If we didn't find any new books this pass, break to avoid infinite loop
            if books_found_this_pass == 0:
                print(f"⚠️  No new books found in this pass. Stopping with {books_found} books processed.")
                # Debug: Show what titles were found on this page
                if page_titles:
                    print(f"📚 Titles found on page (first 10):")
                    for i, t in enumerate(page_titles[:10], 1):
                        print(f"   {i}. {t}")
                break

            # Stop if we found all books we were looking for
            if books_found >= len(books_needing_dates):
                print("✅ Found all books we were looking for!")
                break

        print(f"🎯 Successfully scraped {len(scraped_data)} out of {len(books_needing_dates)} books")
        return scraped_data

    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return {}

    finally:
        driver.quit()


def expand_gr_reading_split(row, columns, col):
    """Splits the rows to have one row per day, with page division"""
    # Handle missing dates
    if pd.isna(row.get('Date started')) or pd.isna(row.get('Date ended')):
        date_df = pd.DataFrame(columns=col)
        new_row = {}
        for column in columns:
            if column not in ['Date started', 'Date ended']:
                new_row[column] = row.get(column, None)
        new_row['Timestamp'] = row.get('Date started', pd.NaT)
        new_row['page_split'] = row.get('Number of Pages', 0)

        date_df = pd.concat([date_df, pd.DataFrame([new_row])], ignore_index=True)
        return date_df

    # Create date range
    try:
        dates = pd.date_range(row['Date started'], row['Date ended'], freq='D')
        date_df = pd.DataFrame({'Timestamp': dates})

        # Add all other columns except dates
        for column in columns:
            if column not in ['Date started', 'Date ended']:
                date_df[column] = row.get(column, None)

        # Calculate page split
        num_pages = row.get('Number of Pages', 0)
        if pd.notna(num_pages) and num_pages > 0:
            date_df['page_split'] = num_pages / len(dates)
        else:
            date_df['page_split'] = 0

        return date_df
    except Exception as e:
        print(f"    ⚠️  Error expanding dates for book: {e}")
        # Return single row with available data
        date_df = pd.DataFrame(columns=col)
        new_row = {}
        for column in columns:
            if column not in ['Date started', 'Date ended']:
                new_row[column] = row.get(column, None)
        new_row['Timestamp'] = row.get('Date started', pd.NaT)
        new_row['page_split'] = row.get('Number of Pages', 0)

        date_df = pd.concat([date_df, pd.DataFrame([new_row])], ignore_index=True)
        return date_df


def create_goodreads_file():
    """Main processing logic using JSON-based reading dates"""
    print("⚙️  Processing Goodreads data with JSON-based dates...")

    try:
        # Load Goodreads export
        df = pd.read_csv("files/exports/goodreads_exports/gr_export.csv")
        print(f"📊 Loaded {len(df)} books from Goodreads export")

        # Load existing reading dates from JSON
        dates_data = load_reading_dates_json()

        # Filter to only 'read' books
        read_books = df[df['Exclusive Shelf'] == 'read'].copy()
        print(f"📚 Found {len(read_books)} books marked as 'read'")

        # Identify books that need date scraping
        books_needing_dates = {}
        books_with_dates = {}

        for _, book in read_books.iterrows():
            book_id = str(book['Book Id'])
            title = str(book['Title']).strip()

            if book_id in dates_data and dates_data[book_id].get('date_started') and dates_data[book_id].get('date_ended'):
                # We have complete data for this book
                books_with_dates[book_id] = dates_data[book_id]
            else:
                # We need to scrape dates for this book
                books_needing_dates[book_id] = {
                    'title': title,
                    'book_data': book
                }

        print(f"✅ {len(books_with_dates)} books already have dates")
        print(f"🔍 {len(books_needing_dates)} books need date scraping")

        # Scrape missing dates if needed
        if books_needing_dates:
            # Print the list of books that need scraping
            print("\n📋 Books that need date scraping:")
            for i, (book_id, book_info) in enumerate(books_needing_dates.items(), 1):
                print(f"   {i}. {book_info['title']} (ID: {book_id})")
            print()  # Add blank line for readability

            print(f"🚀 Starting automatic scraping for {len(books_needing_dates)} books...")
            scraped_data = scrape_missing_reading_dates(books_needing_dates)

            # Merge scraped data into main dates_data
            for book_id, scraped_info in scraped_data.items():
                dates_data[book_id] = scraped_info
                books_with_dates[book_id] = scraped_info

            # Save updated JSON
            save_reading_dates_json(dates_data)
            print(f"✅ Updated JSON with {len(scraped_data)} newly scraped books")

        # Now process all books with available dates
        processed_books = []

        for _, book in read_books.iterrows():
            book_id = str(book['Book Id'])

            if book_id in books_with_dates:
                # Add date information to book data
                date_info = books_with_dates[book_id]

                book_with_dates = book.copy()
                book_with_dates['Date started'] = pd.to_datetime(date_info.get('date_started'), errors='coerce')
                book_with_dates['Date ended'] = pd.to_datetime(date_info.get('date_ended'), errors='coerce')
                book_with_dates['cover_url'] = date_info.get('cover_url', '')

                processed_books.append(book_with_dates)
            else:
                # No dates available, add with NaT dates
                book_without_dates = book.copy()
                book_without_dates['Date started'] = pd.NaT
                book_without_dates['Date ended'] = pd.NaT
                book_without_dates['cover_url'] = ''

                processed_books.append(book_without_dates)

        # Convert to DataFrame
        df_with_dates = pd.DataFrame(processed_books)

        # Calculate reading duration and fiction classification
        df_with_dates['reading_duration'] = (df_with_dates['Date ended'] - df_with_dates['Date started']).dt.days + 1
        df_with_dates['Fiction_yn'] = df_with_dates['Bookshelves'].apply(
            lambda x: "fiction" if str(x).lower() in fiction_genres else "non-fiction"
        )

        # Prepare for expansion
        base_columns = ['Book Id', 'Title', 'Author', 'Original Publication Year', 'My Rating',
                       'Average Rating', 'Bookshelves', 'Fiction_yn', 'reading_duration',
                       'Number of Pages', 'Date started', 'Date ended', 'cover_url']

        # Only keep existing columns
        existing_columns = [col for col in base_columns if col in df_with_dates.columns]
        df_limited = df_with_dates[existing_columns]

        # Prepare final columns
        final_columns = [col for col in existing_columns if col not in ['Date started', 'Date ended']]
        final_columns.extend(['Timestamp', 'page_split'])

        # Expand each book into daily reading records
        expanded_df = pd.DataFrame(columns=final_columns)

        print(f"📖 Expanding {len(df_limited)} books into daily reading records...")

        for _, row in df_limited.iterrows():
            try:
                expanded_row = expand_gr_reading_split(row, existing_columns, final_columns)
                expanded_df = pd.concat([expanded_df, expanded_row], ignore_index=True)
            except Exception as e:
                print(f"⚠️  Error processing book '{row.get('Title', 'Unknown')}': {e}")
                continue

        # Add final required columns
        expanded_df['Seconds'] = np.nan
        expanded_df['Source'] = 'GoodReads'

        # Clean up genre column name
        if 'Bookshelves' in expanded_df.columns:
            expanded_df = expanded_df.rename(columns={'Bookshelves': 'Genre'})

        # Clean up title column
        if 'Title' in expanded_df.columns:
            expanded_df['Title'] = expanded_df['Title'].apply(lambda x: str(x).strip())

        # Sort the df by dates
        expanded_df = expanded_df.sort_values(by="Timestamp", ascending=False)

        # Save processed file to NEW location
        output_path = 'files/source_processed_files/goodreads/goodreads_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        expanded_df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        print(f"✅ Successfully processed {len(expanded_df)} reading records")
        print(f"📊 Final columns: {list(expanded_df.columns)}")
        print(f"📚 Books with complete dates: {len(books_with_dates)}")
        print(f"📈 Books processed: {len(processed_books)}")

        return True

    except Exception as e:
        print(f"❌ Error processing Goodreads data: {e}")
        import traceback
        traceback.print_exc()
        return False




def process_goodreads_export(upload="Y"):
    """Legacy function for backward compatibility"""
    if upload == "Y":
        return full_goodreads_pipeline(auto_full=True)
    else:
        return create_goodreads_file()


def full_goodreads_pipeline(auto_full=False, auto_process_only=False):
    """Complete Goodreads pipeline with JSON-based processing (SOURCE PROCESSOR - NO UPLOAD)"""
    print("\n" + "="*60)
    print("📚 GOODREADS SOURCE PROCESSOR (JSON-BASED)")
    print("="*60)

    # First, migrate any existing Excel data to JSON
    migrate_excel_to_json()

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download and process new data")
        print("2. Process existing data")
        print("3. Scrape dates only (for books missing dates)")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting Goodreads pipeline...")

        # Step 1: Download
        download_success = download_goodreads_data()

        # Step 2: Move files
        if download_success:
            move_success = move_goodreads_files()
        else:
            print("⚠️  Download not confirmed, checking for existing files...")
            move_success = move_goodreads_files()

        # Step 3: Process (includes scraping if needed)
        if move_success or os.path.exists("files/exports/goodreads_exports/gr_export.csv"):
            success = create_goodreads_file()
        else:
            print("❌ No Goodreads export file found")
            success = False

    elif choice == "2":
        print("\n⚙️  Processing existing data...")
        success = create_goodreads_file()

    elif choice == "3":
        print("\n🔍 Scraping dates for books missing dates...")

        # Load current data to identify missing dates
        try:
            df = pd.read_csv("files/exports/goodreads_exports/gr_export.csv")
            dates_data = load_reading_dates_json()
            read_books = df[df['Exclusive Shelf'] == 'read']

            books_needing_dates = {}
            for _, book in read_books.iterrows():
                book_id = str(book['Book Id'])
                title = str(book['Title']).strip()

                if book_id not in dates_data or not dates_data[book_id].get('date_started'):
                    books_needing_dates[book_id] = {
                        'title': title,
                        'book_data': book
                    }

            if books_needing_dates:
                # Print the list of books that need scraping
                print("\n📋 Books that need date scraping:")
                for i, (book_id, book_info) in enumerate(books_needing_dates.items(), 1):
                    print(f"   {i}. {book_info['title']} (ID: {book_id})")
                print()  # Add blank line for readability

                scraped_data = scrape_missing_reading_dates(books_needing_dates)

                # Update JSON with scraped data
                for book_id, scraped_info in scraped_data.items():
                    dates_data[book_id] = scraped_info

                success = save_reading_dates_json(dates_data)
                if success:
                    print(f"✅ Updated {len(scraped_data)} books with scraped dates")
            else:
                print("✅ No books need date scraping")
                success = True

        except Exception as e:
            print(f"❌ Error during scraping: {e}")
            success = False

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Goodreads source processor completed successfully!")
        # Record successful run with new tracking name
        record_successful_run('source_goodreads', 'active')
    else:
        print("❌ Goodreads source processor failed")
    print("="*60)

    return success



if __name__ == "__main__":
    # Allow running this file directly
    print("📚 Goodreads Source Processing Tool")
    print("This tool downloads, scrapes, and processes Goodreads data.")
    print("Note: Upload is handled by the Reading topic coordinator")

    # Run the pipeline
    full_goodreads_pipeline(auto_full=False)
