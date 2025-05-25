# Enhanced goodreads_processing.py with automated scraping pipeline
import pandas as pd
import numpy as np
import requests
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection

# Fiction genres classification
fiction_genres = ['drama', 'horror', 'thriller']


def load_existing_books():
    """Load books that are already processed to avoid duplicates during scraping"""
    try:
        # Check if the processed file exists
        if os.path.exists('files/processed_files/books/gr_processed.csv'):
            df = pd.read_csv('files/processed_files/books/gr_processed.csv', sep='|')
            existing_titles = set(df['Title'].str.strip().str.lower())
            print(f"📚 Found {len(existing_titles)} existing books in processed file")
            return existing_titles
        else:
            print("📚 No existing processed file found - will process all books")
            return set()
    except Exception as e:
        print(f"⚠️  Error reading existing books: {e}")
        return set()


def download_goodreads_data():
    """
    Opens Goodreads export page and prompts user to download data.
    Returns True if user confirms download, False otherwise.
    """
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
    """
    Moves the downloaded Goodreads file from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
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
    """
    Click on a view link and extract Start/End reading dates AND book cover URL from the detailed view
    """
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
            print(f"    ✅ Navigation successful")
        except Exception as e1:
            # Method 2: JavaScript click
            try:
                driver.execute_script("arguments[0].click();", view_link)
                click_successful = True
                print(f"    ✅ JavaScript click successful")
            except Exception as e2:
                # Method 3: Navigate directly to the URL
                try:
                    view_url = view_link.get_attribute('href')
                    if view_url:
                        driver.get(view_url)
                        click_successful = True
                        print(f"    ✅ Direct navigation successful")
                except Exception as e3:
                    print(f"    ❌ All navigation methods failed")

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
                            result['start_date'] = date_part
                            print(f"    📅 Found start date: {date_part}")

                    elif "Finished Reading" in text:
                        date_part = text.replace("Finished Reading", "").replace("–", "").strip()
                        if date_part:
                            result['end_date'] = date_part
                            print(f"    📅 Found end date: {date_part}")

                except Exception as e:
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
                                result['start_date'] = date_part
                        elif "Finished Reading" in line:
                            date_part = line.replace("Finished Reading", "").replace("–", "").strip()
                            if date_part:
                                result['end_date'] = date_part
                except:
                    pass

        except Exception as e:
            print(f"    ⚠️  Error extracting dates: {e}")

        # Extract book cover URL
        try:
            # Look for book cover image
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
                            # Replace with higher resolution
                            cover_url = cover_url.replace('_SX50_', '_SX500_').replace('_SY75_', '_SY500_')

                        result['cover_url'] = cover_url
                        print(f"    🖼️  Found cover URL: {cover_url[:60]}...")
                        break
                except:
                    continue

        except Exception as e:
            print(f"    ⚠️  Error extracting cover: {e}")

        return result

    except Exception as e:
        print(f"    ❌ Error in detailed extraction: {e}")
        return {}


def scrape_goodreads_reading_data():
    """
    Automated scraping of Goodreads reading dates and book covers
    Returns a list of books with extracted data
    """
    print("🤖 Starting automated Goodreads scraping...")

    # Load existing books to avoid duplicates
    existing_books = load_existing_books()

    # Setup Chrome with persistent session
    chrome_options = Options()
    chrome_options.add_argument("--user-data-dir=/Users/valen/chrome_goodreads_profile")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Add headless mode option - comment out if you want to see the browser
    # chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to reading list
        print("🔍 Opening Goodreads reading list...")
        driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")

        # Wait a bit for page to load
        time.sleep(5)

        # Check if we need to log in
        if "Sign in" in driver.title or "sign_in" in driver.current_url:
            print("🔐 Please log into Goodreads...")
            print("⏳ Waiting for login... (you have 60 seconds)")

            # Wait for user to log in (max 60 seconds)
            start_time = time.time()
            while time.time() - start_time < 60:
                if "Sign in" not in driver.title and "sign_in" not in driver.current_url:
                    print("✅ Login detected, continuing...")
                    break
                time.sleep(2)
            else:
                print("❌ Login timeout - continuing anyway")

        print("🎯 Looking for books that need data extraction...")

        # Find all book rows
        book_rows = driver.find_elements(By.CSS_SELECTOR, "tr[id*='review_'], tr.bookalike")
        if not book_rows:
            book_rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
            book_rows = [row for row in book_rows if row.find_elements(By.CSS_SELECTOR, "td")]

        print(f"Found {len(book_rows)} total book rows")

        extracted_books = []

        for i, row in enumerate(book_rows[:20]):  # Limit to first 20 books for testing
            try:
                # Extract book title
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
                book_url = title_element.get_attribute('href')

                # Check if this book is already processed
                if title.lower() in existing_books:
                    print(f"⏭️  Skipping '{title}' (already processed)")
                    continue

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
                    print(f"📖 Processing: '{title}'...")

                    # Extract detailed data (dates + cover)
                    book_data = extract_detailed_reading_dates_and_cover(driver, view_link, title)

                    if book_data:
                        book_data['title'] = title
                        book_data['book_url'] = book_url
                        extracted_books.append(book_data)
                        print(f"  ✅ Extracted data for '{title}'")
                    else:
                        print(f"  ❌ Could not extract data for '{title}'")
                else:
                    print(f"📖 '{title}' - no 'view' link found")

                # Add delay and return to main list
                time.sleep(2)
                driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")
                time.sleep(2)

            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue

        return extracted_books

    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return []

    finally:
        driver.quit()


def update_excel_with_scraped_data(scraped_books):
    """
    Update the Excel file with scraped reading dates and cover URLs
    """
    print("📝 Updating Excel file with scraped data...")

    excel_path = 'files/work_files/gr_work_files/gr_dates_input.xlsx'

    try:
        # Load existing Excel file
        if os.path.exists(excel_path):
            df_dates = pd.read_excel(excel_path)
        else:
            # Create new DataFrame if file doesn't exist
            df_dates = pd.DataFrame(columns=['Book Id', 'Title', 'Date started', 'Date ended', 'Check', 'cover_url'])

        # Load the main Goodreads export to get Book IDs
        df_gr = pd.read_csv("files/exports/goodreads_exports/gr_export.csv")

        updates_made = 0

        for book in scraped_books:
            title = book['title']

            # Find Book ID from main export
            matching_books = df_gr[df_gr['Title'].str.lower() == title.lower()]
            if matching_books.empty:
                print(f"⚠️  Could not find Book ID for '{title}' in main export")
                continue

            book_id = matching_books.iloc[0]['Book Id']

            # Check if this book is already in Excel file
            existing_row = df_dates[df_dates['Book Id'] == book_id]

            if existing_row.empty:
                # Add new row
                new_row = {
                    'Book Id': book_id,
                    'Title': title,
                    'Date started': pd.to_datetime(book.get('start_date'), errors='coerce') if book.get('start_date') else None,
                    'Date ended': pd.to_datetime(book.get('end_date'), errors='coerce') if book.get('end_date') else None,
                    'Check': 'OK' if (book.get('start_date') and book.get('end_date')) else 'Partial',
                    'cover_url': book.get('cover_url', '')
                }

                df_dates = pd.concat([df_dates, pd.DataFrame([new_row])], ignore_index=True)
                updates_made += 1
                print(f"  ✅ Added new entry for '{title}'")

            else:
                # Update existing row
                row_index = existing_row.index[0]
                updated = False

                if book.get('start_date') and pd.isna(df_dates.loc[row_index, 'Date started']):
                    df_dates.loc[row_index, 'Date started'] = pd.to_datetime(book.get('start_date'), errors='coerce')
                    updated = True

                if book.get('end_date') and pd.isna(df_dates.loc[row_index, 'Date ended']):
                    df_dates.loc[row_index, 'Date ended'] = pd.to_datetime(book.get('end_date'), errors='coerce')
                    updated = True

                if book.get('cover_url') and not df_dates.loc[row_index, 'cover_url']:
                    df_dates.loc[row_index, 'cover_url'] = book.get('cover_url', '')
                    updated = True

                if updated:
                    df_dates.loc[row_index, 'Check'] = 'OK'
                    updates_made += 1
                    print(f"  ✅ Updated existing entry for '{title}'")

        # Save updated Excel file
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        df_dates.to_excel(excel_path, index=False)
        print(f"💾 Updated Excel file with {updates_made} changes")

        return updates_made > 0

    except Exception as e:
        print(f"❌ Error updating Excel file: {e}")
        return False


def add_dates_read_from_excel():
    """Load Goodreads export and Excel files - enhanced with cover URLs"""
    df = pd.read_csv("files/exports/goodreads_exports/gr_export.csv")

    excel_path = 'files/work_files/gr_work_files/gr_dates_input.xlsx'
    if os.path.exists(excel_path):
        df2 = pd.read_excel(excel_path)
    else:
        print("⚠️  No Excel file found - creating empty one")
        df2 = pd.DataFrame(columns=['Book Id', 'Title', 'Date started', 'Date ended', 'Check', 'cover_url'])

    # Fill missing Book IDs
    df2["Book Id"].fillna(0, inplace=True)

    # Merge with main export
    d = pd.merge(df, df2, on="Book Id", how='left')
    d = d[(d['Exclusive Shelf'] == 'read') & (d['Check'].isna())].reset_index(drop=True)

    return df, df2


def expand_gr_reading_split(row, columns, col):
    """Splits the rows to have one row per day, with a division of the total pages in the book by the number of days to read it"""
    # Handle missing dates
    if (pd.isna(row['Date started'])) | (pd.isna(row['Date ended'])):
        date_df = pd.DataFrame(columns=col)
        # Create single row with available data
        new_row = {}
        for column in columns:
            if column not in ['Date started', 'Date ended', 'cover_url']:
                new_row[column] = row.get(column, None)
        new_row['Timestamp'] = row.get('Date started', pd.NaT)
        new_row['page_split'] = row.get('Number of Pages', 0)
        new_row['cover_url'] = row.get('cover_url', '')

        date_df = pd.concat([date_df, pd.DataFrame([new_row])], ignore_index=True)
        return date_df

    # Create date range
    dates = pd.date_range(row['Date started'], row['Date ended'], freq='D')
    date_df = pd.DataFrame({'Timestamp': dates})

    # Add all other columns except dates and cover_url
    for column in columns:
        if column not in ['Date started', 'Date ended', 'cover_url']:
            date_df[column] = row.get(column, None)

    # Calculate page split
    num_pages = row.get('Number of Pages', 0)
    if pd.notna(num_pages) and num_pages > 0:
        date_df['page_split'] = num_pages / len(dates)
    else:
        date_df['page_split'] = 0

    # Add cover URL to all rows
    date_df['cover_url'] = row.get('cover_url', '')

    return date_df


def create_goodreads_file():
    """
    Main processing logic for Goodreads data with cover URLs from scraping
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Goodreads data...")

    try:
        df, df2 = add_dates_read_from_excel()

        # Ensure cover_url column exists in df2
        if 'cover_url' not in df2.columns:
            df2['cover_url'] = ''

        # Merge with dates and cover URLs
        merge_columns = ['Title', 'Book Id', 'Date started', 'Date ended', 'Check', 'cover_url']
        # Only use columns that exist
        available_columns = [col for col in merge_columns if col in df2.columns]

        df3 = pd.merge(df2[available_columns],
                      df, on='Book Id', how='left') \
                .rename(columns={'Title_x': 'Title', 'Bookshelves': 'Genre'})

        # Handle case where Title_x doesn't exist (no merge conflict)
        if 'Title_x' not in df3.columns and 'Title_y' in df3.columns:
            df3 = df3.rename(columns={'Title_y': 'Title'})

        # Calculate reading duration
        df3['reading_duration'] = (pd.to_datetime(df3['Date ended']) - pd.to_datetime(df3['Date started'])).dt.days + 1
        df3['Fiction_yn'] = df3['Genre'].apply(lambda x: "fiction" if str(x).lower() in fiction_genres else "non-fiction")

        # Define columns for processing
        base_columns = ['Book Id', 'Title', 'Author', 'Original Publication Year', 'My Rating',
                       'Average Rating', 'Genre', 'Fiction_yn', 'reading_duration', 'Number of Pages']

        # Add date and cover columns if they exist
        if 'Date started' in df3.columns:
            base_columns.append('Date started')
        if 'Date ended' in df3.columns:
            base_columns.append('Date ended')
        if 'cover_url' in df3.columns:
            base_columns.append('cover_url')

        # Filter to only read books
        df_limited = df3[(df3['Exclusive Shelf'] == 'read') | (df3['Check'] == 'OK')]

        # Only keep columns that exist
        existing_columns = [col for col in base_columns if col in df_limited.columns]
        df_limited = df_limited[existing_columns]

        print(f"📊 Processing {len(df_limited)} books...")

        # Prepare columns for expanded dataframe
        final_columns = [col for col in existing_columns if col not in ['Date started', 'Date ended']]
        final_columns.extend(['Timestamp', 'page_split'])

        # Ensure cover_url is in final columns
        if 'cover_url' not in final_columns:
            final_columns.append('cover_url')

        expanded_df = pd.DataFrame(columns=final_columns)

        for _, row in df_limited.iterrows():
            try:
                expanded_row = expand_gr_reading_split(row, existing_columns, final_columns)
                expanded_df = pd.concat([expanded_df, expanded_row], ignore_index=True)
            except Exception as e:
                print(f"⚠️  Error processing book '{row.get('Title', 'Unknown')}': {e}")
                continue

        # Add remaining required columns
        expanded_df['Seconds'] = np.nan
        if 'Fiction_yn' not in expanded_df.columns:
            expanded_df['Fiction_yn'] = expanded_df.get('Genre', '').apply(lambda x: "fiction" if str(x).lower() in fiction_genres else "non-fiction")
        expanded_df['Source'] = 'GoodReads'

        # Clean up title column
        if 'Title' in expanded_df.columns:
            expanded_df['Title'] = expanded_df['Title'].apply(lambda x: str(x).strip())

        # Save to the new location
        output_path = 'files/processed_files/books/gr_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        expanded_df.to_csv(output_path, sep='|', index=False)

        print(f"✅ Successfully processed {len(expanded_df)} reading records")
        print(f"📖 Columns in final file: {list(expanded_df.columns)}")

        return True

    except Exception as e:
        print(f"❌ Error processing Goodreads data: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_goodreads_results():
    """
    Uploads the processed Goodreads files to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading Goodreads results to Google Drive...")

    files_to_upload = [
        'files/processed_files/books/gr_processed.csv',
        'files/work_files/gr_work_files/gr_dates_input.xlsx'
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Goodreads results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


def process_goodreads_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_goodreads_pipeline(auto_full=True)
    else:
        return create_goodreads_file()


def full_goodreads_pipeline(auto_full=False):
    """
    Complete Goodreads pipeline with 3 options.

    Options:
    1. Full pipeline (download → scrape → process → upload)
    2. Process and upload (process existing data → upload)
    3. Scrape → process → upload (scrape missing data → process → upload)

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("📚 GOODREADS DATA PIPELINE")
    print("="*60)

    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → scrape → process → upload)")
        print("2. Process and upload (use existing data)")
        print("3. Scrape → process → upload (scrape missing data first)")

        choice = input("\nEnter your choice (1/5/6): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full Goodreads pipeline...")

        # Step 1: Download
        download_success = download_goodreads_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_goodreads_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_goodreads_files()

        # Step 3: Scrape missing data
        if move_success:
            scraped_books = scrape_goodreads_reading_data()
            if scraped_books:
                scrape_success = update_excel_with_scraped_data(scraped_books)
                print(f"✅ Scraped data for {len(scraped_books)} books")
            else:
                print("⚠️  No new books found to scrape")
                scrape_success = True  # Continue anyway
        else:
            print("⚠️  Using existing files, attempting to scrape...")
            scraped_books = scrape_goodreads_reading_data()
            scrape_success = update_excel_with_scraped_data(scraped_books) if scraped_books else True

        # Step 4: Process
        if scrape_success:
            process_success = create_goodreads_file()
        else:
            print("⚠️  Scraping had issues, attempting to process anyway...")
            process_success = create_goodreads_file()

        # Step 5: Upload
        if process_success:
            upload_success = upload_goodreads_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n⚙️  Processing existing data and uploading...")
        process_success = create_goodreads_file()
        if process_success:
            success = upload_goodreads_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "3":
        print("\n🤖 Scraping → processing → uploading...")

        # Step 1: Scrape
        scraped_books = scrape_goodreads_reading_data()
        if scraped_books:
            scrape_success = update_excel_with_scraped_data(scraped_books)
            print(f"✅ Scraped data for {len(scraped_books)} books")
        else:
            print("⚠️  No new books found to scrape, continuing with existing data...")
            scrape_success = True

        # Step 2: Process
        if scrape_success:
            process_success = create_goodreads_file()
        else:
            print("⚠️  Scraping had issues, attempting to process anyway...")
            process_success = create_goodreads_file()

        # Step 3: Upload
        if process_success:
            success = upload_goodreads_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    else:
        print("❌ Invalid choice. Please select 1, 2, or 3.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Goodreads pipeline completed successfully!")
    else:
        print("❌ Goodreads pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("📚 Enhanced Goodreads Processing Tool")
    print("This tool downloads, scrapes, processes, and uploads Goodreads data.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_goodreads_pipeline(auto_full=False)
