import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
# Drive operations not needed - source processor doesn't upload
from src.utils.utils_functions import record_successful_run

# Fiction genres classification
fiction_genres = ['drama', 'horror', 'thriller', 'classics', 'science-fiction']
from src.utils.logger import log


def clean_isbn(isbn_value):
    """
    Clean ISBN value from Excel format quirks in Goodreads export.

    Args:
        isbn_value: Raw ISBN value (may be like '="9780747582977"')

    Returns:
        str: Cleaned ISBN or None if invalid
    """
    if isbn_value is None or (isinstance(isbn_value, float) and pd.isna(isbn_value)):
        return None

    if not isbn_value:
        return None

    # Convert to string
    isbn_str = str(isbn_value).strip()

    # Remove Excel formatting: ="isbn" or ='isbn'
    if isbn_str.startswith('="') and isbn_str.endswith('"'):
        isbn_str = isbn_str[2:-1]
    elif isbn_str.startswith("='") and isbn_str.endswith("'"):
        isbn_str = isbn_str[2:-1]
    elif isbn_str.startswith('='):
        isbn_str = isbn_str[1:]

    # Remove any remaining quotes
    isbn_str = isbn_str.strip('"\'')

    # Validate: should be 10 or 13 digits (allowing for X in ISBN-10)
    cleaned = isbn_str.replace('-', '').replace(' ', '')
    if len(cleaned) == 10 or len(cleaned) == 13:
        if cleaned[:-1].isdigit() and (cleaned[-1].isdigit() or cleaned[-1].upper() == 'X'):
            return cleaned

    return None


def load_reading_dates_json():
    """Load reading dates from JSON file"""
    json_path = 'files/work_files/gr_work_files/reading_dates.json'

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                dates_data = json.load(f)
            log.progress(f"Loaded {len(dates_data)} book dates from JSON")
            return dates_data
        except Exception as e:
            log.warning(f"Error loading JSON file: {e}")
            return {}
    else:
        log.progress("No existing reading dates JSON found - creating new one")
        return {}


def save_reading_dates_json(dates_data):
    """Save reading dates to JSON file"""
    json_path = 'files/work_files/gr_work_files/reading_dates.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    try:
        with open(json_path, 'w') as f:
            json.dump(dates_data, f, indent=2, default=str)
        log.success(f"Saved {len(dates_data)} book dates to JSON")
        return True
    except Exception as e:
        log.error(f"Error saving JSON file: {e}")
        return False


def migrate_excel_to_json():
    """One-time migration from Excel file to JSON format"""
    log.progress("Checking for Excel to JSON migration...")

    excel_path = 'files/work_files/gr_work_files/gr_dates_input.xlsx'
    json_path = 'files/work_files/gr_work_files/reading_dates.json'

    # If JSON already exists, skip migration
    if os.path.exists(json_path):
        log.success("JSON file already exists, skipping migration")
        return True

    # If Excel doesn't exist, nothing to migrate
    if not os.path.exists(excel_path):
        log.info("No Excel file found to migrate")
        return True

    try:
        log.progress("Migrating Excel data to JSON...")
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
            log.success(f"Successfully migrated {migrated_count} books from Excel to JSON")
            return True
        else:
            return False

    except Exception as e:
        log.error(f"Error during migration: {e}")
        return False


def download_goodreads_data():
    """Opens Goodreads export page and prompts user to download data"""
    log.progress("Starting Goodreads data download...")

    urls = ['https://www.goodreads.com/review/import']
    open_web_urls(urls)

    log.info("Instructions:")
    log.info(" 1. Click 'Export Library'")
    log.info(" 2. Wait for the export to be prepared")
    log.info(" 3. Download the CSV file when ready")
    log.info(" 4. The file will be named 'goodreads_library_export.csv'")

    response = prompt_user_download_status("Goodreads")
    return response


def move_goodreads_files():
    """Moves the downloaded Goodreads file from Downloads to the correct export folder"""
    log.info("📁 Moving Goodreads files...")

    success = clean_rename_move_file(
        export_folder="files/exports/goodreads_exports",
        download_folder="/Users/valen/Downloads",
        file_name="goodreads_library_export.csv",
        new_file_name="gr_export.csv"
    )

    if success:
        log.success("Successfully moved Goodreads export to exports folder")
    else:
        log.error("Failed to move Goodreads files")

    return success


def extract_detailed_reading_dates_and_cover(driver, view_link, book_title):
    """Click on a view link and extract Start/End reading dates AND book cover URL"""
    try:
        log.progress(f"Extracting data for '{book_title}'...")

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
            log.error(f"Could not access detailed view for '{book_title}'")
            return {}

        # Wait for the detailed view to load
        time.sleep(3)

        result = {}

        # Extract reading dates from the edit page's <select> dropdowns
        # The edit page has separate selects for start/finish: month, day, year
        try:
            # Find all select elements on the page
            selects = driver.find_elements(By.TAG_NAME, "select")
            log.debug(f"Found {len(selects)} select elements on page")

            # The edit page has selects in order:
            # [start_month, start_day, start_year, finish_month, finish_day, finish_year]
            # We identify them by their name/id attributes
            date_selects = {}
            for sel in selects:
                name = sel.get_attribute("name") or sel.get_attribute("id") or ""
                name_lower = name.lower()
                log.debug(f"Select element: name='{name}', id='{sel.get_attribute('id')}'")

                # Map select elements to date parts
                if "start" in name_lower and "month" in name_lower:
                    date_selects['start_month'] = sel
                elif "start" in name_lower and "day" in name_lower:
                    date_selects['start_day'] = sel
                elif "start" in name_lower and "year" in name_lower:
                    date_selects['start_year'] = sel
                elif "finish" in name_lower and "month" in name_lower:
                    date_selects['finish_month'] = sel
                elif "finish" in name_lower and "day" in name_lower:
                    date_selects['finish_day'] = sel
                elif "finish" in name_lower and "year" in name_lower:
                    date_selects['finish_year'] = sel

            # If name-based matching didn't work, try positional matching
            # Goodreads edit page typically has 6 date selects in order
            if len(date_selects) < 3 and len(selects) >= 6:
                log.debug("Name-based matching incomplete, trying positional matching")
                # Filter to only date-related selects (those with year/month options)
                date_related = []
                for sel in selects:
                    try:
                        options = [o.text for o in Select(sel).options]
                        # Date selects have months (January, etc.) or years (2024, 2025, etc.)
                        if any(m in ' '.join(options) for m in ['January', 'February', 'March']):
                            date_related.append(('month', sel))
                        elif any(str(y) in ' '.join(options) for y in range(2020, 2030)):
                            date_related.append(('year', sel))
                        elif any(str(d) in options for d in range(28, 32)):
                            date_related.append(('day', sel))
                    except:
                        continue

                log.debug(f"Found {len(date_related)} date-related selects: {[t for t,_ in date_related]}")

                # Assign positionally: first set = start, second set = finish
                month_sels = [s for t, s in date_related if t == 'month']
                day_sels = [s for t, s in date_related if t == 'day']
                year_sels = [s for t, s in date_related if t == 'year']

                if len(month_sels) >= 1:
                    date_selects.setdefault('start_month', month_sels[0])
                if len(day_sels) >= 1:
                    date_selects.setdefault('start_day', day_sels[0])
                if len(year_sels) >= 1:
                    date_selects.setdefault('start_year', year_sels[0])
                if len(month_sels) >= 2:
                    date_selects.setdefault('finish_month', month_sels[1])
                if len(day_sels) >= 2:
                    date_selects.setdefault('finish_day', day_sels[1])
                if len(year_sels) >= 2:
                    date_selects.setdefault('finish_year', year_sels[1])

            # Extract start date
            if all(k in date_selects for k in ['start_month', 'start_day', 'start_year']):
                try:
                    s_month = Select(date_selects['start_month']).first_selected_option.text.strip()
                    s_day = Select(date_selects['start_day']).first_selected_option.text.strip()
                    s_year = Select(date_selects['start_year']).first_selected_option.text.strip()
                    if s_month != '--' and s_year != '--' and s_year.isdigit():
                        day_str = s_day if s_day != '--' and s_day.isdigit() else '1'
                        result['date_started'] = f"{s_month} {day_str}, {s_year}"
                        log.info(f"  Found start date: {result['date_started']}")
                except Exception as e:
                    log.debug(f"Error reading start date selects: {e}")

            # Extract finish date
            if all(k in date_selects for k in ['finish_month', 'finish_day', 'finish_year']):
                try:
                    f_month = Select(date_selects['finish_month']).first_selected_option.text.strip()
                    f_day = Select(date_selects['finish_day']).first_selected_option.text.strip()
                    f_year = Select(date_selects['finish_year']).first_selected_option.text.strip()
                    if f_month != '--' and f_year != '--' and f_year.isdigit():
                        day_str = f_day if f_day != '--' and f_day.isdigit() else '1'
                        result['date_ended'] = f"{f_month} {day_str}, {f_year}"
                        log.info(f"  Found end date: {result['date_ended']}")
                except Exception as e:
                    log.debug(f"Error reading finish date selects: {e}")

            if not result.get('date_started') and not result.get('date_ended'):
                log.warning(f"No dates found in select dropdowns for '{book_title}'")
                # Debug: list all select names found
                for sel in selects:
                    name = sel.get_attribute("name") or sel.get_attribute("id") or "(unnamed)"
                    try:
                        selected = Select(sel).first_selected_option.text
                    except:
                        selected = "?"
                    log.debug(f"  Select '{name}' = '{selected}'")

        except Exception as e:
            log.warning(f"Error extracting dates: {e}")

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
                        log.info(f" 🖼️  Found cover URL")
                        break
                except:
                    continue

        except Exception as e:
            log.warning(f"Error extracting cover: {e}")

        return result

    except Exception as e:
        log.error(f"Error in detailed extraction: {e}")
        return {}


def scrape_missing_reading_dates(books_needing_dates):
    """Scrape reading dates for books that don't have them in the JSON file"""
    if not books_needing_dates:
        log.success("No books need date scraping")
        return {}

    log.info(f"🤖 Starting scraping for {len(books_needing_dates)} books...")
    log.normal(f"Starting Goodreads date scraping for {len(books_needing_dates)} books")

    # Setup Chrome with persistent session (undetected to bypass Google login blocks)
    chrome_options = uc.ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--user-data-dir=/Users/valen/chrome_goodreads_profile")

    driver = uc.Chrome(options=chrome_options, version_main=144)
    scraped_data = {}

    try:
        # Navigate to reading list
        log.progress("Opening Goodreads reading list...")
        driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")
        time.sleep(5)

        # Check if we need to log in
        if "Sign in" in driver.title or "sign_in" in driver.current_url:
            log.info("Please log into Goodreads...")
            log.progress("Waiting for login... (you have 120 seconds)")

            start_time = time.time()
            while time.time() - start_time < 120:
                if "Sign in" not in driver.title and "sign_in" not in driver.current_url:
                    log.success("Login detected, continuing...")
                    break
                time.sleep(2)
            else:
                log.error("Login timeout - continuing anyway")

        books_found = 0
        processed_books = set()  # Track which books we've already processed to avoid duplicates

        while books_found < len(books_needing_dates):
            # Find all book rows on current page
            book_rows = driver.find_elements(By.CSS_SELECTOR, "tr[id*='review_'], tr.bookalike")
            if not book_rows:
                book_rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
                book_rows = [row for row in book_rows if row.find_elements(By.CSS_SELECTOR, "td")]

            log.success(f"Found {len(book_rows)} total book rows on page")

            # Debug: Show what we're looking for
            if books_found < len(books_needing_dates):
                log.progress(f"Looking for: {list(books_needing_dates.values())[0]['title']}")

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
                            log.progress(f"Fuzzy match: '{found_title}'≈ '{expected_title}'")
                            matching_book_id = book_id
                            break

                    if not matching_book_id or matching_book_id in processed_books:
                        continue

                    log.progress(f"Found book needing dates: '{title}'(ID: {matching_book_id})")

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
                            log.success(f"Scraped data for '{title}'")
                            # NORMAL-level per-book summary
                            dates_found = []
                            if scraped_info.get('date_started'):
                                dates_found.append('start')
                            if scraped_info.get('date_ended'):
                                dates_found.append('end')
                            dates_summary = ' + '.join(dates_found) if dates_found else 'no dates'
                            log.normal_success(f"'{title}' - found {dates_summary}")
                        else:
                            log.error(f"Could not extract data for '{title}'")
                            log.normal(f"'{title}' - no data extracted")
                            processed_books.add(matching_book_id)  # Mark as processed even if failed

                        # Return to main list
                        time.sleep(2)
                        driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")
                        time.sleep(2)

                        # Break from current row loop to re-find book rows
                        break

                except Exception as e:
                    log.error(f"Error processing row {i}: {e}")
                    continue

            # If we didn't find any new books this pass, break to avoid infinite loop
            if books_found_this_pass == 0:
                log.warning(f"No new books found in this pass. Stopping with {books_found} books processed.")
                # Debug: Show what titles were found on this page
                if page_titles:
                    log.progress(f"Titles found on page (first 10):")
                    for i, t in enumerate(page_titles[:10], 1):
                        log.info(f"{i}. {t}")
                break

            # Stop if we found all books we were looking for
            if books_found >= len(books_needing_dates):
                log.success("Found all books we were looking for!")
                break

        log.success(f"Successfully scraped {len(scraped_data)} out of {len(books_needing_dates)} books")
        log.normal_success(f"Scraping complete: {len(scraped_data)} out of {len(books_needing_dates)} books scraped successfully")
        return scraped_data

    except Exception as e:
        log.error(f"Scraping failed: {e}")
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
        log.warning(f"Error expanding dates for book: {e}")
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
    log.info("⚙️  Processing Goodreads data with JSON-based dates...")

    try:
        # Load Goodreads export
        df = pd.read_csv("files/exports/goodreads_exports/gr_export.csv")
        log.progress(f"Loaded {len(df)} books from Goodreads export")

        # Load existing reading dates from JSON
        dates_data = load_reading_dates_json()

        # Filter to only 'read' books
        read_books = df[df['Exclusive Shelf'] == 'read'].copy()
        log.progress(f"Found {len(read_books)} books marked as 'read'")

        # Identify books that need date scraping
        books_needing_dates = {}
        books_with_dates = {}

        for _, book in read_books.iterrows():
            book_id = str(book['Book Id'])
            title = str(book['Title']).strip()

            # If JSON has date_started but no date_ended, fill from CSV 'Date Read'
            if book_id in dates_data and dates_data[book_id].get('date_started') and not dates_data[book_id].get('date_ended'):
                csv_date_read = book.get('Date Read', '')
                if pd.notna(csv_date_read) and str(csv_date_read).strip():
                    dates_data[book_id]['date_ended'] = str(csv_date_read).strip()
                    log.info(f"Filled end date from CSV for '{title}': {csv_date_read}")

            if book_id in dates_data and dates_data[book_id].get('date_started') and dates_data[book_id].get('date_ended'):
                # We have complete data for this book in JSON
                books_with_dates[book_id] = dates_data[book_id]
            else:
                # Need to scrape dates for this book
                books_needing_dates[book_id] = {
                    'title': title,
                    'book_data': book
                }

        log.success(f"{len(books_with_dates)} books already have dates")
        log.progress(f"{len(books_needing_dates)} books need date scraping")

        # Scrape missing dates if needed
        if books_needing_dates:
            # Print the list of books that need scraping
            log.progress("\n Books that need date scraping:")
            for i, (book_id, book_info) in enumerate(books_needing_dates.items(), 1):
                log.info(f"{i}. {book_info['title']} (ID: {book_id})")
            log.blank()  # Add blank line for readability

            log.milestone(f"Starting automatic scraping for {len(books_needing_dates)} books...")
            scraped_data = scrape_missing_reading_dates(books_needing_dates)

            # Merge scraped data into main dates_data
            # Use CSV 'Date Read' as fallback for date_ended when scraper didn't find it
            for book_id, scraped_info in scraped_data.items():
                if not scraped_info.get('date_ended') and book_id in books_needing_dates:
                    csv_date_read = books_needing_dates[book_id]['book_data'].get('Date Read', '')
                    if pd.notna(csv_date_read) and str(csv_date_read).strip():
                        scraped_info['date_ended'] = str(csv_date_read).strip()
                        log.info(f"Using CSV 'Date Read' as end date for '{scraped_info.get('title', book_id)}': {scraped_info['date_ended']}")
                dates_data[book_id] = scraped_info
                books_with_dates[book_id] = scraped_info

            # Save updated JSON
            save_reading_dates_json(dates_data)
            log.success(f"Updated JSON with {len(scraped_data)} newly scraped books")

        # Now process all books with available dates
        processed_books = []

        for _, book in read_books.iterrows():
            book_id = str(book['Book Id'])

            # Clean ISBN values from Excel format quirks
            isbn10 = clean_isbn(book.get('ISBN', ''))
            isbn13 = clean_isbn(book.get('ISBN13', ''))

            if book_id in books_with_dates:
                # Add date information to book data
                date_info = books_with_dates[book_id]

                book_with_dates = book.copy()
                book_with_dates['Date started'] = pd.to_datetime(date_info.get('date_started'), errors='coerce')
                book_with_dates['Date ended'] = pd.to_datetime(date_info.get('date_ended'), errors='coerce')
                book_with_dates['cover_url'] = date_info.get('cover_url', '')
                book_with_dates['isbn'] = isbn10
                book_with_dates['isbn13'] = isbn13

                # Update JSON cache with ISBN if not already present
                if isbn10 or isbn13:
                    if 'isbn' not in dates_data.get(book_id, {}) or not dates_data[book_id].get('isbn'):
                        dates_data[book_id] = dates_data.get(book_id, {})
                        dates_data[book_id]['isbn'] = isbn10
                        dates_data[book_id]['isbn13'] = isbn13

                processed_books.append(book_with_dates)
            else:
                # No dates available, add with NaT dates
                book_without_dates = book.copy()
                book_without_dates['Date started'] = pd.NaT
                book_without_dates['Date ended'] = pd.NaT
                book_without_dates['cover_url'] = ''
                book_without_dates['isbn'] = isbn10
                book_without_dates['isbn13'] = isbn13

                processed_books.append(book_without_dates)

        # Convert to DataFrame
        df_with_dates = pd.DataFrame(processed_books)

        # Calculate reading duration and fiction classification
        df_with_dates['reading_duration'] = (df_with_dates['Date ended'] - df_with_dates['Date started']).dt.days + 1
        df_with_dates['Fiction_yn'] = df_with_dates['Bookshelves'].apply(
            lambda x: "fiction" if str(x).lower() in fiction_genres else "non-fiction"
        )

        # Save updated JSON with ISBN data
        save_reading_dates_json(dates_data)

        # Prepare for expansion
        base_columns = ['Book Id', 'Title', 'Author', 'Original Publication Year', 'My Rating',
                       'Average Rating', 'Bookshelves', 'Fiction_yn', 'reading_duration',
                       'Number of Pages', 'Date started', 'Date ended', 'cover_url', 'isbn', 'isbn13']

        # Only keep existing columns
        existing_columns = [col for col in base_columns if col in df_with_dates.columns]
        df_limited = df_with_dates[existing_columns]

        # Prepare final columns
        final_columns = [col for col in existing_columns if col not in ['Date started', 'Date ended']]
        final_columns.extend(['Timestamp', 'page_split'])

        # Expand each book into daily reading records
        expanded_df = pd.DataFrame(columns=final_columns)

        log.progress(f"Expanding {len(df_limited)} books into daily reading records...")

        for _, row in df_limited.iterrows():
            try:
                expanded_row = expand_gr_reading_split(row, existing_columns, final_columns)
                expanded_df = pd.concat([expanded_df, expanded_row], ignore_index=True)
            except Exception as e:
                log.warning(f"Error processing book '{row.get('Title', 'Unknown')}': {e}")
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

        log.success(f"Successfully processed {len(expanded_df)} reading records")
        log.debug(f"Final columns: {list(expanded_df.columns)}")
        log.progress(f"Books with complete dates: {len(books_with_dates)}")
        log.progress(f"Books processed: {len(processed_books)}")

        return True

    except Exception as e:
        log.error(f"Error processing Goodreads data: {e}")
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
    log.info("\n" + "="*60)
    log.progress("GOODREADS SOURCE PROCESSOR (JSON-BASED)")
    log.info("="*60)

    # First, migrate any existing Excel data to JSON
    migrate_excel_to_json()

    if auto_process_only:
        log.info("🤖 Auto process mode: Processing existing data...")
        choice = "2"
    elif auto_full:
        log.info("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        log.prompt("\nSelect an option:")
        log.prompt("1. Download and process new data")
        log.prompt("2. Process existing data")
        log.prompt("3. Scrape dates only (for books missing dates)")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        log.milestone("\n Starting Goodreads pipeline...")

        # Step 1: Download
        download_success = download_goodreads_data()

        # Step 2: Move files
        if download_success:
            move_success = move_goodreads_files()
        else:
            log.warning("Download not confirmed, checking for existing files...")
            move_success = move_goodreads_files()

        # Step 3: Process (includes scraping if needed)
        if move_success or os.path.exists("files/exports/goodreads_exports/gr_export.csv"):
            success = create_goodreads_file()
        else:
            log.error("No Goodreads export file found")
            success = False

    elif choice == "2":
        log.info("\n⚙️  Processing existing data...")
        success = create_goodreads_file()

    elif choice == "3":
        log.progress("\n Scraping dates for books missing dates...")

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
                log.progress("\n Books that need date scraping:")
                for i, (book_id, book_info) in enumerate(books_needing_dates.items(), 1):
                    log.info(f"{i}. {book_info['title']} (ID: {book_id})")
                log.blank()  # Add blank line for readability

                scraped_data = scrape_missing_reading_dates(books_needing_dates)

                # Update JSON with scraped data
                for book_id, scraped_info in scraped_data.items():
                    dates_data[book_id] = scraped_info

                success = save_reading_dates_json(dates_data)
                if success:
                    log.success(f"Updated {len(scraped_data)} books with scraped dates")
            else:
                log.success("No books need date scraping")
                success = True

        except Exception as e:
            log.error(f"Error during scraping: {e}")
            success = False

    else:
        log.error("Invalid choice. Please select 1-3.")
        return False

    # Final status
    log.info("\n" + "="*60)
    if success:
        log.success("Goodreads source processor completed successfully!")
        # Record successful run with new tracking name
        record_successful_run('source_goodreads', 'active')
    else:
        log.error("Goodreads source processor failed")
    log.info("="*60)

    return success



if __name__ == "__main__":
    # Allow running this file directly — default to VERBOSE when run standalone
    log.set_verbosity(1)  # VERBOSE
    log.progress("Goodreads Source Processing Tool")
    log.info("This tool downloads, scrapes, and processes Goodreads data.")
    log.info("Note: Upload is handled by the Reading topic coordinator")

    # Run the pipeline
    full_goodreads_pipeline(auto_full=False)
