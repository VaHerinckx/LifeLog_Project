# Fixed Goodreads Reading Dates Scraper - Get Start/End Dates
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import os

def load_existing_books():
    """Load books that are already processed to avoid duplicates"""
    try:
        # Check if the processed file exists
        if os.path.exists('files/processed_files/gr_processed.csv'):
            df = pd.read_csv('files/processed_files/gr_processed.csv', sep='|')
            existing_titles = set(df['Title'].str.strip().str.lower())
            print(f"📚 Found {len(existing_titles)} existing books in processed file")
            return existing_titles
        else:
            print("📚 No existing processed file found - will process all books")
            return set()
    except Exception as e:
        print(f"⚠️  Error reading existing books: {e}")
        return set()

def test_goodreads_scraping():
    """
    Test function to extract start/end reading dates from Goodreads
    Only processes books not already in the system
    """

    # Load existing books to avoid duplicates
    existing_books = load_existing_books()

    # Setup Chrome with persistent session
    chrome_options = Options()
    chrome_options.add_argument("--user-data-dir=/Users/valen/chrome_goodreads_profile")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to your reading list
        print("🔍 Opening Goodreads reading list...")
        driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")

        # Wait for login and page interaction
        print("⏳ Waiting for you to log in if needed...")
        print("📝 Please log into Goodreads if needed, then press Enter...")
        input()

        print("🎯 Looking for new books that need date extraction...")

        # Find all book rows
        book_rows = driver.find_elements(By.CSS_SELECTOR, "tr[id*='review_'], tr.bookalike")
        if not book_rows:
            book_rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
            book_rows = [row for row in book_rows if row.find_elements(By.CSS_SELECTOR, "td")]

        print(f"Found {len(book_rows)} total book rows")

        new_books = []

        for i, row in enumerate(book_rows):
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

                # Look for "view" link in this row to get detailed dates
                view_link = None
                try:
                    # The view link is usually in the same row, often in a review column
                    view_link = row.find_element(By.CSS_SELECTOR, "a[href*='#review_']")
                except:
                    try:
                        # Alternative: look for any link with "view" text
                        view_link = row.find_element(By.XPATH, ".//a[contains(text(), 'view')]")
                    except:
                        pass

                if view_link:
                    print(f"📖 Found new book: '{title}' - extracting detailed dates...")

                    # Extract detailed reading dates
                    reading_dates = extract_detailed_reading_dates(driver, view_link, title)

                    if reading_dates:
                        new_books.append({
                            'title': title,
                            'book_url': book_url,
                            'start_date': reading_dates.get('start_date'),
                            'end_date': reading_dates.get('end_date')
                        })
                        print(f"  ✅ Start: {reading_dates.get('start_date', 'Not found')}")
                        print(f"  ✅ End: {reading_dates.get('end_date', 'Not found')}")
                    else:
                        print(f"  ❌ Could not extract dates for '{title}'")
                else:
                    print(f"📖 '{title}' - no 'view' link found (dates might already be set)")

                # Add a small delay to be respectful
                time.sleep(2)  # Increased delay for page stability

                # After each book, return to the main list
                try:
                    driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")
                    time.sleep(2)
                except:
                    pass

            except Exception as e:
                print(f"Error processing row {i}: {e}")

        return new_books

    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return None

    finally:
        try:
            print("🔄 Returning to main books list...")
            driver.get("https://www.goodreads.com/review/list/143865509?shelf=read&sort=date_read&order=d")
            time.sleep(2)
        except:
            pass
        driver.quit()

def extract_detailed_reading_dates(driver, view_link, book_title):
    """
    Click on a view link and extract Start/End reading dates from the detailed view
    Like in your screenshot: "May 7, 2025 – Started Reading" and "May 14, 2025 – Finished Reading"
    """
    try:
        print(f"    🔍 Clicking view link for '{book_title}'...")

        # Handle potential overlays/banners that might intercept clicks
        try:
            # First try to dismiss any potential overlays
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
                        print(f"    🚫 Dismissed overlay: {selector}")
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
            print(f"    ✅ Regular click successful")
        except Exception as e1:
            print(f"    ⚠️  Regular click failed: {str(e1)[:100]}...")

            # Method 2: JavaScript click
            try:
                driver.execute_script("arguments[0].click();", view_link)
                click_successful = True
                print(f"    ✅ JavaScript click successful")
            except Exception as e2:
                print(f"    ⚠️  JavaScript click failed: {str(e2)[:100]}...")

                # Method 3: Navigate directly to the URL
                try:
                    view_url = view_link.get_attribute('href')
                    if view_url:
                        driver.get(view_url)
                        click_successful = True
                        print(f"    ✅ Direct navigation successful")
                except Exception as e3:
                    print(f"    ❌ All click methods failed: {str(e3)[:100]}...")

        if not click_successful:
            print(f"    ❌ Could not click view link for '{book_title}'")
            return {}

        # Wait for the detailed view to load
        time.sleep(3)

        dates = {}

        # Look for the "READING PROGRESS" section like in your screenshot
        try:
            # Try different selectors for the reading progress section
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
                        # Get all text from this section
                        for element in elements:
                            progress_items.extend(element.find_elements(By.XPATH, ".//*"))
                        break
                except:
                    continue

            # If we couldn't find a section, just search the whole page
            if not progress_items:
                progress_items = driver.find_elements(By.XPATH, "//*[contains(text(), 'Started Reading') or contains(text(), 'Finished Reading')]")

            print(f"    🔍 Found {len(progress_items)} progress elements")

            # Parse the progress items for dates
            for item in progress_items:
                try:
                    text = item.text.strip()

                    if "Started Reading" in text:
                        # Extract date before "Started Reading"
                        date_part = text.replace("Started Reading", "").replace("–", "").strip()
                        if date_part:
                            dates['start_date'] = date_part
                            print(f"    📅 Found start date: {date_part}")

                    elif "Finished Reading" in text:
                        # Extract date before "Finished Reading"
                        date_part = text.replace("Finished Reading", "").replace("–", "").strip()
                        if date_part:
                            dates['end_date'] = date_part
                            print(f"    📅 Found end date: {date_part}")

                except Exception as e:
                    continue

            # If we still don't have dates, try a broader search
            if not dates:
                print("    🔍 Trying broader date search...")
                # Look for any date patterns on the page
                all_text = driver.find_element(By.TAG_NAME, "body").text
                lines = all_text.split('\n')

                for line in lines:
                    line = line.strip()
                    if "Started Reading" in line:
                        date_part = line.replace("Started Reading", "").replace("–", "").strip()
                        if date_part:
                            dates['start_date'] = date_part
                    elif "Finished Reading" in line:
                        date_part = line.replace("Finished Reading", "").replace("–", "").strip()
                        if date_part:
                            dates['end_date'] = date_part

        except Exception as e:
            print(f"    ❌ Error extracting dates: {e}")

        return dates

    except Exception as e:
        print(f"    ❌ Error in detailed extraction: {e}")
        return {}

# Test function you can run
if __name__ == "__main__":
    print("🧪 Testing Goodreads detailed date scraping...")
    print("This will:")
    print("  1. Check what books you already have processed")
    print("  2. Find new books on your Goodreads 'Read' shelf")
    print("  3. Click 'view' links to get Start/End reading dates")
    print("  4. Only process books not already in your system")

    new_books = test_goodreads_scraping()

    if new_books:
        print(f"\n✅ Successfully extracted dates for {len(new_books)} NEW books:")
        for book in new_books:
            print(f"  📖 {book['title']}")
            print(f"     Start: {book['start_date']}")
            print(f"     End: {book['end_date']}")
            print()
    else:
        print("❌ No new books found or scraping failed")
        print("This could mean:")
        print("  - All your books are already processed")
        print("  - No books have 'view' links (dates already set)")
        print("  - There was an error in scraping")
