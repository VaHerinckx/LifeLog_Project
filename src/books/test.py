def manual_cover_input(books_missing, reading_dates_data, reading_dates_path):
    """
    Interactive function to manually add cover URLs for books that couldn't be automatically matched

    Args:
        books_missing: List of dictionaries with book info that need covers
        reading_dates_data: The reading dates data dictionary to update
        reading_dates_path: Path to save the updated JSON file

    Returns:
        Number of covers successfully added
    """
    import json

    print("\n" + "="*80)
    print("🎨 MANUAL COVER URL INPUT")
    print("="*80)
    print("For each book below, you can:")
    print("  • Enter a cover URL (from Google Books, Goodreads, Amazon, etc.)")
    print("  • Press Enter to skip the book")
    print("  • Type 'quit' to stop and save progress")
    print("  • Type 'skip all' to skip all remaining books")
    print("-"*80)

    covers_added = 0

    for i, book in enumerate(books_missing, 1):
        book_id = book['book_id']
        title = book['title']
        author = book['author']

        print(f"\n📖 Book {i}/{len(books_missing)}")
        print(f"Title: {title}")
        print(f"Author: {author}")
        print(f"Book ID: {book_id}")

        # Show some URL suggestions
        print("\n💡 Suggestions for finding cover URLs:")
        print(f"  • Google Books: https://books.google.com/books?q={title.replace(' ', '+')}")
        print(f"  • Goodreads: Search for '{title}' on goodreads.com")
        print(f"  • Amazon: Search for '{title}' on amazon.com")

        while True:
            user_input = input(f"\nEnter cover URL (or Enter to skip, 'quit' to stop, 'skip all' to skip remaining): ").strip()

            if user_input.lower() == 'quit':
                print(f"🛑 Stopping manual input. Added {covers_added} covers so far.")
                break
            elif user_input.lower() == 'skip all':
                print(f"⏭️  Skipping all remaining books. Added {covers_added} covers so far.")
                break
            elif user_input == '':
                print(f"⏭️  Skipping '{title}'")
                break
            elif user_input.startswith(('http://', 'https://')):
                # Validate that it looks like a URL
                try:
                    # Add the cover URL to the book
                    reading_dates_data[book_id]['cover_url'] = user_input
                    covers_added += 1
                    print(f"✅ Added cover URL for '{title}'")

                    # Save progress after each addition
                    with open(reading_dates_path, 'w', encoding='utf-8') as f:
                        json.dump(reading_dates_data, f, indent=2, ensure_ascii=False)

                    break
                except Exception as e:
                    print(f"❌ Error saving cover URL: {e}")
                    print("Please try again or press Enter to skip.")
            else:
                print("❌ Please enter a valid URL starting with http:// or https://")
                print("Or press Enter to skip this book.")

        # Check if user wants to quit or skip all
        if user_input.lower() in ['quit', 'skip all']:
            break

    if covers_added > 0:
        print(f"\n💾 Final save of {covers_added} new cover URLs...")
        try:
            with open(reading_dates_path, 'w', encoding='utf-8') as f:
                json.dump(reading_dates_data, f, indent=2, ensure_ascii=False)
            print("✅ Successfully saved all manual cover additions!")
        except Exception as e:
            print(f"❌ Error during final save: {e}")

    return covers_added


def fuzzy_match_cover(title, author, cover_lookup, threshold=80):
    """
    Perform fuzzy matching to find the best cover match for a book

    Args:
        title: Book title to search for
        author: Book author to search for
        cover_lookup: Dictionary of normalized_key -> cover_url
        threshold: Minimum similarity score (0-100) to consider a match

    Returns:
        Tuple of (cover_url, match_score, matched_key) or (None, 0, None) if no match
    """
    try:
        from difflib import SequenceMatcher
    except ImportError:
        print("⚠️  difflib not available for fuzzy matching")
        return None, 0, None

    def similarity(a, b):
        """Calculate similarity ratio between two strings"""
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() * 100

    def normalize_text(text):
        """Normalize text for better matching"""
        import re
        if not text or str(text).lower() == 'nan':
            return ""
        # Remove extra whitespace, punctuation, and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', str(text))
        text = ' '.join(text.split())  # Normalize whitespace
        return text.lower().strip()

    normalized_title = normalize_text(title)
    normalized_author = normalize_text(author)

    best_match = None
    best_score = 0
    best_key = None

    print(f"🔍 Fuzzy searching for: '{title}' by '{author}'")

    for lookup_key, cover_url in cover_lookup.items():
        if "||" in lookup_key:
            lookup_title, lookup_author = lookup_key.split("||", 1)
        else:
            lookup_title = lookup_key.replace("||", "")
            lookup_author = ""

        lookup_title = normalize_text(lookup_title)
        lookup_author = normalize_text(lookup_author)

        # Calculate title similarity (most important)
        title_similarity = similarity(normalized_title, lookup_title)

        # Calculate author similarity (secondary)
        author_similarity = 0
        if normalized_author and lookup_author:
            author_similarity = similarity(normalized_author, lookup_author)
        elif not normalized_author and not lookup_author:
            # Both authors are empty/missing - consider this a match
            author_similarity = 100
        elif not normalized_author or not lookup_author:
            # One author is missing - moderate penalty
            author_similarity = 50

        # Weighted score: title is 70% of score, author is 30%
        combined_score = (title_similarity * 0.7) + (author_similarity * 0.3)

        if combined_score > best_score and combined_score >= threshold:
            best_score = combined_score
            best_match = cover_url
            best_key = lookup_key

    if best_match:
        print(f"✨ Fuzzy match found: '{best_key}' (score: {best_score:.1f})")

    return best_match, best_score, best_key


def update_reading_dates_with_covers():
    """
    Updates reading_dates.json with cover URLs from book_covers.json
    Uses gr_export.csv to get the proper Title/Author mapping for each Book ID
    Only updates books that don't have a cover_url (null, empty, or "NaN")
    Includes fuzzy matching for books that don't have exact matches
    """
    import json
    import os
    import pandas as pd

    # File paths
    gr_export_path = "files/exports/goodreads_exports/gr_export.csv"
    book_covers_path = 'files/work_files/gr_work_files/book_covers.json'
    reading_dates_path = 'files/work_files/gr_work_files/reading_dates.json'

    print("📚 Starting cover URL update process...")

    # Check if files exist
    required_files = [gr_export_path, book_covers_path, reading_dates_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found")
            return False

    try:
        # Load Goodreads export CSV
        print("📊 Loading Goodreads export CSV...")
        df_gr = pd.read_csv(gr_export_path)
        print(f"✅ Loaded {len(df_gr)} books from Goodreads export")

        # Load book covers data
        print("📖 Loading book covers data...")
        with open(book_covers_path, 'r', encoding='utf-8') as f:
            book_covers_data = json.load(f)

        # Load reading dates data
        print("📅 Loading reading dates data...")
        with open(reading_dates_path, 'r', encoding='utf-8') as f:
            reading_dates_data = json.load(f)

        print(f"✅ Loaded {len(book_covers_data)} entries from book_covers.json")
        print(f"✅ Loaded {len(reading_dates_data)} entries from reading_dates.json")

        # Create a Book ID to Title/Author mapping from Goodreads export
        print("🔗 Creating Book ID to Title/Author mapping...")
        book_id_to_info = {}

        for _, row in df_gr.iterrows():
            book_id = str(row['Book Id'])  # Convert to string to match JSON keys
            title = str(row['Title']).strip()
            author = str(row['Author']).strip()

            book_id_to_info[book_id] = {
                'title': title,
                'author': author
            }

        print(f"📋 Created mapping for {len(book_id_to_info)} books")

        # Parse book_covers.json to create a title-author lookup dictionary
        print("🔍 Creating title-author lookup from book covers...")
        cover_lookup = {}

        for key, cover_info in book_covers_data.items():
            if "||" in key:
                title, author = key.split("||", 1)  # Split only on first ||
                # Normalize for exact matching
                normalized_key = f"{title.strip().lower()}||{author.strip().lower()}"
                cover_url = cover_info.get('cover_url', '')
                if cover_url:  # Only add if we have a valid URL
                    cover_lookup[normalized_key] = cover_url
            else:
                # Handle cases where there's no author separator
                title = key.strip()
                normalized_key = f"{title.lower()}||"
                cover_url = cover_info.get('cover_url', '')
                if cover_url:
                    cover_lookup[normalized_key] = cover_url

        print(f"📋 Created cover lookup table with {len(cover_lookup)} entries")

        # Update reading_dates.json with cover URLs
        books_needing_update = 0
        books_updated_exact = 0
        books_updated_fuzzy = 0
        books_with_existing_covers = 0
        books_no_gr_match = 0

        for book_id, book_info in reading_dates_data.items():
            # Check if book already has a valid cover URL
            current_cover = book_info.get('cover_url', '')

            # Consider these as "no cover URL": null, empty string, "NaN", or missing key
            needs_cover = (
                current_cover is None or
                current_cover == "" or
                current_cover == "NaN" or
                str(current_cover).lower() == "nan"
            )

            if not needs_cover:
                books_with_existing_covers += 1
                continue

            books_needing_update += 1

            # Get Title/Author from Goodreads export using Book ID
            if book_id not in book_id_to_info:
                print(f"⚠️  Book ID {book_id} not found in Goodreads export")
                books_no_gr_match += 1
                continue

            gr_info = book_id_to_info[book_id]
            title = gr_info['title']
            author = gr_info['author']

            if not title:
                print(f"⚠️  Empty title for Book ID {book_id}")
                continue

            print(f"🔍 Looking for cover: '{title}' by '{author}'")

            # STEP 1: Try exact matching first
            found_exact_match = False
            search_keys = []

            if author and author.lower() != 'nan':
                # Try with author first (exact match)
                search_keys.append(f"{title.lower()}||{author.lower()}")

            # Try without author (title only)
            search_keys.append(f"{title.lower()}||")

            # Look for exact matches
            for search_key in search_keys:
                if search_key in cover_lookup:
                    cover_url = cover_lookup[search_key]
                    book_info['cover_url'] = cover_url
                    books_updated_exact += 1
                    found_exact_match = True
                    print(f"✅ Exact match found for '{title}'")
                    break

            # STEP 2: If no exact match, try fuzzy matching
            if not found_exact_match:
                print(f"🔍 No exact match found, trying fuzzy matching...")
                cover_url, match_score, matched_key = fuzzy_match_cover(title, author, cover_lookup, threshold=70)

                if cover_url:
                    book_info['cover_url'] = cover_url
                    books_updated_fuzzy += 1
                    print(f"✅ Fuzzy match found for '{title}' (score: {match_score:.1f})")
                else:
                    print(f"❌ No fuzzy match found for '{title}'" + (f" by {author}" if author and author.lower() != 'nan' else ""))

        # Save updated reading_dates.json
        print(f"\n💾 Saving updated reading_dates.json...")
        with open(reading_dates_path, 'w', encoding='utf-8') as f:
            json.dump(reading_dates_data, f, indent=2, ensure_ascii=False)

        # Collect books that still need covers for manual input
        books_still_missing = []
        for book_id, book_info in reading_dates_data.items():
            current_cover = book_info.get('cover_url', '')
            needs_cover = (
                current_cover is None or
                current_cover == "" or
                current_cover == "NaN" or
                str(current_cover).lower() == "nan"
            )

            if needs_cover and book_id in book_id_to_info:
                gr_info = book_id_to_info[book_id]
                books_still_missing.append({
                    'book_id': book_id,
                    'title': gr_info['title'],
                    'author': gr_info['author'],
                    'book_info': book_info
                })

        # Print summary statistics
        print(f"\n📊 Update Summary:")
        print(f"✅ Books with existing covers: {books_with_existing_covers}")
        print(f"🔍 Books that needed covers: {books_needing_update}")
        print(f"📖 Books updated with exact matches: {books_updated_exact}")
        print(f"🎯 Books updated with fuzzy matches: {books_updated_fuzzy}")
        print(f"📚 Total books successfully updated: {books_updated_exact + books_updated_fuzzy}")
        print(f"⚠️  Books not in GR export: {books_no_gr_match}")
        print(f"❌ Books still missing covers: {len(books_still_missing)}")

        if books_needing_update > 0:
            success_rate = ((books_updated_exact + books_updated_fuzzy) / books_needing_update * 100)
            print(f"📈 Overall success rate: {success_rate:.1f}%")
            if books_updated_fuzzy > 0:
                fuzzy_contribution = (books_updated_fuzzy / (books_updated_exact + books_updated_fuzzy) * 100)
                print(f"🎯 Fuzzy matching contribution: {fuzzy_contribution:.1f}% of successful matches")
        else:
            print("📈 Success rate: N/A (no books needed updating)")

        # Ask user if they want to manually add covers for missing books
        if books_still_missing:
            print(f"\n🖼️  There are {len(books_still_missing)} books still missing covers.")
            user_input = input("Would you like to manually add cover URLs? (y/N): ").strip().lower()

            if user_input == 'y':
                manually_added = manual_cover_input(books_still_missing, reading_dates_data, reading_dates_path)
                print(f"✅ Manually added {manually_added} cover URLs")

        return True

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


# You can call this function like this:
if __name__ == "__main__":
    success = update_reading_dates_with_covers()
    if success:
        print("\n🎉 Cover URL update completed successfully!")
    else:
        print("\n💥 Cover URL update failed!")
