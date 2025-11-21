import pandas as pd
import string
import unidecode
import os
import json
from dotenv import load_dotenv
from src.utils.utils_functions import time_difference_correction, clean_rename_move_folder, find_unzip_folder
from lingua import Language, LanguageDetectorBuilder
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils.file_operations import check_file_exists
from src.utils.web_operations import open_web_urls
from src.utils.utils_functions import record_successful_run, enforce_snake_case
import requests
import time
load_dotenv()

# ============================================================================
# FILE PATHS - Centralized configuration
# ============================================================================
WORK_DIR = 'files/work_files/pocket_casts_work_files'
EXPORT_DIR = 'files/exports/pocket_casts_exports'
OUTPUT_DIR = 'files/source_processed_files/pocket_casts'

LANGUAGE_CACHE = f'{WORK_DIR}/podcasts_titles_language.json'
TRANSLATION_CACHE = f'{WORK_DIR}/podcasts_cleaned_translated.json'
PODCAST_MAPPING_FILE = f'{WORK_DIR}/podcast_mapping.xlsx'
ITUNES_CACHE_FILE = f'{WORK_DIR}/itunes_api_podcasts.csv'
ARTWORK_CACHE_FILE = f'{WORK_DIR}/podcast_artwork_cache.json'
OUTPUT_FILE = f'{OUTPUT_DIR}/pocket_casts_processed.csv'
EXPORT_DATA_FILE = f'{EXPORT_DIR}/data.txt'

# Legacy path variables for backward compatibility
path_dict_language = LANGUAGE_CACHE
path_dict_translation = TRANSLATION_CACHE

google_lang = {'FRENCH' : 'fr','DUTCH' : 'nl','ENGLISH': 'en', 'CHINESE' : 'zh-CN'}

def detect_podcast_language(episodes):
    """Detects language of podcast using lingua library (free, no API calls)"""
    # Build language detector for English, French, Dutch, and Chinese
    detector = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH,
        Language.FRENCH,
        Language.DUTCH,
        Language.CHINESE
    ).build()

    # Combine all episode titles into one text sample for better detection
    combined_text = " ".join([str(ep) for ep in episodes if ep and str(ep) != 'nan'])

    if not combined_text.strip():
        return "ENGLISH"  # Default fallback

    # Detect language
    detected = detector.detect_language_of(combined_text)

    if detected:
        # Map lingua Language enum to our format
        language_map = {
            Language.ENGLISH: "ENGLISH",
            Language.FRENCH: "FRENCH",
            Language.DUTCH: "DUTCH",
            Language.CHINESE: "CHINESE"
        }
        return language_map.get(detected, "ENGLISH")

    return "ENGLISH"  # Default fallback

def identify_language_new_pods(df, list_new, dict_pod_language):
    """Identify language for new podcasts using free lingua library"""
    print(f"📝 Detecting language for {len(list_new)} new podcasts...")

    for pod in list_new:
        list_episodes = list(df[df["podcast_name"] == pod].title.unique())
        detected_language = detect_podcast_language(list_episodes)
        dict_pod_language[pod] = detected_language
        print(f"   • {pod}: {detected_language}")

    with open(path_dict_language, 'w') as f:
        json.dump(dict_pod_language, f)

    print("✅ Language detection completed")
    return dict_pod_language

def identify_language(df, list_new):
    with open(path_dict_language, 'r') as f:
        dict_pod_language = json.load(f)
    if len(list_new) > 0:
        dict_pod_language = identify_language_new_pods(df, list_new, dict_pod_language)
    df['language'] = df['podcast_name'].map(dict_pod_language)
    return df

def clean(text_input):
    """Prepare the translated text for NLP models by removing useless parts"""
    if isinstance(text_input, float):
        return str(text_input)
    if text_input is None:
        return ''
    text_input = ''.join(char for char in text_input if not char.isdigit())
    for punctuation in string.punctuation:
        text_input = text_input.replace(punctuation, ' ') # Remove Punctuation
    lowercased = text_input.lower() # Lower Case
    unaccented_string = unidecode.unidecode(lowercased) # remove accents
    tokenized = word_tokenize(unaccented_string) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words(('english'))) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    wnl = WordNetLemmatizer() # Lemmatizing the words (keeping only the root of the words)
    verb_lemmatized = [wnl.lemmatize(word, pos = "v") for word in without_stopwords]
    adj_lemmatized = [wnl.lemmatize(word, pos = "a") for word in verb_lemmatized]
    adv_lemmatized = [wnl.lemmatize(word, pos = "r") for word in adj_lemmatized]
    noun_lemmatized = [wnl.lemmatize(word, pos = "n") for word in adv_lemmatized]
    return " ".join(noun_lemmatized)

def translate_clean_new_episodes(df, new_episodes, dict_pod_translation):
    for title in new_episodes:
        if (title is None) | (title== ''):
            dict_pod_translation[title] = ''
        else:
            language = df[df["title"]==title].language.iloc[0]
            if language == 'nan' :
                language_translation = google_lang[language]
            else:
                language_translation = 'en'
            translation = GoogleTranslator(source=language_translation, target='en').translate(title + " ")
            cleaned_translation = clean(translation)
            dict_pod_translation[title] = cleaned_translation
    with open(path_dict_translation, 'w') as f:
        json.dump(dict_pod_translation, f)
    return dict_pod_translation

def translate_clean(df):
    """Detects the language of the input, then apply google translator on it"""
    with open(path_dict_translation, 'r') as f:
        dict_pod_translation = json.load(f)
    podcast_titles = list(df["title"].unique())
    new_episodes = []
    for title in podcast_titles:
        if title not in dict_pod_translation.keys():
            new_episodes.append(title)
    dict_pod_translation = translate_clean_new_episodes(df, new_episodes, dict_pod_translation)
    df['title_cleaned_t'] = df['title'].map(dict_pod_translation)
    return df

def identify_translate_clean(df, list_new):
    """Translate & clean the titles that haven't been cleaned/translated in the past"""
    df = identify_language(df, list_new)
    df = translate_clean(df)
    return df

def open_txt_file(path):
    with open(path, 'r') as file:
        data = file.read()
    tables = data.split('\n--------')
    tables = [table.strip() for table in tables]
    return tables

def parse_table_episodes(table_str):
    # Split the string into lines
    lines = table_str.strip().split('\n')
    # The first line contains the column names
    columns = lines[0].split(',')
    # The remaining lines contain the data
    data = [line.split(',') for line in lines[1:]]
    df_episodes = pd.DataFrame(data, columns=columns)
    return df_episodes

def parse_table_history(table_str):
    # Split the string into lines
    lines = table_str.strip().split('\n')
    # The first line contains the column names
    columns = lines[0].split(',')
    # The remaining lines contain the data
    data = []
    for line in lines[1:]:
        data_line = []
        original_split = line.split(',')
        #If there is a comma in the episode title, extra step is needed
        if len(original_split) > 6:
            data_line = original_split[:4]
            data_line.append(",".join(original_split[4:-1]))
            data_line.append(original_split[-1])
        else:
            data_line = line.split(',')
        data.append(data_line)
    df_history = pd.DataFrame(data, columns=columns)

    # Turning Unix Timestamp into proper date format (UTC)
    # Convert to numeric first to handle empty strings and invalid values
    df_history['modified at'] = pd.to_numeric(df_history['modified at'], errors='coerce')
    df_history['published at'] = pd.to_numeric(df_history['published at'], errors='coerce')

    # Convert numeric timestamps to datetime (timezone-naive UTC for timezone correction)
    # Note: We create timezone-naive datetimes because time_difference_correction() expects naive timestamps
    df_history['modified at'] = pd.to_datetime(df_history['modified at'], unit='ms', utc=True, errors='coerce').dt.tz_localize(None).dt.floor('S')
    df_history['published at'] = pd.to_datetime(df_history['published at'], unit='s', utc=True, errors='coerce').dt.tz_localize(None)

    # Remove rows with null 'modified at' timestamps (required for timezone correction)
    initial_count = len(df_history)
    df_history = df_history[df_history['modified at'].notna()].copy()
    dropped_count = initial_count - len(df_history)

    if dropped_count > 0:
        print(f"⚠️  Dropped {dropped_count} rows with invalid 'modified at' timestamps")

    if len(df_history) == 0:
        print("❌ Error: No valid episodes found in history data")
        return df_history

    # Apply timezone correction to convert from UTC to local time based on location data
    df_history = time_difference_correction(df_history, 'modified at', source_timezone='UTC')
    df_history = time_difference_correction(df_history, 'published at', source_timezone='UTC')

    # Sort by modified at timestamp
    df_history = df_history.sort_values('modified at')
    return df_history

def merge_history_episodes(df_history,df_episodes):
    return df_history.merge(df_episodes[['uuid', 'played up to', 'duration']], on = 'uuid')

def completion_calculation(listen_time, duration):
    if listen_time == "":
        return "Unknown"
    elif duration == "":
        #Assumption that episode was listened in entierety if total duration is missing
        return 100
    else:
        return int((float(listen_time)/float(duration))*100)

def retrieve_name_genre(df):
    df_pod = pd.read_excel(PODCAST_MAPPING_FILE)
    dict_genre = df_pod.set_index('podcast_id')['genre'].to_dict()
    dict_podcast_name = df_pod.set_index('podcast_id')['podcast_name'].to_dict()
    df["podcast_name"] = df["podcast"].apply(lambda x: dict_podcast_name[x] if x in dict_podcast_name.keys() else "Check")
    df["genre"] = df["podcast"].apply(lambda x: dict_genre[x] if x in dict_genre.keys() else "Check")
    return df

def missing_name_genre(df, list_new):
    """
    Process new podcasts with fallback hierarchy:
    1. iTunes API auto-discovery
    2. Existing genre from mapping file backup
    3. Manual input with numbered list selection
    """
    df_pod = pd.read_excel(PODCAST_MAPPING_FILE)

    # Get existing unique genres from mapping file for selection
    existing_genres = sorted(df_pod['genre'].dropna().unique().tolist())

    print(f"\n🎙️  Processing {len(list_new)} new podcast(s)...")

    for podcast_uuid in list_new:
        new_podcast_data = {"podcast_id": podcast_uuid}

        print(f"\n{'='*60}")
        print(f"New podcast UUID: {podcast_uuid}")
        print(f"{'='*60}")

        # Show sample episode titles
        episodes = df[df['podcast'] == podcast_uuid]
        sample_episodes = list(episodes.title.unique())[:3]
        print(f"\n📝 Sample episodes:")
        for i, ep in enumerate(sample_episodes, 1):
            print(f"   {i}. {ep[:70]}..." if len(ep) > 70 else f"   {i}. {ep}")

        # Try auto-discovery first (iTunes API)
        print(f"\n{'='*60}")
        print("ATTEMPTING AUTO-DISCOVERY (using iTunes API)...")
        print(f"{'='*60}")

        discovered_info = auto_discover_podcast_info(podcast_uuid, df)

        if discovered_info:
            # Auto-discovery successful!
            new_podcast_data["podcast_name"] = discovered_info['podcast_name']
            new_podcast_data["genre"] = discovered_info['genre']  # Use iTunes genre
            print(f"\n🎉 Auto-discovery successful!")
            print(f"   Name: {new_podcast_data['podcast_name']}")
            print(f"   Genre: {new_podcast_data['genre']}")
        else:
            # Fall back to manual input
            print(f"\n{'='*60}")
            print("AUTO-DISCOVERY FAILED - MANUAL INPUT REQUIRED")
            print(f"{'='*60}")

            # Get podcast name
            new_podcast_data["podcast_name"] = input("\n❓ What's the podcast's name? ")

            # Check if mapping file has a backup genre for this podcast UUID
            # (This handles cases where name was entered before but iTunes failed)
            backup_genre = None
            if podcast_uuid in df_pod['podcast_id'].values:
                backup_genre = df_pod[df_pod['podcast_id'] == podcast_uuid]['genre'].iloc[0]
                if pd.notna(backup_genre) and backup_genre != "Check":
                    print(f"\n✅ Found backup genre in mapping file: {backup_genre}")
                    new_podcast_data["genre"] = backup_genre
                else:
                    backup_genre = None

            # If no backup genre, prompt for manual selection
            if backup_genre is None:
                print(f"\n📂 Select genre from existing genres:")
                print(f"   0. [Enter new genre manually]")
                for i, genre in enumerate(existing_genres, 1):
                    print(f"   {i}. {genre}")

                while True:
                    try:
                        genre_choice = input(f"\n❓ Select genre (0-{len(existing_genres)}): ").strip()
                        choice_num = int(genre_choice)

                        if choice_num == 0:
                            # Manual genre entry
                            new_podcast_data["genre"] = input("   Enter new genre: ").strip()
                            break
                        elif 1 <= choice_num <= len(existing_genres):
                            # Select from existing genres
                            new_podcast_data["genre"] = existing_genres[choice_num - 1]
                            break
                        else:
                            print(f"   ❌ Invalid choice. Please enter a number between 0 and {len(existing_genres)}")
                    except ValueError:
                        print(f"   ❌ Invalid input. Please enter a number.")

        # Save the new podcast to mapping file
        df_pod = pd.concat([df_pod, pd.DataFrame([new_podcast_data])], ignore_index=True)
        df_pod.to_excel(PODCAST_MAPPING_FILE, index=False)
        print(f"✅ Saved to mapping file: {new_podcast_data['podcast_name']}\n")

def add_columns(df):
    """Adds the podcast name and genre to each episode in the dataset. If missing, it is computed by user input"""
    df = retrieve_name_genre(df)
    list_new = list(df[df["podcast_name"] == "Check"]["podcast"].unique())
    if len(list_new) > 0:
        missing_name_genre(df, list_new)
        df = retrieve_name_genre(df)
    return df, list_new

def get_podcast_info(podcast_name):
    """
    Get podcast information from iTunes API.
    Returns a dictionary with podcast metadata or None if not found.
    """
    try:
        encoded_name = requests.utils.quote(podcast_name)
        url = f"https://itunes.apple.com/search?term={encoded_name}&entity=podcast"

        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

        response = requests.get(url)
        data = response.json()

        if data['resultCount'] > 0:
            podcast = data['results'][0]
            return {
                'podcast_name': podcast_name,  # Original name from our data
                'itunes_name': podcast.get('collectionName'),
                'artist': podcast.get('artistName'),
                'artwork_large': podcast.get('artworkUrl600'),
                'genre': podcast.get('primaryGenreName'),
                'feed_url': podcast.get('feedUrl')
            }
        return None

    except Exception as e:
        print(f"Error fetching info for {podcast_name}: {str(e)}")
        return None


def auto_discover_podcast_info(podcast_uuid, df):
    """
    Auto-discover podcast information using iTunes API search based on episode titles.
    Returns podcast info dict if found, None otherwise.
    """
    print(f"\n🔍 Auto-discovering podcast info for UUID: {podcast_uuid}")

    # Get sample episode titles for this podcast
    episodes = df[df['podcast'] == podcast_uuid]
    sample_titles = episodes['title'].head(5).tolist()

    print(f"📋 Found {len(sample_titles)} episode titles to search:")
    for i, title in enumerate(sample_titles[:3], 1):
        print(f"   {i}. {title[:60]}..." if len(title) > 60 else f"   {i}. {title}")

    # Try searching iTunes with each episode title
    # (Podcasts often include their name in episode metadata)
    for title in sample_titles:
        if not title or pd.isna(title):
            continue

        info = get_podcast_info(title)

        if info and info.get('itunes_name'):
            print(f"\n✅ Found potential match:")
            print(f"   Podcast: {info['itunes_name']}")
            print(f"   Creator: {info['artist']}")
            print(f"   Genre: {info['genre']}")

            # Ask user to confirm
            confirm = input(f"\n   Is this the correct podcast? (y/n): ").strip().lower()

            if confirm == 'y':
                print(f"✅ Confirmed: {info['itunes_name']}")
                # Use iTunes name as the podcast name
                info['podcast_name'] = info['itunes_name']
                return info
            else:
                print("   Trying next search...")
                continue

    print("❌ Auto-discovery failed - no match found")
    return None

def load_artwork_cache():
    """Load artwork cache from JSON file"""
    try:
        with open(ARTWORK_CACHE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ℹ️  Artwork cache file not found, creating new one")
        return {}
    except json.JSONDecodeError:
        print("⚠️  Artwork cache file corrupted, starting fresh")
        return {}


def save_artwork_cache(cache_data):
    """Save artwork cache to JSON file"""
    with open(ARTWORK_CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)


def manual_genre_override():
    """
    Interactive function to manually update podcast genres.
    Searches podcasts by name and allows updating genre using numbered list from existing genres in processed CSV.
    """
    print("\n" + "="*60)
    print("📂 MANUAL GENRE OVERRIDE")
    print("="*60)

    # Load processed CSV
    try:
        df = pd.read_csv(OUTPUT_FILE, sep="|", encoding="utf-8")
    except FileNotFoundError:
        print("❌ Processed file not found. Please process data first.")
        return False

    # Get existing unique genres from processed CSV (not mapping file)
    existing_genres = sorted(df['genre'].dropna().unique().tolist())

    # Get unique podcasts
    podcasts = df['podcast_name'].dropna().unique()
    print(f"\n📋 Found {len(podcasts)} podcasts in processed data")

    # Search for podcast
    search_term = input("\n🔍 Enter podcast name to search (partial match): ").strip().lower()

    matches = [p for p in podcasts if search_term in p.lower()]

    if not matches:
        print(f"❌ No podcasts found matching '{search_term}'")
        return False

    if len(matches) > 1:
        print(f"\n📋 Found {len(matches)} matching podcasts:")
        for i, podcast in enumerate(matches, 1):
            current_genre = df[df['podcast_name'] == podcast]['genre'].iloc[0] if 'genre' in df.columns else 'N/A'
            print(f"   {i}. {podcast}")
            print(f"      Current genre: {current_genre}")

        try:
            choice = int(input(f"\n❓ Select podcast (1-{len(matches)}): ").strip())
            if 1 <= choice <= len(matches):
                podcast_name = matches[choice - 1]
            else:
                print("❌ Invalid choice")
                return False
        except ValueError:
            print("❌ Invalid input")
            return False
    else:
        podcast_name = matches[0]

    # Show current genre
    current_genre = df[df['podcast_name'] == podcast_name]['genre'].iloc[0] if 'genre' in df.columns else 'N/A'
    print(f"\n📺 Podcast: {podcast_name}")
    print(f"   Current genre: {current_genre}")

    # Genre selection with numbered list
    print(f"\n📂 Select new genre from existing genres:")
    print(f"   0. [Enter new genre manually]")
    for i, genre in enumerate(existing_genres, 1):
        print(f"   {i}. {genre}")

    while True:
        try:
            genre_choice = input(f"\n❓ Select genre (0-{len(existing_genres)}, or 'cancel' to abort): ").strip()

            if genre_choice.lower() == 'cancel':
                print("❌ Cancelled")
                return False

            choice_num = int(genre_choice)

            if choice_num == 0:
                # Manual genre entry
                new_genre = input("   Enter new genre: ").strip()
                break
            elif 1 <= choice_num <= len(existing_genres):
                # Select from existing genres
                new_genre = existing_genres[choice_num - 1]
                break
            else:
                print(f"   ❌ Invalid choice. Please enter a number between 0 and {len(existing_genres)}")
        except ValueError:
            print(f"   ❌ Invalid input. Please enter a number.")

    # Update all rows for this podcast in the CSV
    df.loc[df['podcast_name'] == podcast_name, 'genre'] = new_genre

    # Save updated CSV
    df.to_csv(OUTPUT_FILE, sep="|", encoding="utf-8", index=False)

    # Update podcast mapping file
    df_pod = pd.read_excel(PODCAST_MAPPING_FILE)
    df_pod.loc[df_pod['podcast_name'] == podcast_name, 'genre'] = new_genre
    df_pod.to_excel(PODCAST_MAPPING_FILE, index=False)

    print(f"\n✅ Genre updated successfully!")
    print(f"   Podcast: {podcast_name}")
    print(f"   Old genre: {current_genre}")
    print(f"   New genre: {new_genre}")

    # Ask if user wants to upload to Drive
    upload_choice = input("\n❓ Upload updated file to Google Drive? (y/n): ").strip().lower()
    if upload_choice == 'y':
        upload_pocket_casts_results()

    return True


def manual_artwork_override():
    """
    Interactive function to manually update podcast artwork URLs.
    Searches podcasts by name and allows updating artwork URL.
    """
    print("\n" + "="*60)
    print("🎨 MANUAL ARTWORK OVERRIDE")
    print("="*60)

    # Load processed CSV
    try:
        df = pd.read_csv(OUTPUT_FILE, sep="|", encoding="utf-8")
    except FileNotFoundError:
        print("❌ Processed file not found. Please process data first.")
        return False

    # Load artwork cache
    artwork_cache = load_artwork_cache()

    # Get unique podcasts
    podcasts = df['podcast_name'].dropna().unique()
    print(f"\n📋 Found {len(podcasts)} podcasts in processed data")

    # Search for podcast
    search_term = input("\n🔍 Enter podcast name to search (partial match): ").strip().lower()

    matches = [p for p in podcasts if search_term in p.lower()]

    if not matches:
        print(f"❌ No podcasts found matching '{search_term}'")
        return False

    if len(matches) > 1:
        print(f"\n📋 Found {len(matches)} matching podcasts:")
        for i, podcast in enumerate(matches, 1):
            current_url = df[df['podcast_name'] == podcast]['artwork_url'].iloc[0] if 'artwork_url' in df.columns else 'N/A'
            url_display = current_url[:50] + "..." if len(str(current_url)) > 50 else current_url
            print(f"   {i}. {podcast}")
            print(f"      Current: {url_display}")

        try:
            choice = int(input(f"\n❓ Select podcast (1-{len(matches)}): ").strip())
            if 1 <= choice <= len(matches):
                podcast_name = matches[choice - 1]
            else:
                print("❌ Invalid choice")
                return False
        except ValueError:
            print("❌ Invalid input")
            return False
    else:
        podcast_name = matches[0]

    # Show current artwork URL
    current_url = df[df['podcast_name'] == podcast_name]['artwork_url'].iloc[0] if 'artwork_url' in df.columns else 'N/A'
    print(f"\n📺 Podcast: {podcast_name}")
    print(f"   Current artwork URL: {current_url}")

    # Get new URL
    new_url = input("\n❓ Enter new artwork URL (or 'cancel' to abort): ").strip()

    if new_url.lower() == 'cancel':
        print("❌ Cancelled")
        return False

    # Update cache
    artwork_cache[podcast_name] = new_url
    save_artwork_cache(artwork_cache)

    # Update all rows for this podcast in the CSV
    df.loc[df['podcast_name'] == podcast_name, 'artwork_url'] = new_url

    # Save updated CSV
    df.to_csv(OUTPUT_FILE, sep="|", encoding="utf-8", index=False)

    print(f"\n✅ Artwork updated successfully!")
    print(f"   Podcast: {podcast_name}")
    print(f"   New URL: {new_url[:60]}..." if len(new_url) > 60 else f"   New URL: {new_url}")

    # Ask if user wants to upload to Drive
    upload_choice = input("\n❓ Upload updated file to Google Drive? (y/n): ").strip().lower()
    if upload_choice == 'y':
        upload_pocket_casts_results()

    return True


def manual_itunes_search():
    """
    Interactive function to manually search iTunes API and select the correct podcast from multiple results.
    Useful when automatic discovery selected the wrong podcast.
    """
    print("\n" + "="*60)
    print("🔍 MANUAL ITUNES API SEARCH")
    print("="*60)

    # Load processed CSV
    try:
        df = pd.read_csv(OUTPUT_FILE, sep="|", encoding="utf-8")
    except FileNotFoundError:
        print("❌ Processed file not found. Please process data first.")
        return False

    # Load iTunes API cache
    try:
        api_results_df = pd.read_csv(ITUNES_CACHE_FILE, sep="|", quoting=1, escapechar='\\')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("❌ iTunes API cache not found. Please run the pipeline first.")
        return False

    # Get unique podcasts
    podcasts = df['podcast_name'].dropna().unique()
    print(f"\n📋 Found {len(podcasts)} podcasts in processed data")

    # Search for podcast
    search_term = input("\n🔍 Enter podcast name to search (partial match): ").strip().lower()

    matches = [p for p in podcasts if search_term in p.lower()]

    if not matches:
        print(f"❌ No podcasts found matching '{search_term}'")
        return False

    if len(matches) > 1:
        print(f"\n📋 Found {len(matches)} matching podcasts:")
        for i, podcast in enumerate(matches, 1):
            current_itunes = df[df['podcast_name'] == podcast]['itunes_name'].iloc[0] if 'itunes_name' in df.columns else 'N/A'
            print(f"   {i}. {podcast}")
            print(f"      Current iTunes match: {current_itunes}")

        try:
            choice = int(input(f"\n❓ Select podcast (1-{len(matches)}): ").strip())
            if 1 <= choice <= len(matches):
                podcast_name = matches[choice - 1]
            else:
                print("❌ Invalid choice")
                return False
        except ValueError:
            print("❌ Invalid input")
            return False
    else:
        podcast_name = matches[0]

    # Show current iTunes match
    current_itunes = df[df['podcast_name'] == podcast_name]['itunes_name'].iloc[0] if 'itunes_name' in df.columns else 'N/A'
    current_genre = df[df['podcast_name'] == podcast_name]['genre'].iloc[0] if 'genre' in df.columns else 'N/A'
    print(f"\n📺 Podcast: {podcast_name}")
    print(f"   Current iTunes match: {current_itunes}")
    print(f"   Current genre: {current_genre}")

    # Get search query from user
    search_query = input("\n🔍 Enter search term for iTunes API (or press Enter to use podcast name): ").strip()
    if not search_query:
        search_query = podcast_name

    # Fetch multiple results from iTunes API
    print(f"\n🔄 Searching iTunes API for: {search_query}")
    try:
        encoded_name = requests.utils.quote(search_query)
        url = f"https://itunes.apple.com/search?term={encoded_name}&entity=podcast&limit=10"
        time.sleep(0.5)
        response = requests.get(url)
        data = response.json()

        if data['resultCount'] == 0:
            print("❌ No results found from iTunes API")
            return False

        # Display all results
        print(f"\n📋 Found {data['resultCount']} result(s) from iTunes API:")
        print(f"   0. [Cancel - don't update]")

        results = []
        for i, podcast in enumerate(data['results'][:10], 1):  # Limit to 10 results
            itunes_name = podcast.get('collectionName', 'N/A')
            artist = podcast.get('artistName', 'N/A')
            genre = podcast.get('primaryGenreName', 'N/A')
            print(f"   {i}. {itunes_name}")
            print(f"      Creator: {artist}")
            print(f"      Genre: {genre}")
            results.append({
                'podcast_name': podcast_name,
                'itunes_name': itunes_name,
                'artist': artist,
                'artwork_large': podcast.get('artworkUrl600'),
                'genre': genre,
                'feed_url': podcast.get('feedUrl')
            })

        # User selection
        while True:
            try:
                choice = input(f"\n❓ Select the correct podcast (0-{len(results)}, or 'cancel'): ").strip()

                if choice.lower() == 'cancel':
                    print("❌ Cancelled")
                    return False

                choice_num = int(choice)

                if choice_num == 0:
                    print("❌ Cancelled - no updates made")
                    return False
                elif 1 <= choice_num <= len(results):
                    selected_info = results[choice_num - 1]
                    break
                else:
                    print(f"   ❌ Invalid choice. Please enter a number between 0 and {len(results)}")
            except ValueError:
                print(f"   ❌ Invalid input. Please enter a number.")

    except Exception as e:
        print(f"❌ Error fetching from iTunes API: {str(e)}")
        return False

    # Update iTunes API cache
    if podcast_name in api_results_df['podcast_name'].values:
        # Update existing entry
        for col, value in selected_info.items():
            api_results_df.loc[api_results_df['podcast_name'] == podcast_name, col] = value
    else:
        # Add new entry
        api_results_df = pd.concat([api_results_df, pd.DataFrame([selected_info])], ignore_index=True)

    api_results_df.to_csv(ITUNES_CACHE_FILE, index=False, sep="|", quoting=1, escapechar='\\')

    # Update processed CSV
    df.loc[df['podcast_name'] == podcast_name, 'itunes_name'] = selected_info['itunes_name']
    df.loc[df['podcast_name'] == podcast_name, 'artist'] = selected_info['artist']
    df.loc[df['podcast_name'] == podcast_name, 'genre'] = selected_info['genre']
    df.loc[df['podcast_name'] == podcast_name, 'artwork_url'] = selected_info['artwork_large']
    df.loc[df['podcast_name'] == podcast_name, 'feed_url'] = selected_info['feed_url']
    df.to_csv(OUTPUT_FILE, sep="|", encoding="utf-8", index=False)

    # Update podcast mapping file
    df_pod = pd.read_excel(PODCAST_MAPPING_FILE)
    df_pod.loc[df_pod['podcast_name'] == podcast_name, 'genre'] = selected_info['genre']
    df_pod.to_excel(PODCAST_MAPPING_FILE, index=False)

    print(f"\n✅ iTunes match updated successfully!")
    print(f"   Podcast: {podcast_name}")
    print(f"   New iTunes match: {selected_info['itunes_name']}")
    print(f"   New genre: {selected_info['genre']}")

    # Ask if user wants to upload to Drive
    upload_choice = input("\n❓ Upload updated file to Google Drive? (y/n): ").strip().lower()
    if upload_choice == 'y':
        upload_pocket_casts_results()

    return True


def enrich_podcast_data(df):
    """
    Enrich podcast episodes data with iTunes metadata.
    Also applies manual artwork overrides from cache.
    """
    # Load artwork cache for manual overrides
    artwork_cache = load_artwork_cache()

    # 1. Get unique podcast names from episodes
    unique_podcasts = df['podcast_name'].unique()

    # 2. Load existing API results if file exists
    try:
        # Try reading with standard settings first
        api_results_df = pd.read_csv(ITUNES_CACHE_FILE, sep="|")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Create empty DataFrame if file doesn't exist or is empty
        print("⚠️  iTunes API cache file not found, creating new one")
        api_results_df = pd.DataFrame(columns=['podcast_name', 'itunes_name', 'artist', 'artwork_large', 'genre', 'feed_url'])
    except pd.errors.ParserError:
        # If parsing fails, try with quoting to handle special characters
        print("⚠️  iTunes API cache has formatting issues, attempting recovery...")
        try:
            api_results_df = pd.read_csv(ITUNES_CACHE_FILE, sep="|", quoting=1, escapechar='\\')
        except:
            # If still fails, start fresh
            print("❌ Could not parse existing iTunes API cache, starting fresh")
            api_results_df = pd.DataFrame(columns=['podcast_name', 'itunes_name', 'artist', 'artwork_large', 'genre', 'feed_url'])

    # 3. Get API data for new podcasts
    counter = 0
    for podcast in unique_podcasts:
        counter += 1
        if pd.isna(podcast):
            continue
        # Check if podcast is not in existing results
        if api_results_df.empty or podcast not in api_results_df['podcast_name'].values:
            print(f"Fetching Itunes API data for {podcast}")
            info = get_podcast_info(podcast)
            if info:
                # Convert dict to DataFrame and append to existing results
                new_row_df = pd.DataFrame([info])
                api_results_df = pd.concat([api_results_df, new_row_df], ignore_index=True)
                print(f"Results for {podcast} were fetched")
        if counter%10 == 0:
            api_results_df.to_csv(ITUNES_CACHE_FILE, index=False, sep="|", quoting=1, escapechar='\\')
            print("Temporary API results saved")

    # Save updated API results with proper quoting to handle special characters
    api_results_df.to_csv(ITUNES_CACHE_FILE, index=False, sep="|", quoting=1, escapechar='\\')
    print("All API results saved")

    # 4. Join episodes with API results
    enriched_df = df.merge(api_results_df, on='podcast_name', how='left', suffixes=('_mapping', '_itunes'))

    # 5. Merge genre columns: iTunes genre takes precedence, falls back to mapping file genre
    if 'genre_itunes' in enriched_df.columns and 'genre_mapping' in enriched_df.columns:
        # Use iTunes genre when available (non-null), otherwise use mapping file backup
        enriched_df['genre'] = enriched_df['genre_itunes'].fillna(enriched_df['genre_mapping'])
        # Drop the temporary genre columns
        enriched_df = enriched_df.drop(columns=['genre_itunes', 'genre_mapping'])
    elif 'genre_itunes' in enriched_df.columns:
        # Only iTunes genre exists (shouldn't happen, but handle it)
        enriched_df['genre'] = enriched_df['genre_itunes']
        enriched_df = enriched_df.drop(columns=['genre_itunes'])
    elif 'genre_mapping' in enriched_df.columns:
        # Only mapping genre exists (shouldn't happen, but handle it)
        enriched_df['genre'] = enriched_df['genre_mapping']
        enriched_df = enriched_df.drop(columns=['genre_mapping'])

    # 6. Apply manual artwork overrides from cache
    if artwork_cache:
        print(f"\n🎨 Applying {len(artwork_cache)} manual artwork override(s)...")
        for podcast_name, artwork_url in artwork_cache.items():
            if podcast_name in enriched_df['podcast_name'].values:
                enriched_df.loc[enriched_df['podcast_name'] == podcast_name, 'artwork_large'] = artwork_url
                print(f"   ✅ {podcast_name}")

    return enriched_df


def create_pocket_casts_file():
    """Step 3: Process Pocket Casts data and create processed CSV file"""
    print("\n🔄 Processing Pocket Casts data...")

    try:
        # Verify export file exists
        if not os.path.exists(EXPORT_DATA_FILE):
            print(f"❌ Export file not found: {EXPORT_DATA_FILE}")
            return False

        tables = open_txt_file(EXPORT_DATA_FILE)

        # Get status of the different episodes I listened
        table_episodes = tables[3]
        df_episodes = parse_table_episodes(table_episodes).drop(parse_table_episodes(table_episodes).index[-2:])

        # Retrieve my listening history + additional information about the episodes
        table_history = tables[5].split('\n-------')[2]
        df_history = parse_table_history(table_history).drop(parse_table_history(table_history).index[-2:])
        df = merge_history_episodes(df_history,df_episodes)
        df["completion_percent"] = df.apply(lambda x: completion_calculation(x["played up to"], x["duration"]),axis = 1)

        # Add podcast names and genres
        df, list_new = add_columns(df)
        df = identify_translate_clean(df, list_new)
        df.sort_values('modified at', ascending=True, inplace = True)
        df['is_new_podcast'] = df.groupby('podcast_name').cumcount() == 0
        df['is_new_podcast'] = df['is_new_podcast'].astype(int)
        df['is_recurring_podcast'] = df.groupby('podcast_name').cumcount() == 5
        df['is_recurring_podcast'] = df['is_recurring_podcast'].astype(int)
        df = enrich_podcast_data(df)
        df.sort_values('modified at', ascending=False, inplace = True)

        # Rename columns to snake_case standard
        df = df.rename(columns={
            'uuid': 'episode_uuid',
            'modified at': 'listened_date',
            'podcast': 'podcast_id',
            'published at': 'published_date',
            'title': 'episode_title',
            'url': 'episode_url',
            'played up to': 'listened_seconds',
            'duration': 'duration_seconds',
            'title_cleaned_t': 'title_cleaned_translated',
            'artwork_large': 'artwork_url'
        })

        # Reorder columns logically: identifiers → descriptive → numerical → dates → booleans
        column_order = [
            'episode_uuid', 'podcast_id', 'podcast_name', 'episode_title', 'episode_url',
            'genre', 'duration_seconds', 'listened_seconds', 'completion_percent',
            'published_date', 'listened_date', 'is_new_podcast', 'is_recurring_podcast',
            'language', 'title_cleaned_translated', 'itunes_name', 'artist', 'artwork_url',
            'feed_url'
        ]

        # Only include columns that exist in the dataframe
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]

        # Enforce snake_case before saving
        df = enforce_snake_case(df, "processed file")

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save with UTF-8 encoding (critical for website compatibility)
        df.to_csv(OUTPUT_FILE, sep="|", encoding="utf-8", index=False)

        print("✅ Processing completed")

        # Generate website files
        return True

    except Exception as e:
        print(f"❌ Error processing Pocket Casts data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def download_pocket_casts_data():
    """Step 1: Instructions for obtaining Pocket Casts GDPR export"""
    print("\n🎙️  POCKET CASTS DATA EXPORT INSTRUCTIONS")
    print("="*60)
    print("\nPocket Casts requires a GDPR data request to export your listening history.")
    print("\nSteps to request your data:")
    print("1. Email Pocket Casts support or use their GDPR request form")
    print("2. Wait for them to prepare your data export (usually 1-3 days)")
    print("3. Download the zip file they email you")
    print("4. Save the zip file to your Downloads folder")
    print("\nOnce you have the export file in Downloads, you can proceed with processing.")
    print("="*60)

    choice = input("\nHave you already downloaded the export to your Downloads folder? (y/n): ").strip().lower()

    if choice == 'y':
        print("✅ Ready to proceed with processing")
        return True
    else:
        print("⚠️  Please download the export file first, then run this pipeline again")
        return False


def move_pocket_casts_files():
    """Step 2: Find, unzip, and move Pocket Casts export files"""
    print("\n📁 Finding and organizing Pocket Casts export files...")

    try:
        find_unzip_folder("pocket_casts")
        clean_rename_move_folder("files/exports", os.path.expanduser("~/Downloads"), "pocket_casts_export_unzipped", "pocket_casts_exports")
        print("✅ Files moved successfully")
        return True
    except Exception as e:
        print(f"❌ Error moving files: {str(e)}")
        return False


def full_pocket_casts_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Pocket Casts SOURCE pipeline with 4 options.

    Options:
    1. Download new data and process
    2. Process existing data
    3. Manual genre override
    4. Manual artwork override
    """
    print("\n" + "="*60)
    print("🎙️  POCKET CASTS SOURCE DATA PIPELINE")
    print("="*60)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new data and process")
        print("2. Process existing data")
        print("3. Manual genre override")
        print("4. Manual artwork override")

        choice = input("\nEnter your choice (1-4): ").strip()

    success = False

    try:
        if choice == "1":
            print("\n🚀 Download new data and process...")
            download_success = download_pocket_casts_data()

            if download_success:
                move_success = move_pocket_casts_files()
            else:
                print("⚠️  Download not confirmed, but checking for existing files...")
                move_success = move_pocket_casts_files()

            if move_success:
                process_success = create_pocket_casts_file()
            else:
                print("⚠️  No new files found, attempting to process existing files...")
                process_success = create_pocket_casts_file()

            success = process_success

        elif choice == "2":
            print("\n⚙️  Process existing data...")
            success = create_pocket_casts_file()

        elif choice == "3":
            print("\n🎨 Manual genre override...")
            manual_genre_override()
            print("\n✅ Genre overrides applied")
            print("⚠️  Run option 2 to reprocess with new genres")
            return True

        elif choice == "4":
            print("\n🎨 Manual artwork override...")
            manual_artwork_override()
            print("\n✅ Artwork overrides applied")
            print("⚠️  Run option 2 to reprocess with new artwork")
            return True

        else:
            print("❌ Invalid choice. Please select 1-4.")
            return False

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Pocket Casts source pipeline completed successfully!")
        print("ℹ️  Note: To upload to Drive, run the Podcasts topic pipeline.")
        record_successful_run('source_pocket_casts', 'active')
    else:
        print("❌ Pocket Casts source pipeline failed")
    print("="*60)

    return success


# ============================================================================
# LEGACY FUNCTION (for backward compatibility)
# ============================================================================

def process_pocket_casts_export(upload="Y"):
    """Legacy function - use full_pocket_casts_pipeline() instead"""
    print('⚠️  This is a legacy function. Please use full_pocket_casts_pipeline() instead.')
    print('Starting the processing of the Pocket Casts export \n')
    find_unzip_folder("pocket_casts")
    clean_rename_move_folder("files/exports", os.path.expanduser("~/Downloads"), "pocket_casts_export_unzipped", "pocket_casts_exports")
    create_pocket_casts_file()
    print('Pocket Cast processed files were created \n')
    print(f'📁 Output: {OUTPUT_FILE}')

if __name__ == "__main__":
    full_pocket_casts_pipeline()
