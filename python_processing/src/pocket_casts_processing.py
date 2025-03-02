import pandas as pd
import string
import unidecode
import os
import json
from dotenv import load_dotenv
from utils import time_difference_correction, get_response, clean_rename_move_folder, find_unzip_folder
from lingua import Language, LanguageDetectorBuilder
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from langdetect import detect
#from googletrans import Translator
from openai import OpenAI
from drive_storage import update_drive
import requests
import time
load_dotenv()

path_dict_language = 'files/work_files/pocket_casts_work_files/podcasts_titles_language.json'
path_dict_translation = 'files/work_files/pocket_casts_work_files/podcasts_cleaned_translated.json'
google_lang = {'FRENCH' : 'fr','DUTCH' : 'nl','ENGLISH': 'en', 'CHINESE' : 'zh-CN'}

def gpt_podcast_language(client, podcasts):
    """Generates a ChatGPT prompt to get language of the different podcasts"""
    system_prompt = """You are a helpful linguistic expert who is able to identify the language of
    the texts you are given as input.
    You are able to understand french, chinese (traditionnal and simplified), dutch
    and english alike."""
    user_prompt = f"""Please identify the language used in the podcast episodes names
    delimited by triple backticks below.
    Please just answer with only the information, in capital letters, and it should always be one of these 4 languages:
    ENGLISH, FRENCH, DUTCH or CHINESE. If you are unsure, say ENGLISH.
    Below are examples of how your answer should look like:
    Podcast list 1: ['Ep. 183 | The Nanjing Massacre (Part 2)', 'Ep. 182 | The Nanjing Massacre (Part 1)',
                     'CHP-006 The Opium War'] Output: ENGLISH
    Podcast list 2: ['The Leftovers, la fin est proche', 'Que faut-il attendre du retour de Twin Peaks ?',
                     'Quel avenir pour HBO ?'] Output: FRENCH
    Podcast list 3: ['#23: “Voor het echte verhaal van Thiago’s transfer is het wachten op de memoires van Bart Verhaeghe”',
                     '#11: "Over Anderlechts mislukte transfers is veel gesproken, maar die van Club Brugge de laatste jaren kunnen ook tellen"',
                     '#8:  “Brandon Mechele is een degelijke verdediger, maar iemand van het niveau Jan Vertonghen heeft Club Brugge niet”']
                    Output: DUTCH
    Podcast list 4: ['2.2.42B《此心安处是吾乡》', '3.1.9B《我的父亲》','2.3.15AB《爱的代价》'] Output: CHINESE
    Podcast list 5: ['John Locke', 'Petain','XYZ'] Output: ENGLISH
    Now provide the same for the following podcast episodes ```{podcasts}```"""
    return get_response(client, system_prompt, user_prompt)

def identify_language_new_pods(df, list_new, dict_pod_language):
    api_key = os.environ['OpenAI_Key']
    client = OpenAI(api_key = api_key)
    for pod in list_new:
        list_episodes = list(df[df["podcast_name"] == pod].title.unique())
        dict_pod_language[pod] = gpt_podcast_language(client, list_episodes)
    with open(path_dict_language, 'w') as f:
        json.dump(dict_pod_language, f)
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
    #Turning Unix Timestamp into proper date format
    df_history['modified at'] = pd.to_datetime(df_history['modified at'], unit = 'ms')
    #Removing milliseconds and adjusting to time difference in Taiwan
    df_history['modified at'] = pd.to_datetime(df_history['modified at'], utc = True).dt.floor('S').apply(lambda x: time_difference_correction(x)).sort_values()
    df_history['published at'] = pd.to_datetime(df_history['published at'], unit = 's', utc = True).apply(lambda x: time_difference_correction(x)).sort_values()
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
    df_pod = pd.read_excel('files/work_files/pocket_casts_work_files/podcast_mapping.xlsx')
    dict_podcast_genre = df_pod.set_index('podcast_id')['podcast_genre'].to_dict()
    dict_podcast_name = df_pod.set_index('podcast_id')['podcast_name'].to_dict()
    df["podcast_name"] = df["podcast"].apply(lambda x: dict_podcast_name[x] if x in dict_podcast_name.keys() else "Check")
    df["podcast_genre"] = df["podcast"].apply(lambda x: dict_podcast_genre[x] if x in dict_podcast_genre.keys() else "Check")
    return df

def missing_name_genre(df, list_new):
    df_pod = pd.read_excel('files/work_files/pocket_casts_work_files/podcast_mapping.xlsx')
    print(f"{len(list_new)} new podcasts")
    for new in list_new:
        #Retrieve the
        new_podcast_data = {}
        print(new)
        new_podcasts = list(df[df['podcast'] == new].title.unique())
        print(f"These are the episodes from this new podcast : {new_podcasts}" + "\n")
        new_podcast_data["podcast_id"] = new
        new_podcast_data["podcast_name"] = input("What's the podcast's name ? " + "\n")
        new_podcast_data["podcast_genre"] = input("What's the podcast's genre ? Please choose from the following list: " + "\n" +
                                                  "News and Current Affairs" + "\n"  + "Real-life stories" + "\n"  + "Educational" + "\n"  +
                                                  "Sports" + "\n"  + "History" + "\n"  + "Humor" + "\n"  + "Technology" + "\n"  + "Horror" + "\n"  +
                                                  "Culture" + "\n"  + "Self-Improvement" + " : ")
        df_pod = df_pod.append(new_podcast_data, ignore_index=True)
        df_pod.to_excel('files/work_files/pocket_casts_work_files/podcast_mapping.xlsx', index = False)

def add_columns(df):
    """Adds the podcast name and genre to each episode in the dataset. If missing, it is computed by user input"""
    df = retrieve_name_genre(df)
    list_new = list(df[df["podcast_name"] == "Check"]["podcast"].unique())
    if len(list_new) > 0:
        missing_name_genre(df, list_new)
        df = retrieve_name_genre(df)
    return df, list_new

def get_podcast_info_from_pocketcast(uuid):
    """
    Get podcast information from Pocket Casts private API.
    """
    try:
        url = f"https://api.pocketcasts.com/discover/show/{uuid}"

        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

        # Make request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse response
        data = response.json()

        return {
            'uuid': uuid,
            'title': data.get('title'),
            'author': data.get('author'),
            'description': data.get('description'),
            'category': data.get('category'),
            'language': data.get('language'),
            'website': data.get('website'),
            'artwork_url': data.get('artwork'),
            'subscribers': data.get('subscribers')
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching info for UUID {uuid}: {str(e)}")
        return None


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

def enrich_podcast_data(df, api_results_path):
    """
    Enrich podcast episodes data with iTunes metadata.
    """
    # 1. Get unique podcast names from episodes
    unique_podcasts = df['podcast_name'].unique()

    # 2. Load existing API results if file exists
    api_results_df = pd.read_csv(api_results_path, sep = "|")

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
            api_results_df.to_csv(api_results_path, index=False, sep = "|")
            print("Temporary API results saved")

    # Save updated API results
    api_results_df.to_csv(api_results_path, index=False)
    print("All API results saved")
    # 4. Join episodes with API results
    enriched_df = df.merge(api_results_df, on='podcast_name', how='left')
    return enriched_df


def create_pocket_cast_file():
    tables = open_txt_file('files/exports/pocket_casts_exports/data.txt')
    #Get status of the different episodes I listened
    table_episodes = tables[3]
    df_episodes = parse_table_episodes(table_episodes).drop(parse_table_episodes(table_episodes).index[-2:])
    #Retrieve my listening history + additional information about the episodes
    table_history = tables[5].split('\n-------')[2]
    df_history = parse_table_history(table_history).drop(parse_table_history(table_history).index[-2:])
    df = merge_history_episodes(df_history,df_episodes)
    df["completion_%"] = df.apply(lambda x: completion_calculation(x["played up to"], x["duration"]),axis = 1)
    #print(df_episodes)
    #list_new = list(df[df["podcast_name"] == "Check"]["podcast"].unique())
    df, list_new = add_columns(df)
    df = identify_translate_clean(df, list_new)
    df.sort_values('modified at', ascending=True, inplace = True)
    df['new_podcast_yn'] = df.groupby('podcast_name').cumcount() == 0
    df['new_podcast_yn'] = df['new_podcast_yn'].astype(int)
    df['new_recurring_podcast_yn'] = df.groupby('podcast_name').cumcount() == 5
    df['new_recurring_podcast_yn'] = df['new_recurring_podcast_yn'].astype(int)
    df = enrich_podcast_data(df, "files/work_files/pocket_casts_work_files/itunes_api_podcasts.csv")
    df.sort_values('modified at', ascending=False, inplace = True)
    df.to_csv('files/processed_files/pocket_casts_processed.csv', sep = "|", encoding = "utf-16")

def process_pocket_casts_export(upload="Y"):
    file_names = []
    print('Starting the processing of the Pocket Casts export \n')
    find_unzip_folder("pocket_casts")
    clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "pocket_casts_export_unzipped", "pocket_casts_exports")
    create_pocket_cast_file()
    file_names.append('files/processed_files/pocket_casts_processed.csv')
    if upload == "Y":
        update_drive(file_names)
        print('Pocket Cast processed files were created and uploaded to the Drive \n')
    else:
        print('Pocket Cast processed files were created \n')

#process_pocket_casts_export()
