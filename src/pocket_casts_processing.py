import pandas as pd
from dotenv import load_dotenv
from utils import time_difference_correction
from lingua import Language, LanguageDetectorBuilder
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import unidecode
from drive_storage import update_file
import os
from langdetect import detect
from googletrans import Translator
import time
load_dotenv()

path_dict_language = 'files/work_files/pocket_casts_work_files/podcasts_language.csv'
path_dict_translation = 'files/work_files/pocket_casts_work_files/podcasts_cleaned_translated.csv'

def import_dict(option = '1'):
    """Imports the already translated records for the different features to translate"""
    if option == '1':
        cleaned_t_podcast = {row['Title']: row['Language'] for _, row in pd.read_csv(path_dict_language, sep='|').iterrows()}
    if option == '2':
        cleaned_t_podcast = {row['Original']: row['Cleaned&Translated'] for _, row in pd.read_csv(path_dict_translation, sep='|').iterrows()}
    return cleaned_t_podcast

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

def find_language(title, podcast):
    """This function first checks the result of the LanguageDetectorBuilder.
    As it is quite good for English, French and Chinese, it keeps result if so identified.
    If DUTCH, it checks with langdetect, if it agrees, then DUTCH.
    If they have different opinion, we use googletrans (more reliable but fails after too many requests)"""
    if (title is None) | (title== ''):
        return ''
    languages = [Language.DUTCH, Language.FRENCH, Language.ENGLISH, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    language_detection_input = f"{str(title)} - {str(podcast)}"
    language_LDB = str(detector.detect_language_of(language_detection_input))[9:]
    if language_LDB in ['CHINESE', 'FRENCH', 'ENGLISH']:
        return language_LDB
    language_LD = detect(language_detection_input)
    if (language_LD == 'nl') & (language_LDB == 'DUTCH'):
        return language_LDB
    else:
        return find_specific_cases_language(title,podcast)

def find_specific_cases_language(title, podcast):
    if (title is None) | (title== ''):
        return ''
    google_lang = {'fr' : 'FRENCH','nl' : 'DUTCH','en': 'ENGLISH', 'zh-TW' : 'CHINESE', 'zh-CN' : 'CHINESE'}
    translator = Translator()
    print(podcast, title)
    language_detection_input = f"{str(title)} - {str(podcast)}"
    if language_detection_input:
        translator = Translator()
        detected_lang = translator.detect(language_detection_input).lang
        print(detected_lang)
        return google_lang[detected_lang] if detected_lang in google_lang.keys() else 'ENGLISH'
    else:
        # Handle the case when the input text is None or empty
        return None

def translate(title, language):
    """Detects the language of the input, then apply google translator on it"""
    if (title is None) | (title== ''):
        return ''
    google_lang = {'FRENCH' : 'fr','DUTCH' : 'nl','ENGLISH': 'en', 'CHINESE' : 'zh-CN'}
    language_translation = google_lang[language]
    translation = GoogleTranslator(source=language_translation, target='en').translate(title + " ")
    return translation

def save_dict(dict_pod, option = '1'):
    """Save the new translations in the translated records dictionnaries"""
    new_cleaned_t_podcast= {}
    if option == '1':
        new_cleaned_t_podcast['Title'] = list(dict_pod.keys())
        new_cleaned_t_podcast['Language'] = list(dict_pod.values())
        pd.DataFrame(new_cleaned_t_podcast).to_csv(path_dict_language, sep = '|')
    if option == '2':
        new_cleaned_t_podcast['Original'] = list(dict_pod.keys())
        new_cleaned_t_podcast['Cleaned&Translated'] = list(dict_pod.values())
        pd.DataFrame(new_cleaned_t_podcast).to_csv(path_dict_translation, sep = '|')

def translate_clean(df):
    """Translate & clean the titles that haven't been cleaned/translated in the past"""
    dict_pod_language = import_dict('1')
    dict_pod_translated = import_dict('2')
    for _, row in df.iterrows():
        if row['title'] not in dict_pod_language.keys():
            language = find_language(row['title'], row['podcast_name'])
            dict_pod_language[row['title']] = language
            save_dict(dict_pod_language, '1')
        if row['title'] not in dict_pod_translated.keys():
            language = find_language(row['title'], row['podcast_name'])
            translated = translate(row['title'], language)
            cleaned = clean(translated)
            dict_pod_translated[row['title']] = cleaned
    df['title_cleaned_t'] = df['title'].map(dict_pod_translated)
    df['language'] = df['title'].map(dict_pod_language)
    save_dict(dict_pod_language, '1')
    save_dict(dict_pod_translated, '2')
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
    df_history['modified at'] = df_history['modified at'].dt.floor('S').apply(lambda x: time_difference_correction(x)).sort_values()
    df_history['published at'] = pd.to_datetime(df_history['published at'], unit = 's').apply(lambda x: time_difference_correction(x)).sort_values()
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

def retrieve_podcast_genre(podcast_id):
    df = pd.read_excel('files/work_files/pocket_casts_work_files/podcast_mapping.xlsx')
    dict_podcast_genre = df.set_index('podcast_id')['podcast_genre'].to_dict()
    if podcast_id in dict_podcast_genre.keys():
        return dict_podcast_genre[podcast_id]
    else:
        return "Unknown"

def retrieve_podcast_name(podcast_id, list_new = []):
    df = pd.read_excel('files/work_files/pocket_casts_work_files/podcast_mapping.xlsx')
    dict_podcast_name = df.set_index('podcast_id')['podcast_name'].to_dict()
    if podcast_id in dict_podcast_name.keys():
        return dict_podcast_name[podcast_id]
    else:
        if podcast_id not in list_new:
            list_new.append(podcast_id)
        return "Unknown"

def process_pocket_casts_export():
    tables = open_txt_file('files/exports/pocket_casts_exports/data.txt')
    #Get status of the different episodes I listened
    table_episodes = tables[3]
    df_episodes = parse_table_episodes(table_episodes).drop(parse_table_episodes(table_episodes).index[-2:])
    #GRetrieve my listening history + additionnal information about the episodes
    table_history = tables[5].split('\n-------')[2]
    df_history = parse_table_history(table_history).drop(parse_table_history(table_history).index[-2:])
    df = merge_history_episodes(df_history,df_episodes)
    df["completion_%"] = df.apply(lambda x: completion_calculation(x["played up to"], x["duration"]),axis = 1)
    list_new = []
    df["podcast_name"] = df["podcast"].apply(lambda x: retrieve_podcast_name(x, list_new))
    df["podcast_genre"] = df["podcast"].apply(lambda x: retrieve_podcast_genre(x))
    if len(list_new) > 0:
        print(f"{len(list_new)} new podcast, add them in excel file to continue")
        for new in list_new:
            print(new)
        add = input("New podcasts added to excel? (Y/N) ")
        if add == 'Y':
            list_new = []
            df["podcast_name"] = df["podcast"].apply(lambda x: retrieve_podcast_name(x, list_new))
            df["podcast_genre"] = df["podcast"].apply(lambda x: retrieve_podcast_genre(x))
    df = translate_clean(df)
    df.sort_values('modified at', ascending=True, inplace = True)
    df['new_podcast_yn'] = df.groupby('podcast_name').cumcount() == 0
    df['new_podcast_yn'] = df['new_podcast_yn'].astype(int)
    df['new_recurring_podcast_yn'] = df.groupby('podcast_name').cumcount() == 5
    df['new_recurring_podcast_yn'] = df['new_recurring_podcast_yn'].astype(int)
    df.sort_values('modified at', ascending=False, inplace = True)
    df.to_csv('files/processed_files/pocket_casts_processed.csv', sep = "|", encoding = "utf-16")

#process_pocket_casts_export()
#update_file('processed_files/pocket_casts_processed.csv')
#df = pd.read_csv('processed_files/pocket_casts_processed.csv', sep = "|", encoding = "utf-16")
#print(df[df['podcast_name'] == 'Hard Fork'])
