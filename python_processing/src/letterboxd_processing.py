import os
import requests
import pandas as pd
import re
from dotenv import load_dotenv
from utils import clean_rename_move_folder, find_unzip_folder
from drive_storage import update_drive
load_dotenv()

def get_genre(title, release_year):
    """Retrievs the genre of a movie using OMDB_API"""
    df_processed = pd.read_csv('files/processed_files/letterboxd_processed.csv', sep = '|')
    df_processed["Key"] = df_processed["Name"] + df_processed["Year"].astype(str)
    key_input = str(title) + str(release_year)
    if key_input in list(df_processed["Key"].unique()):
        return df_processed[df_processed["Key"] == key_input]['Genre'].iloc[0]
    else:
        api_key = os.environ['OMDB_API_KEY']
        response = requests.get(f'http://www.omdbapi.com/?apikey={api_key}&t={title}&y={release_year}')
        if response.status_code == 200:
            data = response.json()
            if data['Response'] == 'True':
                return data['Genre']
        else:
            return 'Unknown'

def get_watched_rating(path_watched, path_ratings):
    """Merges the watched & ratings dfs"""
    df_watched = pd.read_csv(path_watched)
    df_ratings = pd.read_csv(path_ratings)
    return df_watched.merge(df_ratings[['Name', 'Year', 'Rating']], on = ['Name', 'Year'], how = 'left')

def create_letterboxd_file():
    path_watched = "files/exports/letterboxd_exports/watched.csv"
    path_ratings = "files/exports/letterboxd_exports/ratings.csv"
    df = get_watched_rating(path_watched, path_ratings)
    df['Genre'] = df.apply(lambda x: get_genre(x.Name, x.Year), axis = 1)
    df['Date'] = pd.to_datetime(df['Date'])
    df.to_csv('files/processed_files/letterboxd_processed.csv', sep = '|')


def process_letterboxd_export(upload="Y"):
    file_names = []
    print('Starting the processing of the Letterboxd export \n')
    csv_regex = r'letterboxd-vaherinckx-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-utc.zip'
    for f in os.listdir('/Users/valen/Downloads'):
        if re.match(csv_regex, f):
            find_unzip_folder('letterboxd', zip_file_path = os.path.join("/Users/valen/Downloads", f))
    clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "letterboxd_export_unzipped", "letterboxd_exports")
    process_letterboxd_export()
    file_names.append('files/processed_files/letterboxd_processed.csv')
    if upload == "Y":
        update_drive(file_names)
        print('Letterboxd processed files were created and uploaded to the Drive \n')
    else:
        print('Letterboxd processed files were created \n')
