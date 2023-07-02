import os
import requests
import pandas as pd
import shutil
from dotenv import load_dotenv
load_dotenv()

def get_genre(title, release_year):
    df_processed = pd.read_csv('processed_files/letterboxd_processed.csv', sep = '|')
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
    df_watched = pd.read_csv(path_watched)
    df_ratings = pd.read_csv(path_ratings)
    return df_watched.merge(df_ratings[['Name', 'Year', 'Rating']], on = ['Name', 'Year'], how = 'left')

def process_letterboxd_export():
    path_watched = "exports/letterboxd_exports/watched.csv"
    path_ratings = "exports/letterboxd_exports/ratings.csv"
    df = get_watched_rating(path_watched, path_ratings)
    df['Genre'] = df.apply(lambda x: get_genre(x.Name, x.Year), axis = 1)
    df['Date'] = pd.to_datetime(df['Date'])
    df.to_csv('processed_files/letterboxd_processed.csv', sep = '|')
