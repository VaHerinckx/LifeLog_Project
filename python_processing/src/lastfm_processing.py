import pandas as pd
import requests
import base64
import numpy as np
import os
import math
from dotenv import load_dotenv
from datetime import timedelta
from utils import time_difference_correction, clean_rename_move_file
from drive_storage import update_drive
load_dotenv()

def add_spotify_legacy(df):
    """Adds the spotify legacy extract made, and removes all the lastFm records that are before the maximum date in this
    extract to avoid duplicates"""
    df_spot = pd.read_csv("files/processed_files/spotify_processed.csv", sep = "|")
    df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'], utc = True)
    max_timestamp = df_spot["timestamp"].max()
    filtered_df = df[df["timestamp"] > max_timestamp]
    concat_df = pd.concat([df_spot,filtered_df], ignore_index = True)
    concat_df['timestamp'] = pd.to_datetime(concat_df['timestamp'], utc = True).dt.floor('T')
    return concat_df

def authentification(client_id, client_secret):
    """Generate the access token necessary to call the Spotify API"""
    auth_url = "https://accounts.spotify.com/api/token"
    # Encode the client ID and client secret as base64
    client_creds = f"{client_id}:{client_secret}"
    client_creds_b64 = base64.b64encode(client_creds.encode())
    headers = {"Authorization": f"Basic {client_creds_b64.decode()}"}
    params = {"grant_type": "client_credentials"}
    response = requests.post(auth_url, headers=headers, data=params)
    data = response.json()
    access_token = data["access_token"]
    return access_token

def compute_completion(df):
    """Adds a column "completion" to the df, to see what percentage of the song was listened t before skipping"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['skip_next_track'] = 0
    df['completion'] = 0
    for i in range(1, len(df)):
        curr_row = df.loc[i]
        prev_row = df.loc[i-1]
        time_diff = prev_row['timestamp'] - curr_row['timestamp']
        minutes_diff = time_diff.total_seconds() / 60
        original_duration = curr_row['track_duration']/1000/60
        if (isinstance(curr_row['track_duration'], (int, float))) & (curr_row['track_duration'] == curr_row['track_duration']):
            duration = math.floor(curr_row['track_duration']/1000/60)
        else:
            duration = 0
        if time_diff >= timedelta(minutes=duration):
            df.at[i, 'skip_next_track'] = duration
            df.at[i, 'completion'] = 100
        elif duration > 0:
            if minutes_diff != minutes_diff:
                continue
            df.at[i, 'skip_next_track'] = minutes_diff
            df.at[i, 'completion'] = (minutes_diff / original_duration)*100
    # Set the last row's completion to 1 since there's no row below it
    df.at[0, 'skip_next_track'] = 0
    df.at[0, 'completion'] = 100
    df['completion'] = df['completion']/100
    return df

def artist_info(token, artist_names):
    """Retrieves information about the artist genre, followers, popularity, etc."""
    #Importing the current dictionnary with all artists listened to & their characteristics
    artist_df = pd.read_csv('files/work_files/lfm_work_files/artists_infos.csv', sep = '|')
    count = 0
    dict_artists = {}
    for _, row in artist_df.iterrows():
        dict_info_artists = {}
        for col in list(artist_df.columns)[1:]:
            dict_info_artists[col] = row[col]
        dict_artists[row['artist_name']] = dict_info_artists
    for artist_name in artist_names:
        if str(artist_name) in dict_artists.keys():
            pass
        else:
            count += 1
            print(f"new artist {count} : {artist_name}")
            endpoint_url = 'https://api.spotify.com/v1/search'
            headers = {'Authorization': f'Bearer {token}'}
            params = {'q': artist_name,'type': 'artist', 'limit': 1}
            response = requests.get(endpoint_url, headers=headers, params=params)
            response_json = response.json()
            dict_info_artists = {}
            if response_json['artists']['items'] == []:
                dict_info_artists['followers'] = "Unknown"
                dict_artists[artist_name] = dict_info_artists
                continue
            artist_id = response_json['artists']['items'][0]['id']
            endpoint_url = f'https://api.spotify.com/v1/artists/{artist_id}'
            response = requests.get(endpoint_url, headers=headers)
            artist_info = response.json()
            dict_info_artists['followers'] = artist_info['followers']['total']
            dict_info_artists['artist_popularity'] = artist_info['popularity']
            for i in range(len(artist_info['genres'])):
                dict_info_artists[f'genre_{i+1}'] = artist_info['genres'][i]
            dict_artists[artist_name] = dict_info_artists
            df_artist = pd.DataFrame.from_dict(dict_artists, orient='index').reset_index().rename(columns={'index':'artist_name'})
            df_artist.to_csv('files/work_files/lfm_work_files/artists_infos.csv', sep = '|', index = False)
    print (f"{count} new artist(s) were added to the artist dictionnary")
    df_artist = pd.DataFrame.from_dict(dict_artists, orient='index').reset_index().rename(columns={'index':'artist_name'})
    df_artist.drop_duplicates().to_csv('files/work_files/lfm_work_files/artists_infos.csv', sep = '|', index = False)
    return df_artist

def track_info(token, song_keys):
    """
    Retrieve information about the track, followers, popularity, etc.
    """
    #Importing the current dictionnary with all tracks listened to & their characteristics
    track_df = pd.read_csv('files/work_files/lfm_work_files/tracks_infos.csv', sep = '|')
    count = 0
    #Rebuilding the dictionnary
    dict_tracks = {}
    for _, row in track_df.iterrows():
        dict_info_tracks = {}
        for col in list(track_df.columns)[1:]:
            dict_info_tracks[col] = row[col]
        dict_tracks[row['song_key']] = dict_info_tracks
    #Checking how many songs are already in the dictionnary
    count_API_requests = 0
    for song_key in song_keys:
        if (song_key in dict_tracks.keys()) | (song_key == ''):
            pass
        else:
            count_API_requests += 1
    for song_key in song_keys:
        if (song_key in dict_tracks.keys()) | (song_key == ''):
            pass
        else:
            count += 1
            print(f"Info gathered for {count} out of {count_API_requests} new songs")
            search_url = "https://api.spotify.com/v1/search"
            track_name = song_key.split("/:")[0].strip()
            artist_name = song_key.split("/:")[1].strip()
            params = {"q": f"{track_name} artist:{artist_name}", "type": "track"}
            headers = {"Authorization": "Bearer " + token}
            response = requests.get(search_url, params=params, headers=headers).json()
            # Get the first track from the search results
            dict_info_tracks = {}
            if not response["tracks"]["items"]:
                print(f"No API result for {track_name} - {artist_name}")
                dict_info_tracks['track_name'] = track_name
                dict_info_tracks['artist_name'] = artist_name
                list_keys = ['album_name', 'album_release_date', 'track_duration', 'track_popularity', \
                             'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',\
                             'instrumentalness', 'liveness', 'valence', 'tempo']
                for key in list_keys:
                    dict_info_tracks[key] = "No API result"
                dict_tracks[song_key] = dict_info_tracks
            else:
                track_id = response["tracks"]["items"][0]["id"]
                # Get the track info using the track ID
                track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
                response = requests.get(track_url, headers=headers).json()
                dict_info_tracks['track_name'] = track_name
                dict_info_tracks['artist_name'] = artist_name
                dict_info_tracks['album_name'] = response['album']['name']
                dict_info_tracks['album_release_date'] = response['album']['release_date']
                dict_info_tracks['track_duration'] = response['duration_ms']
                dict_info_tracks['track_popularity'] = response['popularity']
                track_url = f"https://api.spotify.com/v1/audio-features/{track_id}"
                response_track_details = requests.get(track_url, headers=headers).json()
                for info in response_track_details.keys():
                    if info == 'type':
                        break
                    dict_info_tracks[info] = response_track_details[info]
                dict_tracks[song_key] = dict_info_tracks
                if count % 50 ==0:
                    df_tracks = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                    df_tracks.to_csv('files/work_files/lfm_work_files/tracks_infos.csv', sep = '|', index = False)
    print (f"{count} new tracks were added to the track dictionnary \n")
    df_tracks = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
    df_tracks.to_csv('files/work_files/lfm_work_files/tracks_infos.csv', sep = '|', index = False)
    return df_tracks

def merge_dfs(df, artist_df, track_df):
    """Merging the export df, with the artist info and track info dfs"""
    df_merge_artist = pd.merge(df, artist_df, how = 'left', on= 'artist_name')
    cols_to_use = list(track_df.columns.difference(df_merge_artist.columns))
    cols_to_use.append('song_key')
    df_merge_artist_track = pd.merge(df_merge_artist, track_df[cols_to_use], how = 'left', on = 'song_key')
    return df_merge_artist_track

def power_bi_processing(df):
    """Changes some results for better display in PBI"""
    df['genre_1'].fillna('Unknown')
    df['track_duration'] = df['track_duration'].replace('No API result', '0').astype(float)
    return df

def create_lfm_file():
    df = pd.read_csv(f"files/exports/lfm_exports/lfm_export.csv", header = None)
    df.columns = ['artist_name', 'album_name', 'track_name', 'timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc = True).apply(lambda x: time_difference_correction(x, 'GMT'))
    df = add_spotify_legacy(df)
    client_id = os.environ['Spotify_API_Client_ID']
    client_secret = os.environ['Spotify_API_Client_Secret']
    token = authentification(client_id, client_secret)
    unique_artists = list(df.artist_name.astype(str).replace("nan", "nan_").unique())
    df['song_key'] = (df['track_name'] + " /: " + df['artist_name']).replace(np.nan, '')
    unique_tracks = list(df.song_key.astype(str).unique())
    #Adding the new artists since last export using the Spotify API, and saving them in the dictionnary
    artist_df = artist_info(token, unique_artists)
    #Adding the new tracks since last export using the Spotify API, and saving them in the dictionnary
    track_df = track_info(token, unique_tracks)
    #Merging the export with the detailled artist & track infos
    df = merge_dfs(df, artist_df, track_df)
    df = power_bi_processing(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', ascending=True, inplace = True)
    df['new_artist_yn'] = df.groupby('artist_name').cumcount() == 0
    df['new_recurring_artist_yn'] = df.groupby('artist_name').cumcount() == 10
    df['new_track_yn'] = df.groupby('track_name').cumcount() == 0
    df['new_recurring_track_yn'] = df.groupby('track_name').cumcount() == 5
    df['new_artist_yn'] = df['new_artist_yn'].astype(int)
    df['new_recurring_artist_yn'] = df['new_recurring_artist_yn'].astype(int)
    df['new_track_yn'] = df['new_track_yn'].astype(int)
    df['new_recurring_track_yn'] = df['new_recurring_track_yn'].astype(int)
    df.sort_values('timestamp', ascending=False, inplace = True)
    df = compute_completion(df.reset_index(drop=True))
    df.to_csv('files/processed_files/lfm_processed.csv', sep = '|', index = False)


def process_lfm_export(upload="Y"):
    file_names = []
    print('Starting the processing of the lfm export \n')
    clean_rename_move_file("files/exports/lfm_exports", "/Users/valen/Downloads", "entinval.csv", "lfm_export.csv")
    create_lfm_file()
    file_names.append('files/processed_files/lfm_processed.csv')
    if upload == "Y":
        update_drive(file_names)
        print('LFM processed files were created and uploaded to the Drive \n')
    else:
        print('LFM processed files were created \n')
