import os
import pandas as pd
import json
from utils import time_difference_correction

folder_path = 'files/exports/spotify_exports/'
artist_names = []
album_names = []
track_names = []
timestamps = []

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r') as json_file:
            data = json.load(json_file)
            tracks = len(data)
            for t in range(tracks):
                artist_names.append(data[t]['master_metadata_album_artist_name'])
                album_names.append(data[t]['master_metadata_album_album_name'])
                track_names.append(data[t]['master_metadata_track_name'])
                timestamps.append(data[t]['ts'])
data_dict = {
    'artist_name': artist_names,
    'album_name': album_names,
    'track_name': track_names,
    'timestamp' : timestamps
}

df_spot = pd.DataFrame(data_dict)
df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'], utc = True).apply(lambda x: time_difference_correction(x, 'GMT'))
df_spot.to_csv("files/processed_files/spotify_processed.csv", sep = "|", index = False)
