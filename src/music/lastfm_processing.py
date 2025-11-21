#!/usr/bin/env python3
"""
Last.fm API Processing Module

This module handles incremental updates to the Last.fm processed data file by:
1. Reading the latest timestamp from existing lastfm_export.csv
2. Fetching new tracks from Last.fm API since that timestamp (saved incrementally per page)
3. Applying the complete pipeline processing (Spotify API enrichment, etc.)
4. Merging new data with existing data and removing duplicates
5. Saving the updated data back to lastfm_processed.csv

This replaces the manual download process while maintaining all existing processing logic.

Author: LifeLog Project
"""

import requests
import pandas as pd
import time
import os
import json
import sys
import base64
import math
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to sys.path to import utils
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.drive_operations import upload_multiple_files, verify_drive_connection
from utils.utils_functions import record_successful_run, enforce_snake_case
from music.genre_mapping import get_simplified_genre, analyze_unmapped_genres
from dotenv import load_dotenv

load_dotenv()




class LastFmAPIProcessor:
    """Handles Last.fm API data processing and incremental updates."""

    def __init__(self):
        """Initialize the processor with API credentials from environment variables."""
        # Load LastFM credentials from environment variables
        self.api_key = os.environ.get('LAST_FM_API_KEY')
        self.api_secret = os.environ.get('LAST_FM_API_SECRET')  # Available if needed for write operations
        self.username = os.environ.get('LAST_FM_API_USERNAME')

        if not self.api_key:
            raise ValueError("LAST_FM_API_KEY environment variable is required")
        if not self.username:
            raise ValueError("LAST_FM_API_USERNAME environment variable is required")

        self.base_url = "http://ws.audioscrobbler.com/2.0/"

        # File paths
        self.export_file_path = "files/exports/lastfm_exports/lastfm_export.csv"
        self.processed_file_path = "files/processed_files/music/lastfm_processed.csv"
        self.spotify_file_path = "files/processed_files/music/spotify_processed.csv"
        self.artists_work_file = "files/work_files/lastfm_work_files/artists_infos.csv"
        self.tracks_work_file = "files/work_files/lastfm_work_files/tracks_infos.csv"

    def get_latest_timestamp_from_file(self):
        """
        Read the existing raw export file and return the latest timestamp.
        This is used to determine the checkpoint for resuming downloads.

        Returns:
            datetime: Latest timestamp from the file, or None if file doesn't exist
        """
        try:
            if not os.path.exists(self.export_file_path):
                print(f"File {self.export_file_path} not found. Will fetch all available data.")
                return None

            # Try different encodings to read the file (UTF-8 first as it's the new standard)
            encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be']
            df = None

            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(self.export_file_path, sep='|', encoding=encoding, low_memory=False)
                    print(f"Successfully read export file with {encoding} encoding")
                    break
                except UnicodeError:
                    continue
                except Exception as e:
                    if encoding == encodings_to_try[-1]:  # Last encoding attempt
                        raise e
                    continue

            if df is None:
                raise Exception("Could not read file with any supported encoding")

            if df.empty:
                print("Existing export file is empty. Will fetch all available data.")
                return None

            # Convert timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Get the latest timestamp
            latest_timestamp = df['timestamp'].max()
            print(f"Latest timestamp in export file: {latest_timestamp}")

            return latest_timestamp

        except Exception as e:
            print(f"Error reading existing export file: {e}")
            print("Will fetch all available data.")
            return None

    def fetch_tracks_since_timestamp(self, since_timestamp=None):
        """
        Fetch tracks from Last.fm API since the given timestamp.
        Saves each page incrementally to lastfm_export.csv for crash recovery.

        Args:
            since_timestamp (datetime): Fetch tracks after this timestamp

        Returns:
            bool: True if successful, False otherwise
        """
        # Ensure export directory exists
        os.makedirs(os.path.dirname(self.export_file_path), exist_ok=True)

        page = 1
        total_pages = None
        printed_currently_playing = set()  # Track already-printed "currently playing" messages
        total_saved = 0

        print(f"Fetching tracks from Last.fm API with incremental checkpointing...")
        if since_timestamp:
            print(f"Fetching tracks since: {since_timestamp}")
        else:
            print("Fetching all available tracks (no existing data found)")

        # Check if export file exists to determine if we should write header
        file_exists = os.path.exists(self.export_file_path)

        while True:
            retry_count = 0
            max_retries = 3
            success = False

            while retry_count < max_retries and not success:
                try:
                    params = {
                        'method': 'user.getrecenttracks',
                        'user': self.username,
                        'api_key': self.api_key,
                        'format': 'json',
                        'limit': 200,  # Maximum allowed by API
                        'page': page
                    }

                    # Add timestamp filter if provided
                    if since_timestamp:
                        # Convert to Unix timestamp
                        unix_timestamp = int(since_timestamp.timestamp())
                        params['from'] = unix_timestamp

                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()

                    data = response.json()

                    # Check for API errors
                    if 'error' in data:
                        print(f"API Error: {data['message']}")
                        print(f"✅ Saved {total_saved} tracks before error")
                        return True  # Partial success - data is checkpointed

                    # Check if we have track data
                    if 'recenttracks' not in data or 'track' not in data['recenttracks']:
                        print("No track data found in API response")
                        print(f"✅ Saved {total_saved} tracks before response issue")
                        return True  # Partial success - data is checkpointed

                    tracks = data['recenttracks']['track']
                    attr = data['recenttracks']['@attr']

                    # Get total pages from first response
                    if total_pages is None:
                        total_pages = int(attr['totalPages'])
                        total_tracks = int(attr['total'])
                        print(f"Total tracks to fetch: {total_tracks} across {total_pages} pages")

                    # If no tracks on this page, we're done
                    if not tracks:
                        print(f"✅ Complete: Saved {total_saved} tracks")
                        return True  # Success

                    # Handle case where only one track is returned (not a list)
                    if isinstance(tracks, dict):
                        tracks = [tracks]

                    # Filter out currently playing tracks (they have no timestamp)
                    valid_tracks = []
                    for track in tracks:
                        if 'date' in track and 'uts' in track['date']:
                            valid_tracks.append(track)
                        else:
                            # Only print message once per track to avoid spam
                            track_name = track.get('name', 'Unknown')
                            if track_name not in printed_currently_playing:
                                print(f"Skipping currently playing track: {track_name}")
                                printed_currently_playing.add(track_name)

                    # Save this page immediately to export file (checkpoint)
                    if valid_tracks:
                        page_df = self.parse_api_tracks_to_dataframe(valid_tracks)

                        # Append to CSV (write header only if file doesn't exist or is first page)
                        mode = 'w' if not file_exists else 'a'
                        header = not file_exists
                        page_df.to_csv(self.export_file_path, sep='|', encoding='utf-8',
                                      index=False, mode=mode, header=header)

                        # After first write, file exists
                        file_exists = True
                        total_saved += len(valid_tracks)

                    print(f"Fetched page {page}/{total_pages} ({len(valid_tracks)} tracks) ✓ Saved checkpoint")

                    success = True  # Mark as successful

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"❌ Network error on page {page} after {max_retries} retries: {e}")
                        if total_pages:
                            estimated_missing = (total_pages - page + 1) * 200
                            print(f"⚠️  WARNING: Incomplete fetch - successfully saved {total_saved} tracks")
                            print(f"    Failed at page {page} of {total_pages}")
                            print(f"    Estimated missing tracks: ~{estimated_missing}")
                            print(f"    ✓ Checkpointed data saved - next run will resume from {total_saved} tracks")
                        return True  # Partial success - checkpointed data is recoverable
                    else:
                        wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                        print(f"⚠️  Error on page {page}, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                        time.sleep(wait_time)

                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"❌ Error processing page {page} after {max_retries} retries: {e}")
                        if total_pages:
                            print(f"⚠️  WARNING: Saved {total_saved} tracks before failure")
                            print(f"    ✓ Checkpointed data saved - next run will resume")
                        return True  # Partial success - checkpointed data is recoverable
                    else:
                        wait_time = 2 ** retry_count
                        print(f"⚠️  Error on page {page}, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                        time.sleep(wait_time)

            # If we didn't succeed after retries, move on
            if not success:
                break

            # Check if we've reached the end
            if page >= total_pages:
                break

            page += 1

            # Be nice to the API - small delay between requests
            time.sleep(0.2)

        print(f"✅ Successfully fetched and saved {total_saved} new tracks to {self.export_file_path}")
        return True

    def parse_api_tracks_to_dataframe(self, tracks):
        """
        Parse raw API track data into a DataFrame with the basic structure.

        Args:
            tracks (list): List of track dictionaries from API

        Returns:
            pandas.DataFrame: Parsed track data in basic format
        """
        parsed_tracks = []

        for track in tracks:
            try:
                # Extract basic track information
                track_data = {
                    'artist_name': track['artist']['#text'],
                    'album_name': track['album']['#text'] if track['album']['#text'] else 'Unknown Album',
                    'track_name': track['name'],
                    'timestamp': datetime.fromtimestamp(int(track['date']['uts']))
                }

                parsed_tracks.append(track_data)

            except Exception as e:
                print(f"Error parsing track {track.get('name', 'Unknown')}: {e}")
                continue

        df = pd.DataFrame(parsed_tracks)
        return df

    # Copy all the processing functions from the original file
    def add_spotify_legacy(self, df):
        """Adds the spotify legacy extract made, and removes all the lastFm records that are before the maximum date in this
        extract to avoid duplicates"""
        if not os.path.exists(self.spotify_file_path):
            print("⚠️  Spotify legacy file not found, proceeding without merging")
            return df

        df_spot = pd.read_csv(self.spotify_file_path, sep="|")
        df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'], utc=True)

        # Convert to timezone-naive for consistency
        df_spot['timestamp'] = df_spot['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

        # Ensure both DataFrames have timezone-naive timestamps
        if hasattr(df['timestamp'].dtype, 'tz') and df['timestamp'].dtype.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

        max_timestamp = df_spot["timestamp"].max()
        filtered_df = df[df["timestamp"] > max_timestamp]
        concat_df = pd.concat([df_spot, filtered_df], ignore_index=True)
        concat_df['timestamp'] = pd.to_datetime(concat_df['timestamp']).dt.floor('T')
        return concat_df

    def authentification(self, client_id, client_secret):
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

    def compute_completion(self, df):
        """Adds a column "completion" to the df, to see what percentage of the song was listened t before skipping"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['skip_next_track'] = 0
        df['completion'] = 0
        for i in range(1, len(df)):
            curr_row = df.loc[i]
            prev_row = df.loc[i-1]
            time_diff = prev_row['timestamp'] - curr_row['timestamp']
            minutes_diff = time_diff.total_seconds() / 60
            original_duration = curr_row['track_duration']/1000/60 if isinstance(curr_row['track_duration'], (int, float)) and not pd.isna(curr_row['track_duration']) else 0
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

    def artist_info(self, token, artist_names):
        """Retrieves information about the artist genre, followers, popularity, etc."""
        # Ensure work directory exists
        os.makedirs(os.path.dirname(self.artists_work_file), exist_ok=True)

        # Initialize or load existing artist data
        if os.path.exists(self.artists_work_file):
            artist_df = pd.read_csv(self.artists_work_file, sep='|', low_memory=False)
        else:
            artist_df = pd.DataFrame(columns=['artist_name'])

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
                dict_info_artists = {}

                try:
                    endpoint_url = 'https://api.spotify.com/v1/search'
                    headers = {'Authorization': f'Bearer {token}'}
                    params = {'q': artist_name,'type': 'artist', 'limit': 1}
                    response = requests.get(endpoint_url, headers=headers, params=params)

                    # Check for rate limiting FIRST (before raise_for_status which would throw exception)
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        print(f"⚠️  Rate limited by Spotify. Waiting {retry_after} seconds before retry...")
                        # Save progress before waiting
                        df_artist = pd.DataFrame.from_dict(dict_artists, orient='index').reset_index().rename(columns={'index':'artist_name'})
                        df_artist.to_csv(self.artists_work_file, sep='|', index=False)
                        time.sleep(retry_after)
                        # Decrement count so we retry this artist
                        count -= 1
                        continue

                    # Validate response before parsing JSON
                    if not response.text or response.text.strip() == '':
                        print(f"⚠️  Empty response for artist: {artist_name}, skipping...")
                        dict_info_artists['followers'] = "No API result"
                        dict_artists[artist_name] = dict_info_artists
                        continue

                    # Raise for other HTTP errors (4xx, 5xx) - after checking 429
                    response.raise_for_status()

                    response_json = response.json()

                    if response_json['artists']['items'] == []:
                        dict_info_artists['followers'] = "Unknown"
                        dict_artists[artist_name] = dict_info_artists
                        # Add small delay to avoid rate limiting
                        time.sleep(0.15)
                        continue

                    artist_id = response_json['artists']['items'][0]['id']
                    endpoint_url = f'https://api.spotify.com/v1/artists/{artist_id}'
                    response = requests.get(endpoint_url, headers=headers)

                    # Check for rate limiting on second API call
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        print(f"⚠️  Rate limited by Spotify. Waiting {retry_after} seconds before retry...")
                        # Save progress before waiting
                        df_artist = pd.DataFrame.from_dict(dict_artists, orient='index').reset_index().rename(columns={'index':'artist_name'})
                        df_artist.to_csv(self.artists_work_file, sep='|', index=False)
                        time.sleep(retry_after)
                        # Decrement count so we retry this artist
                        count -= 1
                        continue

                    # Validate second API call response
                    if not response.text or response.text.strip() == '':
                        print(f"⚠️  Empty response for artist details: {artist_name}, skipping...")
                        dict_info_artists['followers'] = "No API result"
                        dict_artists[artist_name] = dict_info_artists
                        continue

                    # Raise for other HTTP errors - after checking 429
                    response.raise_for_status()
                    artist_info = response.json()

                except requests.exceptions.JSONDecodeError as e:
                    print(f"❌ JSON parsing error for artist '{artist_name}'")
                    print(f"   Error: {e}")
                    print(f"   Response status: {response.status_code}")
                    print(f"   Response headers: {dict(response.headers)}")
                    print(f"   Response body (first 500 chars): {response.text[:500]}")
                    dict_info_artists['followers'] = "API Error - JSON Parse Failed"
                    dict_artists[artist_name] = dict_info_artists
                    continue

                except requests.exceptions.RequestException as e:
                    print(f"❌ Network error for artist '{artist_name}': {e}")
                    dict_info_artists['followers'] = "Network Error"
                    dict_artists[artist_name] = dict_info_artists
                    continue

                except Exception as e:
                    print(f"❌ Unexpected error for artist '{artist_name}': {e}")
                    dict_info_artists['followers'] = "Unknown Error"
                    dict_artists[artist_name] = dict_info_artists
                    continue

                # Store ALL Spotify artist fields (future-proof)
                dict_info_artists['spotify_id'] = artist_info.get('id')
                dict_info_artists['spotify_url'] = artist_info.get('external_urls', {}).get('spotify')
                dict_info_artists['artist_type'] = artist_info.get('type')
                dict_info_artists['spotify_uri'] = artist_info.get('uri')
                dict_info_artists['href'] = artist_info.get('href')

                # Followers (keep original column name for backward compatibility)
                dict_info_artists['followers'] = artist_info.get('followers', {}).get('total')
                dict_info_artists['followers_total'] = artist_info.get('followers', {}).get('total')

                # Popularity (keep both names for backward compatibility)
                dict_info_artists['artist_popularity'] = artist_info.get('popularity')
                dict_info_artists['popularity'] = artist_info.get('popularity')

                # Genres - store as both individual columns AND JSON array
                genres = artist_info.get('genres', [])
                dict_info_artists['genres_json'] = json.dumps(genres)
                for i in range(len(genres[:14])):  # Keep first 14 for backward compatibility
                    dict_info_artists[f'genre_{i+1}'] = genres[i]

                # Images - store as JSON array AND flattened by size
                images = artist_info.get('images', [])
                dict_info_artists['images_json'] = json.dumps(images)
                dict_info_artists['artist_artwork_url'] = images[0].get('url') if images else None  # Backward compat
                for idx, img in enumerate(images[:3]):  # Store up to 3 image sizes
                    size = img.get('height', f'size{idx}')
                    dict_info_artists[f'artist_artwork_{size}'] = img.get('url')

                dict_artists[artist_name] = dict_info_artists

                # Add small delay to avoid rate limiting
                time.sleep(0.15)

                # Save progress every 50 artists (performance optimization)
                if count % 50 == 0:
                    df_artist = pd.DataFrame.from_dict(dict_artists, orient='index').reset_index().rename(columns={'index':'artist_name'})
                    df_artist.to_csv(self.artists_work_file, sep='|', index=False)
        print(f"{count} new artist(s) were added to the artist dictionnary")
        df_artist = pd.DataFrame.from_dict(dict_artists, orient='index').reset_index().rename(columns={'index':'artist_name'})
        df_artist.drop_duplicates().to_csv(self.artists_work_file, sep='|', index=False)
        return df_artist

    def track_info(self, token, song_keys):
        """
        Retrieve information about the track, followers, popularity, etc.
        """
        # Ensure work directory exists
        os.makedirs(os.path.dirname(self.tracks_work_file), exist_ok=True)

        # Initialize or load existing track data
        if os.path.exists(self.tracks_work_file):
            track_df = pd.read_csv(self.tracks_work_file, sep='|', low_memory=False)
        else:
            track_df = pd.DataFrame(columns=['song_key'])

        count = 0
        # Rebuilding the dictionnary
        dict_tracks = {}
        for _, row in track_df.iterrows():
            dict_info_tracks = {}
            for col in list(track_df.columns)[1:]:
                dict_info_tracks[col] = row[col]
            dict_tracks[row['song_key']] = dict_info_tracks
        # Checking how many songs are already in the dictionnary
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
                dict_info_tracks = {}

                try:
                    search_url = "https://api.spotify.com/v1/search"
                    track_name = song_key.split("/:")[0].strip()
                    artist_name = song_key.split("/:")[1].strip()
                    params = {"q": f"{track_name} artist:{artist_name}", "type": "track"}
                    headers = {"Authorization": "Bearer " + token}
                    response = requests.get(search_url, params=params, headers=headers)

                    # Check for rate limiting FIRST (before raise_for_status which would throw exception)
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        print(f"⚠️  Rate limited by Spotify. Waiting {retry_after} seconds before retry...")
                        # Save progress before waiting
                        df_tracks = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                        df_tracks.to_csv(self.tracks_work_file, sep='|', index=False)
                        time.sleep(retry_after)
                        # Decrement count so we retry this track
                        count -= 1
                        continue

                    # Validate response before parsing JSON
                    if not response.text or response.text.strip() == '':
                        print(f"⚠️  Empty response for track: {track_name} - {artist_name}, skipping...")
                        dict_info_tracks['track_name'] = track_name
                        dict_info_tracks['artist_name'] = artist_name
                        # Set all new fields to "No API result" for consistency
                        list_keys = [
                            # Track identifiers
                            'spotify_track_id', 'spotify_album_id', 'spotify_track_url', 'spotify_track_uri',
                            'track_href', 'isrc', 'preview_url',
                            # Track properties
                            'track_number', 'disc_number', 'track_duration', 'track_popularity', 'explicit', 'is_local',
                            # Album info
                            'album_name', 'album_type', 'album_release_date', 'album_release_date_precision',
                            'album_total_tracks', 'album_spotify_url', 'album_uri',
                            # Album images
                            'album_images_json', 'album_artwork_url',
                            # Audio features
                            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                            'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'analysis_url'
                        ]
                        for key in list_keys:
                            dict_info_tracks[key] = "No API result"
                        dict_tracks[song_key] = dict_info_tracks
                        continue

                    # Raise for other HTTP errors (4xx, 5xx) - after checking 429
                    response.raise_for_status()

                    response_json = response.json()

                    # Get the first track from the search results
                    if not response_json["tracks"]["items"]:
                        print(f"No API result for {track_name} - {artist_name}")
                        dict_info_tracks['track_name'] = track_name
                        dict_info_tracks['artist_name'] = artist_name
                        # Set all new fields to "No API result" for consistency
                        list_keys = [
                            # Track identifiers
                            'spotify_track_id', 'spotify_album_id', 'spotify_track_url', 'spotify_track_uri',
                            'track_href', 'isrc', 'preview_url',
                            # Track properties
                            'track_number', 'disc_number', 'track_duration', 'track_popularity', 'explicit', 'is_local',
                            # Album info
                            'album_name', 'album_type', 'album_release_date', 'album_release_date_precision',
                            'album_total_tracks', 'album_spotify_url', 'album_uri',
                            # Album images
                            'album_images_json', 'album_artwork_url',
                            # Audio features
                            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                            'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'analysis_url'
                        ]
                        for key in list_keys:
                            dict_info_tracks[key] = "No API result"
                        dict_tracks[song_key] = dict_info_tracks
                        # Add small delay to avoid rate limiting
                        time.sleep(0.15)
                        continue

                    # Process successful track search result
                    track_id = response_json["tracks"]["items"][0]["id"]
                    # Get the track info using the track ID
                    track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
                    response = requests.get(track_url, headers=headers)

                    # Check for rate limiting on track details API call
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        print(f"⚠️  Rate limited by Spotify. Waiting {retry_after} seconds before retry...")
                        # Save progress before waiting
                        df_tracks = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                        df_tracks.to_csv(self.tracks_work_file, sep='|', index=False)
                        time.sleep(retry_after)
                        # Decrement count so we retry this track
                        count -= 1
                        continue

                    # Validate track details response
                    if not response.text or response.text.strip() == '':
                        print(f"⚠️  Empty response for track details: {track_name} - {artist_name}, skipping...")
                        dict_info_tracks['track_name'] = track_name
                        dict_info_tracks['artist_name'] = artist_name
                        list_keys = [
                            'spotify_track_id', 'spotify_album_id', 'spotify_track_url', 'spotify_track_uri',
                            'track_href', 'isrc', 'preview_url', 'track_number', 'disc_number', 'track_duration',
                            'track_popularity', 'explicit', 'is_local', 'album_name', 'album_type',
                            'album_release_date', 'album_release_date_precision', 'album_total_tracks',
                            'album_spotify_url', 'album_uri', 'album_images_json', 'album_artwork_url',
                            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                            'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'analysis_url'
                        ]
                        for key in list_keys:
                            dict_info_tracks[key] = "No API result"
                        dict_tracks[song_key] = dict_info_tracks
                        continue

                    response.raise_for_status()
                    response = response.json()

                    # Store ALL Spotify track fields (future-proof)
                    dict_info_tracks['track_name'] = track_name
                    dict_info_tracks['artist_name'] = artist_name

                    # Track identifiers and metadata
                    dict_info_tracks['spotify_track_id'] = response.get('id')
                    dict_info_tracks['spotify_album_id'] = response.get('album', {}).get('id')
                    dict_info_tracks['spotify_track_url'] = response.get('external_urls', {}).get('spotify')
                    dict_info_tracks['spotify_track_uri'] = response.get('uri')
                    dict_info_tracks['track_href'] = response.get('href')
                    dict_info_tracks['isrc'] = response.get('external_ids', {}).get('isrc')
                    dict_info_tracks['preview_url'] = response.get('preview_url')

                    # Track properties
                    dict_info_tracks['track_number'] = response.get('track_number')
                    dict_info_tracks['disc_number'] = response.get('disc_number')
                    dict_info_tracks['track_duration'] = response.get('duration_ms')
                    dict_info_tracks['track_popularity'] = response.get('popularity')
                    dict_info_tracks['explicit'] = response.get('explicit')
                    dict_info_tracks['is_local'] = response.get('is_local')

                    # Album information
                    album = response.get('album', {})
                    dict_info_tracks['album_name'] = album.get('name')
                    dict_info_tracks['album_type'] = album.get('album_type')
                    dict_info_tracks['album_release_date'] = album.get('release_date')
                    dict_info_tracks['album_release_date_precision'] = album.get('release_date_precision')
                    dict_info_tracks['album_total_tracks'] = album.get('total_tracks')
                    dict_info_tracks['album_spotify_url'] = album.get('external_urls', {}).get('spotify')
                    dict_info_tracks['album_uri'] = album.get('uri')

                    # Album images - store as JSON array AND flattened by size
                    album_images = album.get('images', [])
                    dict_info_tracks['album_images_json'] = json.dumps(album_images)
                    dict_info_tracks['album_artwork_url'] = album_images[0].get('url') if album_images else None  # Backward compat
                    for idx, img in enumerate(album_images[:3]):  # Store up to 3 image sizes
                        size = img.get('height', f'size{idx}')
                        dict_info_tracks[f'album_artwork_{size}'] = img.get('url')

                    # Get audio features
                    track_url = f"https://api.spotify.com/v1/audio-features/{track_id}"
                    response_track_details = requests.get(track_url, headers=headers)

                    # Check for rate limiting on audio features API call
                    if response_track_details.status_code == 429:
                        retry_after = int(response_track_details.headers.get('Retry-After', 60))
                        print(f"⚠️  Rate limited by Spotify. Waiting {retry_after} seconds before retry...")
                        # Save progress before waiting
                        df_tracks = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                        df_tracks.to_csv(self.tracks_work_file, sep='|', index=False)
                        time.sleep(retry_after)
                        # Decrement count so we retry this track
                        count -= 1
                        continue

                    # Validate audio features response
                    if response_track_details.text and response_track_details.text.strip() != '':
                        response_track_details.raise_for_status()
                        response_track_details = response_track_details.json()

                        # Store ALL audio features (not just some)
                        audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                                         'time_signature', 'duration_ms', 'analysis_url']
                        for feature in audio_features:
                            if feature in response_track_details:
                                dict_info_tracks[feature] = response_track_details[feature]

                    dict_tracks[song_key] = dict_info_tracks

                    # Add small delay to avoid rate limiting
                    time.sleep(0.15)

                    if count % 50 == 0:
                        df_tracks = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                        df_tracks.to_csv(self.tracks_work_file, sep='|', index=False)

                except requests.exceptions.JSONDecodeError as e:
                    print(f"❌ JSON parsing error for track '{track_name} - {artist_name}'")
                    print(f"   Error: {e}")
                    print(f"   Response status: {response.status_code}")
                    print(f"   Response headers: {dict(response.headers)}")
                    print(f"   Response body (first 500 chars): {response.text[:500]}")
                    dict_info_tracks['track_name'] = track_name
                    dict_info_tracks['artist_name'] = artist_name
                    list_keys = [
                        'spotify_track_id', 'spotify_album_id', 'spotify_track_url', 'spotify_track_uri',
                        'track_href', 'isrc', 'preview_url', 'track_number', 'disc_number', 'track_duration',
                        'track_popularity', 'explicit', 'is_local', 'album_name', 'album_type',
                        'album_release_date', 'album_release_date_precision', 'album_total_tracks',
                        'album_spotify_url', 'album_uri', 'album_images_json', 'album_artwork_url',
                        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'analysis_url'
                    ]
                    for key in list_keys:
                        dict_info_tracks[key] = "API Error - JSON Parse Failed"
                    dict_tracks[song_key] = dict_info_tracks
                    continue

                except requests.exceptions.RequestException as e:
                    print(f"❌ Network error for track '{track_name} - {artist_name}': {e}")
                    dict_info_tracks['track_name'] = track_name
                    dict_info_tracks['artist_name'] = artist_name
                    list_keys = [
                        'spotify_track_id', 'spotify_album_id', 'spotify_track_url', 'spotify_track_uri',
                        'track_href', 'isrc', 'preview_url', 'track_number', 'disc_number', 'track_duration',
                        'track_popularity', 'explicit', 'is_local', 'album_name', 'album_type',
                        'album_release_date', 'album_release_date_precision', 'album_total_tracks',
                        'album_spotify_url', 'album_uri', 'album_images_json', 'album_artwork_url',
                        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'analysis_url'
                    ]
                    for key in list_keys:
                        dict_info_tracks[key] = "Network Error"
                    dict_tracks[song_key] = dict_info_tracks
                    continue

                except Exception as e:
                    print(f"❌ Unexpected error for track '{track_name} - {artist_name}': {e}")
                    dict_info_tracks['track_name'] = track_name
                    dict_info_tracks['artist_name'] = artist_name
                    list_keys = [
                        'spotify_track_id', 'spotify_album_id', 'spotify_track_url', 'spotify_track_uri',
                        'track_href', 'isrc', 'preview_url', 'track_number', 'disc_number', 'track_duration',
                        'track_popularity', 'explicit', 'is_local', 'album_name', 'album_type',
                        'album_release_date', 'album_release_date_precision', 'album_total_tracks',
                        'album_spotify_url', 'album_uri', 'album_images_json', 'album_artwork_url',
                        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'analysis_url'
                    ]
                    for key in list_keys:
                        dict_info_tracks[key] = "Unknown Error"
                    dict_tracks[song_key] = dict_info_tracks
                    continue
        print(f"{count} new tracks were added to the track dictionnary \n")
        df_tracks = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
        df_tracks.to_csv(self.tracks_work_file, sep='|', index=False)
        return df_tracks

    def merge_dfs(self, df, artist_df, track_df):
        """Merging the export df, with the artist info and track info dfs"""
        df_merge_artist = pd.merge(df, artist_df, how='left', on='artist_name')
        cols_to_use = list(track_df.columns.difference(df_merge_artist.columns))
        cols_to_use.append('song_key')
        df_merge_artist_track = pd.merge(df_merge_artist, track_df[cols_to_use], how='left', on='song_key')
        return df_merge_artist_track

    def select_output_columns(self, df):
        """
        Select only the columns needed for final output from the full work file data.
        This allows work files to store ALL API fields while keeping output files clean.

        Work files contain ~60+ columns (all Spotify API fields).
        Output files contain ~30-35 columns (only what's needed for website/analysis).

        Args:
            df (pandas.DataFrame): Merged dataframe with all columns from work files

        Returns:
            pandas.DataFrame: Dataframe with only output columns
        """
        # Define columns needed for processed output file
        output_columns = [
            # Core identifiers
            'timestamp', 'song_key', 'artist_name', 'album_name', 'track_name',

            # Album metadata
            'album_release_date', 'album_type',

            # Artist metadata
            'followers', 'followers_total', 'artist_popularity', 'popularity',

            # Genres (keep first 14 for display, plus JSON for future use)
            'genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6',
            'genre_7', 'genre_8', 'genre_9', 'genre_10', 'genre_11', 'genre_12',
            'genre_13', 'genre_14', 'genres_json',

            # Track metadata
            'track_duration', 'track_popularity', 'track_number', 'explicit',

            # Audio features
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',

            # Artwork URLs
            'album_artwork_url', 'artist_artwork_url',

            # Listening behavior (calculated fields)
            'completion', 'skip_next_track',

            # Discovery flags
            'new_artist_yn', 'new_track_yn', 'new_recurring_artist_yn', 'new_recurring_track_yn'
        ]

        # Keep only columns that exist in the dataframe
        existing_output_columns = [col for col in output_columns if col in df.columns]

        # Log how many columns are being filtered out
        removed_count = len(df.columns) - len(existing_output_columns)
        if removed_count > 0:
            print(f"📊 Filtered output: keeping {len(existing_output_columns)} columns, removing {removed_count} work-file-only columns")

        return df[existing_output_columns]

    def power_bi_processing(self, df):
        """Changes some results for better display in PBI"""
        df['genre_1'].fillna('Unknown')
        df['track_duration'] = df['track_duration'].replace('No API result', '0').astype(float)
        return df

    def merge_and_deduplicate(self, new_data_df):
        """
        Merge new data with existing data and remove duplicates.

        Args:
            new_data_df (pandas.DataFrame): New track data

        Returns:
            pandas.DataFrame: Merged and deduplicated data
        """
        try:
            # Read existing data
            if os.path.exists(self.processed_file_path):
                # Try different encodings to read the file (UTF-8 first as it's the new standard)
                encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be']
                existing_df = None

                for encoding in encodings_to_try:
                    try:
                        existing_df = pd.read_csv(self.processed_file_path, sep='|', encoding=encoding, low_memory=False)
                        break
                    except UnicodeError:
                        continue
                    except Exception as e:
                        if encoding == encodings_to_try[-1]:  # Last encoding attempt
                            raise e
                        continue

                if existing_df is None:
                    raise Exception("Could not read existing file with any supported encoding")

                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                print(f"Loaded {len(existing_df)} existing tracks")
            else:
                existing_df = pd.DataFrame()
                print("No existing data file found")

            if new_data_df.empty:
                print("No new data to merge")
                return existing_df

            print(f"Merging {len(new_data_df)} new tracks")

            # Combine dataframes
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            else:
                combined_df = new_data_df.copy()

            # Remove duplicates based on song_key and timestamp
            # Keep the first occurrence (existing data takes precedence)
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['song_key', 'timestamp'], keep='first')
            final_count = len(combined_df)

            duplicates_removed = initial_count - final_count
            if duplicates_removed > 0:
                print(f"Removed {duplicates_removed} duplicate tracks")

            # Sort by timestamp (newest first, matching your existing file structure)
            combined_df = combined_df.sort_values('timestamp', ascending=False)

            print(f"Final dataset contains {len(combined_df)} tracks")
            return combined_df

        except Exception as e:
            print(f"Error merging data: {e}")
            return new_data_df

    def save_data(self, df):
        """
        Save the processed data to the CSV file.

        Args:
            df (pandas.DataFrame): Data to save
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.processed_file_path), exist_ok=True)

            # Enforce snake_case before saving
            df = enforce_snake_case(df, "processed file")

            # Save with pipe separator and UTF-8 encoding (required for website parsing)
            df.to_csv(self.processed_file_path, sep='|', encoding='utf-8', index=False)
            print(f"Successfully saved {len(df)} tracks to {self.processed_file_path}")

            # Generate website files
            self.generate_music_website_page_files(df)

        except Exception as e:
            print(f"Error saving data: {e}")
            raise

    def generate_music_website_page_files(self, df):
        """
        Generate website-optimized files for the Music page.

        Args:
            df: Processed dataframe (already in snake_case)

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n🌐 Generating website files for Music page...")

        try:
            # Ensure output directory exists
            website_dir = 'files/website_files/music'
            os.makedirs(website_dir, exist_ok=True)

            # Work with copy to avoid modifying original
            df_web = df.copy()

            # Enforce snake_case before saving
            df_web = enforce_snake_case(df_web, "music_page_data")

            # Add toggle_id - unique identifier for each listening event
            df_web['toggle_id'] = range(1, len(df_web) + 1)
            print(f"✅ Added toggle_id column ({len(df_web):,} toggles)")

            # Add listening_seconds - calculated from completion and track_duration
            # listening_seconds = (completion * track_duration_ms) / 1000
            df_web['listening_seconds'] = (df_web['completion'] * df_web['track_duration']) / 1000
            df_web['listening_seconds'] = df_web['listening_seconds'].fillna(0).astype(int)
            print(f"✅ Added listening_seconds column")

            # Combine genres from genre_1 through genre_14 into single 'genres' column
            # Use comma separator (not pipe, which is the CSV delimiter)
            genre_cols = [f'genre_{i}' for i in range(1, 15)]
            df_web['genres'] = df_web[genre_cols].apply(
                lambda row: ', '.join([str(g) for g in row if pd.notna(g) and str(g) != '' and str(g) != 'nan']),
                axis=1
            )
            print(f"✅ Combined genres from genre_1-14 into single column (comma-separated)")

            # Add simplified_genre column using genre_1
            df_web['simplified_genre'] = df_web['genre_1'].apply(get_simplified_genre)
            print(f"✅ Created simplified_genre from genre_1 using genre mapping")

            # Optionally run analysis to show unmapped genres
            # Uncomment to see which genres need mapping:
            # analyze_unmapped_genres(df_web, top_n=20)

            # Select columns for website file
            website_columns = [
                'toggle_id',
                'song_key',
                'artist_name',
                'album_name',
                'track_name',
                'timestamp',
                'album_release_date',
                'followers',
                'artist_popularity',
                'genre_1',
                'simplified_genre',
                'genres',
                'track_duration',
                'track_popularity',
                'completion',
                'skip_next_track',
                'listening_seconds',
                'new_artist_yn',
                'new_track_yn'
            ]

            # Add album_artwork_url if it exists in the data
            if 'album_artwork_url' in df_web.columns:
                website_columns.append('album_artwork_url')
                print(f"✅ Including album_artwork_url column")

            # Add artist_artwork_url if it exists in the data
            if 'artist_artwork_url' in df_web.columns:
                website_columns.append('artist_artwork_url')
                print(f"✅ Including artist_artwork_url column")

            # Filter to only website columns
            df_web = df_web[website_columns]
            print(f"✅ Removed {len(df.columns) - len(website_columns)} unused columns")

            # Save website file
            website_path = f'{website_dir}/music_page_data.csv'
            df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
            print(f"✅ Website file: {len(df_web):,} records → {website_path}")

            return True

        except Exception as e:
            print(f"❌ Error generating website files: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_new_data_with_pipeline(self, raw_df):
        """
        Apply the complete processing pipeline to new data from the API.
        This replicates the processing logic from create_lastfm_file().

        Args:
            raw_df (pandas.DataFrame): Raw data from Last.fm API

        Returns:
            pandas.DataFrame: Fully processed data
        """
        if raw_df.empty:
            print("No new data to process")
            return raw_df

        print("🚀 Processing new Last.fm data with full pipeline...")

        # Apply timezone correction
        print("🕐 Converting timestamps...")
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

        # Add Spotify legacy data if available
        if os.path.exists(self.spotify_file_path):
            print("🎧 Merging with Spotify legacy data...")
            raw_df = self.add_spotify_legacy(raw_df)
        else:
            print("⚠️  Spotify legacy file not found, proceeding without merging")

        # Get Spotify API credentials
        client_id = os.environ.get('Spotify_API_Client_ID')
        client_secret = os.environ.get('Spotify_API_Client_Secret')

        if not client_id or not client_secret:
            print("❌ Spotify API credentials not found in environment variables")
            print("Proceeding with basic Last.fm data only...")
            # Return basic data without Spotify enrichment
            raw_df['song_key'] = (raw_df['track_name'] + " /: " + raw_df['artist_name']).replace(np.nan, '')
            return raw_df

        # Authenticate with Spotify API
        print("🔐 Authenticating with Spotify API...")
        token = self.authentification(client_id, client_secret)

        # Prepare data for API calls
        unique_artists = list(raw_df.artist_name.astype(str).replace("nan", "nan_").unique())
        raw_df['song_key'] = (raw_df['track_name'] + " /: " + raw_df['artist_name']).replace(np.nan, '')
        unique_tracks = list(raw_df.song_key.astype(str).unique())

        # Get artist information from Spotify API
        print("🎤 Gathering artist information from Spotify API...")
        artist_df = self.artist_info(token, unique_artists)

        # Get track information from Spotify API
        print("🎶 Gathering track information from Spotify API...")
        track_df = self.track_info(token, unique_tracks)

        # Merge all data together
        print("🔄 Merging data...")
        processed_df = self.merge_dfs(raw_df, artist_df, track_df)

        # Select only output columns (reduces from ~60 work file columns to ~35 output columns)
        processed_df = self.select_output_columns(processed_df)

        processed_df = self.power_bi_processing(processed_df)

        # Calculate listening statistics
        print("📊 Calculating listening statistics...")
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
        processed_df.sort_values('timestamp', ascending=True, inplace=True)

        # Add new artist/track flags
        processed_df['new_artist_yn'] = processed_df.groupby('artist_name').cumcount() == 0
        processed_df['new_recurring_artist_yn'] = processed_df.groupby('artist_name').cumcount() == 10
        processed_df['new_track_yn'] = processed_df.groupby('track_name').cumcount() == 0
        processed_df['new_recurring_track_yn'] = processed_df.groupby('track_name').cumcount() == 5

        # Convert boolean flags to integers
        processed_df['new_artist_yn'] = processed_df['new_artist_yn'].astype(int)
        processed_df['new_recurring_artist_yn'] = processed_df['new_recurring_artist_yn'].astype(int)
        processed_df['new_track_yn'] = processed_df['new_track_yn'].astype(int)
        processed_df['new_recurring_track_yn'] = processed_df['new_recurring_track_yn'].astype(int)

        # Sort by timestamp descending and compute completion
        processed_df.sort_values('timestamp', ascending=False, inplace=True)
        processed_df = self.compute_completion(processed_df.reset_index(drop=True))

        return processed_df

    def process_incremental_update(self):
        """
        Main method to perform incremental update of Last.fm data with full pipeline processing.
        Uses checkpointed export file for crash recovery.
        """
        print("Starting Last.fm incremental update with full pipeline...")
        print("=" * 60)

        try:
            # Step 1: Get latest timestamp from export file (checkpoint)
            latest_timestamp = self.get_latest_timestamp_from_file()

            # Step 2: Fetch new tracks from API (saves incrementally to export file)
            fetch_success = self.fetch_tracks_since_timestamp(latest_timestamp)

            if not fetch_success:
                print("❌ API fetch failed")
                return False

            # Step 3: Check if export file has data to process
            if not os.path.exists(self.export_file_path):
                print("No export data found. Nothing to process!")
                print("Re-generating website files from existing processed data...")
                # Still regenerate website files even if no new data
                if os.path.exists(self.processed_file_path):
                    df = pd.read_csv(self.processed_file_path, sep='|', encoding='utf-8', low_memory=False)
                    self.generate_music_website_page_files(df)
                    print("✅ Website files regenerated")
                else:
                    print("⚠️  No existing processed file found")
                return True

            # Step 4: Read the export file (raw API data)
            print(f"📖 Reading export file: {self.export_file_path}")
            new_data_df = pd.read_csv(self.export_file_path, sep='|', encoding='utf-8', low_memory=False)

            if new_data_df.empty:
                print("Export file is empty. Data is up to date!")
                print("Re-generating website files from existing processed data...")
                if os.path.exists(self.processed_file_path):
                    df = pd.read_csv(self.processed_file_path, sep='|', encoding='utf-8', low_memory=False)
                    self.generate_music_website_page_files(df)
                    print("✅ Website files regenerated")
                return True

            # Step 5: Apply full processing pipeline to export data
            processed_new_data = self.process_new_data_with_pipeline(new_data_df)

            # Step 6: Merge with existing processed data and remove duplicates
            final_df = self.merge_and_deduplicate(processed_new_data)

            # Step 7: Save the updated processed data
            self.save_data(final_df)

            print("=" * 60)
            print("Last.fm incremental update completed successfully!")
            print(f"📊 Processed {len(new_data_df)} tracks from export file")
            print(f"📊 Final dataset contains {len(final_df)} total tracks")

            return True

        except Exception as e:
            print(f"Error during incremental update: {e}")
            import traceback
            traceback.print_exc()
            return False

    def upload_results(self):
        """
        Uploads the processed Last.fm files to Google Drive.
        Returns True if successful, False otherwise.
        """
        print("⬆️  Uploading Last.fm results to Google Drive...")

        files_to_upload = ['files/website_files/music/music_page_data.csv']

        # Filter to only existing files
        existing_files = [f for f in files_to_upload if os.path.exists(f)]

        if not existing_files:
            print("❌ No files found to upload")
            return False

        print(f"📤 Uploading {len(existing_files)} files...")
        success = upload_multiple_files(existing_files)

        if success:
            print("✅ Last.fm results uploaded successfully!")
        else:
            print("❌ Some files failed to upload")

        return success


def full_lastfm_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Last.fm API pipeline with 3 standard options.
    Uses Last.fm API for automatic incremental updates.

    Options:
    1. Fetch new data from API, process, and upload to Drive
    2. Process existing data and upload to Drive
    3. Upload existing processed files to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🎵 LAST.FM DATA PIPELINE")
    print("="*60)

    try:
        if auto_process_only:
            print("🤖 Auto process mode: Processing existing data and uploading...")
            choice = "2"
        elif auto_full:
            print("🤖 Auto mode: Fetching new data from API...")
            choice = "1"
        else:
            print("\nSelect an option:")
            print("1. Fetch new data from API, process, and upload to Drive")
            print("2. Process existing data and upload to Drive")
            print("3. Upload existing processed files to Drive")

            choice = input("\nEnter your choice (1-3): ").strip()

        success = False
        processor = LastFmAPIProcessor()

        if choice == "1":
            print("\n🚀 Fetch new data from API, process, and upload to Drive...")

            # Fetch and process new data from API
            process_success = processor.process_incremental_update()

            if not process_success:
                print("❌ Processing failed, skipping upload")
                return False

            # Test drive connection before upload
            if not verify_drive_connection():
                print("⚠️  Warning: Google Drive connection issues detected")
                proceed = input("Continue with upload anyway? (Y/N): ").upper() == 'Y'
                if not proceed:
                    print("✅ Processing completed successfully (upload skipped)")
                    return True

            # Upload results
            success = processor.upload_results()

        elif choice == "2":
            print("\n⚙️  Process existing data and upload to Drive...")

            # Check if existing file exists
            if not os.path.exists(processor.processed_file_path):
                print("❌ No existing processed file found")
                return False

            # Re-process existing file to regenerate website files
            print("📊 Reading existing processed file...")
            df = pd.read_csv(processor.processed_file_path, sep='|', encoding='utf-8', low_memory=False)
            print(f"✅ Loaded {len(df):,} existing tracks")

            # Regenerate website files
            processor.generate_music_website_page_files(df)

            # Upload results
            success = processor.upload_results()

        elif choice == "3":
            print("\n⬆️  Upload existing processed files to Drive...")
            success = processor.upload_results()

        else:
            print("❌ Invalid choice. Please select 1-3.")
            return False

        # Final status
        print("\n" + "="*60)
        if success:
            print("✅ Last.fm pipeline completed successfully!")
            record_successful_run('music_lastfm', 'active')
        else:
            print("❌ Last.fm pipeline failed")
        print("="*60)

        return success

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False


def main():
    """Main function to run the Last.fm API processing."""
    try:
        print("🎵 Last.fm Processing Tool")
        print("This tool fetches new data from Last.fm API and processes it with the full pipeline.")

        # Run the pipeline (interactive mode)
        success = full_lastfm_pipeline(auto_full=False)

        if success:
            print("\n🎉 All done! Your Last.fm data has been updated.")
        else:
            print("\n❌ Pipeline failed. Check the output above for details.")
            return 1

    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
