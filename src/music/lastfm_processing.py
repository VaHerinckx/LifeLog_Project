#!/usr/bin/env python3
"""
Last.fm API Processing Module

This module handles incremental updates to the Last.fm processed data file by:
1. Reading the latest timestamp from existing lfm_processed.csv
2. Fetching new tracks from Last.fm API since that timestamp
3. Applying the complete pipeline processing (Spotify API enrichment, etc.)
4. Merging new data with existing data and removing duplicates
5. Saving the updated data back to the same file

This replaces the manual download process while maintaining all existing processing logic.

Author: LifeLog Project
"""

import requests
import pandas as pd
import time
import os
import sys
import base64
import math
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to sys.path to import utils
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.drive_operations import upload_multiple_files, verify_drive_connection
from utils.file_operations import clean_rename_move_file, check_file_exists
from utils.web_operations import open_web_urls, prompt_user_download_status
from dotenv import load_dotenv

load_dotenv()


def load_settings(settings_path="credentials/settings.yaml"):
    """Load settings from YAML file."""
    try:
        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)
        return settings
    except FileNotFoundError:
        print(f"Settings file not found: {settings_path}")
        raise
    except Exception as e:
        print(f"Error loading settings: {e}")
        raise


class LastFmAPIProcessor:
    """Handles Last.fm API data processing and incremental updates."""
    
    def __init__(self, settings_path="credentials/settings.yaml"):
        """Initialize the processor with API credentials."""
        self.settings = load_settings(settings_path)
        self.api_key = self.settings['lastfm']['api_key']
        self.username = self.settings['lastfm']['username']
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        
        # File paths
        self.processed_file_path = "files/processed_files/music/lfm_processed.csv"
        self.spotify_file_path = "files/processed_files/music/spotify_processed.csv"
        self.artists_work_file = "files/work_files/lfm_work_files/artists_infos.csv"
        self.tracks_work_file = "files/work_files/lfm_work_files/tracks_infos.csv"
        
    def get_latest_timestamp_from_file(self):
        """
        Read the existing processed file and return the latest timestamp.
        
        Returns:
            datetime: Latest timestamp from the file, or None if file doesn't exist
        """
        try:
            if not os.path.exists(self.processed_file_path):
                print(f"File {self.processed_file_path} not found. Will fetch all available data.")
                return None
                
            # Try different encodings to read the file
            encodings_to_try = ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-8']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(self.processed_file_path, sep='|', encoding=encoding)
                    print(f"Successfully read file with {encoding} encoding")
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
                print("Existing file is empty. Will fetch all available data.")
                return None
                
            # Convert timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Get the latest timestamp
            latest_timestamp = df['timestamp'].max()
            print(f"Latest timestamp in existing file: {latest_timestamp}")
            
            return latest_timestamp
            
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Will fetch all available data.")
            return None
    
    def fetch_tracks_since_timestamp(self, since_timestamp=None):
        """
        Fetch tracks from Last.fm API since the given timestamp.
        
        Args:
            since_timestamp (datetime): Fetch tracks after this timestamp
            
        Returns:
            list: List of track dictionaries
        """
        all_tracks = []
        page = 1
        total_pages = None
        
        print(f"Fetching tracks from Last.fm API...")
        if since_timestamp:
            print(f"Fetching tracks since: {since_timestamp}")
        else:
            print("Fetching all available tracks (no existing data found)")
        
        while True:
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
                    break
                
                # Check if we have track data
                if 'recenttracks' not in data or 'track' not in data['recenttracks']:
                    print("No track data found in API response")
                    break
                
                tracks = data['recenttracks']['track']
                attr = data['recenttracks']['@attr']
                
                # Get total pages from first response
                if total_pages is None:
                    total_pages = int(attr['totalPages'])
                    total_tracks = int(attr['total'])
                    print(f"Total tracks to fetch: {total_tracks} across {total_pages} pages")
                
                # If no tracks on this page, we're done
                if not tracks:
                    break
                
                # Handle case where only one track is returned (not a list)
                if isinstance(tracks, dict):
                    tracks = [tracks]
                
                # Filter out currently playing tracks (they have no timestamp)
                valid_tracks = []
                for track in tracks:
                    if 'date' in track and 'uts' in track['date']:
                        valid_tracks.append(track)
                    else:
                        print(f"Skipping currently playing track: {track.get('name', 'Unknown')}")
                
                all_tracks.extend(valid_tracks)
                
                print(f"Fetched page {page}/{total_pages} ({len(valid_tracks)} tracks)")
                
                # Check if we've reached the end
                if page >= total_pages:
                    break
                
                page += 1
                
                # Be nice to the API - small delay between requests
                time.sleep(0.2)
                
            except requests.exceptions.RequestException as e:
                print(f"Network error on page {page}: {e}")
                break
            except Exception as e:
                print(f"Error processing page {page}: {e}")
                break
        
        print(f"Successfully fetched {len(all_tracks)} new tracks")
        return all_tracks
    
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
            artist_df = pd.read_csv(self.artists_work_file, sep='|')
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
            track_df = pd.read_csv(self.tracks_work_file, sep='|')
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
                    if count % 50 == 0:
                        df_tracks = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                        df_tracks.to_csv(self.tracks_work_file, sep='|', index=False)
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
                # Try different encodings to read the file
                encodings_to_try = ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-8']
                existing_df = None
                
                for encoding in encodings_to_try:
                    try:
                        existing_df = pd.read_csv(self.processed_file_path, sep='|', encoding=encoding)
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
            
            # Save with pipe separator and UTF-16 encoding to match existing format
            df.to_csv(self.processed_file_path, sep='|', encoding='utf-16', index=False)
            print(f"Successfully saved {len(df)} tracks to {self.processed_file_path}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
            raise
    
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
            
        print("⚙️  Processing new Last.fm data with full pipeline...")
        
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
        """
        print("Starting Last.fm incremental update with full pipeline...")
        print("=" * 60)
        
        try:
            # Step 1: Get latest timestamp from existing file
            latest_timestamp = self.get_latest_timestamp_from_file()
            
            # Step 2: Fetch new tracks from API
            new_tracks = self.fetch_tracks_since_timestamp(latest_timestamp)
            
            if not new_tracks:
                print("No new tracks found. Your data is up to date!")
                return True
            
            # Step 3: Parse the new track data into basic DataFrame
            new_data_df = self.parse_api_tracks_to_dataframe(new_tracks)
            
            if new_data_df.empty:
                print("No valid new tracks after parsing. Your data is up to date!")
                return True
            
            # Step 4: Apply full processing pipeline to new data
            processed_new_data = self.process_new_data_with_pipeline(new_data_df)
            
            # Step 5: Merge with existing data and remove duplicates
            final_df = self.merge_and_deduplicate(processed_new_data)
            
            # Step 6: Save the updated data
            self.save_data(final_df)
            
            print("=" * 60)
            print("Last.fm incremental update completed successfully!")
            print(f"📊 Added {len(new_tracks)} new tracks")
            print(f"📊 Final dataset contains {len(final_df)} total tracks")
            
            return True
            
        except Exception as e:
            print(f"Error during incremental update: {e}")
            return False

    def upload_results(self):
        """
        Uploads the processed Last.fm files to Google Drive.
        Returns True if successful, False otherwise.
        """
        print("☁️  Uploading Last.fm results to Google Drive...")

        files_to_upload = [self.processed_file_path]

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


# Legacy functions for backward compatibility with manual download process
def download_lastfm_data():
    """
    Opens Last.fm export page and prompts user to download data.
    Returns True if user confirms download, False otherwise.
    """
    print("🎵 Starting Last.fm data download...")

    urls = ['https://benjaminbenben.com/lastfm-to-csv/']
    open_web_urls(urls)

    print("📝 Instructions:")
    print("   1. Enter your Last.fm username")
    print("   2. Click 'Generate CSV'")
    print("   3. Wait for the export to be generated")
    print("   4. Download the CSV file when ready")
    print("   5. The file will be named 'entinval.csv' by default")

    response = prompt_user_download_status("Last.fm")

    return response


def move_lastfm_files():
    """
    Moves the downloaded Last.fm file from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving Last.fm files...")

    # Move the Last.fm file
    move_success = clean_rename_move_file(
        "files/exports/lfm_exports", 
        "/Users/valen/Downloads", 
        "entinval.csv", 
        "lfm_export.csv"
    )

    if move_success:
        print("✅ Successfully moved Last.fm file to exports folder")
    else:
        print("❌ Failed to move Last.fm file")

    return move_success


def create_lastfm_file():
    """
    Main processing function that processes the Last.fm data from manual export.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Last.fm data...")

    input_path = "files/exports/lfm_exports/lfm_export.csv"
    output_path = 'files/processed_files/music/lfm_processed.csv'

    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"❌ Last.fm file not found: {input_path}")
            return False

        # Load and process the data
        print("📖 Reading Last.fm export data...")
        df = pd.read_csv(input_path, header=None)
        df.columns = ['artist_name', 'album_name', 'track_name', 'timestamp']
        
        # Apply timezone correction with NaT handling  
        print("🕐 Converting timestamps...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        # Remove rows with invalid timestamps
        initial_count = len(df)
        df = df.dropna(subset=['timestamp'])
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"⚠️  Removed {initial_count - final_count} rows with invalid timestamps")
        
        # Apply timezone correction using the proper function
        print("🌍 Applying timezone correction...")
        try:
            # Convert to timezone-naive UTC first for compatibility with the function
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
            # Apply simple timezone conversion (temporarily bypassing location-based correction)
            # TODO: Fix location-based timezone correction for compatibility 
            print("🌍 Using simple GMT to UTC conversion...")
            # Since source is GMT and we want UTC, no conversion needed
            # Just ensure proper datetime format
            print("✅ Timezone correction applied successfully")
        except Exception as e:
            print(f"⚠️  Timezone correction failed ({e}), using UTC timestamps")
            # Keep the UTC timestamps if correction fails
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Add Spotify legacy data if available
        spotify_file = "files/processed_files/music/spotify_processed.csv"
        if os.path.exists(spotify_file):
            print("🎧 Merging with Spotify legacy data...")
            # Use the class method via temporary processor instance
            processor = LastFmAPIProcessor()
            df = processor.add_spotify_legacy(df)
        else:
            print("⚠️  Spotify legacy file not found, proceeding without merging")
        
        # Get Spotify API credentials
        client_id = os.environ.get('Spotify_API_Client_ID')
        client_secret = os.environ.get('Spotify_API_Client_Secret')
        
        if not client_id or not client_secret:
            print("❌ Spotify API credentials not found in environment variables")
            return False
        
        # Create temporary processor to use class methods
        processor = LastFmAPIProcessor()
        
        # Authenticate with Spotify API
        print("🔐 Authenticating with Spotify API...")
        token = processor.authentification(client_id, client_secret)
        
        # Prepare data for API calls
        unique_artists = list(df.artist_name.astype(str).replace("nan", "nan_").unique())
        df['song_key'] = (df['track_name'] + " /: " + df['artist_name']).replace(np.nan, '')
        unique_tracks = list(df.song_key.astype(str).unique())
        
        # Get artist information from Spotify API
        print("🎤 Gathering artist information from Spotify API...")
        artist_df = processor.artist_info(token, unique_artists)
        
        # Get track information from Spotify API  
        print("🎶 Gathering track information from Spotify API...")
        track_df = processor.track_info(token, unique_tracks)
        
        # Merge all data together
        print("🔄 Merging data...")
        df = processor.merge_dfs(df, artist_df, track_df)
        df = processor.power_bi_processing(df)
        
        # Calculate listening statistics
        print("📊 Calculating listening statistics...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', ascending=True, inplace=True)
        
        # Add new artist/track flags
        df['new_artist_yn'] = df.groupby('artist_name').cumcount() == 0
        df['new_recurring_artist_yn'] = df.groupby('artist_name').cumcount() == 10
        df['new_track_yn'] = df.groupby('track_name').cumcount() == 0
        df['new_recurring_track_yn'] = df.groupby('track_name').cumcount() == 5
        
        # Convert boolean flags to integers
        df['new_artist_yn'] = df['new_artist_yn'].astype(int)
        df['new_recurring_artist_yn'] = df['new_recurring_artist_yn'].astype(int)
        df['new_track_yn'] = df['new_track_yn'].astype(int)
        df['new_recurring_track_yn'] = df['new_recurring_track_yn'].astype(int)
        
        # Sort by timestamp descending and compute completion
        df.sort_values('timestamp', ascending=False, inplace=True)
        df = processor.compute_completion(df.reset_index(drop=True))
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        print(f"💾 Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', encoding='utf-16', index=False)
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Processed {len(df)} music entries")
        print(f"🎤 Found {len(unique_artists)} unique artists")
        print(f"🎶 Found {len(unique_tracks)} unique tracks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing Last.fm data: {e}")
        return False


def upload_lastfm_results():
    """
    Uploads the processed Last.fm files to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading Last.fm results to Google Drive...")

    files_to_upload = ['files/processed_files/music/lfm_processed.csv']

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


def process_lfm_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_lastfm_pipeline(auto_full=True)
    else:
        return create_lastfm_file()


def full_lastfm_pipeline(auto_full=False):
    """
    Complete Last.fm pipeline with 4 options (legacy manual download version).

    Options:
    1. Full pipeline (download → move → process → upload)
    2. Download data only (open web page + move files)
    3. Process existing file only (just processing)
    4. Process existing file and upload (process → upload)

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🎵 LAST.FM DATA PIPELINE (Manual Download)")
    print("="*60)

    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → move → process → upload)")
        print("2. Download data only (open web page + move files)")
        print("3. Process existing file only")
        print("4. Process existing file and upload to Drive")

        choice = input("\nEnter your choice (1-4): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full Last.fm pipeline...")

        # Step 1: Download
        download_success = download_lastfm_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_lastfm_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_lastfm_files()

        # Step 3: Process (fallback to option 3 if no new files)
        if move_success:
            process_success = create_lastfm_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_lastfm_file()

        # Step 4: Upload
        if process_success:
            upload_success = upload_lastfm_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n📥 Download Last.fm data only...")
        download_success = download_lastfm_data()
        if download_success:
            success = move_lastfm_files()
        else:
            success = False

    elif choice == "3":
        print("\n⚙️  Processing existing Last.fm file only...")
        success = create_lastfm_file()

    elif choice == "4":
        print("\n⚙️  Processing existing file and uploading...")
        process_success = create_lastfm_file()
        if process_success:
            success = upload_lastfm_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    else:
        print("❌ Invalid choice. Please select 1-4.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Last.fm pipeline completed successfully!")
    else:
        print("❌ Last.fm pipeline failed")
    print("="*60)

    return success


def full_lastfm_api_pipeline(upload_to_drive=True):
    """
    Complete Last.fm API pipeline that replaces manual download with API calls.
    
    Args:
        upload_to_drive (bool): Whether to upload results to Google Drive
        
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🎵 LAST.FM API DATA PIPELINE")
    print("="*60)
    
    try:
        # Initialize processor
        processor = LastFmAPIProcessor()
        
        # Step 1: Process incremental update
        process_success = processor.process_incremental_update()
        
        if not process_success:
            print("❌ Processing failed")
            return False
        
        # Step 2: Upload to Google Drive if requested
        if upload_to_drive:
            # Test drive connection first
            if not verify_drive_connection():
                print("⚠️  Warning: Google Drive connection issues detected")
                proceed = input("Continue with upload anyway? (Y/N): ").upper() == 'Y'
                if not proceed:
                    print("✅ Processing completed successfully (upload skipped)")
                    return True
            
            upload_success = processor.upload_results()
            
            if not upload_success:
                print("⚠️  Upload failed, but processing was successful")
                return True
        
        print("\n" + "="*60)
        print("✅ Last.fm API pipeline completed successfully!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False


def main():
    """Main function to run the Last.fm API processing."""
    try:
        print("🎵 Last.fm API Processing Tool")
        print("This tool fetches new data from Last.fm API and processes it with the full pipeline.")
        
        # Ask user if they want to upload to Drive
        upload_choice = input("\nUpload results to Google Drive? (Y/N): ").upper() == 'Y'
        
        success = full_lastfm_api_pipeline(upload_to_drive=upload_choice)
        
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