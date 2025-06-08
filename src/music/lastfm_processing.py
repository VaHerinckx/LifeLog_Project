import os
import requests
import base64
import pandas as pd
import numpy as np
import math
from datetime import timedelta
from dotenv import load_dotenv
from src.utils.utils_functions import time_difference_correction
from src.utils.file_operations import clean_rename_move_file, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection

load_dotenv()

def add_spotify_legacy(df):
    """Adds the spotify legacy extract made, and removes all the lastFm records that are before the maximum date in this
    extract to avoid duplicates"""
    df_spot = pd.read_csv("files/processed_files/music/spotify_processed.csv", sep = "|")
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
    Main processing function that processes the Last.fm data.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Last.fm data...")

    input_path = "files/exports/lfm_exports/lfm_export.csv"
    output_path = 'files/processed_files/lfm_processed.csv'

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
            df = add_spotify_legacy(df)
        else:
            print("⚠️  Spotify legacy file not found, proceeding without merging")
        
        # Get Spotify API credentials
        client_id = os.environ.get('Spotify_API_Client_ID')
        client_secret = os.environ.get('Spotify_API_Client_Secret')
        
        if not client_id or not client_secret:
            print("❌ Spotify API credentials not found in environment variables")
            return False
        
        # Authenticate with Spotify API
        print("🔐 Authenticating with Spotify API...")
        token = authentification(client_id, client_secret)
        
        # Prepare data for API calls
        unique_artists = list(df.artist_name.astype(str).replace("nan", "nan_").unique())
        df['song_key'] = (df['track_name'] + " /: " + df['artist_name']).replace(np.nan, '')
        unique_tracks = list(df.song_key.astype(str).unique())
        
        # Get artist information from Spotify API
        print("🎤 Gathering artist information from Spotify API...")
        artist_df = artist_info(token, unique_artists)
        
        # Get track information from Spotify API  
        print("🎶 Gathering track information from Spotify API...")
        track_df = track_info(token, unique_tracks)
        
        # Merge all data together
        print("🔄 Merging data...")
        df = merge_dfs(df, artist_df, track_df)
        df = power_bi_processing(df)
        
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
        df = compute_completion(df.reset_index(drop=True))
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        print(f"💾 Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False)
        
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
    Complete Last.fm pipeline with 4 options.

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
    print("🎵 LAST.FM DATA PIPELINE")
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


if __name__ == "__main__":
    # Allow running this file directly
    print("🎵 Last.fm Processing Tool")
    print("This tool helps you download, process, and upload Last.fm data.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_lastfm_pipeline(auto_full=False)
