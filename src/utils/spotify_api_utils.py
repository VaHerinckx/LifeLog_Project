"""
Spotify API Utilities

Shared utilities for Spotify API authentication and data enrichment.
Used by Music source and topic processors.
"""

import requests
import pandas as pd
import time
import os
import json
import base64


def spotify_authentication(client_id, client_secret):
    """
    Generate the access token necessary to call the Spotify API.

    Args:
        client_id: Spotify API client ID
        client_secret: Spotify API client secret

    Returns:
        str: Access token for Spotify API
    """
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


def get_artist_info(token, artist_names, artists_work_file):
    """
    Retrieves information about artists from Spotify API (genre, followers, popularity, etc.).

    Args:
        token: Spotify API access token
        artist_names: List of artist names to fetch info for
        artists_work_file: Path to CSV file for caching artist data

    Returns:
        DataFrame: Artist information with Spotify metadata
    """
    # Ensure work directory exists
    os.makedirs(os.path.dirname(artists_work_file), exist_ok=True)

    # Initialize or load existing artist data
    if os.path.exists(artists_work_file):
        artist_df = pd.read_csv(artists_work_file, sep='|', low_memory=False)
    else:
        artist_df = pd.DataFrame(columns=['artist_name'])

    count = 0
    dict_artists = {}
    for _, row in artist_df.iterrows():
        dict_info_artists = {}
        for col in list(artist_df.columns)[1:]:
            dict_info_artists[col] = row[col]
        # Use lowercase for case-insensitive lookup
        dict_artists[str(row['artist_name']).lower()] = dict_info_artists

    for artist_name in artist_names:
        # Case-insensitive lookup using lowercase
        if str(artist_name).lower() in dict_artists.keys():
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
                    df_artist.to_csv(artists_work_file, sep='|', index=False)
                    time.sleep(retry_after)
                    # Decrement count so we retry this artist
                    count -= 1
                    continue

                # Validate response before parsing JSON
                if not response.text or response.text.strip() == '':
                    print(f"⚠️  Empty response for artist: {artist_name}, skipping...")
                    dict_info_artists['followers'] = "No API result"
                    dict_artists[str(artist_name).lower()] = dict_info_artists
                    continue

                # Raise for other HTTP errors (4xx, 5xx) - after checking 429
                response.raise_for_status()

                response_json = response.json()

                if response_json['artists']['items'] == []:
                    dict_info_artists['followers'] = "Unknown"
                    dict_artists[str(artist_name).lower()] = dict_info_artists
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
                    df_artist.to_csv(artists_work_file, sep='|', index=False)
                    time.sleep(retry_after)
                    # Decrement count so we retry this artist
                    count -= 1
                    continue

                # Validate second API call response
                if not response.text or response.text.strip() == '':
                    print(f"⚠️  Empty response for artist details: {artist_name}, skipping...")
                    dict_info_artists['followers'] = "No API result"
                    dict_artists[str(artist_name).lower()] = dict_info_artists
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
                dict_artists[str(artist_name).lower()] = dict_info_artists
                continue

            except requests.exceptions.RequestException as e:
                print(f"❌ Network error for artist '{artist_name}': {e}")
                dict_info_artists['followers'] = "Network Error"
                dict_artists[str(artist_name).lower()] = dict_info_artists
                continue

            except Exception as e:
                print(f"❌ Unexpected error for artist '{artist_name}': {e}")
                dict_info_artists['followers'] = "Unknown Error"
                dict_artists[str(artist_name).lower()] = dict_info_artists
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

            dict_artists[str(artist_name).lower()] = dict_info_artists

            # Add small delay to avoid rate limiting
            time.sleep(0.15)

            # Save progress every 50 artists (performance optimization)
            if count % 50 == 0:
                df_artist = pd.DataFrame.from_dict(dict_artists, orient='index').reset_index().rename(columns={'index':'artist_name'})
                df_artist.to_csv(artists_work_file, sep='|', index=False)
    print(f"{count} new artist(s) were added to the artist dictionary")
    df_artist = pd.DataFrame.from_dict(dict_artists, orient='index').reset_index().rename(columns={'index':'artist_name'})
    df_artist.drop_duplicates().to_csv(artists_work_file, sep='|', index=False)
    return df_artist


def get_track_info(token, song_keys, tracks_work_file):
    """
    Retrieve information about tracks from Spotify API (duration, popularity, audio features, etc.).

    Args:
        token: Spotify API access token
        song_keys: List of "track_name /: artist_name" keys
        tracks_work_file: Path to CSV file for caching track data

    Returns:
        DataFrame: Track information with Spotify metadata
    """
    # Ensure work directory exists
    os.makedirs(os.path.dirname(tracks_work_file), exist_ok=True)

    # Initialize or load existing track data
    if os.path.exists(tracks_work_file):
        track_df = pd.read_csv(tracks_work_file, sep='|', low_memory=False)
    else:
        track_df = pd.DataFrame(columns=['song_key'])

    count = 0
    # Rebuilding the dictionary
    dict_tracks = {}
    for _, row in track_df.iterrows():
        dict_info_tracks = {}
        for col in list(track_df.columns)[1:]:
            dict_info_tracks[col] = row[col]
        # Use lowercase for case-insensitive lookup
        dict_tracks[str(row['song_key']).lower()] = dict_info_tracks
    # Checking how many songs are already in the dictionary
    count_API_requests = 0
    for song_key in song_keys:
        # Case-insensitive lookup using lowercase
        if (str(song_key).lower() in dict_tracks.keys()) | (song_key == ''):
            pass
        else:
            count_API_requests += 1
    for song_key in song_keys:
        # Case-insensitive lookup using lowercase
        if (str(song_key).lower() in dict_tracks.keys()) | (song_key == ''):
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
                    df_track = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                    df_track.to_csv(tracks_work_file, sep='|', index=False)
                    time.sleep(retry_after)
                    # Decrement count so we retry this track
                    count -= 1
                    continue

                # Validate response before parsing JSON
                if not response.text or response.text.strip() == '':
                    print(f"⚠️  Empty response for track: {song_key}, skipping...")
                    dict_info_tracks['track_duration'] = "No API result"
                    dict_tracks[str(song_key).lower()] = dict_info_tracks
                    continue

                # Raise for other HTTP errors - after checking 429
                response.raise_for_status()
                response_json = response.json()

                if response_json['tracks']['items'] == []:
                    dict_info_tracks['track_duration'] = "Unknown"
                    dict_tracks[str(song_key).lower()] = dict_info_tracks
                    # Add small delay to avoid rate limiting
                    time.sleep(0.15)
                    continue

                track_id = response_json['tracks']['items'][0]['id']

                # Get track info
                track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
                response = requests.get(track_url, headers=headers)

                # Check for rate limiting on second API call
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"⚠️  Rate limited by Spotify. Waiting {retry_after} seconds before retry...")
                    # Save progress before waiting
                    df_track = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                    df_track.to_csv(tracks_work_file, sep='|', index=False)
                    time.sleep(retry_after)
                    # Decrement count so we retry this track
                    count -= 1
                    continue

                # Validate second API call response
                if not response.text or response.text.strip() == '':
                    print(f"⚠️  Empty response for track details: {song_key}, skipping...")
                    dict_info_tracks['track_duration'] = "No API result"
                    dict_tracks[str(song_key).lower()] = dict_info_tracks
                    continue

                # Raise for other HTTP errors - after checking 429
                response.raise_for_status()
                track_info = response.json()

                # Get audio features
                audio_features_url = f"https://api.spotify.com/v1/audio-features/{track_id}"
                response = requests.get(audio_features_url, headers=headers)

                # Check for rate limiting on third API call
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"⚠️  Rate limited by Spotify. Waiting {retry_after} seconds before retry...")
                    # Save progress before waiting
                    df_track = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                    df_track.to_csv(tracks_work_file, sep='|', index=False)
                    time.sleep(retry_after)
                    # Decrement count so we retry this track
                    count -= 1
                    continue

                # Audio features might not be available for all tracks
                audio_features = {}
                if response.status_code == 200 and response.text and response.text.strip() != '':
                    try:
                        audio_features = response.json()
                    except:
                        print(f"⚠️  Could not parse audio features for {song_key}")

            except requests.exceptions.JSONDecodeError as e:
                print(f"❌ JSON parsing error for track '{song_key}'")
                print(f"   Error: {e}")
                print(f"   Response status: {response.status_code}")
                dict_info_tracks['track_duration'] = "API Error - JSON Parse Failed"
                dict_tracks[str(song_key).lower()] = dict_info_tracks
                continue

            except requests.exceptions.RequestException as e:
                print(f"❌ Network error for track '{song_key}': {e}")
                dict_info_tracks['track_duration'] = "Network Error"
                dict_tracks[str(song_key).lower()] = dict_info_tracks
                continue

            except Exception as e:
                print(f"❌ Unexpected error for track '{song_key}': {e}")
                dict_info_tracks['track_duration'] = "Unknown Error"
                dict_tracks[str(song_key).lower()] = dict_info_tracks
                continue

            # Store track metadata
            dict_info_tracks['spotify_track_id'] = track_info.get('id')
            dict_info_tracks['spotify_track_url'] = track_info.get('external_urls', {}).get('spotify')
            dict_info_tracks['track_duration'] = track_info.get('duration_ms')
            dict_info_tracks['explicit'] = track_info.get('explicit')
            dict_info_tracks['track_popularity'] = track_info.get('popularity')
            dict_info_tracks['track_number'] = track_info.get('track_number')
            dict_info_tracks['disc_number'] = track_info.get('disc_number')

            # Album info
            album = track_info.get('album', {})
            dict_info_tracks['album_name'] = album.get('name')
            dict_info_tracks['album_type'] = album.get('album_type')
            dict_info_tracks['album_release_date'] = album.get('release_date')
            dict_info_tracks['album_total_tracks'] = album.get('total_tracks')
            album_images = album.get('images', [])
            dict_info_tracks['album_artwork_url'] = album_images[0].get('url') if album_images else None

            # Audio features (if available)
            if audio_features and isinstance(audio_features, dict):
                dict_info_tracks['danceability'] = audio_features.get('danceability')
                dict_info_tracks['energy'] = audio_features.get('energy')
                dict_info_tracks['key'] = audio_features.get('key')
                dict_info_tracks['loudness'] = audio_features.get('loudness')
                dict_info_tracks['mode'] = audio_features.get('mode')
                dict_info_tracks['speechiness'] = audio_features.get('speechiness')
                dict_info_tracks['acousticness'] = audio_features.get('acousticness')
                dict_info_tracks['instrumentalness'] = audio_features.get('instrumentalness')
                dict_info_tracks['liveness'] = audio_features.get('liveness')
                dict_info_tracks['valence'] = audio_features.get('valence')
                dict_info_tracks['tempo'] = audio_features.get('tempo')
                dict_info_tracks['time_signature'] = audio_features.get('time_signature')

            dict_tracks[str(song_key).lower()] = dict_info_tracks

            # Add small delay to avoid rate limiting
            time.sleep(0.15)

            # Save progress every 50 tracks (performance optimization)
            if count % 50 == 0:
                df_track = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
                df_track.to_csv(tracks_work_file, sep='|', index=False)

    print(f"{count} new track(s) were added to the track dictionary")
    df_track = pd.DataFrame.from_dict(dict_tracks, orient='index').reset_index().rename(columns={'index':'song_key'})
    df_track.drop_duplicates().to_csv(tracks_work_file, sep='|', index=False)
    return df_track
