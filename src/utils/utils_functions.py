import pandas as pd
import datetime
from datetime import datetime, timedelta
import os
import shutil
import zipfile
import re

time_diff_excel = pd.read_excel('files/work_files/GMT_timediff.xlsx')


def today_export():
    current_date = datetime.now()
    date_string = current_date.strftime("%d%m%y")
    return date_string


def time_difference_correction(df: pd.DataFrame, timestamp_column: str, source_timezone: str = 'GMT') -> pd.DataFrame:
    """
    Corrects timezone differences by converting timestamps from source timezone to local timezone
    based on location data from combined_timezone_processed.csv.

    Args:
        df: DataFrame containing timestamps to correct
        timestamp_column: Name of the column containing timestamps
        source_timezone: Current timezone of the input timestamps (e.g., 'GMT', 'UTC', 'UTC+08:00')

    Returns:
        DataFrame with corrected timestamps in local timezone

    Raises:
        ValueError: If source_timezone format is not supported
        FileNotFoundError: If combined_timezone_processed.csv is not found
        KeyError: If timestamp_column is not found in DataFrame
    """

    # Validate inputs
    if timestamp_column not in df.columns:
        raise KeyError(f"Column '{timestamp_column}' not found in DataFrame. Available columns: {list(df.columns)}")

    if df.empty:
        print("⚠️  Empty DataFrame provided, returning as-is")
        return df.copy()

    # Validate and normalize source timezone
    def validate_and_parse_timezone(tz_str: str) -> int:
        """
        Validate source timezone and convert to offset in minutes.

        Args:
            tz_str: Timezone string

        Returns:
            Offset in minutes from UTC (positive = ahead of UTC, negative = behind UTC)
        """
        tz_str = tz_str.upper().strip()

        # Standard timezone mappings
        standard_timezones = {
            'GMT': 0,
            'UTC': 0,
            'CET': 60,      # Central European Time
            'EET': 120,     # Eastern European Time
            'WET': 0,       # Western European Time
            'EST': -300,    # Eastern Standard Time
            'CST': -360,    # Central Standard Time
            'MST': -420,    # Mountain Standard Time
            'PST': -480,    # Pacific Standard Time
            'EDT': -240,    # Eastern Daylight Time
            'CDT': -300,    # Central Daylight Time
            'MDT': -360,    # Mountain Daylight Time
            'PDT': -420,    # Pacific Daylight Time
            'JST': 540,     # Japan Standard Time
            'KST': 540,     # Korea Standard Time
            'CST_CHINA': 480,  # China Standard Time
            'IST': 330,     # India Standard Time
            'AEST': 600,    # Australian Eastern Standard Time
            'NZST': 720,    # New Zealand Standard Time
        }

        # Check standard timezones first
        if tz_str in standard_timezones:
            return standard_timezones[tz_str]

        # Parse UTC+XX:XX or UTC-XX:XX format
        utc_pattern = r'^UTC([+-])(\d{1,2}):?(\d{2})?$'
        match = re.match(utc_pattern, tz_str)

        if match:
            sign = 1 if match.group(1) == '+' else -1
            hours = int(match.group(2))
            minutes = int(match.group(3)) if match.group(3) else 0

            if hours > 14 or minutes >= 60:
                raise ValueError(f"Invalid timezone offset: {tz_str}")

            return sign * (hours * 60 + minutes)

        # Parse simple +XX:XX or -XX:XX format
        simple_pattern = r'^([+-])(\d{1,2}):?(\d{2})?$'
        match = re.match(simple_pattern, tz_str)

        if match:
            sign = 1 if match.group(1) == '+' else -1
            hours = int(match.group(2))
            minutes = int(match.group(3)) if match.group(3) else 0

            if hours > 14 or minutes >= 60:
                raise ValueError(f"Invalid timezone offset: {tz_str}")

            return sign * (hours * 60 + minutes)

        # If no match found
        supported_formats = list(standard_timezones.keys()) + ['UTC+XX:XX', 'UTC-XX:XX', '+XX:XX', '-XX:XX']
        raise ValueError(f"Unsupported timezone format: '{tz_str}'. Supported formats: {supported_formats}")


    def parse_timezone_offset(tz_str: str) -> int:
        """
        Parse timezone string like 'UTC+08:00' to offset in minutes.

        Args:
            tz_str: Timezone string from location data

        Returns:
            Offset in minutes from UTC
        """
        if pd.isna(tz_str) or not isinstance(tz_str, str):
            return 0

        # Clean the string
        tz_str = tz_str.strip()

        # Handle UTC+XX:XX format
        pattern = r'UTC([+-])(\d{1,2}):(\d{2})'
        match = re.search(pattern, tz_str)

        if match:
            sign = 1 if match.group(1) == '+' else -1
            hours = int(match.group(2))
            minutes = int(match.group(3))
            return sign * (hours * 60 + minutes)

        # Fallback - try to extract just the offset part
        pattern = r'([+-])(\d{1,2}):(\d{2})'
        match = re.search(pattern, tz_str)

        if match:
            sign = 1 if match.group(1) == '+' else -1
            hours = int(match.group(2))
            minutes = int(match.group(3))
            return sign * (hours * 60 + minutes)

        print(f"⚠️  Could not parse timezone: {tz_str}, assuming UTC")
        return 0


    print(f"🕐 Starting timezone correction for {len(df)} records...")
    print(f"📊 Source timezone: {source_timezone}")
    print(f"📋 Timestamp column: {timestamp_column}")

    # Validate source timezone
    try:
        source_offset_minutes = validate_and_parse_timezone(source_timezone)
        print(f"✅ Source timezone validated: {source_offset_minutes/60:+.1f} hours from UTC")
    except ValueError as e:
        print(f"❌ {e}")
        raise

    # Load location timezone data
    location_file_path = 'files/processed_files/location/combined_timezone_processed.csv'

    try:
        location_df = pd.read_csv(location_file_path, sep='|')
        print(f"✅ Loaded location data: {len(location_df)} records")
    except FileNotFoundError:
        print(f"❌ Location file not found: {location_file_path}")
        print("💡 Make sure to run the Google Maps processing and merge functions first")
        raise FileNotFoundError(f"Could not find {location_file_path}")

    # Validate location data
    required_columns = ['timestamp', 'timezone']
    missing_columns = [col for col in required_columns if col not in location_df.columns]
    if missing_columns:
        raise ValueError(f"Location data missing required columns: {missing_columns}")

    # Parse location timestamps
    location_df['timestamp_parsed'] = pd.to_datetime(location_df['timestamp'])
    location_df = location_df.sort_values('timestamp_parsed')

    # Parse location timezone offsets
    location_df['local_offset_minutes'] = location_df['timezone'].apply(parse_timezone_offset)

    print(f"📅 Location data range: {location_df['timestamp_parsed'].min()} to {location_df['timestamp_parsed'].max()}")

    # Copy input dataframe to avoid modifying original
    result_df = df.copy()

    # Parse input timestamps
    print("🔄 Processing timestamps...")
    result_df['temp_timestamp_parsed'] = pd.to_datetime(result_df[timestamp_column])

    # Convert from source timezone to UTC
    source_offset_timedelta = timedelta(minutes=source_offset_minutes)
    result_df['temp_utc_timestamp'] = result_df['temp_timestamp_parsed'] - source_offset_timedelta

    # Function to find local timezone for a given UTC timestamp
    def find_local_timezone_offset(utc_timestamp):
        """Find the local timezone offset for a given UTC timestamp."""
        if pd.isna(utc_timestamp):
            return 0

        # Find the location record that covers this timestamp
        # Use the latest record that starts before or at this timestamp
        applicable_records = location_df[location_df['timestamp_parsed'] <= utc_timestamp]

        if len(applicable_records) == 0:
            # No records before this timestamp, use the earliest record
            if len(location_df) > 0:
                offset = location_df.iloc[0]['local_offset_minutes']
                print(f"⚠️  Using earliest location data for {utc_timestamp}")
                return offset
            else:
                print(f"⚠️  No location data available, using UTC for {utc_timestamp}")
                return 0

        # Use the most recent applicable record
        latest_record = applicable_records.iloc[-1]
        return latest_record['local_offset_minutes']

    # Apply timezone correction
    print("🌍 Applying location-based timezone corrections...")
    result_df['temp_local_offset'] = result_df['temp_utc_timestamp'].apply(find_local_timezone_offset)

    # Convert from UTC to local timezone
    result_df['temp_corrected_timestamp'] = result_df.apply(
        lambda row: row['temp_utc_timestamp'] + timedelta(minutes=row['temp_local_offset']),
        axis=1
    )

    # Update the original timestamp column
    result_df[timestamp_column] = result_df['temp_corrected_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Clean up temporary columns
    temp_columns = [col for col in result_df.columns if col.startswith('temp_')]
    result_df = result_df.drop(columns=temp_columns)

    # Calculate summary statistics
    total_corrections = len(result_df)

    # Show sample of corrections
    print(f"\n✅ Successfully corrected {total_corrections} timestamps")

    if total_corrections > 0:
        print("📋 Sample corrections:")
        sample_size = min(5, len(df))

        for i in range(sample_size):
            original = df.iloc[i][timestamp_column]
            corrected = result_df.iloc[i][timestamp_column]
            print(f"  • {original} → {corrected}")

    print(f"🎯 Timezone correction completed!")

    return result_df

def get_response(client, system_prompt, user_prompt):
  # Assign the role and content for each message
  messages = [{"role": "system", "content": system_prompt},
      		  {"role": "user", "content": user_prompt}]
  response = client.chat.completions.create(
      model="gpt-3.5-turbo", messages= messages, temperature=0)
  return response.choices[0].message.content


def find_unzip_folder(data_source, zip_file_path = None):
    """Unzips the foler specified in the path"""
    download_folder = "/Users/valen/Downloads"
    # Get a list of all the zip files in the download folder
    zip_files = [f for f in os.listdir(download_folder) if f.endswith('.zip')]

    for zip_file in zip_files:
        if (data_source == 'garmin') & (len(zip_file[:-4]) == 38):
            zip_file_path = os.path.join(download_folder, zip_file)
            break
        elif (data_source == 'kindle') & (zip_file == 'Kindle.zip'):
            zip_file_path = os.path.join(download_folder, zip_file)
            break
        elif (data_source == 'apple') & (zip_file == 'export.zip'):
            zip_file_path = os.path.join(download_folder, zip_file)
            break
        elif (data_source == 'pocket_casts') & (zip_file == 'data_export.zip'):
            zip_file_path = os.path.join(download_folder, zip_file)
            break
    if zip_file_path is not None:
    # Extract the contents of the zip file to a new folder
        unzip_folder = os.path.join(download_folder, f"{data_source}_export_unzipped")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
        os.remove(zip_file_path)
    else:
        return f"No {data_source} file to unzip \n"


def clean_rename_move_folder(export_folder, download_folder, folder_name, new_folder_name):
    """Removes the folder from Download, renames them and sends them within this directory"""
    folder_path = os.path.join(download_folder, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist in Downloads")
        return None
    old_export_folder = os.path.join(export_folder, new_folder_name)
    print(old_export_folder)
    shutil.rmtree(old_export_folder)
    downloaded_folder_path = os.path.join(download_folder, folder_name)
    # Rename the downloaded folder
    renamed_folder_path = os.path.join(download_folder, new_folder_name)
    os.rename(downloaded_folder_path, renamed_folder_path)
    # Move the renamed folder to the export folder
    export_folder_path = os.path.join(export_folder, new_folder_name)
    shutil.move(renamed_folder_path, export_folder_path)


def clean_rename_move_file(export_folder, download_folder, file_name, new_file_name, file_number = 1):
    """Removes the file from Download, renames them and sends them within this directory"""
    file_path = os.path.join(download_folder, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return None
    if file_number == 1:
        for filename in os.listdir(export_folder):
            file_path = os.path.join(export_folder, filename)
            os.remove(file_path)
    filename = 'new_file.csv'  # Change this to the name of your file
    downloaded_file_path = os.path.join(download_folder, file_name)
    # Rename the downloaded file
    renamed_file_path = os.path.join(download_folder, new_file_name)
    os.rename(downloaded_file_path, renamed_file_path)
    # Move the renamed file to the export folder
    export_file_path = os.path.join(export_folder, new_file_name)
    shutil.move(renamed_file_path, export_file_path)
