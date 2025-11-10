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


def time_difference_correction(df: pd.DataFrame, timestamp_column: str, source_timezone: str = 'GMT', debug: bool = False) -> pd.DataFrame:
    """
    Corrects timezone differences by converting timestamps from source timezone to local timezone
    based on location data from combined_timezone_processed.csv.

    Now handles multiple timestamp formats automatically:
    - String timestamps (ISO format)
    - Numeric timestamps (seconds or milliseconds since epoch)
    - Datetime objects (timezone-aware or naive)

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

    # DEBUG: Very first thing - show what we received (only if debug enabled)
    if debug:
        print(f"\n🚨 DEBUG - FUNCTION START")
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Timestamp column: {timestamp_column}")
        print(f"   Source timezone: {source_timezone}")
        print(f"   Available columns: {list(df.columns)}")

        if timestamp_column in df.columns:
            sample_values = df[timestamp_column].head()
            print(f"   Sample timestamp values: {sample_values.tolist()}")
            print(f"   Timestamp column dtype: {df[timestamp_column].dtype}")
            print(f"   Has any null values: {df[timestamp_column].isnull().any()}")
        else:
            print(f"   ❌ TIMESTAMP COLUMN '{timestamp_column}' NOT FOUND!")

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


    def smart_timestamp_conversion(series: pd.Series) -> pd.Series:
        """
        Intelligently convert timestamps from various formats to pandas datetime.
        Handles:
        - String timestamps (ISO format)
        - Numeric timestamps (seconds or milliseconds since epoch)
        - Datetime objects (timezone-aware or naive)

        Returns timezone-naive datetime series.
        """
        # Sample a few values to determine the format
        sample_values = series.dropna().head(10)

        if len(sample_values) == 0:
            print("⚠️  No valid timestamp values found")
            return pd.to_datetime(series, errors='coerce')

        first_value = sample_values.iloc[0]

        # If already datetime, just ensure it's timezone-naive
        if pd.api.types.is_datetime64_any_dtype(series):
            print("🕐 Timestamps are already datetime objects")
            result = pd.to_datetime(series)  # Force conversion to ensure consistency
            # Remove timezone info if present
            if hasattr(result.dtype, 'tz') and result.dtype.tz is not None:
                print("   Removing timezone information...")
                result = result.dt.tz_localize(None)
            return result

        # If numeric, determine if seconds or milliseconds
        elif pd.api.types.is_numeric_dtype(series):
            print("🔢 Numeric timestamps detected")

            # Check if values are too large for seconds (likely milliseconds)
            max_val = series.max()
            if max_val > 2000000000000:  # Milliseconds threshold (year 2033+)
                print("   Converting from milliseconds...")
                return pd.to_datetime(series, unit='ms', errors='coerce')
            else:
                print("   Converting from seconds...")
                return pd.to_datetime(series, unit='s', errors='coerce')

        # If string/object, try parsing as ISO format
        else:
            print("📝 String timestamps detected")
            result = pd.to_datetime(series, errors='coerce')

            # Remove timezone info if present
            if hasattr(result.dtype, 'tz') and result.dtype.tz is not None:
                print("   Removing timezone information...")
                result = result.dt.tz_localize(None)

            return result


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

        # DEBUG: Check location data immediately after loading (only if debug enabled)
        if debug:
            print(f"🚨 DEBUG - RAW LOCATION DATA:")
            print(f"   Location columns: {list(location_df.columns)}")
            if 'timestamp' in location_df.columns:
                sample_loc_timestamps = location_df['timestamp'].head()
                print(f"   Raw location timestamps: {sample_loc_timestamps.tolist()}")
                print(f"   Raw location timestamp dtype: {location_df['timestamp'].dtype}")

        has_location_data = True

    except FileNotFoundError:
        print(f"⚠️  Location file not found: {location_file_path}")
        print("⚠️  Proceeding with simple timezone conversion (no location-based correction)")
        has_location_data = False
        location_df = None

    # Validate location data (only if we have location data)
    if has_location_data:
        required_columns = ['timestamp', 'timezone']
        missing_columns = [col for col in required_columns if col not in location_df.columns]
        if missing_columns:
            print(f"⚠️  Location data missing required columns: {missing_columns}")
            print("⚠️  Proceeding with simple timezone conversion")
            has_location_data = False

    # Process location data only if available
    if has_location_data:
        # Parse location timestamps
        if debug:
            print(f"🚨 DEBUG - PARSING LOCATION TIMESTAMPS:")
            print(f"   Before parsing: {location_df['timestamp'].dtype}")
            print(f"   Sample raw values: {location_df['timestamp'].head().tolist()}")
            print(f"   Type of first value: {type(location_df['timestamp'].iloc[0])}")

        # CRITICAL FIX: Location timestamps contain timezone info (e.g., 2023-01-01T00:00:00+01:00)
        # We need to parse them WITH timezone info and convert to UTC for comparison
        try:
            # Parse timestamps keeping timezone information
            location_df['timestamp_parsed'] = pd.to_datetime(location_df['timestamp'], errors='coerce', utc=False)

            if debug:
                print(f"   After initial parsing: {location_df['timestamp_parsed'].dtype}")
                print(f"   Sample parsed values: {location_df['timestamp_parsed'].head().tolist()}")
                if hasattr(location_df['timestamp_parsed'].dtype, 'tz'):
                    print(f"   Timezone info: {location_df['timestamp_parsed'].dtype.tz}")

            # Convert timezone-aware timestamps to UTC, then remove timezone info for comparison
            if hasattr(location_df['timestamp_parsed'].dtype, 'tz') and location_df['timestamp_parsed'].dtype.tz is not None:
                # Already has timezone, convert to UTC
                print("   📍 Location timestamps have timezone info, converting to UTC...")
                location_df['timestamp_parsed'] = location_df['timestamp_parsed'].dt.tz_convert('UTC').dt.tz_localize(None)
            else:
                # Try to infer timezone from the string format
                # Check if first timestamp has timezone offset (e.g., +01:00)
                first_ts = str(location_df['timestamp'].iloc[0])
                if '+' in first_ts or first_ts.count('-') > 2:  # Has timezone offset
                    print("   📍 Parsing timestamps with timezone information...")
                    # Re-parse with timezone awareness
                    location_df['timestamp_parsed'] = pd.to_datetime(location_df['timestamp'], errors='coerce')
                    # Convert to UTC and remove timezone
                    if hasattr(location_df['timestamp_parsed'].dtype, 'tz') and location_df['timestamp_parsed'].dtype.tz is not None:
                        location_df['timestamp_parsed'] = location_df['timestamp_parsed'].dt.tz_convert('UTC').dt.tz_localize(None)
                    else:
                        # Pandas didn't detect timezone, manually handle
                        print("   ⚠️  Timezone in string but not detected by pandas, converting manually...")
                        location_df['timestamp_parsed'] = pd.to_datetime(location_df['timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
                else:
                    print("   📍 Location timestamps are already timezone-naive (UTC assumed)")

            if debug:
                print(f"   Final location timestamp dtype: {location_df['timestamp_parsed'].dtype}")
                print(f"   Final sample (should be UTC): {location_df['timestamp_parsed'].head().tolist()}")

        except Exception as e:
            print(f"   ❌ Error parsing location timestamps: {e}")
            print("⚠️  Falling back to simple timezone conversion")
            import traceback
            traceback.print_exc()
            has_location_data = False

        if has_location_data:
            location_df = location_df.sort_values('timestamp_parsed')

            # Parse location timezone offsets
            location_df['local_offset_minutes'] = location_df['timezone'].apply(parse_timezone_offset)

            print(f"📅 Location data range: {location_df['timestamp_parsed'].min()} to {location_df['timestamp_parsed'].max()}")

            # DEBUG: Show sample of location data
            if debug:
                print("🔍 DEBUG - Sample location data:")
                print(location_df[['timestamp', 'timestamp_parsed', 'timezone', 'local_offset_minutes']].head())
                print(f"   Location timestamp dtype: {location_df['timestamp_parsed'].dtype}")

    # Copy input dataframe to avoid modifying original
    result_df = df.copy()

    # Smart timestamp conversion
    print("🔄 Processing timestamps...")
    if debug:
        print(f"🚨 DEBUG - SMART CONVERSION INPUT:")
        print(f"   Input dtype: {result_df[timestamp_column].dtype}")
        print(f"   Input sample: {result_df[timestamp_column].head().tolist()}")

    result_df['temp_timestamp_parsed'] = smart_timestamp_conversion(result_df[timestamp_column])

    if debug:
        print(f"🚨 DEBUG - SMART CONVERSION OUTPUT:")
        print(f"   Output dtype: {result_df['temp_timestamp_parsed'].dtype}")
        print(f"   Output sample: {result_df['temp_timestamp_parsed'].head().tolist()}")
        if hasattr(result_df['temp_timestamp_parsed'].dtype, 'tz'):
            print(f"   Output timezone: {result_df['temp_timestamp_parsed'].dtype.tz}")

        # DEBUG: Show sample of input timestamp data
        print("🔍 DEBUG - Sample input timestamp data:")
        print(f"   Original values: {result_df[timestamp_column].head().tolist()}")
        print(f"   Original dtype: {result_df[timestamp_column].dtype}")
        print(f"   Parsed values: {result_df['temp_timestamp_parsed'].head().tolist()}")
        print(f"   Parsed dtype: {result_df['temp_timestamp_parsed'].dtype}")

    # Convert from source timezone to UTC
    source_offset_timedelta = timedelta(minutes=source_offset_minutes)
    result_df['temp_utc_timestamp'] = result_df['temp_timestamp_parsed'] - source_offset_timedelta

    if debug:
        # DEBUG: Show sample of UTC conversion
        print("🔍 DEBUG - Sample UTC conversion:")
        print(f"   UTC values: {result_df['temp_utc_timestamp'].head().tolist()}")
        print(f"   UTC dtype: {result_df['temp_utc_timestamp'].dtype}")
        if hasattr(result_df['temp_utc_timestamp'].dtype, 'tz'):
            print(f"   UTC timezone: {result_df['temp_utc_timestamp'].dtype.tz}")

    # Choose processing path based on location data availability
    if not has_location_data:
        # Simple fallback: just return the UTC timestamps (since source is GMT/UTC, no adjustment needed)
        print("🌍 Using simple timezone conversion (no location-based adjustments)")
        result_df[timestamp_column] = result_df['temp_utc_timestamp']
        
        # Clean up temporary columns
        temp_columns = [col for col in result_df.columns if col.startswith('temp_')]
        result_df = result_df.drop(columns=temp_columns)
        
        print(f"✅ Successfully converted {len(result_df)} timestamps")
        return result_df

    # Location-based timezone correction (only when location data is available)
    print("🌍 Applying location-based timezone corrections...")

    # CRITICAL FIX: Ensure both DataFrames have timezone-naive timestamps before any operations
    if debug:
        print("🚨 DEBUG - FINAL TIMEZONE CHECK BEFORE COMPARISON:")
        print(f"   Input UTC dtype: {result_df['temp_utc_timestamp'].dtype}")
        print(f"   Location parsed dtype: {location_df['timestamp_parsed'].dtype}")

    # Force both to be timezone-naive
    if hasattr(result_df['temp_utc_timestamp'].dtype, 'tz') and result_df['temp_utc_timestamp'].dtype.tz is not None:
        if debug:
            print("   🔧 Removing timezone from input UTC timestamps...")
        result_df['temp_utc_timestamp'] = result_df['temp_utc_timestamp'].dt.tz_localize(None)
        if debug:
            print(f"   Input UTC after fix: {result_df['temp_utc_timestamp'].dtype}")

    if hasattr(location_df['timestamp_parsed'].dtype, 'tz') and location_df['timestamp_parsed'].dtype.tz is not None:
        if debug:
            print("   🔧 Removing timezone from location timestamps...")
        location_df['timestamp_parsed'] = location_df['timestamp_parsed'].dt.tz_localize(None)
        if debug:
            print(f"   Location after fix: {location_df['timestamp_parsed'].dtype}")

    # Show sample values for final comparison
    if debug:
        print(f"   Sample input UTC: {result_df['temp_utc_timestamp'].head().tolist()}")
        print(f"   Sample location: {location_df['timestamp_parsed'].head().tolist()}")

    # OPTIMIZED APPROACH: Use merge_asof instead of apply for massive performance boost
    # This is O(n log n) instead of O(n*m)
    print("⚡ Using optimized merge for timezone matching...")

    # Save original index to preserve row order for final result
    result_df['_original_index'] = result_df.index

    # Ensure both dataframes are sorted by timestamp
    result_df_sorted = result_df.sort_values('temp_utc_timestamp').reset_index(drop=True)
    location_df = location_df.sort_values('timestamp_parsed').reset_index(drop=True)

    # Use merge_asof to find the nearest location record for each timestamp
    # direction='backward' means we take the most recent location record before or at the timestamp
    merged = pd.merge_asof(
        result_df_sorted,
        location_df[['timestamp_parsed', 'local_offset_minutes']],
        left_on='temp_utc_timestamp',
        right_on='timestamp_parsed',
        direction='backward'
    )

    # Fill any NaN offsets (timestamps before first location record) with the first location's offset
    if merged['local_offset_minutes'].isna().any():
        first_offset = location_df.iloc[0]['local_offset_minutes']
        print(f"   ⚠️  {merged['local_offset_minutes'].isna().sum()} timestamps before location data, using offset {first_offset/60:+.1f}h")
        merged['local_offset_minutes'].fillna(first_offset, inplace=True)

    # Add the offset back to the sorted dataframe
    result_df_sorted['temp_local_offset'] = merged['local_offset_minutes']

    # Convert from UTC to local timezone
    result_df_sorted['temp_corrected_timestamp'] = result_df_sorted['temp_utc_timestamp'] + pd.to_timedelta(result_df_sorted['temp_local_offset'], unit='m')

    # Restore original order
    result_df_sorted = result_df_sorted.sort_values('_original_index').reset_index(drop=True)

    # Update the original timestamp column in the result dataframe
    result_df[timestamp_column] = result_df_sorted['temp_corrected_timestamp'].values

    # Clean up temporary columns (including the index marker)
    temp_columns = [col for col in result_df.columns if col.startswith('temp_') or col == '_original_index']
    result_df = result_df.drop(columns=temp_columns)

    # Calculate summary statistics
    total_corrections = len(result_df)

    # Show sample of corrections (now correctly aligned)
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


def record_successful_run(source_name: str, pipeline_type: str = 'active'):
    """
    Record a successful pipeline run for a data source.
    
    Args:
        source_name (str): Name of the data source (e.g., 'music_lastfm', 'books_goodreads')
        pipeline_type (str): Type of pipeline ('active', 'coordination', 'legacy', 'inactive')
    """
    tracking_file = 'files/tracking/last_successful_runs.csv'
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Ensure the tracking directory exists
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
        
        # Read existing tracking data
        if os.path.exists(tracking_file):
            df = pd.read_csv(tracking_file, sep=',')
        else:
            # Create new tracking file if it doesn't exist
            df = pd.DataFrame(columns=['source_name', 'last_successful_run', 'status', 'pipeline_type'])
        
        # Check if source already exists
        if source_name in df['source_name'].values:
            # Update existing record
            df.loc[df['source_name'] == source_name, 'last_successful_run'] = current_timestamp
            df.loc[df['source_name'] == source_name, 'status'] = 'success'
            df.loc[df['source_name'] == source_name, 'pipeline_type'] = pipeline_type
            print(f"📈 Updated tracking for {source_name}: {current_timestamp}")
        else:
            # Add new record
            new_row = {
                'source_name': source_name,
                'last_successful_run': current_timestamp,
                'status': 'success',
                'pipeline_type': pipeline_type
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"📈 Added new tracking for {source_name}: {current_timestamp}")
        
        # Save updated tracking file
        df.to_csv(tracking_file, sep=',', index=False, encoding='utf-8')
        
    except Exception as e:
        print(f"⚠️  Warning: Could not update tracking file: {e}")
        # Don't fail the pipeline if tracking fails


def get_last_successful_runs():
    """
    Get the tracking data for all data sources.
    
    Returns:
        pd.DataFrame: DataFrame with tracking information, or empty DataFrame if file doesn't exist
    """
    tracking_file = 'files/tracking/last_successful_runs.csv'
    
    try:
        if os.path.exists(tracking_file):
            return pd.read_csv(tracking_file, sep=',')
        else:
            print(f"⚠️  Tracking file not found: {tracking_file}")
            return pd.DataFrame(columns=['source_name', 'last_successful_run', 'status', 'pipeline_type'])
    except Exception as e:
        print(f"⚠️  Error reading tracking file: {e}")
        return pd.DataFrame(columns=['source_name', 'last_successful_run', 'status', 'pipeline_type'])


def display_tracking_summary():
    """
    Display a summary of all data source tracking information.
    """
    df = get_last_successful_runs()
    
    if df.empty:
        print("📊 No tracking data available")
        return
    
    print("\n📊 DATA SOURCE TRACKING SUMMARY")
    print("=" * 50)
    
    # Group by pipeline type
    for pipeline_type in ['active', 'coordination', 'legacy', 'inactive']:
        sources = df[df['pipeline_type'] == pipeline_type]
        if not sources.empty:
            print(f"\n{pipeline_type.upper()} SOURCES:")
            for _, row in sources.iterrows():
                last_run = row['last_successful_run']
                if pd.isna(last_run) or last_run == '':
                    status_icon = "❌"
                    last_run_str = "Never run"
                else:
                    # Calculate days since last run
                    try:
                        last_run_date = pd.to_datetime(last_run)
                        days_ago = (datetime.now() - last_run_date).days
                        if days_ago == 0:
                            status_icon = "✅"
                            last_run_str = "Today"
                        elif days_ago <= 7:
                            status_icon = "🟡"
                            last_run_str = f"{days_ago} days ago"
                        else:
                            status_icon = "🟠"
                            last_run_str = f"{days_ago} days ago"
                    except:
                        status_icon = "❓"
                        last_run_str = "Invalid date"
                
                print(f"  {status_icon} {row['source_name']:<20} | {last_run_str}")
    
    print("=" * 50)
