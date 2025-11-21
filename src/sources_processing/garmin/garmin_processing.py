import pandas as pd
import os
import json
from src.utils.utils_functions import time_difference_correction, find_unzip_folder, clean_rename_move_folder
import fnmatch
import fitparse
import re
import zipfile
# Drive operations removed - handled by topic coordinator
from src.utils.file_operations import check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status

# Constants
DICT_STRESS_LEVEL = {0: 'Rest', 26: 'Low', 51: 'Medium', 76: 'High'}
DICT_COL = {
    'startLatitude': 'no agg',
    'startLongitude': 'no agg',
    'messageIndex': 'no agg',
    'type': 'no agg',
    'startIndex': 'no agg',
    'endIndex': 'no agg',
    'totalExerciseReps': 'no agg',
    'lapIndexes': 'no agg',
    'MAX_RUNCADENCE__BPM': 'no agg',
    'LOSS_ELEVATION__M': 'div',
    'WEIGHTED_MEAN_DOUBLE_CADENCE__BPM': 'no agg',
    'SUM_DISTANCE__M': 'div',
    'MAX_SPEED__KMH': 'no agg',
    'SUM_BMR_ENERGY__KILOJOULE': 'div',
    'SUM_ENERGY__KILOJOULE': 'div',
    'WEIGHTED_MEAN_HEARTRATE__BPM': 'no agg',
    'WEIGHTED_MEAN_SPEED__KMH': 'no agg',
    'WEIGHTED_MEAN_VERTICAL_SPEED__KMH': 'no agg',
    'SUM_DURATION__M': 'div',
    'MAX_DOUBLE_CADENCE__BPM': 'no agg',
    'MAX_HEARTRATE__BPM': 'no agg',
    'SUM_ELAPSEDDURATION__M': 'div',
    'WEIGHTED_MEAN_RUNCADENCE__BPM': 'no agg',
    'GAIN_ELEVATION__M': 'div',
    'SUM_MOVINGDURATION__M': 'div',
    'WEIGHTED_MEAN_STRIDE_LENGTH__M': 'no agg',
    'WEIGHTED_MEAN_ELAPSED_DURATION_VERTICAL_SPEED__KMH': 'no agg',
    'WEIGHTED_MEAN_MOVINGSPEED__KMH': 'no agg',
    'GAIN_CORRECTED_ELEVATION__M': 'div',
    'GAIN_UNCORRECTED_ELEVATION__M': 'div',
    'LOSS_CORRECTED_ELEVATION__M': 'div',
    'LOSS_UNCORRECTED_ELEVATION__M': 'div',
    'END_LATITUDE__DECIMAL_DEGREE': 'no agg',
    'END_LONGITUDE__DECIMAL_DEGREE': 'no agg',
    'MAX_ELEVATION__M': 'no agg',
    'MAX_CORRECTED_ELEVATION__M': 'no agg',
    'MAX_UNCORRECTED_ELEVATION__M': 'no agg',
    'MAX_FRACTIONAL_CADENCE__BPM': 'no agg',
    'MAX_VERTICAL_SPEED__KMH': 'no agg',
    'MIN_ELEVATION__M': 'no agg',
    'MIN_CORRECTED_ELEVATION__M': 'no agg',
    'MIN_UNCORRECTED_ELEVATION__M': 'no agg',
    'MIN_HEARTRATE__BPM': 'no agg',
    'MIN_RUNCADENCE__BPM': 'no agg',
    'MIN_SPEED__KMH': 'no agg',
    'WEIGHTED_MEAN_FRACTIONAL_CADENCE__BPM': 'no agg',
    'MIN_ACTIVITY_LAP_DURATION__M': 'no agg',
    'SUM_STEP__STEP': 'div',
    'startTimeLocalACT': 'no agg'
}

DICT_PATH = 'files/work_files/garmin_work_files/garmin_zip_dictionnary.json'


def check_required_files():
    """Check if required work files exist"""
    try:
        # Check if the dictionary file exists, create if not
        dict_dir = os.path.dirname(DICT_PATH)
        if not os.path.exists(dict_dir):
            os.makedirs(dict_dir, exist_ok=True)

        if not os.path.exists(DICT_PATH):
            with open(DICT_PATH, 'w') as f:
                json.dump({}, f)
            print(f"Created empty dictionary file: {DICT_PATH}")

        return True
    except Exception as e:
        print(f"❌ Error checking required files: {e}")
        return False


def load_fit_files_dictionary():
    """Load the FIT files processing dictionary"""
    try:
        with open(DICT_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"⚠️  Dictionary file not found or corrupted, creating new one")
        return {}


def save_fit_files_dictionary(dict_files):
    """Save the FIT files processing dictionary"""
    try:
        with open(DICT_PATH, 'w') as f:
            json.dump(dict_files, f)
        return True
    except Exception as e:
        print(f"❌ Error saving dictionary file: {e}")
        return False


def row_expander_minutes(row):
    """Function to expand rows based on minute differences and return a DataFrame"""
    try:
        minute_diff = (row['endTimeLocalSPLIT'] - row['startTimeLocalSPLIT']).total_seconds() / 60

        if minute_diff <= 1:
            date_df = pd.DataFrame(columns=list(DICT_COL.keys()))
            date_df.loc[0, 'date'] = row['startTimeLocalSPLIT']
            for col in DICT_COL.keys():
                if col in row.index:
                    date_df.loc[0, col] = row[col]
            return date_df

        dates = pd.date_range(
            row['startTimeLocalSPLIT'],
            row['endTimeLocalSPLIT'] - pd.Timedelta(minutes=1),
            freq='T'
        )
        date_df = pd.DataFrame({'date': dates})

        for col in DICT_COL.keys():
            if col in row.index:
                date_df[col] = row[col] / minute_diff if DICT_COL[col] == 'div' else row[col]
            else:
                date_df[col] = None

        return date_df
    except Exception as e:
        print(f"❌ Error expanding row: {e}")
        return pd.DataFrame()


def generate_to_format_columns(columns):
    """Function to generate lists of columns based on their units"""
    col_duration = []
    col_speed = []
    col_distance = []

    for column in columns:
        if '__' not in column or len(column.split('__')) != 2:
            continue

        unit = column.split('__')[1]
        if unit == 'MILLISECOND':
            col_duration.append(column)
        elif unit == 'CENTIMETERS_PER_MILLISECOND':
            col_speed.append(column)
        elif unit == 'CENTIMETER':
            col_distance.append(column)

    return col_duration, col_speed, col_distance


def rename_formatted_columns(df, col_duration, col_speed, col_distance):
    """Function to rename columns based on their units"""
    rename_dict = {}

    for col in col_duration:
        rename_dict[col] = f"{col.split('__')[0]}__M"

    for col in col_speed:
        rename_dict[col] = f"{col.split('__')[0]}__KMH"

    for col in col_distance:
        rename_dict[col] = f"{col.split('__')[0]}__M"

    return df.rename(columns=rename_dict)


def formatting_garmin_df(df, columns_datetime=None, columns_duration=None, columns_speed=None, columns_distance=None):
    """Function to format specific columns in the DataFrame with improved error handling"""
    if df.empty:
        return df

    try:
        if columns_datetime:
            for col in columns_datetime:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')

        if columns_duration:
            for col in columns_duration:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 1000 / 60

        if columns_speed:
            for col in columns_speed:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') * 36

        if columns_distance:
            for col in columns_distance:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100

        return df
    except Exception as e:
        print(f"❌ Error formatting DataFrame: {e}")
        return df


def activity_summary_extract(path):
    """Function to extract activity summaries from a JSON file and save to a CSV file"""
    try:
        print("📊 Processing activity summaries...")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Activity summary file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        if not data or 'summarizedActivitiesExport' not in data[0]:
            raise ValueError("Invalid activity summary file format")

        list_dict = data[0]['summarizedActivitiesExport']
        overall_df = pd.DataFrame(list_dict)

        if overall_df.empty:
            print("⚠️  No activity data found in file")
            return pd.DataFrame()

        print(f"📈 Found {len(overall_df)} activities to process")

        # Format different column types
        col_dt = ['beginTimestamp', 'startTimeLocal', 'startTimeGmt']
        col_ms = ['duration', 'elapsedDuration', 'movingDuration']
        col_speed = ['avgSpeed', 'maxSpeed', 'maxVerticalSpeed']
        col_distance = ['distance']

        df = formatting_garmin_df(overall_df, col_dt, col_ms, col_speed, col_distance)

        # Apply new timezone correction
        if 'startTimeGmt' in df.columns:
            print("🌍 Applying location-based timezone correction...")
            df = time_difference_correction(df, 'startTimeGmt', 'GMT')
            df.rename(columns={'startTimeGmt': 'startTimeGMTACT'}, inplace=True)

        df['Source'] = 'Garmin'

        # Merge with Polar data if it exists
        polar_file = 'files/source_processed_files/garmin/polar_summary_processed.csv'
        if os.path.exists(polar_file):
            try:
                df_polar = pd.read_csv(polar_file, sep='|')
                df = pd.concat([df_polar, df], ignore_index=True).reset_index(drop=True)
                print("✅ Merged with existing Polar data")
            except Exception as e:
                print(f"⚠️  Could not merge with Polar data: {e}")

        # Calculate calories in kcal and fix elevation
        if 'calories' in df.columns:
            df['Calories_kcal'] = df['calories'] / 4.18

        if 'elevationGain' in df.columns:
            df['elevationGain'] = df.apply(
                lambda x: x.elevationGain / 100 if x.Source == "Garmin" else x.elevationGain,
                axis=1
            )

        # Save processed file
        output_path = 'files/source_processed_files/garmin/garmin_activities_list_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        columns_to_drop = ['splits', 'splitSummaries', 'summarizedDiveInfo']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]

        df.drop(columns_to_drop, axis=1).to_csv(output_path, sep='|', index=False)

        print(f"✅ Activity summary processed: {len(df)} records saved")
        return df

    except Exception as e:
        print(f"❌ Error processing activity summary: {e}")
        return pd.DataFrame()


def activity_splits_extract(df):
    """Function to extract activity splits from the DataFrame and save to a CSV file"""
    try:
        print("🏃 Processing activity splits...")

        if df.empty:
            print("⚠️  No activity data provided for splits processing")
            return pd.DataFrame()

        activity_splits_full = pd.DataFrame()
        processed_activities = 0

        for _, row_activity in df.iterrows():
            if row_activity.get('activityType') != 'running':
                continue

            if 'splits' not in row_activity or not row_activity['splits']:
                continue

            try:
                activity_splits = pd.DataFrame(row_activity['splits'])

                if activity_splits.empty:
                    continue

                # Process timestamps
                activity_splits['startTimeGMT'] = pd.to_datetime(
                    activity_splits['startTimeGMT'], unit='ms', errors='coerce'
                )
                activity_splits['endTimeGMT'] = pd.to_datetime(
                    activity_splits['endTimeGMT'], unit='ms', errors='coerce'
                )
                activity_splits['startTimeGMTACT'] = row_activity.get('startTimeGMTACT')

                activity_splits.rename(columns={
                    'startTimeGMT': 'startTimeGMTSPLIT',
                    'endTimeGMT': 'endTimeGMTSPLIT'
                }, inplace=True)

                # Process measurements
                splits_measurements_full = pd.DataFrame()
                for _, row_split in activity_splits.iterrows():
                    if 'measurements' not in row_split or not row_split['measurements']:
                        continue

                    try:
                        split_measurements = pd.json_normalize(row_split['measurements'])

                        if split_measurements.empty:
                            continue

                        dict_measurements = {'startTimeGMTSPLIT': row_split['startTimeGMTSPLIT']}

                        for _, row_measurement in split_measurements.iterrows():
                            field = row_measurement.get('fieldEnum', '')
                            unit = row_measurement.get('unitEnum', '')
                            value = row_measurement.get('value')

                            if field and unit:
                                dict_measurements[f"{field}__{unit}"] = value

                        if len(dict_measurements) > 1:  # More than just the timestamp
                            measurement_df = pd.DataFrame([dict_measurements])
                            splits_measurements_full = pd.concat(
                                [splits_measurements_full, measurement_df],
                                ignore_index=True
                            )

                    except Exception as e:
                        print(f"⚠️  Error processing measurement: {e}")
                        continue

                if not splits_measurements_full.empty:
                    activity_splits = pd.merge(
                        activity_splits, splits_measurements_full,
                        on='startTimeGMTSPLIT', how='left'
                    )

                activity_splits_full = pd.concat(
                    [activity_splits_full, activity_splits],
                    ignore_index=True
                )
                processed_activities += 1

            except Exception as e:
                print(f"⚠️  Error processing activity splits: {e}")
                continue

        if activity_splits_full.empty:
            print("⚠️  No valid splits data found")
            return pd.DataFrame()

        print(f"📊 Processed splits from {processed_activities} running activities")

        # Format columns
        col_dt = ['startTimeGMTSPLIT', 'endTimeGMTSPLIT']
        col_duration, col_speed, col_distance = generate_to_format_columns(activity_splits_full.columns)

        df_formatted = formatting_garmin_df(
            activity_splits_full, col_dt, col_duration, col_speed, col_distance
        )
        df_formatted = rename_formatted_columns(df_formatted, col_duration, col_speed, col_distance)

        # Apply timezone correction
        print("🌍 Applying timezone correction to splits data...")
        if 'startTimeGMTSPLIT' in df_formatted.columns:
            df_formatted = time_difference_correction(df_formatted, 'startTimeGMTSPLIT', 'GMT')
            df_formatted.rename(columns={'startTimeGMTSPLIT': 'startTimeLocalSPLIT'}, inplace=True)

        if 'endTimeGMTSPLIT' in df_formatted.columns:
            df_formatted = time_difference_correction(df_formatted, 'endTimeGMTSPLIT', 'GMT')
            df_formatted.rename(columns={'endTimeGMTSPLIT': 'endTimeLocalSPLIT'}, inplace=True)

        if 'startTimeGMTACT' in df_formatted.columns:
            df_formatted = time_difference_correction(df_formatted, 'startTimeGMTACT', 'GMT')
            df_formatted.rename(columns={'startTimeGMTACT': 'startTimeLocalACT'}, inplace=True)

        # Filter and clean data
        df_filtered = df_formatted[df_formatted['type'] == 17].copy()

        columns_to_drop = [
            'measurements', 'startTimeSource', 'endTimeSource'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in df_filtered.columns]
        df_filtered = df_filtered.drop(columns_to_drop, axis=1).reset_index(drop=True)

        # Expand minutes
        print("⏱️  Expanding data by minutes...")
        new_df = pd.DataFrame()

        for _, row in df_filtered.iterrows():
            try:
                expanded_row = row_expander_minutes(row)
                if not expanded_row.empty:
                    new_df = pd.concat([new_df, expanded_row], ignore_index=True)
            except Exception as e:
                print(f"⚠️  Error expanding row: {e}")
                continue

        new_df['Source'] = 'Garmin'

        # Save processed file
        output_path = 'files/source_processed_files/garmin/garmin_activities_splits_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_df.to_csv(output_path, sep='|', index=False)

        print(f"✅ Activity splits processed: {len(new_df)} minute-level records saved")
        return new_df

    except Exception as e:
        print(f"❌ Error processing activity splits: {e}")
        return pd.DataFrame()


def sleep_extract(x, key):
    """Function to extract sleep data from a dictionary"""
    try:
        if isinstance(x, dict) and key in x:
            return x[key]
        return None
    except Exception:
        return None


def sleep_file(path):
    """Function to process sleep data from a JSON file"""
    try:
        if not os.path.exists(path):
            print(f"⚠️  Sleep file not found: {path}")
            return pd.DataFrame()

        df = pd.read_json(path)

        if df.empty:
            return df

        # Extract SPO2 data
        df['averageSPO2'] = df['spo2SleepSummary'].apply(lambda x: sleep_extract(x, 'averageSPO2'))
        df['averageHR'] = df['spo2SleepSummary'].apply(lambda x: sleep_extract(x, 'averageHR'))

        # Apply timezone correction
        if 'sleepStartTimestampGMT' in df.columns:
            df = time_difference_correction(df, 'sleepStartTimestampGMT', 'GMT')
            df.rename(columns={'sleepStartTimestampGMT': 'sleepStartTimestampLocal'}, inplace=True)

        if 'sleepEndTimestampGMT' in df.columns:
            df = time_difference_correction(df, 'sleepEndTimestampGMT', 'GMT')
            df.rename(columns={'sleepEndTimestampGMT': 'sleepEndTimestampLocal'}, inplace=True)

        return df

    except Exception as e:
        print(f"❌ Error processing sleep file {path}: {e}")
        return pd.DataFrame()


def process_sleep_files():
    """Process all sleep data files"""
    try:
        print("😴 Processing sleep data...")

        folder_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Wellness"

        if not os.path.exists(folder_path):
            print(f"❌ Sleep data folder not found: {folder_path}")
            return False

        file_list = os.listdir(folder_path)
        sleep_files = [f for f in file_list if f.endswith("sleepData.json")]

        if not sleep_files:
            print("⚠️  No sleep data files found")
            return False

        print(f"📁 Found {len(sleep_files)} sleep data files")

        df_sleep = pd.DataFrame()
        processed_files = 0

        for file in sleep_files:
            filepath = os.path.join(folder_path, file)
            file_df = sleep_file(filepath)

            if not file_df.empty:
                df_sleep = pd.concat([df_sleep, file_df], ignore_index=True)
                processed_files += 1

        if df_sleep.empty:
            print("⚠️  No valid sleep data found")
            return False

        # Save processed file
        output_path = 'files/source_processed_files/garmin/garmin_sleep_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_sleep.sort_values(by="calendarDate").to_csv(output_path, sep="|", index=False)

        print(f"✅ Sleep data processed: {processed_files} files, {len(df_sleep)} records saved")
        return True

    except Exception as e:
        print(f"❌ Error processing sleep files: {e}")
        return False


def extract_fit_files_path():
    """Function to extract paths of all .fit files in the specified folder and its subfolders"""
    try:
        fit_files = []
        base_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/FitFiles"

        if not os.path.exists(base_path):
            print(f"⚠️  FIT files folder not found: {base_path}")
            return []

        for root, _, filenames in os.walk(base_path):
            for filename in fnmatch.filter(filenames, '*.fit'):
                fit_files.append(os.path.join(root, filename))

        print(f"📁 Found {len(fit_files)} FIT files")
        return fit_files

    except Exception as e:
        print(f"❌ Error extracting FIT file paths: {e}")
        return []


def stress_level_qualification(x):
    """Function to qualify stress levels based on their values"""
    if pd.isna(x):
        return None

    stress_level = None
    for key, value in DICT_STRESS_LEVEL.items():
        if x >= key:
            stress_level = value
    return stress_level


def process_stress_level():
    """Function to process stress level data from .fit files and save to a CSV file"""
    try:
        print("😰 Processing stress level data...")

        # Extract ZIP file if it exists
        zip_file_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/UploadedFiles_0-_Part1.zip"
        if os.path.exists(zip_file_path):
            print("📦 Extracting FIT files from ZIP...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall("files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/FitFiles")
            os.remove(zip_file_path)

        fit_files = extract_fit_files_path()
        if not fit_files:
            print("⚠️  No FIT files found for stress level processing")
            return False

        # Load existing dictionary
        dict_files = load_fit_files_dictionary()

        data = []
        count_new_files = 0

        for fit_file in fit_files:
            if fit_file in dict_files:
                continue

            try:
                count_new_files += 1
                fitfile = fitparse.FitFile(fit_file)

                for record in fitfile.get_messages("stress_level"):
                    record_data = record.get_values()
                    if record_data:
                        data.append(record_data)

                dict_files[fit_file] = "Yes"

                if count_new_files % 50 == 0:
                    print(f"📊 Processed {count_new_files} new FIT files...")

            except Exception as e:
                print(f"⚠️  Error processing FIT file {fit_file}: {e}")
                continue

        # Save updated dictionary
        save_fit_files_dictionary(dict_files)

        print(f"📈 Processed {count_new_files} new FIT files for stress level data")

        # Load existing stress level data
        stress_file_path = 'files/source_processed_files/garmin/garmin_stress_level_processed.csv'

        try:
            df_stress_level = pd.read_csv(stress_file_path, sep='|')
        except FileNotFoundError:
            print("📝 Creating new stress level file")
            df_stress_level = pd.DataFrame()

        if data:
            new_df = pd.DataFrame(data)
            df_stress_level = pd.concat([df_stress_level, new_df], ignore_index=True).drop_duplicates()

            # Apply timezone correction
            if 'stress_level_time' in df_stress_level.columns:
                print("🌍 Applying timezone correction to stress data...")
                df_stress_level = time_difference_correction(df_stress_level, 'stress_level_time', 'GMT')

            # Qualify stress levels
            if 'stress_level_value' in df_stress_level.columns:
                df_stress_level['stress_level'] = df_stress_level['stress_level_value'].apply(stress_level_qualification)

            df_stress_level.sort_values('stress_level_time', inplace=True)

            # Save processed file
            os.makedirs(os.path.dirname(stress_file_path), exist_ok=True)
            df_stress_level.to_csv(stress_file_path, sep='|', index=False)

            print(f"✅ Stress level data processed: {len(df_stress_level)} records saved")
        else:
            print("ℹ️  No new stress level data found")

        return True

    except Exception as e:
        print(f"❌ Error processing stress level data: {e}")
        return False


def process_training_history():
    """Process training history data"""
    try:
        print("🏋️ Processing training history...")

        folder_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Metrics"

        if not os.path.exists(folder_path):
            print(f"❌ Training history folder not found: {folder_path}")
            return False

        # Find training history files
        pattern = r"^TrainingHistory_\d{8}_\d{8}_\d{9}\.json$"
        filenames = os.listdir(folder_path)
        files = [filename for filename in filenames if re.match(pattern, filename)]

        if not files:
            print("⚠️  No training history files found")
            return False

        print(f"📁 Found {len(files)} training history files")

        dataframes = []
        for file in files:
            filepath = os.path.join(folder_path, file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if data:
                        df = pd.DataFrame(data)
                        if not df.empty:
                            dataframes.append(df)
            except Exception as e:
                print(f"⚠️  Error processing file {file}: {e}")
                continue

        if not dataframes:
            print("⚠️  No valid training history data found")
            return False

        df = pd.concat(dataframes, ignore_index=True)

        # Apply timezone correction
        if 'timestamp' in df.columns:
            print("🌍 Applying timezone correction to training history...")
            df = time_difference_correction(df, 'timestamp', 'GMT')

        # Save processed file
        output_path = 'files/source_processed_files/garmin/garmin_training_history_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.sort_values('timestamp', ascending=False).to_csv(output_path, sep='|', index=False)

        print(f"✅ Training history processed: {len(df)} records saved")
        return True

    except Exception as e:
        print(f"❌ Error processing training history: {e}")
        return False


def download_garmin_data():
    """
    Opens Garmin Connect export page and prompts user to download data.
    Returns True if user confirms download, False otherwise.
    """
    print("⌚ Starting Garmin data download...")

    urls = ['https://connect.garmin.com/modern/settings/dataExport']
    open_web_urls(urls)

    print("📝 Instructions:")
    print("   1. Log into your Garmin Connect account")
    print("   2. Go to Account Settings > Data Export")
    print("   3. Request your data export")
    print("   4. Wait for the export email (can take several hours)")
    print("   5. Download and extract the ZIP file to Downloads folder")
    print("   6. The file should be named like 'DI_CONNECT_FITNESS_METRICS_[timestamp].zip'")

    response = prompt_user_download_status("Garmin")
    return response


def move_garmin_files():
    """
    Moves the downloaded Garmin files from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving Garmin files...")

    # First, try to unzip the garmin file
    unzip_success = find_unzip_folder("garmin")
    if not unzip_success:
        print("❌ Failed to find or unzip Garmin file")
        return False

    # Then move the unzipped folder
    move_success = clean_rename_move_folder(
        export_folder="files/exports",
        download_folder="/Users/valen/Downloads",
        folder_name="garmin_export_unzipped",
        new_folder_name="garmin_exports"
    )

    if move_success:
        print("✅ Successfully moved Garmin files to exports folder")
    else:
        print("❌ Failed to move Garmin files")

    return move_success


def create_garmin_files():
    """
    Main processing function that processes all Garmin data types.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Garmin data...")

    # Check if required files exist
    if not check_required_files():
        return False

    success_count = 0
    total_processes = 5

    try:
        # Process activity summaries
        activity_path = 'files/exports/garmin_exports/DI_CONNECT/DI-Connect-Fitness/valentin.herinckx@gmail.com_0_summarizedActivities.json'

        if os.path.exists(activity_path):
            df = activity_summary_extract(activity_path)
            if not df.empty:
                print("✅ Activity summaries processed successfully")
                success_count += 1

                # Process activity splits
                splits_result = activity_splits_extract(df)
                if not splits_result.empty:
                    print("✅ Activity splits processed successfully")
                    success_count += 1
                else:
                    print("⚠️  Activity splits processing had issues")
            else:
                print("❌ Failed to process activity summaries")
        else:
            print(f"❌ Activity summary file not found: {activity_path}")

        # Process training history
        if process_training_history():
            success_count += 1

        # Process stress level data
        if process_stress_level():
            success_count += 1

        # Process sleep data
        if process_sleep_files():
            success_count += 1

        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successful: {success_count}/{total_processes} processes")
        print(f"   {'🎉 All processes completed successfully!' if success_count == total_processes else '⚠️  Some processes had issues'}")

        return success_count > 0

    except Exception as e:
        print(f"❌ Error in main processing: {e}")
        return False


def process_garmin_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_garmin_pipeline(auto_full=True)
    else:
        return create_garmin_files()


def full_garmin_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Garmin SOURCE pipeline with 2 options.

    Options:
    1. Download new data and process
    2. Process existing data

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("⌚ GARMIN SOURCE PIPELINE")
    print("="*60)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new data and process")
        print("2. Process existing data")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Download new data and process...")

        # Step 1: Download
        download_success = download_garmin_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_garmin_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_garmin_files()

        # Step 3: Process (fallback if no new files)
        if move_success:
            process_success = create_garmin_files()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_garmin_files()

        success = process_success

    elif choice == "2":
        print("\n⚙️  Process existing data...")
        success = create_garmin_files()

    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Garmin source pipeline completed!")
        print("ℹ️  Note: To upload to Drive, run the Fitness topic pipeline.")
        # Record successful run
        from src.utils.utils_functions import record_successful_run
        record_successful_run('source_garmin', 'active')
    else:
        print("❌ Garmin pipeline failed")
    print("="*60)

    return success


def process_individual_data_type(data_type):
    """
    Process individual Garmin data types for debugging or selective processing.

    Args:
        data_type (str): Type of data to process
                        ('activities', 'splits', 'sleep', 'stress', 'training')

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n🔧 Processing individual data type: {data_type}")

    if not check_required_files():
        return False

    try:
        if data_type == 'activities':
            activity_path = 'files/exports/garmin_exports/DI_CONNECT/DI-Connect-Fitness/valentin.herinckx@gmail.com_0_summarizedActivities.json'
            if os.path.exists(activity_path):
                df = activity_summary_extract(activity_path)
                return not df.empty
            else:
                print(f"❌ File not found: {activity_path}")
                return False

        elif data_type == 'splits':
            # First need to load activities
            activity_path = 'files/exports/garmin_exports/DI_CONNECT/DI-Connect-Fitness/valentin.herinckx@gmail.com_0_summarizedActivities.json'
            if os.path.exists(activity_path):
                df = activity_summary_extract(activity_path)
                if not df.empty:
                    splits_df = activity_splits_extract(df)
                    return not splits_df.empty
            return False

        elif data_type == 'sleep':
            return process_sleep_files()

        elif data_type == 'stress':
            return process_stress_level()

        elif data_type == 'training':
            return process_training_history()

        else:
            print(f"❌ Unknown data type: {data_type}")
            print("Available types: activities, splits, sleep, stress, training")
            return False

    except Exception as e:
        print(f"❌ Error processing {data_type}: {e}")
        return False


if __name__ == "__main__":
    # Allow running this file directly
    print("⌚ Garmin Processing Tool")
    print("This tool helps you download, process, and upload Garmin data.")

    # Run the pipeline
    full_garmin_pipeline(auto_full=False)
