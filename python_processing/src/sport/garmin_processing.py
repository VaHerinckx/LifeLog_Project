import pandas as pd
import os
import json
from src.utils.utils_functions import time_difference_correction, find_unzip_folder, clean_rename_move_folder
import os
import fnmatch
import fitparse
import re
import zipfile
from src.utils.drive_storage import update_drive

dict_stress_level = {0: 'Rest', 26: 'Low', 51: 'Medium', 76: 'High'}
dict_col = {
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

dict_path = 'files/work_files/garmin_work_files/garmin_zip_dictionnary.json'
with open(dict_path, 'r') as f:
        dict_files = json.load(f)

def row_expander_minutes(row):
    """Function to expand rows based on minute differences and return a DataFrame"""
    minute_diff = (row['endTimeLocalSPLIT'] - row['startTimeLocalSPLIT']).total_seconds()/60
    if minute_diff <=1:
        date_df = pd.DataFrame(columns=list(dict_col.keys()))
        date_df['date'] = row['startTimeLocalSPLIT']
        for col in dict_col.keys():
            date_df[col] = row[col]
        return date_df
    dates = pd.date_range(row['startTimeLocalSPLIT'], row['endTimeLocalSPLIT']- pd.Timedelta(minutes=1), freq='T')
    date_df = pd.DataFrame({'date': dates})
    for col in dict_col.keys():
        date_df[col] = row[col]/minute_diff if dict_col[col] == 'div' else row[col]
    return date_df

def generate_to_format_columns(columns):
    """Function to generate lists of columns based on their units"""
    col_duration = []
    col_speed = []
    col_distance = []
    for column in columns:
        if len(column.split('__'))!=2:
            continue
        unit = column.split('__')[1]
        if unit == 'MILLISECOND':
            col_duration.append(column)
        if unit == 'CENTIMETERS_PER_MILLISECOND':
            col_speed.append(column)
        if unit == 'CENTIMETER':
            col_distance.append(column)
    return col_duration, col_speed, col_distance

def rename_formatted_columns(df, col_duration, col_speed, col_distance):
    """Function to rename columns based on their units"""
    for col in col_duration:
        df.rename(columns={col:f"{col.split('__')[0]}__M"}, inplace = True)
    for col in col_speed:
        df.rename(columns={col:f"{col.split('__')[0]}__KMH"}, inplace = True)
    for col in col_distance:
        df.rename(columns={col:f"{col.split('__')[0]}__M"}, inplace = True)
    return df

def formatting_garmin_df(df, columns_datetime=None, columns_duration=None, columns_speed=None, columns_distance=None):
    """Function to format specific columns in the DataFrame"""
    if columns_datetime:
        for col in columns_datetime:
            df[col] = pd.to_datetime(df[col], unit = 'ms')
    if columns_duration:
        for col in columns_duration:
            df[col] = df[col]/1000/60
    if columns_speed:
        for col in columns_speed:
            df[col] = df[col]*36
    if columns_distance:
        for col in columns_distance:
            df[col] = df[col]/100
    return df

def activity_summary_extract(path):
    """Function to extract activity summaries from a JSON file and save to a CSV file"""
    with open(path,'r') as f:
        data = json.load(f)
    list_dict = data[0]['summarizedActivitiesExport']
    overall_df = pd.DataFrame(list_dict)
    col_dt = ['beginTimestamp', 'startTimeLocal', 'startTimeGmt'] #the columns to be switched to datetime format
    col_ms = ['duration', 'elapsedDuration', 'movingDuration']
    col_speed = ['avgSpeed', 'maxSpeed', 'maxVerticalSpeed']
    col_distance = ['distance']
    df = formatting_garmin_df(overall_df, col_dt, col_ms, col_speed, col_distance)
    df['TimeDiffGMT'] = df['startTimeLocal'] - df['startTimeGmt']
    df.rename(columns = {'startTimeGmt' : 'startTimeGMTACT'}, inplace = True)
    df['Source'] = 'Garmin'
    df_polar = pd.read_csv('files/processed_files/polar_summary_processed.csv', sep = '|')
    df = pd.concat([df_polar, df], ignore_index=False).reset_index(drop=True)
    df['Calories_kcal'] = df['calories']/4.18
    df['elevationGain'] = df.apply(lambda x: x.elevationGain / 100 if x.Source == "Garmin" else x.elevationGain, axis = 1)
    df.drop(['splits', 'splitSummaries', 'summarizedDiveInfo'], axis =1)\
      .to_csv('files/processed_files/garmin_activities_list_processed.csv', sep = '|', index = False)
    return df

def activity_splits_extract(df):
    """Function to extract activity splits from the DataFrame and save to a CSV file"""
    activity_splits_full = pd.DataFrame()
    for _, row_activity in df.iterrows():
        if row_activity['activityType'] != 'running':
            continue
        activity_splits = pd.DataFrame(row_activity['splits'])
        activity_splits['startTimeGMT'] = pd.to_datetime(activity_splits['startTimeGMT'], unit = 'ms')
        activity_splits['endTimeGMT'] = pd.to_datetime(activity_splits['endTimeGMT'], unit = 'ms')
        activity_splits['startTimeGMTACT'] = row_activity['startTimeGMTACT']
        activity_splits.rename(columns = {'startTimeGMT': 'startTimeGMTSPLIT'}, inplace = True)
        activity_splits.rename(columns = {'endTimeGMT': 'endTimeGMTSPLIT'}, inplace = True)
        splits_measurements_full = pd.DataFrame()
        for _, row_split in activity_splits.iterrows():
            dict_measurements = {}
            split_measurements = pd.json_normalize(row_split['measurements'])
            for _, row_measurement in split_measurements.iterrows():
                dict_measurements['startTimeGMTSPLIT'] = row_split['startTimeGMTSPLIT']
                dict_measurements[f"{row_measurement['fieldEnum']}__{row_measurement['unitEnum']}"] = row_measurement['value']
            split_measurements = pd.DataFrame(dict_measurements, index = [0])
            splits_measurements_full = pd.concat([splits_measurements_full, split_measurements], ignore_index=True)
        activity_splits = pd.merge(activity_splits, splits_measurements_full, on = 'startTimeGMTSPLIT')
        activity_splits_full = pd.concat([activity_splits_full, activity_splits], ignore_index=True)
    col_dt = ['startTimeGMTSPLIT', 'endTimeGMTSPLIT']
    col_duration, col_speed, col_distance = generate_to_format_columns(activity_splits_full.columns)
    df = formatting_garmin_df(activity_splits_full, col_dt, col_duration, col_speed, col_distance)
    df = rename_formatted_columns(df, col_duration, col_speed, col_distance)
    df['startTimeLocalSPLIT'] = pd.to_datetime(df['startTimeGMTSPLIT'], utc=True).apply(lambda x: time_difference_correction(x, 'GMT+1'))
    df['endTimeLocalSPLIT'] = pd.to_datetime(df['endTimeGMTSPLIT'], utc=True).apply(lambda x: time_difference_correction(x, 'GMT+1'))
    df['startTimeLocalACT'] = pd.to_datetime(df['startTimeGMTACT'], utc=True).apply(lambda x: time_difference_correction(x, 'GMT+1'))
    df = df[df['type']==17].drop(['measurements', 'startTimeGMTACT', 'startTimeGMTSPLIT', 'startTimeSource', 'endTimeSource',\
                                  'endTimeGMTSPLIT'], axis = 1).reset_index(drop=True)
    new_df = pd.DataFrame(columns=list(dict_col.keys()))
    #old_df = pd.read_csv(path, sep = '|').rename(columns = {name_val : 'val'})
    #old_df['date'] = pd.to_datetime(old_df['date'])
    #new_df = pd.concat([new_df, old_df], ignore_index=True)
    for _, row in df.iterrows():
        new_df = pd.concat([new_df, row_expander_minutes(row)], ignore_index=True)
    new_df['Source'] = 'Garmin'
    new_df.to_csv('files/processed_files/garmin_activities_splits_processed.csv', sep = '|', index = False)
    return new_df

def sleep_extract(x, key):
    """Function to extract sleep data from a JSON file"""
    if isinstance(x, dict):
        if key in x.keys():
            return x[key]
        else:
            return None

def sleep_file(path):
    """Function to process sleep data from a JSON file and save to a CSV file"""
    df = pd.read_json(path)
    df['averageSPO2'] = df['spo2SleepSummary'].apply(lambda x : sleep_extract(x, 'averageSPO2'))
    df['averageHR'] = df['spo2SleepSummary'].apply(lambda x : sleep_extract(x, 'averageHR'))
    df['sleepStartTimestampLocal'] = pd.to_datetime(df['sleepStartTimestampGMT'], utc=True).apply(lambda x: time_difference_correction(x, 'GMT+1'))
    df['sleepEndTimestampLocal'] = pd.to_datetime(df['sleepEndTimestampGMT'], utc=True).apply(lambda x: time_difference_correction(x, 'GMT+1'))
    return df

def process_sleep_files():
    folder_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Wellness"
    file_list = os.listdir(folder_path)
    df_sleep = pd.DataFrame()
    for file in file_list:
        if file.endswith("sleepData.json"):
            filepath = os.path.join(folder_path, file)
            df_sleep = pd.concat([df_sleep, sleep_file(filepath)], ignore_index=True)
    df_sleep.sort_values(by="calendarDate").to_csv('files/processed_files/garmin_sleep_processed.csv', sep = "|")

#def process_rename_sleep_file():
#    folder_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Wellness"
#    filenames = []
#    file_dates = []
#    file_list = os.listdir(folder_path)
#    for file in file_list:
#        if file.endswith("sleepData.json"):
#            filenames.append(file)
#    for filename in filenames:
#        date_str = filename.split("_")[1]
#        date = int(date_str.replace("-", ""))
#        file_dates.append((date, filename))
#    file_dates_sorted = sorted(file_dates, reverse=True)
#    if file_dates_sorted:
#        biggest_date_filename = file_dates_sorted[0][1]
#        biggest_date_filepath = os.path.join(folder_path, biggest_date_filename)
#    df = sleep_file(biggest_date_filepath)
#    df.to_csv('files/processed_files/garmin_sleep_processed.csv', sep = "|")

def extract_fit_files_path():
    """Function to extract paths of all .fit files in the specified folder and its subfolders"""
    fit_files = []
    for root, _, filenames in os.walk("files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/FitFiles"):
        for filename in fnmatch.filter(filenames, '*.fit'):
            fit_files.append(os.path.join(root, filename))
    return fit_files

def stress_level_qualification(x):
    """Function to qualify stress levels based on their values"""
    stress_level = None
    for key, value in dict_stress_level.items():
        if x >= key:
            stress_level = value
    return stress_level

def process_stress_level():
    """Function to process stress level data from .fit files and save to a CSV file"""
    zip_file_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/UploadedFiles_0-_Part1.zip"
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall("files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/FitFiles")
        os.remove(zip_file_path)
    fit_files = extract_fit_files_path()
    data = []
    count_new_files = 0
    for fit_file in fit_files:
        if fit_file in dict_files.keys():
            continue
        count_new_files +=1
        fitfile = fitparse.FitFile(fit_file)
        for record in fitfile.get_messages("stress_level"):
            data.append(record.get_values())
        dict_files[fit_file] = "Yes"
    with open(dict_path, 'w') as f:
        json.dump(dict_files, f)
    print(f"{count_new_files} new fit files to process for stress level data")
    df_stress_level = pd.read_csv('files/processed_files/garmin_stress_level_processed.csv', sep = '|')
    if len(data) > 0:
        df = pd.DataFrame(data)
        df_stress_level = pd.concat([df_stress_level,df], ignore_index=True).reset_index(drop = True).drop_duplicates()
        df_stress_level['stress_level_time'] = pd.to_datetime(df_stress_level['stress_level_time'], utc=True).apply(lambda x: time_difference_correction(x, 'GMT'))
        df_stress_level['stress_level'] = df_stress_level['stress_level_value'].apply(lambda x: stress_level_qualification(x))
        df_stress_level.sort_values('stress_level_time', inplace = True)
        df_stress_level.to_csv('files/processed_files/garmin_stress_level_processed.csv', sep = '|', index = False)

def process_training_history():
    folder_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Metrics"
    #Finding the .json file with the latest end date in the folder
    pattern = r"^TrainingHistory_\d{8}_\d{8}_\d{9}\.json$"
    filenames = os.listdir(folder_path)
    files = []
    for filename in filenames:
        if re.match(pattern, filename):
            files.append(filename)
    dataframes = []
    for file in files:
        filepath = f"{folder_path}/{file}"
        with open(filepath,'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            dataframes.append(df)
    df = pd.concat(dataframes, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).apply(lambda x: time_difference_correction(x, 'GMT'))
    df.sort_values('timestamp', ascending = False).to_csv('files/processed_files/garmin_training_history_processed.csv', sep = '|', index = False)

def create_garmin_files():
    df = activity_summary_extract('files/exports/garmin_exports/DI_CONNECT/DI-Connect-Fitness/valentin.herinckx@gmail.com_0_summarizedActivities.json')
    print("garmin_activies_list_processed.csv was generated \n")
    activity_splits_extract(df)
    print("garmin_activities_splits_processed.csv was generated \n")
    process_training_history()
    print("garmin_training_history_processed.csv was generated \n")
    process_stress_level()
    print("garmin_stress_level_processed.csv was generated \n")
    process_sleep_files()
    print("garmin_sleep_processed.csv was generated \n")


def process_garmin_export(upload="Y"):
    file_names = []
    print('Starting the processing of the Garmin export \n')
    find_unzip_folder("garmin")
    clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "garmin_export_unzipped", "garmin_exports")
    create_garmin_files()
    file_names.append('files/processed_files/garmin_activities_list_processed.csv')
    file_names.append('files/processed_files/garmin_activities_splits_processed.csv')
    file_names.append('files/processed_files/garmin_sleep_processed.csv')
    file_names.append('files/processed_files/garmin_stress_level_processed.csv')
    file_names.append('files/processed_files/garmin_training_history_processed.csv')
    if upload == "Y":
        update_drive(file_names)
        print('Garmin processed files were created and uploaded to the Drive \n')
    else:
        print('Garmin processed files were created \n')
