import pandas as pd
import os
import xmltodict
import subprocess
from utils import clean_rename_move_folder, find_unzip_folder
from drive_storage import update_drive

# Disable pandas warning about chained assignment
pd.options.mode.chained_assignment = None  # default='warn'

# Dictionary mapping column names to their respective identifiers in the export data
dict_identifier = {
    'step_count': 'HKQuantityTypeIdentifierStepCount',
    'step_length': 'HKQuantityTypeIdentifierWalkingStepLength',
    'walking_dist': 'HKQuantityTypeIdentifierDistanceWalkingRunning',
    'flights_climbed': 'HKQuantityTypeIdentifierFlightsClimbed',
    'walking_speed': 'HKQuantityTypeIdentifierWalkingSpeed',
    'heart_rate': 'HKQuantityTypeIdentifierHeartRate',
    'audio_exposure': 'HKQuantityTypeIdentifierHeadphoneAudioExposure',
    'resting_energy': 'HKQuantityTypeIdentifierBasalEnergyBurned',
    'active_energy': 'HKQuantityTypeIdentifierActiveEnergyBurned',
    'body_weight' : 'HKQuantityTypeIdentifierBodyMass',
    'sleep_analysis' : 'HKCategoryTypeIdentifierSleepAnalysis',
    'body_fat_%' : 'HKQuantityTypeIdentifierBodyFatPercentage'
}

#Dictionary to change the values for the sleep categorization
dict_sleep_analysis = {
    "HKCategoryValueSleepAnalysisAsleepUnspecified" : "Unspecified",
    "HKCategoryValueSleepAnalysisInBed" : "In bed",
    "HKCategoryValueSleepAnalysisAsleepDeep" : "Deep sleep",
    "HKCategoryValueSleepAnalysisAsleepREM" : "REM sleep",
    "HKCategoryValueSleepAnalysisAsleepCore" : "Core sleep",
    "HKCategoryValueSleepAnalysisAwake" : "Awake"
}

def clean_import_file():
    """Function to clean the import file by removing certain lines that are creating bugs"""
    path = 'files/exports/apple_exports/apple_health_export/export.xml'
    new_path = 'files/exports/apple_exports/apple_health_export/cleaned_export.xml'
    command = f"sed -e '156,211d' {path} > {new_path}"
    subprocess.run(command, shell=True)

def apple_df_formatting(path):
    """Function to format the Apple export XML file into a DataFrame"""
    with open(path, 'r') as xml_file:
        input_data = xmltodict.parse(xml_file.read())
    records_list = input_data['HealthData']['Record']
    df = pd.DataFrame(records_list)
    df.to_csv('files/exports/apple_exports/apple_health_export/cleaned_export.csv', sep='|', index=False)
    return df

def select_columns(df, name_val, data_type):
    """Function to select columns from the DataFrame based on the column name value"""
    path = f'files/processed_files/apple_{name_val}.csv'
    df = df[df['@type'] == dict_identifier[name_val]].reset_index(drop=True)
    df["@value"] = df["@value"].astype(data_type)
    df.rename(columns={'@startDate': 'date', '@sourceName': 'source', '@value': name_val}, inplace=True)
    if data_type == float:
        df = df[['date', name_val]].groupby('date').mean().reset_index()
    elif name_val == "sleep_analysis":
        df = df[['date', name_val]].groupby('date').max().reset_index()
        df["sleep_analysis"] = df["sleep_analysis"].map(dict_sleep_analysis)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(path, sep='|', index=False)
    return df

def expand_df(df, name_val, aggreg_method='sum'):
    """Function to expand the DataFrame by adding rows for each minute within the given time range"""
    path = f'files/processed_files/apple_processed_files/apple_{name_val}.csv'
    df = df[df['@type'] == dict_identifier[name_val]].reset_index(drop=True)
    df["@value"] = df["@value"].astype(float)
    new_df = pd.DataFrame(columns=['date', 'val', 'source'])
    old_df = pd.read_csv(path, sep='|').rename(columns={name_val: 'val'})
    old_df['date'] = pd.to_datetime(old_df['date'])
    new_df = pd.concat([new_df, old_df], ignore_index=True)
    df = df[df["@startDate"] > max(old_df["date"])].reset_index(drop=True)
    print(f'{df.shape[0]} new rows to expand for {name_val}')
    for _, row in df.iterrows():
        new_df = pd.concat([new_df, row_expander_minutes(row, aggreg_method)], ignore_index=True)
    print(f'{df.shape[0]} rows expanded for {name_val} \n')
    new_df = new_df[['date', 'val']].groupby('date').mean().rename(columns={'val': name_val}).reset_index()
    new_df['date'] = pd.to_datetime(new_df['date'], utc=True)
    new_df.drop_duplicates(inplace=True)
    new_df.to_csv(path, sep='|', index=False)
    return new_df

def row_expander_minutes(row, aggreg_method):
    """Function to expand a single row into multiple rows, each representing a minute"""
    minute_diff = (row['@endDate'] - row['@startDate']).total_seconds() / 60
    if minute_diff <= 1:
        date_df = pd.DataFrame(columns=['date', 'val', 'source'])
        new_row = {'date': row['@startDate'], 'val': row["@value"], 'source': row['@sourceName']}
        date_df = date_df.append(new_row, ignore_index=True)
        return date_df
    dates = pd.date_range(row['@startDate'], row['@endDate'] - pd.Timedelta(minutes=1), freq='T')
    date_df = pd.DataFrame({'date': dates})
    if aggreg_method == 'sum':
        date_df['val'] = row["@value"] / minute_diff
    elif aggreg_method == 'avg':
        date_df['val'] = row["@value"]
    date_df['source'] = row['@sourceName']
    return date_df

def create_apple_files():
    """Function to process the Apple export and generate the final processed DataFrame"""
    path_cleaned_xml = 'files/exports/apple_exports/apple_health_export/cleaned_export.xml'
    path_csv_export = 'files/exports/apple_exports/apple_health_export/cleaned_export.csv'
    if not os.path.isfile(path_csv_export):
        if not os.path.isfile(path_cleaned_xml):
            clean_import_file()
        apple_df_formatting(path_cleaned_xml)
    df = pd.read_csv(path_csv_export, sep='|', low_memory=False)
    df['@startDate'] = pd.to_datetime(df['@startDate']).dt.floor("T")
    df['@endDate'] = pd.to_datetime(df['@endDate']).dt.floor("T")
    df_step_count = expand_df(df, 'step_count', 'sum')
    df_step_length = expand_df(df, 'step_length', 'sum')
    df_walking_dist = expand_df(df, 'walking_dist', 'sum')
    df_flights_climbed = expand_df(df, 'flights_climbed', 'sum')
    df_resting_energy = expand_df(df, 'resting_energy', 'sum')
    df_active_energy = expand_df(df, 'active_energy', 'sum')
    df_walking_speed = expand_df(df, 'walking_speed', 'avg')
    df_audio_exposure = expand_df(df, 'audio_exposure', 'avg')
    df_heart_rate = select_columns(df, 'heart_rate', float)
    df_body_weight = select_columns(df, 'body_weight', float)
    df_body_fat_perc =  select_columns(df, 'body_fat_%', float)
    df_sleep_analysis = select_columns(df, 'sleep_analysis', str)
    apple_df = df_step_count.merge(df_step_length, how='outer', on='date') \
        .merge(df_walking_dist, how='outer', on='date') \
        .merge(df_flights_climbed, how='outer', on='date') \
        .merge(df_resting_energy, how='outer', on='date') \
        .merge(df_active_energy, how='outer', on='date') \
        .merge(df_walking_speed, how='outer', on='date') \
        .merge(df_heart_rate, how='outer', on='date') \
        .merge(df_body_weight, how='outer', on='date') \
        .merge(df_body_fat_perc, how='outer', on='date') \
        .merge(df_audio_exposure, how='outer', on='date') \
        .merge(df_sleep_analysis, how='outer', on='date')
    for col in list(apple_df.columns[1:-2]):
        apple_df[col] = apple_df[col].astype(float)
    apple_df.sort_values('date', inplace=True)
    apple_df.to_csv('files/processed_files/apple_processed.csv', sep='|', index=False)

def process_apple_export(upload="Y"):
    file_names = []
    print('Starting the processing of the Apple export \n')
    find_unzip_folder("apple")
    clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "apple_export_unzipped", "apple_exports")
    create_apple_files()
    file_names.append("files/processed_files/apple_processed.csv")
    if upload == "Y":
        update_drive(file_names)
        print('Apple processed files were created and uploaded to the Drive \n')
    else:
        print('Apple processed files were created \n')
