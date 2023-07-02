import os
import zipfile
import pandas as pd

def calculate_incremental_distance(column):
    column = column.diff()
    column.fillna(0, inplace=True)
    column = column.clip(lower=0)
    return column


def garmin_processing_summary(df):
    #Adding new columns
    df['startTimeLocal'] = pd.to_datetime(df['Date'] + ' ' + df['Start time'])
    #Adapting units
    df['Calories'] = df['Calories'] * 4.182
    #Removing columns
    df.drop(columns = ['Name', 'Average pace (min/km)', 'Max pace (min/km)'\
                     , 'Fat percentage of calories(%)', 'Average cadence (rpm)'\
                     , 'Running index', 'Training load', 'Average power (W)'\
                     , 'Max power (W)', 'Notes', 'Height (cm)', 'Weight (kg)'\
                     , 'Unnamed: 27', 'Date', 'Start time', 'HR sit'], inplace = True)
    #Renaming columns
    df.rename(columns = {'Sport' : 'sportType',
                         'Duration' : 'duration',
                         'Total distance (km)' : 'distance',
                         'Average heart rate (bpm)' : 'avgHr',
                         'Average speed (km/h)' : 'avgSpeed',
                         'Max speed (km/h)' : 'maxSpeed',
                         'Calories' : 'calories',
                         'Average stride length (cm)' : 'avgStrideLength',
                         'Ascent (m)' : 'elevationGain',
                         'Descent (m)' : 'elevationLoss',
                         'HR max' : 'maxHr',
                         'VO2max' : 'vO2MaxValue'}, inplace = True)
    df['Source'] = 'Polar'
    df.sort_values('startTimeLocal').to_csv('processed_files/polar_summary_processed.csv', sep = '|', index = False)

def garmin_processing_splits(df):
    #Adding new columns
    df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values(['Date', 'Time'])
    df['SUM_DISTANCE__M'] = calculate_incremental_distance(df['Distances (m)'])
    df['GAIN_ELEVATION__M'] = calculate_incremental_distance(df['Altitude (m)'])
    #Removing columns
    df.drop(columns = ['Sample rate', 'Date', 'Time', 'Pace (min/km)', 'Altitude (m)', 'Temperatures (C)'\
                      ,'Power (W)', 'Distances (m)', 'Unnamed: 11'], inplace = True)
    #Renaming columns
    df.rename(columns = {'HR (bpm)' : 'WEIGHTED_MEAN_HEARTRATE__BPM',
                         'Speed (km/h)' : 'WEIGHTED_MEAN_SPEED__KMH',
                         'Cadence' : 'WEIGHTED_MEAN_RUNCADENCE__BPM',
                         'Stride length (m)' : 'WEIGHTED_MEAN_STRIDE_LENGTH__M',
                         }, inplace = True)
    df['Source'] = 'Polar'
    df.sort_values('date').to_csv('processed_files/polar_details_processed.csv', sep = '|', index = False)


def polar_processing(folder_path = 'exports/polar_exports'):
    # create an empty list to store the dataframes
    summary_dfs = []
    detail_dfs = []
    # iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.zip'):
            file_path = os.path.join(folder_path, filename)
            # create a ZipFile object
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # iterate through each file in the zip archive
                for zip_info in zip_file.infolist():
                    # check if the file is an Excel file
                    if zip_info.filename.endswith('.csv'):
                        # read the Excel file into a pandas DataFrame
                        with zip_file.open(zip_info.filename) as excel_file:
                            df = pd.read_csv(excel_file, nrows=1)
                            Sport = df['Sport'][0]
                            Date = df['Date'][0]
                            summary_dfs.append(df)
                        with zip_file.open(zip_info.filename) as excel_file:
                            detail_df = pd.read_csv(excel_file, skiprows=2)
                            detail_df['Sport'] = Sport
                            detail_df['Date'] = Date
                            detail_dfs.append(detail_df)
    summary_df = pd.concat(summary_dfs, ignore_index=True)
    detail_df = pd.concat(detail_dfs, ignore_index=True)
    garmin_processing_summary(summary_df)
    garmin_processing_splits(detail_df)
