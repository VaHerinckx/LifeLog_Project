import pandas as pd
from src.utils.utils_functions import clean_rename_move_file
from src.utils.drive_storage import update_drive
import re
import os

def row_expander_seconds(row):
    """Expands the dataframe to have one row per minute, to remove the aggregation in the original df"""
    lower_bound = (row['start']-pd.Timedelta(seconds = 30)).round('T')
    upper_bound = (row['end']-pd.Timedelta(seconds = 30)).round('T')
    rows_needed = (int((upper_bound-lower_bound).total_seconds()/60))+1
    if rows_needed ==1:
        date_df = pd.DataFrame(columns=['date', 'screen_time'])
        new_row = {'date': lower_bound, 'screen_time' : row["duration_s"], 'pickups' : 1}
        date_df = date_df.append(new_row, ignore_index=True)
        return date_df
    dates = pd.date_range(lower_bound, upper_bound, freq='T')
    date_df = pd.DataFrame({'date': dates})
    for minute in range(1, rows_needed+1, 1):
        if minute == 1:
            seconds = (lower_bound + pd.Timedelta(minutes = 1) - row["start"]).total_seconds()
            pickup = 1
        elif minute != rows_needed:
            seconds = 60
            pickup = 0
        else:
            seconds = (row["end"] - upper_bound).total_seconds()
            pickup = 0
        date_df.loc[minute-1,'screen_time'] = seconds
        date_df.loc[minute-1,'pickups'] = pickup
    return date_df

def expand_df(df):
    """Expands the dataframe to have one row per minute, to remove the aggregation in the original df"""
    df['start'] = pd.to_datetime(df['start']) + pd.Timedelta(hours = 8)
    df['end'] = pd.to_datetime(df['end']) + pd.Timedelta(hours = 8)
    df['duration_s'] = (df.end - df.start).dt.total_seconds().astype(float)
    new_df = pd.DataFrame(columns=['date', 'screen_time', 'pickups'])
    old_df = pd.read_csv('files/processed_files/offscreen_processed.csv', sep = '|')
    #old_df['date'] = pd.to_datetime(old_df['date'])
    new_df = pd.concat([new_df, old_df], ignore_index=True)
    df = df[(df['start']-pd.Timedelta(seconds = 30)).round('T')>max(new_df["date"])].reset_index(drop=True)
    print(f'{df.shape[0]} new rows to expand')
    for _, row in df.iterrows():
        new_df = pd.concat([new_df, row_expander_seconds(row)], ignore_index=True)
    #new_df['date'] = pd.to_datetime(new_df['date'])
    new_df = new_df.groupby('date').sum().reset_index()
    return new_df

def screentime_before_sleep(df):
    """Computes the screentime in the hour before sleep"""
    df2 = pd.read_csv('files/processed_files/garmin_sleep_processed.csv', sep = '|')
    df['date'] = pd.to_datetime(df.date, utc = True).dt.tz_localize(None)
    df['sleepDate'] = df.date.apply(lambda x: x - pd.Timedelta(days = 1) if x.hour < 6 else x).dt.date
    df['sleepDate'] = pd.to_datetime(df.sleepDate, utc = True).dt.tz_localize(None)
    df2['sleepDate'] = pd.to_datetime(df2.calendarDate, utc = True).dt.tz_localize(None) - pd.Timedelta(days = 1)
    merged_df = pd.merge(df, df2[['sleepDate', "sleepStartTimestampLocal"]], on='sleepDate', how='left')
    merged_df['time_diff'] = pd.to_datetime(merged_df['sleepStartTimestampLocal']).dt.tz_localize(None) - merged_df['date']
    merged_df['within_hour_before_sleep'] = merged_df['time_diff'].apply(lambda x: 1 if x <= pd.Timedelta(hours=1) else 0)
    return merged_df.drop(["sleepDate", "time_diff"], axis = 1)

def create_offscreen_file():
    df = pd.read_csv("files/exports/offscreen_exports/Pickup.csv")
    df = expand_df(df)
    df = screentime_before_sleep(df)
    df.sort_values("date", ascending = False).to_csv('files/processed_files/offscreen_processed.csv', sep = '|', index = False)

def process_offscreen_export(upload="Y"):
    file_names = []
    print('Starting the processing of the Offscreen export \n')
    csv_regex = r'\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}.*\.csv$'
    count_file = 0
    for f in os.listdir('/Users/valen/Downloads'):
        if re.match(csv_regex, f):
            count_file += 1
            clean_rename_move_file("files/exports/offscreen_exports", "/Users/valen/Downloads", f, f.split('000-')[1], count_file)
    print("No new offscreen export to process") if count_file == 0 else None
    create_offscreen_file()
    file_names.append('files/processed_files/offscreen_processed.csv')
    if upload == "Y":
        update_drive(file_names)
        print('Offscreen processed files were created and uploaded to the Drive \n')
    else:
        print('Offscreen processed files were created \n')
