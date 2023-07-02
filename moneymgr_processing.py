import pandas as pd
from utils import today_export
from drive_storage import update_file


def add_sorting_columns(df):
    df['Year_Week'] = df['Period'].apply(lambda x: str(x.year) + ' - ' + str(str(x.week)))
    df['Year_Month'] = df['Period'].apply(lambda x: str(x.year) + ' - ' + str(str(x.month)))
    df['sorting_week'] = df['Period'].dt.year * 100 + df['Period'].dt.isocalendar().week
    df['sorting_month'] = df['Period'].dt.year * 100 + df['Period'].dt.month
    df['sorting_day'] = df['Period'].dt.year * 100 + df['Period'].dt.isocalendar().day
    return df

def process_moneymgr_export():
    df = pd.read_excel(f"exports/moneymgr_exports/moneymgr_export.xlsx")
    df.sort_values(by = "Period", inplace = True)
    df.drop('Accounts.1', axis = 1, inplace=True)
    df = add_sorting_columns(df)
    df.to_csv('processed_files/moneymgr_processed.csv', sep = '|', index=False)
