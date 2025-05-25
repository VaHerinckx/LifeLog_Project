import pandas as pd
from utils import clean_rename_move_file
from drive_storage import update_drive
from datetime import date

def add_sorting_columns(df):
    """Adds some columns used for sorting in the PBI report"""
    df['Year_Week'] = df['Period'].apply(lambda x: str(x.year) + ' - ' + str(str(x.week)))
    df['Year_Month'] = df['Period'].apply(lambda x: str(x.year) + ' - ' + str(str(x.month)))
    df['sorting_week'] = df['Period'].dt.year * 100 + df['Period'].dt.isocalendar().week
    df['sorting_month'] = df['Period'].dt.year * 100 + df['Period'].dt.month
    df['sorting_day'] = df['Period'].dt.year * 100 + df['Period'].dt.isocalendar().day
    return df

def create_moneymgr_file():
    df = pd.read_excel(f"files/exports/moneymgr_exports/moneymgr_export.xlsx")
    df.sort_values(by = "Period", inplace = True)
    df.drop('Accounts.1', axis = 1, inplace=True)
    df = add_sorting_columns(df)
    df.to_csv('files/processed_files/moneymgr_processed.csv', sep = '|', index=False)


def process_moneymgr_export(upload="Y"):
    file_names = []
    print('Starting the processing of the Money Mgr export \n')
    clean_rename_move_file("files/exports/moneymgr_exports", "/Users/valen/Downloads", f"{date.today().strftime('%Y-%m-%d')}.xlsx", "moneymgr_export.xlsx")
    create_moneymgr_file()
    file_names.append('files/processed_files/moneymgr_processed.csv')
    if upload == "Y":
        update_drive(file_names)
        print('Money Mgr processed files were created and uploaded to the Drive \n')
    else:
        print('Money Mgr processed files were created \n')
