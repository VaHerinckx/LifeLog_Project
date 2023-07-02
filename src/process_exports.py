import subprocess
import os
import re
import shutil
import zipfile
from datetime import date
from lfm_processing import process_lfm_export
from gr_processing import process_gr_export, merge_gr_kindle
from pocket_casts_processing import process_pocket_casts_export
from garmin_processing import process_garmin_export
from kindle_processing import process_kindle_export
from moneymgr_processing import process_moneymgr_export
from nutrilio_processing import process_nutrilio_export
from apple_processing import process_apple_export
from offscreen_processing import process_offscreen_export
from weather_processing import get_weather_data
from letterboxd_processing import process_letterboxd_export
from drive_storage import update_drive

def find_unzip_folder(data_source, zip_file_path = None):
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
    if zip_file_path is not None:
    # Extract the contents of the zip file to a new folder
        unzip_folder = os.path.join(download_folder, f"{data_source}_export_unzipped")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
        os.remove(zip_file_path)
    else:
        return f"No {data_source} file to unzip \n"

def clean_rename_move_folder(export_folder, download_folder, folder_name, new_folder_name):
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

# List of URLs to open in new tabs
def download_web_data():
    download_YN = input("Do you want to download new data from websites? (Y/N) ")
    urls = [
        'https://www.goodreads.com/review/import',
        'https://benjaminbenben.com/lastfm-to-csv/',
        'https://www.amazon.com/hz/privacy-central/data-requests/preview.html',
        'https://www.garmin.com/fr-BE/account/datamanagement/exportdata/',
        'https://letterboxd.com/settings/data/']
    if download_YN == 'N':
        GR = 'N'
        LFM = 'N'
        LBX = 'N'
        return GR, LFM, LBX
    for url in urls:
        subprocess.run(['open', '-a', 'Firefox', '-g', url])
    GR = input("Did GR export got downloaded ? (Y/N) ")
    LFM = input("Did LFM export got downloaded ? (Y/N) ")
    LBX = input("Did Letterboxd export got downloaded ? (Y/N) ")
    input("Was Pocket cast export requested ? (Y/N) ")
    input("Was Garmin & Kindle data requested? (Y/N) ")
    return GR, LFM, LBX

def download_app_data():
    download_YN = input("Did you download new data from apps? (Y/N) ")
    if download_YN == 'N':
        MM = 'N'
        NUT = 'N'
        APH = 'N'
        OFF = 'N'
        return MM, NUT, APH, OFF
    MM = input("Did Money MGR export got downloaded ? (Y/N) ")
    NUT = input("Did Nutrilio export got downloaded ? (Y/N) ")
    APH = input("Did Apple Health export got downloaded ? (Y/N) ")
    OFF = input("Did Offscreen export got downloaded ? (Y/N) ")
    return MM, NUT, APH, OFF

def execute_processing():
    GR, LFM, LBX = download_web_data()
    MM, NUT, APH, OFF = download_app_data()
    file_names = []
    print('\n')
    if GR == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the goodreads export \n')
        clean_rename_move_file("files/exports/goodreads_exports", "/Users/valen/Downloads", "goodreads_library_export.csv", "gr_export.csv")
        process_gr_export()
        print('gr_processed.csv was created \n')
        print('----------------------------------------------')
    if LFM == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the lfm export \n')
        clean_rename_move_file("files/exports/lfm_exports", "/Users/valen/Downloads", "entinval.csv", "lfm_export.csv")
        process_lfm_export()
        file_names.append('processed_files/lfm_processed.csv')
        print('lfm_processed.csv was created\n')
        print('----------------------------------------------')
    PCC = input("Is there new Pocket Cast data available? (Y/N) ")
    if PCC == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the Pocket Cast export \n')
        clean_rename_move_file("files/exports/pocket_casts_exports", "/Users/valen/Downloads", "data.txt", "data.txt")
        process_pocket_casts_export()
        file_names.append('files/processed_files/pocket_casts_processed.csv')
        print('pocket_casts_processed.csv was created \n')
        print('----------------------------------------------')
    GAR = input("Is there new Garmin data available? (Y/N) ")
    if GAR == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the Garmin export \n')
        find_unzip_folder("garmin")
        clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "garmin_export_unzipped", "garmin_exports")
        process_garmin_export()
        file_names.append('files/processed_files/garmin_activities_list_processed.csv')
        file_names.append('files/processed_files/garmin_activities_splits_processed.csv')
        file_names.append('files/processed_files/garmin_sleep_processed.csv')
        file_names.append('files/processed_files/garmin_training_history_processed.csv')
        print('garmin processed files were created \n')
        print('----------------------------------------------')
    KIN = input("Is there new Kindle data available? (Y/N) ")
    if KIN == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the Kindle export \n')
        find_unzip_folder("kindle")
        clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "kindle_export_unzipped", "kindle_exports")
        process_kindle_export()
        print('kindle_processed.csv was created\n')
        print('----------------------------------------------')
    if (KIN == "Y") | (GR == 'Y'):
        print('----------------------------------------------')
        print('Merging the Kindle & Goodreads processed files \n')
        merge_gr_kindle()
        file_names.append('files/processed_files/kindle_gr_processed.csv')
        print('kindle_gr_processed.csv was created\n')
        print('----------------------------------------------')
    if MM == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the Money Mgr export \n')
        clean_rename_move_file("files/exports/moneymgr_exports", "/Users/valen/Downloads", f"{date.today().strftime('%Y-%m-%d')}.xlsx", "moneymgr_export.xlsx")
        process_moneymgr_export()
        file_names.append('files/processed_files/moneymgr_processed.csv')
        print('moneymgr_processed.csv was created \n')
        print('----------------------------------------------')
    if NUT == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the Nutrilio export \n')
        clean_rename_move_file("files/exports/nutrilio_exports", "/Users/valen/Downloads",\
                      f"Nutrilio-export-{date.today().strftime('%Y-%m-%d')}.csv", "nutrilio_export.csv")
        nutrilio_files = process_nutrilio_export()
        for file in nutrilio_files:
            file_names.append(file)
        print('nutrilio_processed.csv & other nutrilio files were created \n')
        print('----------------------------------------------')
    if APH == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the Apple export \n')
        find_unzip_folder("apple")
        clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "apple_export_unzipped", "apple_exports")
        process_apple_export()
        file_names.append('files/processed_files/apple_processed.csv')
        print('apple_processed.csv was created \n')
        print('----------------------------------------------')
    if OFF == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the Offscreen export \n')
        csv_regex = r'\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}.*\.csv$'
        count_file = 0
        for f in os.listdir('/Users/valen/Downloads'):
            if re.match(csv_regex, f):
                count_file += 1
                clean_rename_move_file("files/exports/offscreen_exports", "/Users/valen/Downloads", f, f.split('0-')[1], count_file)
        print("No new offscreen export to process") if count_file == 0 else None
        process_offscreen_export()
        file_names.append('files/processed_files/offscreen_processed.csv')
        print('offscreen_processed.csv was created \n')
        print('----------------------------------------------')
    if LBX == 'Y':
        print('----------------------------------------------')
        print('Starting the processing of the Letterboxd export \n')
        csv_regex = r'letterboxd-vaherinckx-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-utc.zip'
        for f in os.listdir('/Users/valen/Downloads'):
            if re.match(csv_regex, f):
                find_unzip_folder('letterboxd', zip_file_path = os.path.join("/Users/valen/Downloads", f))
        clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "letterboxd_export_unzipped", "letterboxd_exports")
        process_letterboxd_export()
        file_names.append('files/processed_files/letterboxd_processed.csv')
        print('letterboxd_processed.csv was created \n')
        print('----------------------------------------------')
    WEA = input("Use API to download latest weather data? (Y/N) ")
    if WEA == 'Y':
        print('----------------------------------------------')
        print('Adding the latest Weather data \n')
        get_weather_data()
        file_names.append('files/processed_files/weather_processed.csv')
        print('weather_processed.csv was created \n')
        print('----------------------------------------------')
    update_drive(file_names)

execute_processing()
