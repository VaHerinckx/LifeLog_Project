import subprocess
import pandas as pd
import chardet
from lastfm_processing import process_lfm_export
from goodreads_processing import process_gr_export
from books_processing import process_book_exports
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


def download_web_data():
    """Opens all the web urls in Firefox, then asks the user if each export was made, to later decide if processing is needed"""
    urls = [
        'https://www.goodreads.com/review/import',
        'https://benjaminbenben.com/lastfm-to-csv/',
        'https://www.amazon.com/hz/privacy-central/data-requests/preview.html',
        'https://www.garmin.com/fr-BE/account/datamanagement/exportdata/',
        'https://letterboxd.com/settings/data/']
    for url in urls:
        subprocess.run(['open', '-a', 'Firefox', '-g', url])
    GR = input("Did GR export got downloaded ? (Y/N) ")
    LFM = input("Did LFM export got downloaded ? (Y/N) ")
    LBX = input("Did Letterboxd export got downloaded ? (Y/N) ")
    input("Was Pocket cast export requested ? (Y/N) ")
    input("Was Garmin & Kindle data requested? (Y/N) ")
    return GR, LFM, LBX

def download_app_data():
    """Asks the user if each export was made from the different apps, to later decide if processing is needed"""
    MM = input("Did Money MGR export got downloaded ? (Y/N) ")
    NUT = input("Did Nutrilio export got downloaded ? (Y/N) ")
    APH = input("Did Apple Health export got downloaded ? (Y/N) ")
    OFF = input("Did Offscreen export got downloaded ? (Y/N) ")
    return MM, NUT, APH, OFF

def upload_files():
    file_names = ['files/processed_files/lfm_processed.csv', 'files/processed_files/pocket_casts_processed.csv',
                  'files/processed_files/garmin_activities_list_processed.csv', 'files/processed_files/garmin_activities_splits_processed.csv',
                  'files/processed_files/garmin_sleep_processed.csv', 'files/processed_files/garmin_stress_level_processed.csv',
                  'files/processed_files/garmin_training_history_processed.csv', 'files/processed_files/kindle_gr_processed.csv',
                  'files/processed_files/moneymgr_processed.csv',
                  'files/processed_files/apple_processed.csv',
                  'files/processed_files/offscreen_processed.csv',
                  'files/processed_files/letterboxd_processed.csv',
                  'files/processed_files/weather_processed.csv',
                  'files/processed_files/nutrilio_body_sensations_pbi_processed_file.csv', 'files/processed_files/nutrilio_dreams_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_drinks_pbi_processed_file.csv', 'files/processed_files/nutrilio_food_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_self_improvement_pbi_processed_file.csv', 'files/processed_files/nutrilio_social_activity_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_work_content_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_processed.csv',
                  'files/work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx',
                  'files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx',
                  'files/work_files/Objectives.xlsx'
                  ]
    update_drive(file_names)

def download_process_upload():
    web_download = input("Do you want to download new data from websites? (Y/N) ")
    if web_download == "Y":
        GR, LFM, LBX = download_web_data()
    else:
        GR = input("New Goodreads file? (Y/N) ")
        LFM = input("New LFM file? (Y/N) ")
        LBX = input("New Letterboxd file? (Y/N) ")
    app_download = input("Do you want to download new data from mobile apps? (Y/N) ")
    if app_download == "Y":
        MM, NUT, APH, OFF = download_app_data()
    else:
        MM = input("New Money Mgr file? (Y/N) ")
        NUT = input("New Nutrilio file? (Y/N) ")
        APH = input("New Apple Health file? (Y/N) ")
        OFF = input("New OffScreen file? (Y/N) ")
    PCC = input("New Pocket Cast file? (Y/N) ")
    GAR = input("New Garmin file? (Y/N) ")
    KIN = input("New Kindle file? (Y/N) ")
    WEA = input("Use API to download latest weather data? (Y/N) ")
    file_names = []
    print('\n')
    if GR == 'Y':
        print('----------------------------------------------')
        process_gr_export()
        print('----------------------------------------------')
    if LFM == 'Y':
        print('----------------------------------------------')
        process_lfm_export(upload="N")
        print('----------------------------------------------')
    if PCC == 'Y':
        print('----------------------------------------------')
        process_pocket_casts_export(upload="N")
        print('----------------------------------------------')
    if GAR == 'Y':
        print('----------------------------------------------')
        process_garmin_export(upload="N")
        print('----------------------------------------------')
    if KIN == 'Y':
        print('----------------------------------------------')
        process_kindle_export()
        print('----------------------------------------------')
    if (KIN == "Y") | (GR == 'Y'):
        print('----------------------------------------------')
        process_book_exports(upload="N")
        print('----------------------------------------------')
    if MM == 'Y':
        print('----------------------------------------------')
        process_moneymgr_export(upload="N")
        print('----------------------------------------------')
    if NUT == 'Y':
        print('----------------------------------------------')
        process_nutrilio_export(upload="N")
        print('----------------------------------------------')
    if APH == 'Y':
        print('----------------------------------------------')
        process_apple_export(upload="N")
        print('----------------------------------------------')
    if OFF == 'Y':
        print('----------------------------------------------')
        process_offscreen_export(upload="N")
        print('----------------------------------------------')
    if LBX == 'Y':
        print('----------------------------------------------')
        process_letterboxd_export(upload="N")
        print('----------------------------------------------')
    if WEA == 'Y':
        print('----------------------------------------------')
        get_weather_data(upload="N")
        print('----------------------------------------------')


def read_csv_with_encoding(filepath):
    # Detect the encoding
    with open(filepath, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # Read the CSV with detected encoding
    return pd.read_csv(filepath, encoding=encoding, sep='|')


def make_sample_files():
    file_paths = [#'files/processed_files/lfm_processed.csv',
                  #'files/processed_files/pocket_casts_processed.csv',
                  #'files/processed_files/garmin_activities_list_processed.csv',
                  #'files/processed_files/garmin_activities_splits_processed.csv',
                  #'files/processed_files/garmin_sleep_processed.csv',
                  #'files/processed_files/garmin_stress_level_processed.csv',
                  #'files/processed_files/garmin_training_history_processed.csv',
                  'files/processed_files/kindle_gr_processed.csv',
                  #'files/processed_files/moneymgr_processed.csv',
                  #'files/processed_files/apple_processed.csv',
                  #'files/processed_files/offscreen_processed.csv',
                  #'files/processed_files/letterboxd_processed.csv',
                  #'files/processed_files/weather_processed.csv',
                  #'files/processed_files/nutrilio_body_sensations_pbi_processed_file.csv',
                  #'files/processed_files/nutrilio_dreams_pbi_processed_file.csv',
                  #'files/processed_files/nutrilio_drinks_pbi_processed_file.csv',
                  #'files/processed_files/nutrilio_food_pbi_processed_file.csv',
                  #'files/processed_files/nutrilio_self_improvement_pbi_processed_file.csv',
                  #'files/processed_files/nutrilio_social_activity_pbi_processed_file.csv',
                  #'files/processed_files/nutrilio_work_content_pbi_processed_file.csv',
                  #'files/processed_files/nutrilio_processed.csv',
                  ]

    for file_path in file_paths:
        file_name = file_path.split('/')[2].split(".")[0]
        print(file_name)
        try:
           df = pd.read_csv(file_path, sep='|')

        except UnicodeDecodeError:
           print(f"Warning: {file_path} might have a different encoding")
           # Optionally fall back to encoding detection for this specific file
           with open(file_path, 'rb') as file:
               encoding = chardet.detect(file.read())['encoding']
           df = pd.read_csv(file_path, encoding=encoding, sep='|', na_values=[''])

        df.head(20).to_csv(f"files/sample_files/{file_name}_sample.csv", sep = "|", encoding='utf-16', index = False)



#process_gr_export()
#process_lfm_export(upload="Y")
#process_pocket_casts_export(upload="Y")
#process_garmin_export(upload="Y")
#process_kindle_export()
#process_book_exports(upload="Y")
#process_moneymgr_export(upload="Y")
#process_nutrilio_export(upload="Y")
#process_apple_export(upload="Y")
#process_offscreen_export(upload="Y")
#process_letterboxd_export(upload="Y")
#get_weather_data(upload="Y")


#download = input("Do you want to download/process/upload (1) or just upload all files (2)? (1/2) ")
#if download == "1":
#    download_process_upload()
#    upload_files()
#else:
#    upload_files()

make_sample_files()
