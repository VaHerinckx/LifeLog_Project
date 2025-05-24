import subprocess
import pandas as pd
import chardet
import sys
import traceback
import logging
from datetime import datetime
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
from drive_storage import update_drive, get_google_auth, update_file


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

drive = get_google_auth()

reading_files = ['files/processed_files/kindle_gr_processed.csv']
finance_files = ['files/processed_files/moneymgr_processed.csv']
health_files = ['files/processed_files/apple_processed.csv', 'files/processed_files/garmin_activities_list_processed.csv',
                'files/processed_files/garmin_activities_splits_processed.csv', 'files/processed_files/garmin_sleep_processed.csv',
                'files/processed_files/garmin_stress_level_processed.csv', 'files/processed_files/garmin_training_history_processed.csv']
podcast_files = ['files/processed_files/pocket_casts_processed.csv']
music_files = ['files/processed_files/lfm_processed.csv']
movies_files = ['files/processed_files/letterboxd_processed.csv']
nutrilio_files = ['files/processed_files/nutrilio_body_sensations_pbi_processed_file.csv', 'files/processed_files/nutrilio_dreams_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_drinks_pbi_processed_file.csv', 'files/processed_files/nutrilio_food_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_self_improvement_pbi_processed_file.csv', 'files/processed_files/nutrilio_social_activity_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_work_content_pbi_processed_file.csv', 'files/processed_files/nutrilio_processed.csv',
                  'files/work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx', 'files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx',]
screentime_files = ['files/processed_files/offscreen_processed.csv']
weather_files = ['files/processed_files/weather_processed.csv']
all_files = reading_files + finance_files+ health_files + podcast_files + music_files + movies_files + nutrilio_files + screentime_files

dict_upload = {
    "podcast" : podcast_files,
    "reading" : reading_files,
    "health" : health_files,
    "finance" : finance_files,
    "music" : music_files,
    "movies" : movies_files,
    "nutrilio" : nutrilio_files,
    "screentime" : screentime_files,
    "weather" : weather_files
}


def download_web_data():
    """Opens all the web urls in Firefox, then asks the user if each export was made, to later decide if processing is needed"""
    logging.info("Opening web URLs for data exports")
    urls = [
        'https://www.goodreads.com/review/import',
        'https://benjaminbenben.com/lastfm-to-csv/',
        'https://www.amazon.com/hz/privacy-central/data-requests/preview.html',
        'https://www.garmin.com/fr-BE/account/datamanagement/exportdata/',
        'https://letterboxd.com/settings/data/']
    for url in urls:
        try:
            subprocess.run(['open', '-a', 'Firefox', '-g', url])
        except Exception as e:
            logging.warning(f"Failed to open URL {url}: {e}")
            print(f"Warning: Failed to open {url}")

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
    """Upload processed files to Google Drive"""
    logging.info("Starting file upload to Google Drive")
    file_names = all_files

    # Track which files were successfully uploaded
    upload_results = []

    for file_name in file_names:
        try:
            logging.info(f"Uploading {file_name}")
            update_drive([file_name], drive)
            upload_results.append((file_name, "Success"))
        except Exception as e:
            error_msg = f"Failed to upload {file_name}: {str(e)}"
            logging.error(error_msg)
            upload_results.append((file_name, f"Failed: {str(e)}"))

    # Log summary of upload results
    logging.info("Upload summary:")
    for file_name, result in upload_results:
        logging.info(f"{file_name}: {result}")

    # Count successes and failures
    successes = sum(1 for _, result in upload_results if result == "Success")
    failures = len(upload_results) - successes

    logging.info(f"Upload complete. {successes} files uploaded successfully, {failures} failed.")
    print(f"Upload complete. {successes} files uploaded successfully, {failures} failed.")

    if failures > 0:
        print("The following files failed to upload:")
        for file_name, result in upload_results:
            if result != "Success":
                print(f"  - {file_name}")


def download_process_upload():
    """Download, process, and prepare data for upload with error handling"""
    # Setup for tracking failures
    failed_steps = []

    # Get user inputs as before
    web_download = input("Do you want to download new data from websites? (Y/N) ")
    if web_download == "Y":
        try:
            GR, LFM, LBX = download_web_data()
        except Exception as e:
            logging.error(f"Error during web data download: {str(e)}")
            logging.error(traceback.format_exc())
            print(f"ERROR during web data download: {str(e)}")
            GR, LFM, LBX = 'N', 'N', 'N'
    else:
        GR = input("New Goodreads file? (Y/N) ")
        LFM = input("New LFM file? (Y/N) ")
        LBX = input("New Letterboxd file? (Y/N) ")

    app_download = input("Do you want to download new data from mobile apps? (Y/N) ")
    if app_download == "Y":
        try:
            MM, NUT, APH, OFF = download_app_data()
        except Exception as e:
            logging.error(f"Error during app data download: {str(e)}")
            logging.error(traceback.format_exc())
            print(f"ERROR during app data download: {str(e)}")
            MM, NUT, APH, OFF = 'N', 'N', 'N', 'N'
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

    # Define a helper function for processing steps with error handling
    def run_process(condition, process_function, process_name, *args, **kwargs):
        if condition == 'Y':
            try:
                logging.info(f"Starting {process_name} processing")
                print('----------------------------------------------')
                process_function(*args, **kwargs)
                print('----------------------------------------------')
                logging.info(f"Completed {process_name} processing")
                logging.info(f"\n")
                return True
            except Exception as e:
                error_msg = f"Error during {process_name} processing: {str(e)}"
                logging.error(error_msg)
                logging.error(traceback.format_exc())
                print(f"ERROR: {error_msg}")
                print("Continuing with next step...")
                print('----------------------------------------------')
                logging.info(f"\n")
                failed_steps.append(process_name)
                return False
        return None  # If condition wasn't 'Y', nothing to do

    # Run each process with error handling
    gr_success = run_process(GR, process_gr_export, "Goodreads")
    kin_success = run_process(KIN, process_kindle_export, "Kindle")
    update_drive(reading_files, drive)


    # For steps that depend on other steps being successful
    if (KIN == "Y" or GR == 'Y') and (kin_success is not False or gr_success is not False):
        try:
            logging.info("Starting Books processing")
            print('----------------------------------------------')
            process_book_exports(upload="N")
            print('----------------------------------------------')
            logging.info("Completed Books processing")
        except Exception as e:
            error_msg = f"Error during Books processing: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            print(f"ERROR: {error_msg}")
            print("Continuing with next step...")
            print('----------------------------------------------')
            failed_steps.append("Books")

    # Continue with independent processes
    run_process(LFM, process_lfm_export, "Last.fm", upload="N")
    run_process(PCC, process_pocket_casts_export, "Pocket Casts", upload="N")
    run_process(GAR, process_garmin_export, "Garmin", upload="N")
    run_process(MM, process_moneymgr_export, "Money Manager", upload="N")
    run_process(NUT, process_nutrilio_export, "Nutrilio", upload="N")
    run_process(APH, process_apple_export, "Apple Health", upload="N")
    run_process(OFF, process_offscreen_export, "Offscreen", upload="N")
    run_process(LBX, process_letterboxd_export, "Letterboxd", upload="N")
    run_process(WEA, get_weather_data, "Weather", upload="N")

    # Report on any failures
    if failed_steps:
        logging.warning(f"The following processing steps failed: {', '.join(failed_steps)}")
        print("\nWARNING: The following steps had errors:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nCheck the log file for details.")

        proceed = input("\nSome steps failed. Do you still want to upload the files? (Y/N) ")
        if proceed.upper() != 'Y':
            logging.info("Upload canceled by user after processing failures")
            return

    # Upload files regardless of individual processing failures
    try:
        logging.info("Starting file upload")
        upload_files()
        logging.info("Completed file upload")
    except Exception as e:
        logging.error(f"Error during file upload: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"ERROR during upload: {str(e)}")


def make_sample_files():
    """Create sample files for testing or documentation"""
    file_paths = [
        'files/processed_files/kindle_gr_processed.csv',
        # Add other file paths as needed
    ]

    for file_path in file_paths:
        try:
            file_name = file_path.split('/')[2].split(".")[0]
            logging.info(f"Creating sample file for {file_name}")
            print(file_name)

            try:
                df = pd.read_csv(file_path, sep='|')
            except UnicodeDecodeError:
                logging.warning(f"Warning: {file_path} might have a different encoding")
                # Fall back to encoding detection
                with open(file_path, 'rb') as file:
                    encoding = chardet.detect(file.read())['encoding']
                df = pd.read_csv(file_path, encoding=encoding, sep='|', na_values=[''])

            sample_path = f"files/sample_files/{file_name}_sample.csv"
            df.head(20).to_csv(sample_path, sep="|", encoding='utf-16', index=False)
            logging.info(f"Successfully created sample file: {sample_path}")
        except Exception as e:
            logging.error(f"Error creating sample file for {file_path}: {str(e)}")
            logging.error(traceback.format_exc())
            print(f"ERROR creating sample for {file_path}: {str(e)}")


def upload_single_file():
    print("-------------------------------------------------")
    print("Choose the data you want to upload from this list")
    for key in dict_upload.keys():
        print(key)
    print("-------------------------------------------------")
    choice = input("Your choice : ")
    if choice in dict_upload.keys():
        update_drive(dict_upload[choice], drive)
    else:
        logging.warning(f"Invalid option selected: {choice}")
        print("Invalid option. Please run the script again and select a valid option")


# Main execution block
try:
    download = input("""Do you want to download/process/upload (1), just upload all files (2), create sample files (3), or upload one file (4)? (1/2/3/4) """)

    if download == "1":
        logging.info("Starting download, process, and upload workflow")
        download_process_upload()
    elif download == "2":
        logging.info("Starting upload-only workflow")
        try:
            upload_files()
        except Exception as e:
            logging.error(f"Error during file upload: {str(e)}")
            logging.error(traceback.format_exc())
            print(f"ERROR during upload: {str(e)}")
    if download == "3":
        logging.info("Creating sample files")
        try:
            make_sample_files()
        except Exception as e:
            logging.error(f"Error creating sample files: {str(e)}")
            logging.error(traceback.format_exc())
            print(f"ERROR creating sample files: {str(e)}")
    if download == "4":
        logging.info("Creating sample files")
        try:
            upload_single_file()
        except Exception as e:
            logging.error(f"Error uploading files: {str(e)}")
            logging.error(traceback.format_exc())
            print(f"ERROR uploading files: {str(e)}")
    else:
        logging.warning(f"Invalid option selected: {download}")
        print("Invalid option. Please run the script again and select 1, 2, or 3.")

except KeyboardInterrupt:
    print("\nProcess interrupted by user.")
    logging.info("Process interrupted by user")
except Exception as e:
    logging.error(f"Unexpected error in main process: {str(e)}")
    logging.error(traceback.format_exc())
    print(f"CRITICAL ERROR: {str(e)}")
finally:
    print("\nProcessing complete. Check log file for details.")
    logging.info("Script execution finished")
