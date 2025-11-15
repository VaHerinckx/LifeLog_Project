import subprocess
import pandas as pd
import chardet
import sys
import traceback
import logging
from datetime import datetime
from src.music.lastfm_processing import process_lfm_export
from src.books.goodreads_processing import process_gr_export
from src.books.books_processing import process_book_exports
from src.books.kindle_processing import process_kindle_export
from src.podcasts.pocket_casts_processing import full_pocket_casts_pipeline
from src.sport.garmin_processing import process_garmin_export
from src.finance.moneymgr_processing import process_moneymgr_export
from src.nutrilio.nutrilio_processing import process_nutrilio_export
from src.health.apple_processing import process_apple_export
from src.screentime.offscreen_processing import process_offscreen_export
from src.weather.weather_processing import get_weather_data
from src.movies.letterboxd_processing import process_letterboxd_export
from lifelog_python_processing.src.shows.trakt_processing import create_trakt_processed_file
from src.location.location_processing import full_location_pipeline

# Updated imports for enhanced authentication
from src.utils.drive_storage import (
    update_drive,
    check_credentials_status,
    test_drive_connection,
)


# Initialize Drive connection at startup
def initialize_drive_connection():
    """Initialize and test Google Drive connection with enhanced authentication"""
    print("🔗 Initializing Google Drive connection...")
    print("=" * 60)

    # Check current credential status
    print("📋 Checking credential status...")
    check_credentials_status()

    # Test the connection
    print("\n🧪 Testing Drive connection...")
    connection_success = test_drive_connection()

    if connection_success:
        print("✅ Google Drive initialization successful!")
        print("=" * 60)
        return True
    else:
        print("❌ Google Drive initialization failed!")
        print("=" * 60)
        return False

# Initialize connection once at startup
DRIVE_INITIALIZED = initialize_drive_connection()
if not DRIVE_INITIALIZED:
    print("❌ Cannot proceed without Google Drive connection. Exiting.")
    sys.exit(1)

# File organization
reading_files = ['files/processed_files/kindle_gr_processed.csv']
finance_files = ['files/processed_files/moneymgr_processed.csv']
health_files = ['files/processed_files/apple_processed.csv', 'files/processed_files/garmin_activities_list_processed.csv',
                'files/processed_files/garmin_activities_splits_processed.csv', 'files/processed_files/garmin_sleep_processed.csv',
                'files/processed_files/garmin_stress_level_processed.csv', 'files/processed_files/garmin_training_history_processed.csv']
podcast_files = ['files/processed_files/pocket_casts_processed.csv']
music_files = ['files/processed_files/lfm_processed.csv']
movies_files = ['files/processed_files/letterboxd_processed.csv', 'files/processed_files/movies/trakt_processed.csv']
nutrilio_files = ['files/processed_files/nutrilio_body_sensations_pbi_processed_file.csv', 'files/processed_files/nutrilio_dreams_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_drinks_pbi_processed_file.csv', 'files/processed_files/nutrilio_food_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_self_improvement_pbi_processed_file.csv', 'files/processed_files/nutrilio_social_activity_pbi_processed_file.csv',
                  'files/processed_files/nutrilio_work_content_pbi_processed_file.csv', 'files/processed_files/nutrilio_processed.csv',
                  'files/work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx', 'files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx',]
screentime_files = ['files/processed_files/offscreen_processed.csv']
weather_files = ['files/processed_files/weather_processed.csv']
location_files = ['files/processed_files/location/combined_timezone_processed.csv']
all_files = reading_files + finance_files+ health_files + podcast_files + music_files + movies_files + nutrilio_files + screentime_files + weather_files + location_files

dict_upload = {
    "podcast" : podcast_files,
    "reading" : reading_files,
    "health" : health_files,
    "finance" : finance_files,
    "music" : music_files,
    "movies" : movies_files,
    "nutrilio" : nutrilio_files,
    "screentime" : screentime_files,
    "weather" : weather_files,
    "location" : location_files
}


def download_web_data():
    """Opens all the web urls in Firefox, then asks the user if each export was made, to later decide if processing is needed"""
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
    """Upload processed files to Google Drive using enhanced authentication"""
    file_names = all_files

    print(f"\n📤 Starting upload of {len(file_names)} files...")
    print("=" * 50)

    # Use the enhanced update_drive function
    success = update_drive(file_names)

    if success:
        print("🎉 All files uploaded successfully!")
    else:
        print("⚠️  Some files failed to upload. Check the log above for details.")

    print("=" * 50)
    return success


def upload_file_list(file_list):
    """Upload a specific list of files to Google Drive"""

    print(f"\n📤 Starting upload of {len(file_list)} files...")
    print("=" * 40)

    # Use the enhanced update_drive function
    success = update_drive(file_list)

    if success:
        print("✅ Files uploaded successfully!")
    else:
        print("⚠️  Some files failed to upload. Check the log above for details.")

    print("=" * 40)
    return success


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
    TRK = input("New Trakt file? (Y/N) ")
    WEA = input("Use API to download latest weather data? (Y/N) ")
    LOC = input("Process location data (Google Maps + Manual Excel)? (Y/N) ")

    file_names = []
    print('\n')

    # Define a helper function for processing steps with error handling
    def run_process(condition, process_function, process_name, *args, **kwargs):
        if condition == 'Y':
            try:
                print('----------------------------------------------')
                print(f'Starting the processing of the {process_name} export')
                process_function(*args, **kwargs)
                print(f'✅ {process_name} processing completed successfully')
                print('----------------------------------------------')
                return True
            except Exception as e:
                error_msg = f"Error during {process_name} processing: {str(e)}"


                print(f"❌ ERROR: {error_msg}")
                print("Continuing with next step...")
                print('----------------------------------------------')
                failed_steps.append(process_name)
                return False
        return None  # If condition wasn't 'Y', nothing to do

    # Run each process with error handling
    gr_success = run_process(GR, process_gr_export, "Goodreads")
    kin_success = run_process(KIN, process_kindle_export, "Kindle")

    # Upload reading files if either Goodreads or Kindle was processed
    if gr_success or kin_success:
        upload_file_list(reading_files)

    # For steps that depend on other steps being successful
    if (KIN == "Y" or GR == 'Y') and (kin_success is not False or gr_success is not False):
        try:
            print('----------------------------------------------')
            print('Starting the processing of the Books export (combining Goodreads + Kindle)')
            process_book_exports(upload="N")
            print('✅ Books processing completed successfully')
            print('----------------------------------------------')
            upload_file_list(reading_files)
        except Exception as e:
            error_msg = f"Error during Books processing: {str(e)}"


            print(f"❌ ERROR: {error_msg}")
            print("Continuing with next step...")
            print('----------------------------------------------')
            failed_steps.append("Books")

    # Continue with independent processes
    if run_process(LFM, process_lfm_export, "Last.fm", upload="N"):
        upload_file_list(music_files)

    if run_process(PCC, full_pocket_casts_pipeline, "Pocket Casts", auto_full=False):
        upload_file_list(podcast_files)

    if run_process(GAR, process_garmin_export, "Garmin", upload="N"):
        upload_file_list(health_files)

    if run_process(MM, process_moneymgr_export, "Money Manager", upload="N"):
        upload_file_list(finance_files)

    if run_process(NUT, process_nutrilio_export, "Nutrilio", upload="N"):
        upload_file_list(nutrilio_files)

    if run_process(APH, process_apple_export, "Apple Health", upload="N"):
        upload_file_list(health_files)

    if run_process(OFF, process_offscreen_export, "Offscreen", upload="N"):
        upload_file_list(screentime_files)

    if run_process(LBX, process_letterboxd_export, "Letterboxd", upload="N"):
        upload_file_list(movies_files)

    if run_process(TRK, create_trakt_processed_file, "Trakt", upload="N"):
        upload_file_list(movies_files)

    if run_process(WEA, get_weather_data, "Weather", upload="N"):
        upload_file_list(weather_files)

    if run_process(LOC, full_location_pipeline, "Location", auto_full=False):
        upload_file_list(location_files)

    # Report on any failures
    if failed_steps:
        print(f"\n⚠️  WARNING: The following steps had errors:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nCheck the log file for details.")
    else:
        print("\n🎉 All processing steps completed successfully!")


def make_sample_files():
    """Create sample files for testing or documentation"""
    file_paths = [
        'files/processed_files/kindle_gr_processed.csv',
        'files/processed_files/pocket_casts_processed.csv',
        'files/processed_files/lfm_processed.csv',
        'files/processed_files/garmin_activities_list_processed.csv',
        'files/processed_files/letterboxd_processed.csv',
        'files/processed_files/moneymgr_processed.csv',
        'files/processed_files/nutrilio_processed.csv',
        # Add other file paths as needed
    ]

    for file_path in file_paths:
        try:
            file_name = file_path.split('/')[2].split(".")[0]
            print(f"📝 Creating sample for {file_name}")

            try:
                df = pd.read_csv(file_path, sep='|')
            except UnicodeDecodeError:
                # Fall back to encoding detection
                with open(file_path, 'rb') as file:
                    encoding = chardet.detect(file.read())['encoding']
                df = pd.read_csv(file_path, encoding=encoding, sep='|', na_values=[''])

            sample_path = f"files/sample_files/{file_name}_sample.csv"
            df.head(20).to_csv(sample_path, sep="|", encoding='utf-16', index=False)
            print(f"✅ Sample created: {sample_path}")
        except Exception as e:


            print(f"❌ ERROR creating sample for {file_path}: {str(e)}")


def upload_single_file():
    """Upload files for a specific data category"""
    print("-" * 60)
    print("📤 Choose the data category you want to upload from this list:")
    for i, key in enumerate(dict_upload.keys(), 1):
        print(f"  {i}. {key.title()}")
    print("-" * 60)

    choice = input("Your choice (name or number): ").lower().strip()

    # Handle both numeric and text input
    if choice.isdigit():
        choices = list(dict_upload.keys())
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(choices):
            choice = choices[choice_index]
        else:
            print("❌ Invalid option. Please run the script again and select a valid option")
            return

    if choice in dict_upload.keys():
        print(f"📤 Uploading {choice} files...")
        success = upload_file_list(dict_upload[choice])
        if success:
            print(f"✅ {choice.title()} files uploaded successfully!")
        else:
            print(f"⚠️  Some {choice} files failed to upload.")
    else:
        print("❌ Invalid option. Please run the script again and select a valid option")


# Main execution block
def main():
    """Main execution function with enhanced error handling"""
    try:
        print("\n🚀 LifeLog Data Processing System")
        print("=" * 60)

        download = input("""Choose an option:
    1. Download/process/upload all data
    2. Upload all existing files
    3. Create sample files for testing
    4. Upload files for specific category

    Your choice (1/2/3/4): """)

        if download == "1":
            print("\n📋 Starting complete data processing workflow...")
            download_process_upload()
        elif download == "2":
            print("\n📤 Starting upload of all processed files...")
            try:
                upload_files()
            except Exception as e:


                print(f"❌ ERROR during upload: {str(e)}")
        elif download == "3":
            print("\n📝 Creating sample files for testing...")
            try:
                make_sample_files()
            except Exception as e:


                print(f"❌ ERROR creating sample files: {str(e)}")
        elif download == "4":
            print("\n📤 Starting category-specific upload...")
            try:
                upload_single_file()
            except Exception as e:


                print(f"❌ ERROR uploading files: {str(e)}")
        else:
            print("❌ Invalid option. Please run the script again and select 1, 2, 3, or 4.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {str(e)}")
    finally:
        print(f"\n📋 Processing complete. Check log file for details.")


if __name__ == "__main__":
    main()
