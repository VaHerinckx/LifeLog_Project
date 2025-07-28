import pandas as pd
import os
import xmltodict
import subprocess
from src.utils.file_operations import find_unzip_folder, clean_rename_move_folder, check_file_exists
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection

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
    path = f'files/processed_files/apple/apple_{name_val}.csv'
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

def download_apple_data():
    """
    Opens Apple Health export instructions and prompts user to download data.
    Returns True if user confirms download, False otherwise.
    """
    print("🍎 Starting Apple Health data download...")

    urls = ['https://support.apple.com/en-us/102203']
    open_web_urls(urls)

    print("📝 Instructions:")
    print("   1. Open the Health app on your iPhone")
    print("   2. Tap your profile picture in the top right")
    print("   3. Tap 'Export All Health Data'")
    print("   4. Choose 'Export' and then select how to share the data")
    print("   5. Save the export.zip file to your Downloads folder")

    response = prompt_user_download_status("Apple Health")

    return response


def move_apple_files():
    """
    Moves the downloaded Apple Health files from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving Apple Health files...")

    # First, try to unzip the apple health file
    unzip_success = find_unzip_folder("apple")
    if not unzip_success:
        print("❌ Failed to find or unzip Apple Health file")
        return False

    # Then move the unzipped folder
    move_success = clean_rename_move_folder(
        export_folder="files/exports",
        download_folder="/Users/valen/Downloads",
        folder_name="apple_export_unzipped",
        new_folder_name="apple_exports"
    )

    if move_success:
        print("✅ Successfully moved Apple Health files to exports folder")
    else:
        print("❌ Failed to move Apple Health files")

    return move_success


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


def create_apple_file():
    """
    Main processing function that processes Apple Health export data.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Apple Health data...")

    try:
        # Check if input files exist
        path_export = 'files/exports/apple_exports/apple_health_export/export.xml'
        if not os.path.exists(path_export):
            print(f"❌ Apple Health export file not found: {path_export}")
            return False

        # Process the files using existing logic
        create_apple_files()

        print("✅ Apple Health data processing complete!")
        return True

    except Exception as e:
        print(f"❌ Error processing Apple Health data: {e}")
        return False


def upload_apple_results():
    """
    Uploads the processed Apple Health files to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️  Uploading Apple Health results to Google Drive...")

    files_to_upload = ['files/processed_files/apple_processed.csv']

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Apple Health results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


def process_apple_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_apple_pipeline(auto_full=True)
    else:
        return create_apple_file()


def full_apple_pipeline(auto_full=False):
    """
    Complete Apple Health pipeline with 3 options.

    Options:
    1. Full pipeline (download → move → process → upload)
    2. Process existing file only
    3. Process existing file and upload

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🍎 APPLE HEALTH DATA PIPELINE")
    print("="*60)

    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → move → process → upload)")
        print("2. Process existing file only")
        print("3. Process existing file and upload to Drive")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Starting full Apple Health pipeline...")

        # Step 1: Download
        download_success = download_apple_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_apple_files()
        else:
            print("⚠️  Download not confirmed, but checking for existing files...")
            move_success = move_apple_files()

        # Step 3: Process (fallback to option 2 if no new files)
        if move_success:
            process_success = create_apple_file()
        else:
            print("⚠️  No new files found, attempting to process existing files...")
            process_success = create_apple_file()

        # Step 4: Upload
        if process_success:
            upload_success = upload_apple_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "2":
        print("\n⚙️  Processing existing Apple Health file only...")
        success = create_apple_file()

    elif choice == "3":
        print("\n⚙️  Processing existing file and uploading...")
        process_success = create_apple_file()
        if process_success:
            success = upload_apple_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Apple Health pipeline completed successfully!")
    else:
        print("❌ Apple Health pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("🍎 Apple Health Processing Tool")
    print("This tool helps you download, process, and upload Apple Health data.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_apple_pipeline(auto_full=False)
