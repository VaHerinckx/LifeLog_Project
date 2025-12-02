import os
import shutil
import zipfile
import re


def find_unzip_folder(data_source, zip_file_path=None):
    """Unzips the folder specified in the path, or returns True if already unzipped"""
    download_folder = "/Users/valen/Downloads"
    unzip_folder = os.path.join(download_folder, f"{data_source}_export_unzipped")

    # Check if already unzipped folder exists in Downloads
    if os.path.exists(unzip_folder) and os.path.isdir(unzip_folder):
        print(f"✅ Found existing unzipped folder: {unzip_folder}")
        return True

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
        elif (data_source == 'pocket_casts') & (zip_file == 'data_export.zip'):
            zip_file_path = os.path.join(download_folder, zip_file)
            break
        elif (data_source == 'letterboxd'):
            # Handle letterboxd pattern separately
            csv_regex = r'letterboxd-vaherinckx-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-utc.zip'
            if re.match(csv_regex, zip_file):
                zip_file_path = os.path.join(download_folder, zip_file)
                break

    if zip_file_path is not None:
        # Extract the contents of the zip file to a new folder
        print(f"📦 Extracting {os.path.basename(zip_file_path)}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
        os.remove(zip_file_path)
        return True
    else:
        print(f"No {data_source} file to unzip")
        return False


def clean_rename_move_folder(export_folder, download_folder, folder_name, new_folder_name):
    """Removes the folder from Download, renames them and sends them within this directory"""
    folder_path = os.path.join(download_folder, folder_name)
    export_folder_path = os.path.join(export_folder, new_folder_name)

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist in Downloads")
        return False

    # Remove old export folder if it exists
    if os.path.exists(export_folder_path):
        shutil.rmtree(export_folder_path)

    downloaded_folder_path = os.path.join(download_folder, folder_name)
    # Rename the downloaded folder
    renamed_folder_path = os.path.join(download_folder, new_folder_name)
    os.rename(downloaded_folder_path, renamed_folder_path)
    # Move the renamed folder to the export folder
    shutil.move(renamed_folder_path, export_folder_path)
    return True


def clean_rename_move_file(export_folder, download_folder, file_name, new_file_name, file_number=1):
    """Removes the file from Download, renames them and sends them within this directory"""
    file_path = os.path.join(download_folder, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return False

    if file_number == 1:
        # Clear the export folder of old files
        for filename in os.listdir(export_folder):
            old_file_path = os.path.join(export_folder, filename)
            if os.path.isfile(old_file_path):
                os.remove(old_file_path)

    downloaded_file_path = os.path.join(download_folder, file_name)
    # Rename the downloaded file
    renamed_file_path = os.path.join(download_folder, new_file_name)
    os.rename(downloaded_file_path, renamed_file_path)
    # Move the renamed file to the export folder
    export_file_path = os.path.join(export_folder, new_file_name)
    shutil.move(renamed_file_path, export_file_path)
    return True


def check_file_exists(download_folder, file_pattern):
    """Check if a file matching the pattern exists in the download folder"""
    if not os.path.exists(download_folder):
        return False

    files = os.listdir(download_folder)

    # Handle different pattern types
    if isinstance(file_pattern, str):
        # Simple string match
        return file_pattern in files
    elif hasattr(file_pattern, 'match'):
        # Regex pattern
        return any(file_pattern.match(f) for f in files)
    elif callable(file_pattern):
        # Custom function
        return any(file_pattern(f) for f in files)

    return False


def get_matching_files(download_folder, file_pattern):
    """Get all files matching the pattern in the download folder"""
    if not os.path.exists(download_folder):
        return []

    files = os.listdir(download_folder)
    matching_files = []

    # Handle different pattern types
    if isinstance(file_pattern, str):
        # Simple string match
        matching_files = [f for f in files if file_pattern in f]
    elif hasattr(file_pattern, 'match'):
        # Regex pattern
        matching_files = [f for f in files if file_pattern.match(f)]
    elif callable(file_pattern):
        # Custom function
        matching_files = [f for f in files if file_pattern(f)]

    return matching_files
