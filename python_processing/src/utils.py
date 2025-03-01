import pandas as pd
import datetime
from datetime import datetime
import os
import shutil
import zipfile

time_diff_excel = pd.read_excel('files/work_files/GMT_timediff.xlsx')


def today_export():
    current_date = datetime.now()
    date_string = current_date.strftime("%d%m%y")
    return date_string

def time_difference_correction(date, timezone_source='GMT'):
    """Corrects timezone differences"""
    time_diff_excel["Date"] = pd.to_datetime(time_diff_excel["Date"], utc = True)
    for index, row_datefile in time_diff_excel.iterrows():
        if date >= row_datefile['Date']:
            continue
        elif index == 0:
            return date + pd.Timedelta(hours=time_diff_excel.loc[index][f'Time_diff_{timezone_source}'])
        else:
            return date + pd.Timedelta(hours=time_diff_excel.loc[index-1][f'Time_diff_{timezone_source}'])
    return date + pd.Timedelta(hours=time_diff_excel.loc[time_diff_excel.index[-1]][f'Time_diff_{timezone_source}'])

def get_response(client, system_prompt, user_prompt):
  # Assign the role and content for each message
  messages = [{"role": "system", "content": system_prompt},
      		  {"role": "user", "content": user_prompt}]
  response = client.chat.completions.create(
      model="gpt-3.5-turbo", messages= messages, temperature=0)
  return response.choices[0].message.content


def find_unzip_folder(data_source, zip_file_path = None):
    """Unzips the foler specified in the path"""
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
        elif (data_source == 'pocket_casts') & (zip_file == 'data_export.zip'):
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
    """Removes the folder from Download, renames them and sends them within this directory"""
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
    """Removes the file from Download, renames them and sends them within this directory"""
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
