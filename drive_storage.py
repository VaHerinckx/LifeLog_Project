import os
from dotenv import load_dotenv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
load_dotenv()
folder_id = os.environ['Drive_folder_id']

def update_file(file_name):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for file in fileList:
        if(file['title'] == file_name.split('/')[-1]):
            file_to_update = drive.CreateFile({'id': file['id']})
            file_to_update.SetContentFile(file_name)
            file_to_update.Upload()
            print(f"{file_name} was updated in Google Drive")

def update_drive(file_names):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for file_name in file_names:
        for file in fileList:
            if(file['title'] == file_name.split('/')[-1]):
                file_to_update = drive.CreateFile({'id': file['id']})
                file_to_update.SetContentFile(file_name)
                file_to_update.Upload()
                print(f"{file_name} was updated in Google Drive")
