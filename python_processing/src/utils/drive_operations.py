# This file simply imports the existing drive_storage.py functions
# Your drive_storage.py already handles authentication, error handling, and uploading perfectly

from drive_storage import update_file, update_drive, check_credentials_status, test_drive_connection
import os


def upload_single_file(file_path):
    """Upload a single file to Google Drive using existing drive_storage logic"""
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} does not exist")
        return False

    return update_file(file_path)


def upload_multiple_files(file_paths):
    """Upload multiple files to Google Drive using existing drive_storage logic"""
    if not file_paths:
        print("ℹ️  No files to upload")
        return True

    # Filter out files that don't exist
    existing_files = [f for f in file_paths if os.path.exists(f)]

    if not existing_files:
        print("❌ No existing files found to upload")
        return False

    # Use your existing update_drive function which already handles everything
    return update_drive(existing_files)


def verify_drive_connection():
    """Test Google Drive connection before uploading"""
    print("🔍 Checking Google Drive connection...")
    check_credentials_status()
    return test_drive_connection()
