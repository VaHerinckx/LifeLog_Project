# Enhanced drive_storage.py with persistent authentication
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

load_dotenv()
folder_id = os.environ['Drive_folder_id']

# Path to store credentials - you can change these paths
CREDENTIALS_DIR = 'credentials'  # Create a dedicated folder
CREDENTIALS_FILE = os.path.join(CREDENTIALS_DIR, 'credentials.json')
SETTINGS_FILE = os.path.join(CREDENTIALS_DIR, 'settings.yaml')
CLIENT_SECRETS_FILE = os.path.join(CREDENTIALS_DIR, 'client_secrets.json')

def create_settings_file():
    """Create a settings.yaml file for PyDrive configuration using client_secrets.json"""
    # Create credentials directory if it doesn't exist
    os.makedirs(CREDENTIALS_DIR, exist_ok=True)

    settings_content = f"""
client_config_backend: file
client_config_file: {CLIENT_SECRETS_FILE}

save_credentials: True
save_credentials_backend: file
save_credentials_file: {CREDENTIALS_FILE}

get_refresh_token: True

oauth_scope:
  - https://www.googleapis.com/auth/drive.file

access_type: offline
approval_prompt: force
include_granted_scopes: true
"""

    with open(SETTINGS_FILE, 'w') as f:
        f.write(settings_content)

def get_authenticated_drive():
    """
    Get authenticated Google Drive instance with persistent credentials.
    Only requires manual auth on first run or when refresh token expires.
    """
    # Create settings file if it doesn't exist
    if not os.path.exists(SETTINGS_FILE):
        create_settings_file()

    # Initialize GoogleAuth with settings
    gauth = GoogleAuth(SETTINGS_FILE)

    # Try to load saved client credentials
    if os.path.exists(CREDENTIALS_FILE):
        gauth.LoadCredentialsFile(CREDENTIALS_FILE)

    if gauth.credentials is None:
        # First time authentication - requires user interaction
        print("🔐 First time setup: Opening browser for authentication...")
        print("📝 Note: This will only happen once (or when tokens expire)")
        gauth.LocalWebserverAuth()

    elif gauth.access_token_expired:
        try:
            # Try to refresh the token automatically
            print("🔄 Refreshing expired access token...")
            gauth.Refresh()
            print("✅ Token refreshed successfully!")

        except Exception as e:
            # If refresh fails, need to re-authenticate
            print(f"❌ Token refresh failed: {e}")
            print("🔐 Re-authenticating... (this happens when refresh token expires)")
            gauth.LocalWebserverAuth()
    else:
        # Use existing valid credentials
        print("✅ Using existing valid credentials")
        gauth.Authorize()

    # Save the current credentials
    gauth.SaveCredentialsFile(CREDENTIALS_FILE)

    return GoogleDrive(gauth)

def check_credentials_status():
    """
    Check the status of stored credentials and warn if they're about to expire.
    This helps you know when manual intervention might be needed.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        print("❌ No credentials file found. First authentication required.")
        return False

    try:
        with open(CREDENTIALS_FILE, 'r') as f:
            creds = json.load(f)

        # Check if we have a refresh token (most important)
        if 'refresh_token' in creds and creds['refresh_token']:
            print("✅ Refresh token present - long-term access enabled")

            # Check access token expiry (less important since it auto-refreshes)

            return True
        else:
            print("⚠️  No refresh token found - may need re-authentication soon")
            return False

    except Exception as e:
        print(f"❌ Error checking credentials: {e}")
        return False

def update_file(file_name):
    """Function to update a single file in Google Drive or create it if it doesn't exist"""
    try:
        drive = get_authenticated_drive()

        # Check if local file exists first
        if not os.path.exists(file_name):
            print(f"❌ Local file {file_name} does not exist")
            return False

        fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        file_title = file_name.split('/')[-1]

        # Try to find and update existing file
        for file in fileList:
            if file['title'] == file_title:
                try:
                    file_to_update = drive.CreateFile({'id': file['id']})
                    file_to_update.SetContentFile(file_name)
                    file_to_update.Upload()
                    print(f"✅ {file_name} was updated in Google Drive")
                    return True
                except Exception as update_error:
                    # If update fails due to permissions, try to create new file
                    print(f"⚠️  Update failed for {file_name}, trying to create new file...")
                    print(f"   Update error: {str(update_error)[:100]}...")
                    break

        # File not found or update failed - create new file
        try:
            new_file = drive.CreateFile({
                'title': file_title,
                'parents': [{'id': folder_id}]
            })
            new_file.SetContentFile(file_name)
            new_file.Upload()
            print(f"✅ {file_name} was created in Google Drive")
            return True
        except Exception as create_error:
            print(f"❌ Failed to create {file_name}: {create_error}")
            return False

    except Exception as e:
        print(f"❌ Error with {file_name}: {e}")
        return False

def update_drive(file_names):
    """Function to update multiple files in Google Drive or create them if they don't exist"""
    try:
        drive = get_authenticated_drive()
        fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

        successful_updates = []
        successful_creates = []
        failed_updates = []

        for file_name in file_names:
            try:
                # Check if local file exists first
                if not os.path.exists(file_name):
                    failed_updates.append(f"{file_name} (local file not found)")
                    continue

                file_title = file_name.split('/')[-1]
                file_found = False
                update_successful = False

                # Try to find and update existing file
                for file in fileList:
                    if file['title'] == file_title:
                        file_found = True
                        try:
                            file_to_update = drive.CreateFile({'id': file['id']})
                            file_to_update.SetContentFile(file_name)
                            file_to_update.Upload()
                            successful_updates.append(file_name)
                            update_successful = True
                            break
                        except Exception as update_error:
                            # If update fails (e.g., permissions), we'll try to create new file below
                            print(f"⚠️  Update failed for {file_name}: {str(update_error)[:100]}...")
                            break

                # If file wasn't found or update failed, create new file
                if not update_successful:
                    try:
                        new_file = drive.CreateFile({
                            'title': file_title,
                            'parents': [{'id': folder_id}]
                        })
                        new_file.SetContentFile(file_name)
                        new_file.Upload()
                        successful_creates.append(file_name)
                    except Exception as create_error:
                        failed_updates.append(f"{file_name} (create failed: {str(create_error)[:50]}...)")

            except Exception as e:
                failed_updates.append(f"{file_name} (error: {str(e)[:50]}...)")

        # Print detailed summary
        total_successful = len(successful_updates) + len(successful_creates)
        print(f"\n📊 Upload Summary:")
        print(f"✅ Total Successful: {total_successful}")
        print(f"   - Updated: {len(successful_updates)}")
        print(f"   - Created: {len(successful_creates)}")
        print(f"❌ Failed: {len(failed_updates)}")

        if successful_updates:
            print("\n🔄 Successfully updated:")
            for file in successful_updates:
                print(f"  • {file}")

        if successful_creates:
            print("\n🆕 Successfully created:")
            for file in successful_creates:
                print(f"  • {file}")

        if failed_updates:
            print("\n❌ Failed operations:")
            for file in failed_updates:
                print(f"  • {file}")

        return len(failed_updates) == 0

    except Exception as e:
        print(f"❌ Critical error in drive update: {e}")
        return False

# Additional utility function
def test_drive_connection():
    """Test the Google Drive connection and token refresh"""
    try:
        print("🧪 Testing Google Drive connection...")
        drive = get_authenticated_drive()

        # Try to list files in the folder
        fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        print(f"✅ Connection successful! Found {len(fileList)} files in folder.")

        return True

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Quick test when running this file directly
    check_credentials_status()
    test_drive_connection()
