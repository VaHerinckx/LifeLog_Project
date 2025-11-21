import pandas as pd
import os
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.weather.weather_processing import full_weather_pipeline


def generate_weather_website_page_files(df):
    """Generate website-optimized files for the Weather page."""
    print(f"\n🌐 Generating website files for Weather page...")

    try:
        website_dir = 'files/website_files/weather'
        os.makedirs(website_dir, exist_ok=True)

        df_web = df.copy()
        df_web = enforce_snake_case(df_web, "weather_page_data")

        website_path = f'{website_dir}/weather_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Website file: {len(df_web):,} records → {website_path}")

        return True

    except Exception as e:
        print(f"❌ Error generating website files: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_weather_topic_file():
    """Creates the Weather topic file by reading weather source data."""
    print(f"⚙️  Creating Weather topic files...")

    source_file = "files/source_processed_files/weather/weather_processed.csv"
    topic_output_file = "files/topic_processed_files/weather/weather_processed.csv"

    try:
        if not os.path.exists(source_file):
            print(f"❌ Source file not found: {source_file}")
            print(f"   Run the weather source pipeline first.")
            return False

        print(f"📖 Reading source data from {source_file}...")
        df = pd.read_csv(source_file, sep='|', encoding='utf-8')
        print(f"✅ Loaded {len(df)} records")

        os.makedirs(os.path.dirname(topic_output_file), exist_ok=True)
        df.to_csv(topic_output_file, sep='|', index=False, encoding='utf-8')
        print(f"💾 Saved topic file to {topic_output_file}")

        website_success = generate_weather_website_page_files(df)
        return website_success

    except Exception as e:
        print(f"❌ Error creating Weather topic files: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_weather_results():
    """Uploads the processed Weather files to Google Drive."""
    print(f"☁️  Uploading Weather results to Google Drive...")

    files_to_upload = ['files/website_files/weather/weather_page_data.csv']
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No files found to upload")
        return False

    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)

    if success:
        print(f"✅ Weather results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")

    return success


def full_weather_pipeline(auto_full=False, auto_process_only=False, skip_source=False):
    """Complete Weather TOPIC pipeline with 3 standard options."""
    print("\n" + "="*60)
    print("🌤️ WEATHER TOPIC PIPELINE")
    print("="*60)

    if auto_process_only:
        choice = "2"
    elif auto_full:
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Run source pipeline, create topic files, and upload to Drive")
        print("2. Create topic files from existing source data and upload to Drive")
        print("3. Upload existing topic/website files to Drive")
        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print(f"\n🚀 Running full Weather pipeline...")
        if not skip_source:
            print(f"\n📥 Step 1: Running weather source pipeline...")
            source_success = full_weather_pipeline(auto_process_only=True)
            if not source_success:
                print("⚠️  Source pipeline failed, but attempting to use existing source data...")
        
        print(f"\n📊 Step 2: Creating Weather topic files...")
        topic_success = create_weather_topic_file()

        if topic_success:
            print("\n☁️  Step 3: Uploading to Drive...")
            success = upload_weather_results()
        else:
            print("❌ Topic file creation failed, skipping upload")

    elif choice == "2":
        print("\n⚙️  Creating topic files from existing source data and uploading...")
        topic_success = create_weather_topic_file()
        if topic_success:
            success = upload_weather_results()

    elif choice == "3":
        print("\n⬆️  Uploading existing files to Drive...")
        success = upload_weather_results()

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    print("\n" + "="*60)
    if success:
        print(f"✅ Weather topic pipeline completed successfully!")
        record_successful_run('topic_weather', 'active')
    else:
        print(f"❌ Weather topic pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    print(f"🌤️ Weather Topic Processing Tool")
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()
    full_weather_pipeline(auto_full=False)
