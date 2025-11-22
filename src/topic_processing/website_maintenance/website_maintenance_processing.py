"""
Website Maintenance Processing
Handles uploading utility and tracking files needed by the website.
"""

import os
import pandas as pd
from datetime import datetime
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run


def create_website_maintenance_file():
    """
    Creates website maintenance file with both refresh dates and latest data dates.
    Combines data from last_successful_runs.csv and website files.

    Output: files/website_files/web_maintenance/maintenance_web_data.csv (pipe-delimited)

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n📊 Creating website maintenance file...")

    # Topic file mappings: topic_name -> (file_path, date_column)
    topic_files = {
        'topic_reading': ('files/website_files/reading/reading_page_books.csv', 'timestamp'),
        'topic_movies': ('files/website_files/movies/movies_page_letterboxd_data.csv', 'date'),
        'topic_music': ('files/website_files/music/music_page_data.csv', 'timestamp'),
        'topic_podcasts': ('files/website_files/podcasts/podcasts_page_data.csv', 'listened_date'),
        'topic_nutrition': ('files/website_files/nutrition/nutrition_page_data.csv', 'date'),
        'topic_health': ('files/website_files/health/health_page_daily.csv', 'date'),
        'topic_shows': ('files/website_files/shows/shows_page_data.csv', 'watched_at'),
        'topic_finance': ('files/website_files/finance/finance_page_data.csv', 'date')
    }

    try:
        # Read the tracking file
        tracking_file = 'files/tracking/last_successful_runs.csv'
        if not os.path.exists(tracking_file):
            print(f"❌ Tracking file not found: {tracking_file}")
            return False

        df_tracking = pd.read_csv(tracking_file)
        print(f"✅ Loaded tracking data: {len(df_tracking)} records")

        # Add latest_data_date column
        latest_dates = []

        for _, row in df_tracking.iterrows():
            source_name = row['source_name']
            latest_date = None

            # Only process topic files
            if source_name in topic_files:
                file_path, date_column = topic_files[source_name]

                if os.path.exists(file_path):
                    try:
                        # Read the topic file with pipe delimiter
                        df_topic = pd.read_csv(file_path, sep='|', encoding='utf-8')

                        if date_column in df_topic.columns:
                            # Get the latest date
                            df_topic[date_column] = pd.to_datetime(df_topic[date_column], errors='coerce')
                            max_date = df_topic[date_column].max()

                            if pd.notna(max_date):
                                latest_date = max_date.strftime('%Y-%m-%d %H:%M:%S')
                                print(f"   ✓ {source_name}: Latest data = {latest_date}")
                            else:
                                print(f"   ⚠️  {source_name}: No valid dates found")
                        else:
                            print(f"   ⚠️  {source_name}: Column '{date_column}' not found")
                    except Exception as e:
                        print(f"   ⚠️  {source_name}: Error reading file - {e}")
                else:
                    print(f"   ⚠️  {source_name}: File not found - {file_path}")

            latest_dates.append(latest_date if latest_date else '')

        # Add the latest_data_date column
        df_tracking['latest_data_date'] = latest_dates

        # Reorder columns
        df_tracking = df_tracking[['source_name', 'last_successful_run', 'latest_data_date', 'status', 'pipeline_type']]

        # Create output directory if it doesn't exist
        output_dir = 'files/website_files/web_maintenance'
        os.makedirs(output_dir, exist_ok=True)

        # Save with pipe delimiter
        output_file = os.path.join(output_dir, 'maintenance_web_data.csv')
        df_tracking.to_csv(output_file, sep='|', index=False, encoding='utf-8')

        print(f"\n✅ Website maintenance file created: {output_file}")
        print(f"   Total records: {len(df_tracking)}")
        print(f"   Records with latest data: {len([d for d in latest_dates if d])}")

        return True

    except Exception as e:
        print(f"❌ Error creating website maintenance file: {e}")
        return False


def full_website_maintenance_pipeline(auto_mode=False):
    """
    Website Maintenance pipeline - Creates and uploads maintenance file.

    Creates maintenance_web_data.csv with both last refresh dates and latest data dates,
    then uploads to Google Drive.

    Args:
        auto_mode (bool): If True, runs without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🌐 WEBSITE MAINTENANCE PIPELINE")
    print("="*60)

    # Verify Drive connection first
    print("\n🔍 Verifying Google Drive connection...")
    if not verify_drive_connection():
        print("❌ Cannot connect to Google Drive. Please check credentials.")
        return False

    # Create the maintenance file
    if not create_website_maintenance_file():
        print("\n" + "="*60)
        print("❌ WEBSITE MAINTENANCE PIPELINE FAILED")
        print("="*60)
        return False

    # Upload to Drive
    maintenance_file = 'files/website_files/web_maintenance/maintenance_web_data.csv'
    print(f"\n📤 Uploading {maintenance_file}...")

    success = upload_multiple_files([maintenance_file])

    if success:
        print("\n✅ Maintenance file uploaded successfully!")
        print("\n📝 Next steps:")
        print("   1. Copy the file ID from the upload output above")
        print("   2. Add it to lifelog_website/.env:")
        print("      VITE_TRACKING_FILE_ID=<file_id_from_output>")

        record_successful_run('website_maintenance', 'active')

        print("\n" + "="*60)
        print("✅ WEBSITE MAINTENANCE PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
    else:
        print("❌ Failed to upload maintenance file")
        print("\n" + "="*60)
        print("❌ WEBSITE MAINTENANCE PIPELINE FAILED")
        print("="*60)

    return success


if __name__ == "__main__":
    full_website_maintenance_pipeline()
