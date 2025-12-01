import os
import zipfile
import pandas as pd
from src.utils.utils_functions import record_successful_run


def calculate_incremental_distance(column):
    """Calculate incremental distance from cumulative values"""
    column = column.diff()
    column.fillna(0, inplace=True)
    column = column.clip(lower=0)
    return column


def process_polar_summary(df):
    """Reformats the columns to have similar structure as Garmin output"""
    try:
        # Adding new columns
        df['startTimeLocal'] = pd.to_datetime(df['Date'] + ' ' + df['Start time'])

        # Adapting units (Polar calories are in kcal, convert to joules like Garmin)
        df['Calories'] = df['Calories'] * 4.182

        # Columns to drop (if they exist)
        columns_to_drop = [
            'Name', 'Average pace (min/km)', 'Max pace (min/km)',
            'Fat percentage of calories(%)', 'Average cadence (rpm)',
            'Running index', 'Training load', 'Average power (W)',
            'Max power (W)', 'Notes', 'Height (cm)', 'Weight (kg)',
            'Unnamed: 27', 'Date', 'Start time', 'HR sit'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=columns_to_drop, inplace=True)

        # Renaming columns to match Garmin format
        df.rename(columns={
            'Sport': 'sportType',
            'Duration': 'duration',
            'Total distance (km)': 'distance',
            'Average heart rate (bpm)': 'avgHr',
            'Average speed (km/h)': 'avgSpeed',
            'Max speed (km/h)': 'maxSpeed',
            'Calories': 'calories',
            'Average stride length (cm)': 'avgStrideLength',
            'Ascent (m)': 'elevationGain',
            'Descent (m)': 'elevationLoss',
            'HR max': 'maxHr',
            'VO2max': 'vO2MaxValue'
        }, inplace=True)

        df['Source'] = 'Polar'
        return df

    except Exception as e:
        print(f"❌ Error processing Polar summary: {e}")
        return pd.DataFrame()


def process_polar_splits(df):
    """Reformats the columns to have similar structure as Garmin output"""
    try:
        print("   🔄 Processing splits data (this may take a moment)...")

        # Parse datetime efficiently
        df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Sort by date (already parsed, faster than string sort)
        df.sort_values('date', inplace=True)

        # Calculate incremental values using vectorized operations
        if 'Distances (m)' in df.columns:
            df['SUM_DISTANCE__M'] = df['Distances (m)'].diff().clip(lower=0).fillna(0)
        if 'Altitude (m)' in df.columns:
            df['GAIN_ELEVATION__M'] = df['Altitude (m)'].diff().clip(lower=0).fillna(0)

        # Columns to drop (if they exist)
        columns_to_drop = [
            'Sample rate', 'Date', 'Time', 'Pace (min/km)', 'Altitude (m)',
            'Temperatures (C)', 'Power (W)', 'Distances (m)', 'Unnamed: 11'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=columns_to_drop, inplace=True)

        # Renaming columns to match Garmin format
        df.rename(columns={
            'HR (bpm)': 'WEIGHTED_MEAN_HEARTRATE__BPM',
            'Speed (km/h)': 'WEIGHTED_MEAN_SPEED__KMH',
            'Cadence': 'WEIGHTED_MEAN_RUNCADENCE__BPM',
            'Stride length (m)': 'WEIGHTED_MEAN_STRIDE_LENGTH__M',
        }, inplace=True)

        df['Source'] = 'Polar'
        return df

    except Exception as e:
        print(f"❌ Error processing Polar splits: {e}")
        return pd.DataFrame()


def create_polar_files():
    """
    Main processing function that processes Polar export files.
    Returns True if successful, False otherwise.
    """
    print("⚙️  Processing Polar data...")

    folder_path = 'files/exports/polar_exports'

    if not os.path.exists(folder_path):
        print(f"❌ Polar exports folder not found: {folder_path}")
        return False

    try:
        # Create empty lists to store the dataframes
        summary_dfs = []
        detail_dfs = []

        # Iterate through each file in the folder
        zip_files = [f for f in os.listdir(folder_path) if f.endswith('.zip')]

        if not zip_files:
            print("⚠️  No Polar ZIP files found in exports folder")
            return False

        print(f"📁 Found {len(zip_files)} Polar export files")

        for filename in zip_files:
            file_path = os.path.join(folder_path, filename)

            try:
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    for zip_info in zip_file.infolist():
                        if zip_info.filename.endswith('.csv'):
                            # Read summary data (first row)
                            with zip_file.open(zip_info.filename) as csv_file:
                                df = pd.read_csv(csv_file, nrows=1)
                                Sport = df['Sport'][0]
                                Date = df['Date'][0]
                                summary_dfs.append(df)

                            # Read detailed data (skip first 2 rows)
                            with zip_file.open(zip_info.filename) as csv_file:
                                detail_df = pd.read_csv(csv_file, skiprows=2)
                                detail_df['Sport'] = Sport
                                detail_df['Date'] = Date
                                detail_dfs.append(detail_df)

            except Exception as e:
                print(f"⚠️  Error processing {filename}: {e}")
                continue

        if not summary_dfs:
            print("⚠️  No valid Polar data found")
            return False

        # Combine all dataframes
        summary_df = pd.concat(summary_dfs, ignore_index=True)
        detail_df = pd.concat(detail_dfs, ignore_index=True)

        print(f"📊 Processing {len(summary_df)} activities and {len(detail_df)} detail records")

        # Process and save summary data
        processed_summary = process_polar_summary(summary_df)
        if not processed_summary.empty:
            output_path = 'files/source_processed_files/polar/polar_summary_processed.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_summary.sort_values('startTimeLocal').to_csv(
                output_path, sep='|', index=False, encoding='utf-8'
            )
            print(f"✅ Polar summary processed: {len(processed_summary)} records saved")

        # Process and save splits data
        processed_splits = process_polar_splits(detail_df)
        if not processed_splits.empty:
            output_path = 'files/source_processed_files/polar/polar_details_processed.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_splits.sort_values('date').to_csv(
                output_path, sep='|', index=False, encoding='utf-8'
            )
            print(f"✅ Polar splits processed: {len(processed_splits)} records saved")

        return True

    except Exception as e:
        print(f"❌ Error processing Polar data: {e}")
        return False


def full_polar_pipeline(auto_process_only=False):
    """
    Complete Polar SOURCE pipeline - processes existing export data.

    Args:
        auto_process_only (bool): If True, skips user prompt

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("⌚ POLAR SOURCE PIPELINE")
    print("="*60)

    print("\n⚙️  Processing existing Polar export data...")
    success = create_polar_files()

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Polar source pipeline completed!")
        print("ℹ️  Note: Polar data will be merged when running the Fitness topic pipeline.")
        record_successful_run('source_polar', 'legacy')
    else:
        print("❌ Polar pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    print("⌚ Polar Processing Tool")
    print("This tool processes existing Polar export data.")
    full_polar_pipeline()
