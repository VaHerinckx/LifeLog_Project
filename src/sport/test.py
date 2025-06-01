import os
import zipfile
import json
import pandas as pd
import fitparse
from datetime import datetime

def dump_fit_file_to_csv(fit_file_path, output_csv_path=None, message_types=None):
    """
    Dump all content from a single FIT file to CSV format

    Args:
        fit_file_path: Path to the .fit file
        output_csv_path: Output CSV file path (optional)
        message_types: List of specific message types to extract (optional)
    """
    if not output_csv_path:
        base_name = os.path.splitext(os.path.basename(fit_file_path))[0]
        output_csv_path = f"{base_name}_dump.csv"

    try:
        fitfile = fitparse.FitFile(fit_file_path)
        all_data = []

        # If no specific message types specified, get all available types
        if not message_types:
            # First pass to discover all message types
            message_types = set()
            for record in fitfile.get_messages():
                if hasattr(record, 'name') and record.name:
                    message_types.add(record.name)

            print(f"Found message types: {sorted(message_types)}")
            fitfile = fitparse.FitFile(fit_file_path)  # Reset file pointer

        # Extract data from specified message types
        for message_type in message_types:
            print(f"Processing {message_type} messages...")

            for record in fitfile.get_messages(message_type):
                record_data = record.get_values()
                record_data['message_type'] = message_type
                record_data['timestamp_processed'] = datetime.now().isoformat()
                all_data.append(record_data)

        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(output_csv_path, sep='|', index=False)
            print(f"Data exported to {output_csv_path}")
            print(f"Total records: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            return df
        else:
            print("No data found in FIT file")
            return None

    except Exception as e:
        print(f"Error processing {fit_file_path}: {str(e)}")
        return None

def dump_specific_message_type(fit_file_path, message_type, output_csv_path=None):
    """
    Dump specific message type from FIT file to CSV

    Args:
        fit_file_path: Path to the .fit file
        message_type: Specific message type to extract (e.g., 'record', 'stress_level', 'heart_rate')
        output_csv_path: Output CSV file path (optional)
    """
    if not output_csv_path:
        base_name = os.path.splitext(os.path.basename(fit_file_path))[0]
        output_csv_path = f"{base_name}_{message_type}.csv"

    try:
        fitfile = fitparse.FitFile(fit_file_path)
        data = []

        for record in fitfile.get_messages(message_type):
            record_data = record.get_values()
            data.append(record_data)

        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_csv_path, sep='|', index=False)
            print(f"{message_type} data exported to {output_csv_path}")
            print(f"Records found: {len(df)}")
            return df
        else:
            print(f"No {message_type} messages found in FIT file")
            return None

    except Exception as e:
        print(f"Error processing {fit_file_path}: {str(e)}")
        return None

def explore_fit_file_structure(fit_file_path):
    """
    Explore the structure of a FIT file to see what message types are available
    """
    try:
        fitfile = fitparse.FitFile(fit_file_path)
        message_counts = {}
        sample_records = {}

        for record in fitfile.get_messages():
            if hasattr(record, 'name') and record.name:
                msg_type = record.name
                message_counts[msg_type] = message_counts.get(msg_type, 0) + 1

                # Store a sample record for each message type
                if msg_type not in sample_records:
                    sample_records[msg_type] = record.get_values()

        print(f"\nFIT file structure for: {fit_file_path}")
        print("=" * 50)
        for msg_type, count in sorted(message_counts.items()):
            print(f"{msg_type}: {count} records")
            if msg_type in sample_records:
                sample_fields = list(sample_records[msg_type].keys())
                print(f"  Sample fields: {sample_fields[:5]}{'...' if len(sample_fields) > 5 else ''}")

        return message_counts, sample_records

    except Exception as e:
        print(f"Error exploring {fit_file_path}: {str(e)}")
        return None, None

# Enhanced version of your original function
def process_stress_level_enhanced():
    """Enhanced function to process stress level data from .fit files and save to CSV"""
    zip_file_path = "files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/UploadedFiles_0-_Part1.zip"

    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/FitFiles")
        os.remove(zip_file_path)

    fit_files = extract_fit_files_path()  # Your existing function
    all_stress_data = []
    count_new_files = 0

    for fit_file in fit_files:
        if fit_file in dict_files.keys():  # Your existing dict_files
            continue

        count_new_files += 1
        print(f"Processing: {fit_file}")

        # First explore the file structure
        msg_counts, samples = explore_fit_file_structure(fit_file)

        # Extract stress level data
        stress_data = dump_specific_message_type(fit_file, 'stress_level',
                                                f"stress_data_{count_new_files}.csv")

        if stress_data is not None:
            all_stress_data.append(stress_data)

        # Mark file as processed
        dict_files[fit_file] = "Yes"

    # Save updated file tracking
    with open(dict_path, 'w') as f:  # Your existing dict_path
        json.dump(dict_files, f)

    print(f"{count_new_files} new fit files processed for stress level data")

    # Combine all stress data
    if all_stress_data:
        combined_df = pd.concat(all_stress_data, ignore_index=True)
        combined_df.to_csv('files/processed_files/all_stress_level_data.csv', sep='|', index=False)
        print(f"Combined stress data saved. Total records: {len(combined_df)}")

# Example usage:
if __name__ == "__main__":
    # Example: Explore a single FIT file
    fit_file_path = "lifelog_python_processing/files/exports/garmin_exports/DI_CONNECT/DI-Connect-Uploaded-Files/FitFiles/valentin.herinckx@gmail.com_174892936136.fit"

    # 1. First explore the structure
    explore_fit_file_structure(fit_file_path)

    # 2. Dump all content
    dump_fit_file_to_csv(fit_file_path, "complete_dump.csv")

    # 3. Dump specific message type
    dump_specific_message_type(fit_file_path, "record", "activity_records.csv")
    dump_specific_message_type(fit_file_path, "stress_level", "stress_data.csv")
