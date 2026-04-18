"""
Manual Location Source Processor

This module processes manual Excel travel records to extract hourly location data.
Follows the source processor pattern: 1 option (process only - no download needed).
Does NOT upload to Drive (handled by Location topic coordinator).

Input: files/work_files/GMT_timediff.xlsx
Output: files/source_processed_files/manual_location/manual_location_processed.csv
"""

import pandas as pd
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List

from src.utils.utils_functions import enforce_snake_case
from src.utils.geocoding_utils import (
    load_geocoding_cache,
    save_geocoding_cache,
    geocode_location,
    calculate_timezone_offset
)


# Global geocoding cache for manual processing
_geocoding_cache = {}
from src.utils.logger import log


# ============================================================================
# EXCEL READING FUNCTIONS
# ============================================================================

def read_manual_timezone_excel(file_path: str) -> List[Dict]:
    """
    Read the manual timezone Excel file and parse travel records.

    Args:
        file_path: Path to the Excel file

    Returns:
        List of travel records with parsed dates and locations
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path)

        log.progress(f"Loaded {len(df)} travel records from Excel")
        log.debug(f"Columns: {list(df.columns)}")

        # Parse records
        travel_records = []

        for _, row in df.iterrows():
            try:
                # Parse date (handle different formats)
                date_str = str(row['Date'])

                # Try different date parsing approaches
                try:
                    # Try pandas date parsing first
                    date = pd.to_datetime(date_str).date()
                except:
                    # Try manual parsing for formats like "1/1/23"
                    if '/' in date_str:
                        parts = date_str.split('/')
                        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])

                        # Handle 2-digit years
                        if year < 100:
                            if year < 50:  # Assume 2000s
                                year += 2000
                            else:  # Assume 1900s
                                year += 1900

                        date = datetime(year, month, day).date()
                    else:
                        log.warning(f"Could not parse date: {date_str}")
                        continue

                # Extract location information
                country = str(row['Location']).strip()
                city = str(row['City']).strip() if 'City' in row else country

                travel_records.append({
                    'date': date,
                    'country': country,
                    'city': city
                })

            except Exception as e:
                log.warning(f"Error parsing row: {row} - {e}")
                continue

        # Sort by date
        travel_records.sort(key=lambda x: x['date'])

        log.success(f"Successfully parsed {len(travel_records)} travel records")

        # Show sample
        if travel_records:
            log.debug("Sample records:")
            for record in travel_records[:3]:
                log.info(f"• {record['date']} - {record['city']}, {record['country']}")

        return travel_records

    except Exception as e:
        log.error(f"Error reading Excel file: {e}")
        return []


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def create_hourly_records_from_travel_data(travel_records: List[Dict]) -> List[Dict]:
    """
    Create hourly timezone records from travel data.
    Fills gaps between travel dates.

    Args:
        travel_records: List of travel records from Excel

    Returns:
        List of hourly records with timezone and location information
    """
    global _geocoding_cache

    if not travel_records:
        log.error("No travel records to process")
        return []

    log.progress("Creating hourly records from travel data...")

    # Load existing geocoding cache from disk
    log.progress("Loading geocoding cache...")
    _geocoding_cache = load_geocoding_cache()

    # Pre-process all locations to get coordinates and timezones
    log.progress("Pre-processing location data...")
    location_data = {}
    new_locations_geocoded = 0

    for record in travel_records:
        location_key = f"{record['city']}, {record['country']}"

        if location_key not in location_data:
            # Check cache first
            if location_key in _geocoding_cache:
                log.progress(f"Using cached data for {location_key}")
                location_data[location_key] = _geocoding_cache[location_key]
            else:
                # Get fresh data
                log.progress(f"Geocoding {location_key}...")
                coordinates, timezone_id = geocode_location(record['city'], record['country'])
                location_data[location_key] = {
                    'coordinates': coordinates,
                    'timezone_id': timezone_id
                }
                # Cache the result
                _geocoding_cache[location_key] = location_data[location_key]
                new_locations_geocoded += 1

                # Be respectful to APIs
                time.sleep(1)

    # Save updated cache to disk
    if new_locations_geocoded > 0:
        log.success(f"Saving {new_locations_geocoded} new location(s) to cache...")
        save_geocoding_cache(_geocoding_cache)

    # NORMAL log: geocoding scope
    log.normal(f"Manual location geocoding: {new_locations_geocoded} new locations, {len(location_data) - new_locations_geocoded} cached")

    log.success(f"Pre-processed {len(location_data)} unique locations ({len(location_data) - new_locations_geocoded} cached, {new_locations_geocoded} new)")

    # Determine the overall date range
    start_date = travel_records[0]['date']
    end_date = travel_records[-1]['date']

    log.info(f"📅 Date range: {start_date} to {end_date}")

    # Create hourly records
    hourly_records = []
    current_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Keep track of current location
    current_location = travel_records[0]
    travel_index = 0

    while current_datetime <= end_datetime:
        # Check if we need to update location based on travel dates
        while (travel_index + 1 < len(travel_records) and
               current_datetime.date() >= travel_records[travel_index + 1]['date']):
            travel_index += 1
            current_location = travel_records[travel_index]
            log.info(f"📍 Location changed to {current_location['city']}, {current_location['country']} on {current_datetime.date()}")

        # Get location data
        location_key = f"{current_location['city']}, {current_location['country']}"
        loc_data = location_data[location_key]

        # Calculate timezone offset for this specific datetime
        timezone_offset = calculate_timezone_offset(loc_data['timezone_id'], current_datetime)

        # Create record
        record = {
            'timestamp': current_datetime.isoformat(),
            'timezone': timezone_offset,
            'city': current_location['city'],
            'country': current_location['country'],
            'is_home': False,  # Always False for manual entries
            'coordinates': f"geo:{loc_data['coordinates']}"
        }

        hourly_records.append(record)
        current_datetime += timedelta(hours=1)

    log.success(f"Created {len(hourly_records):,} hourly records")

    # Show timezone summary
    timezone_summary = {}
    for record in hourly_records:
        location_key = f"{record['city']}, {record['country']}"
        if location_key not in timezone_summary:
            timezone_summary[location_key] = set()
        timezone_summary[location_key].add(record['timezone'])

    log.progress("Timezone summary:")
    for location, timezones in timezone_summary.items():
        timezones_str = ", ".join(sorted(timezones))
        log.info(f"• {location}: {timezones_str}")

    return hourly_records


def create_manual_location_file():
    """
    Main function to process manual timezone Excel file and create processed CSV.

    Returns:
        bool: True if successful, False otherwise
    """
    log.info("\n" + "="*70)
    log.info("MANUAL LOCATION PROCESSING")
    log.info("="*70)

    # Define file paths
    input_path = "files/work_files/GMT_timediff.xlsx"
    output_path = 'files/source_processed_files/manual_location/manual_location_processed.csv'

    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            log.error(f"Excel file not found: {input_path}")
            return False

        # Read and parse travel data
        travel_records = read_manual_timezone_excel(input_path)

        if not travel_records:
            log.error("No valid travel records found")
            return False

        # Create hourly records
        hourly_records = create_hourly_records_from_travel_data(travel_records)

        if not hourly_records:
            log.error("No hourly records created")
            return False

        # Convert to DataFrame
        df = pd.DataFrame(hourly_records)

        # Add source column
        df['source'] = 'manual'

        # Enforce snake_case
        df = enforce_snake_case(df, "manual_location_processed")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV with pipe separator and UTF-8 encoding
        log.success(f"Saving processed data to {output_path}...")
        df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

        log.success(f"Successfully processed manual location data!")
        log.progress(f"Created {len(df):,} hourly records")
        log.info(f"📅 Date range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
        log.progress(f"Countries: {', '.join(df['country'].unique())}")
        log.info(f"🏙️  Cities: {', '.join(df['city'].unique())}")

        # Show sample records
        log.debug(f"\n Sample records:")
        sample_df = df.head(5)[['timestamp', 'timezone', 'city', 'country']]
        for _, row in sample_df.iterrows():
            log.info(f"• {row['timestamp'][:16]} | {row['timezone']} | {row['city']}, {row['country']}")

        # NORMAL log: end summary
        log.normal_success(f"Manual location: {len(df):,} hourly records processed")

        return True

    except Exception as e:
        log.error(f"Error processing manual location data: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_manual_location_pipeline():
    """
    Complete Manual Location SOURCE processor pipeline.

    Manual location data is entered directly in Excel, so there is only one option:
    1. Process existing data

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    log.info("\n" + "="*70)
    log.info("MANUAL LOCATION SOURCE PROCESSOR PIPELINE")
    log.info("="*70)

    log.info("\n⚙️  Processing manual travel data from Excel...")

    success = create_manual_location_file()

    # Final status
    log.info("\n" + "="*70)
    if success:
        log.success("Manual location source processor completed successfully!")
        log.progress("Output: files/source_processed_files/manual_location/manual_location_processed.csv")
        log.info("Next: Run Location topic coordinator to merge and upload")
    else:
        log.error("Manual location processing failed")
    log.info("="*70)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    log.info("Manual Location Source Processor")
    log.info("This processor handles manual Excel travel records.")
    log.info("Upload is handled by Location topic coordinator.\n")

    # Run the pipeline
    full_manual_location_pipeline()
