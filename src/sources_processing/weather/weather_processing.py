import requests
import json
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import os
import pandas as pd
from src.utils.utils_functions import time_difference_correction
# Drive operations removed - handled by topic coordinator

load_dotenv()

# Load Meteostat API credentials from .env file
api_key = os.environ['Meteostats_API_KEY']
api_host = os.environ['Meteostats_API_HOST']

# Create a pandas Timedelta object of 30 days
ONE_MONTH = pd.Timedelta('30 days')

def max_30_days_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes a pandas dataframe and return another one where the time difference between the start and
    end dates is limited to 30 days or less, as the API doesn't allow requests for longer time periods"""
    # List to store new rows
    new_rows = []
    for _, row in df.iterrows():
        delta = row['end_date'] - row['start_date']
        # If delta is greater than 30 days, then we split the delta in parts no larger than 30 days
        if delta > ONE_MONTH:
            curr_date = row['start_date']
            while curr_date < row['end_date']:
                new_end = min(curr_date + ONE_MONTH, row['end_date'])
                new_rows.append([curr_date, new_end, row['location']])
                curr_date = new_end + pd.Timedelta('1 day')
        else:
            new_rows.append([row['start_date'], row['end_date'], row['location']])
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    return new_df

def lat_long_location(location_name: str):
    """This function takes a location name and returns its latitude and longitude coordinates using Geopy"""
    # Creating a geolocator object using Nominatim
    geolocator = Nominatim(user_agent='my_application')
    # Getting the location object
    location = geolocator.geocode(location_name)
    # Extracting the latitude and longitude coordinates from the location object
    latitude = location.latitude
    longitude = location.longitude
    return latitude, longitude

def api_call_nearest_station(latitude: float, longitude: float):
    """This function takes a latitude and longitude coordinates of a location and makes an API call to Meteostat API to get the ID of the nearest weather station"""
    # Define the API endpoint and set the query parameters
    url = "https://meteostat.p.rapidapi.com/stations/nearby"
    querystring = {"lat":latitude,"lon":longitude, "radius": 1000000}
    # Set the headers for the API request
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": api_host
    }
    # Create a requests session and make the API call
    with requests.Session() as session:
        response = session.get(url, headers=headers, params=querystring)
    # Extract the station ID and name for the nearest weather station from the API response
    stations = json.loads(response.text)['data']
    stations_dict = {station['id']: station['name']['en'] for station in stations}
    return stations_dict

def api_call_weather_data(stations_dict: dict, start_date: str, end_date: str):
    """Calls the Meteostat API to get hourly weather data for a list of stations within a specified date range"""
    url = "https://meteostat.p.rapidapi.com/stations/hourly"
    with requests.Session() as session:
        for station in stations_dict.keys():
            querystring = {"station":station,"start":start_date,"end":end_date}
            headers = { "X-RapidAPI-Key": api_key,
                        "X-RapidAPI-Host": api_host }
            print(querystring)
            response = session.get(url, headers=headers, params=querystring)
            weather_data = json.loads(response.text)['data']
            if weather_data:
                station_id = station
                station_name = stations_dict[station]
                return weather_data, station_id, station_name

def dict_weather_code():
    dict_weather_code = {1:'Clear', 2:'Fair', 3:'Cloudy', 4:'Overcast', 5:'Fog', 6:'Freezing Fog', 7:'Light Rain',\
                         8:'Rain', 9:'Heavy Rain', 10:'Freezing Rain', 11:'Heavy Freezing Rain', 12:'Sleet',\
                         13:'Heavy Sleet', 14:'Light Snowfall', 15:'Snowfall', 16:'Heavy Snowfall', 17:'Rain Shower',\
                         18:'Heavy Rain Shower', 19:'Sleet Shower', 20:'Heavy Sleet Shower', 21:'Snow Shower', 22:'Heavy Snow Shower',\
                         23:'Lightning', 24:'Hail', 25:'Thunderstorm', 26:'Heavy Thunderstorm', 27:'Storm'}
    return dict_weather_code

def create_weather_file():
    # Read the location log file to obtain the start and end dates for each location
    df = pd.read_excel('files/work_files/weather_work_files/location_log.xlsx')

    # Filter out any rows with end dates greater than 30 days ago
    df = max_30_days_timedeltas(df)

    # Define the column names for the weather data DataFrame
    col_names = ['location', 'closest_station_id', 'closest_station_name', 'utc_timestamp', 'temperature',
                 'dew_point', 'relative_humidity_%', 'precipitation', 'snow_depth', 'wind_direction', 'wind_speed',
                 'peak_wind_gust', 'sea_level_air_pressure', 'sunshine_total_time','weather_condition_code']

    # Create an empty DataFrame to store the weather data
    df_weather = pd.DataFrame(columns=col_names, index=range(0,0))

    # Read in the alreadu processed weather data
    df_weather_processed = pd.read_csv('files/source_processed_files/weather/weather_processed.csv', sep = '|')

    # Iterate over each row in the location log file
    for _, row in df.iterrows():

        # Check if the end date for this location has already been processed
        if (pd.to_datetime(df_weather_processed['utc_timestamp']) == row['end_date']).any():
            continue
        # Retrieve the necessary information for this location
        location_name = row['location']
        start_date = row['start_date'].strftime("%Y-%m-%d")
        end_date = row['end_date'].strftime("%Y-%m-%d")
        latitude, longitude = lat_long_location(location_name)
        stations_dict = api_call_nearest_station(latitude, longitude)
        weather_data, station_id, station_name = api_call_weather_data(stations_dict, start_date, end_date)

        # Convert the weather data for this location into a DataFrame and append it to the main DataFrame
        hourly_data = []
        for data_point in weather_data:
            hourly_data.append([location_name, station_id,station_name,data_point['time'],data_point['temp'],
                                data_point['dwpt'],data_point['rhum'],data_point['prcp'],data_point['snow'],
                                data_point['wdir'],data_point['wspd'],data_point['wpgt'],data_point['pres'],
                                data_point['tsun'],data_point['coco']])
        hourly_df = pd.DataFrame(hourly_data, columns=col_names)
        df_weather = pd.concat([df_weather, hourly_df], ignore_index=True)

    # Add additional columns to the weather data DataFrame
    df_weather['weather_assessment'] = df_weather['weather_condition_code'].map(dict_weather_code())
    df_weather['utc_timestamp'] = pd.to_datetime(df_weather['utc_timestamp'])
    df_weather['tz_timestamp'] = df_weather['utc_timestamp'].apply(lambda x: time_difference_correction(x))

    # Combine the newly processed data with the existing data and remove any duplicate rows
    df_weather = pd.concat([df_weather_processed, df_weather], ignore_index=True)
    df_weather.drop_duplicates(inplace=True)

    # Save to a CSV file
    os.makedirs('files/source_processed_files/weather', exist_ok=True)
    df_weather.to_csv('files/source_processed_files/weather/weather_processed.csv', sep = '|', index=False, encoding='utf-8')


def full_weather_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Weather SOURCE pipeline with 2 options.

    Options:
    1. Download new location data and process weather data
    2. Process weather data from existing location log

    Args:
        auto_full (bool): If True, automatically runs option 1
        auto_process_only (bool): If True, automatically runs option 2

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("🌤️  WEATHER SOURCE PIPELINE (Meteostat)")
    print("="*60)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing location data...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new location data and process weather data")
        print("2. Process weather data from existing location log")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Running full Weather source pipeline...")
        print("⚠️  Note: You need to manually update the location log file first")
        print("   File: files/work_files/weather_work_files/location_log.xlsx")

        proceed = input("\nHave you updated the location log? (Y/N): ").upper()
        if proceed != 'Y':
            print("❌ Please update location log first, then run again")
            return False

        print("\n📊 Processing weather data...")
        try:
            create_weather_file()
            print("✅ Weather source processing completed!")
            success = True
        except Exception as e:
            print(f"❌ Weather processing failed: {e}")
            import traceback
            traceback.print_exc()
            success = False

    elif choice == "2":
        print("\n⚙️  Processing weather data from existing location log...")
        try:
            create_weather_file()
            print("✅ Weather source processing completed!")
            success = True
        except Exception as e:
            print(f"❌ Weather processing failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    print("\n" + "="*60)
    if success:
        print("✅ Weather source pipeline completed!")
        print(f"📁 Output: files/source_processed_files/weather/weather_processed.csv")
    else:
        print("❌ Weather source pipeline failed")
    print("="*60)

    return success


if __name__ == "__main__":
    print("🌤️  Weather Source Processing Tool")
    print("This tool processes weather data from Meteostat API.")
    full_weather_pipeline(auto_full=False)
