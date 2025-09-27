import pandas as pd
import regex as re
import subprocess
import warnings
import os
import json
from datetime import date
from pandas.errors import PerformanceWarning
from dotenv import load_dotenv
from src.utils.file_operations import clean_rename_move_file
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.utils_functions import record_successful_run
load_dotenv()
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.", category=PerformanceWarning)

def load_user_config():
    """
    Load user configuration from environment variables with defaults.
    
    Returns:
        dict: Configuration dictionary with user parameters
    """
    return {
        'age': int(os.environ.get('NUTRILIO_USER_AGE', '28')),
        'height_cm': int(os.environ.get('NUTRILIO_USER_HEIGHT_CM', '182')),
        'weight_kg': int(os.environ.get('NUTRILIO_USER_WEIGHT_KG', '77')),
        'activity_factor': float(os.environ.get('NUTRILIO_USER_ACTIVITY_FACTOR', '2.0'))
    }


# Legacy meal score preservation functions removed
# All meal scoring is now handled by the USDA pipeline only
col_unnecessary_qty = ['Places', 'Origin', 'Work - location', 'Screen before sleep', 'Sleep - waking up', 'Sleep - night time']
dict_extract_data = {"Food" : "food",
                     "Drinks" : "drinks",
                     "Body sensations" : "body_sensations",
                     "Sleep - dreams" : "dreams",
                     "Work - content" : "work_content",
                     "Self improvement" : "self_improvement",
                     "Social activity" : "social_activity"}

dict_work_duration = {"0-2h" : 1, "0-4h" : 2, "2-3h" : 2.5, "3-4h" : 3.5, "4-6h" : 5, "4h" : 4, "6-8h" : 7,
                      "8-10h" : 9, "8h" : 8}

dict_kcal = {
    'Meat': {'A little': 50, 'Medium': 100, 'A lot': 150},
    'Vegetables': {'A little': 50, 'Medium': 100, 'A lot': 150},
    'Fruits': {'A little': 50, 'Medium': 100, 'A lot': 150},
    'Carbs': {'A little': 50, 'Medium': 100, 'A lot': 150},
    'Dairy': {'A little': 50, 'Medium': 100, 'A lot': 150},
    'Sauces/Spices': {'A little': 5, 'Medium': 10, 'A lot': 15},
    'Veggie alternative': {'A little': 50, 'Medium': 100, 'A lot': 150},
    'Fish': {'A little': 50, 'Medium': 100, 'A lot': 150},
    'Meal category': {'A little': 0, 'Medium': 0, 'A lot': 0},
    'Sweets': {'A little': 10, 'Medium': 20, 'A lot': 30}
}


def generate_calory_needs(config=None):
    """
    Generates the calory needs of a person based on user configuration.
    
    Args:
        config (dict): User configuration dictionary. If None, loads from environment.
        
    Returns:
        dict: Calorie breakdown by meal type
    """
    if config is None:
        config = load_user_config()
    
    age = config['age']
    height_cm = config['height_cm'] 
    weight_kg = config['weight_kg']
    activity_factor = config['activity_factor']
    
    #Calculate the BMR using the Harris-Benedict equation
    bmr = 88.36 + (13.4 * weight_kg) + (4.8 * height_cm) - (5.7 * age)
    # Calculate the daily calorie needs by multiplying the BMR by the activity factor
    daily_calorie_needs = round(bmr * activity_factor)
    # Calculate the calorie breakdown for each meal and snack
    breakfast_calories = round(0.2 * daily_calorie_needs)
    morning_snack_calories = round(0.1 * daily_calorie_needs)
    lunch_calories = round(0.3 * daily_calorie_needs)
    afternoon_snack_calories = round(0.1 * daily_calorie_needs)
    dinner_calories = round(0.3 * daily_calorie_needs)
    night_snack_calories = round(0.05 * daily_calorie_needs)
    # Create a dictionary to store the calorie breakdown
    calorie_breakdown = {
        'Breakfast': breakfast_calories,
        'Morning snack': morning_snack_calories,
        'Lunch': lunch_calories,
        'Afternoon snack': afternoon_snack_calories,
        'Dinner': dinner_calories,
        'Night snack': night_snack_calories}
    return calorie_breakdown

# Legacy ingredient categorization functions removed
# All ingredient scoring is now handled by the USDA pipeline

def score_meal_with_usda(ingredient_list, quantity="medium", use_usda=False):
    """
    Score a meal using USDA nutrition data as an alternative to OpenAI.
    
    Args:
        ingredient_list (str): Comma-separated list of ingredients
        quantity (str): Quantity description (not used in USDA scoring currently)
        use_usda (bool): Whether to use USDA scoring (default False for compatibility)
        
    Returns:
        float: Meal score from 1-10, or None if scoring disabled
    """
    if not use_usda:
        return None
    
    if pd.isna(ingredient_list) or not ingredient_list:
        return None
    
    try:
        # Import USDA scorer - try multiple import paths
        try:
            from src.nutrilio.usda_nutrition_scoring import create_usda_nutrition_scorer
        except ImportError:
            try:
                from .usda_nutrition_scoring import create_usda_nutrition_scorer
            except ImportError:
                import sys
                import os
                # Add current directory to path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                from usda_nutrition_scoring import create_usda_nutrition_scorer
        
        # Create scorer instance
        scorer = create_usda_nutrition_scorer()
        
        # Parse ingredients
        if ',' in ingredient_list:
            if "'" in ingredient_list or '"' in ingredient_list:
                # Format with quotes: "'ingredient1', 'ingredient2'"
                ingredients = [ingredient.strip()[1:-1] for ingredient in ingredient_list.split(',')]
            else:
                # Format without quotes: "ingredient1, ingredient2"
                ingredients = [ingredient.strip() for ingredient in ingredient_list.split(',')]
        else:
            ingredients = [ingredient_list.strip()]
        
        # Score each ingredient and calculate weighted average
        ingredient_scores = []
        sources = []
        
        for ingredient in ingredients:
            if ingredient.strip():
                score, source = scorer.get_ingredient_score(ingredient.strip())
                ingredient_scores.append(score)
                sources.append(source)
        
        if not ingredient_scores:
            return None
        
        # Calculate weighted average (simple average for now)
        meal_score = sum(ingredient_scores) / len(ingredient_scores)
        
        # Log scoring info for debugging
        usda_count = sources.count('usda')
        fallback_count = sources.count('fallback')
        category_count = sources.count('category')
        default_count = sources.count('default')
        
        score_info = f"USDA:{usda_count}, Fallback:{fallback_count}, Category:{category_count}, Default:{default_count}"
        
        return round(meal_score, 2)
        
    except Exception as e:
        print(f"⚠️  Error in USDA meal scoring: {e}")
        return None


def add_usda_meal_scoring_efficient(df, use_usda_scoring=False):
    """
    Add USDA-based meal scoring using efficient ingredient-first approach.
    
    Args:
        df (DataFrame): Input dataframe with meal data
        use_usda_scoring (bool): Whether to enable USDA scoring
        
    Returns:
        DataFrame: Dataframe with USDA scores added (if enabled)
    """
    if not use_usda_scoring:
        print("ℹ️  USDA meal scoring disabled (use_usda_scoring=False)")
        return df
    
    print("🥗 Starting efficient USDA-based meal scoring...")
    
    meal_rows = df[df['Meal'].notna() & (df['Meal'] != '')].copy()
    
    if len(meal_rows) == 0:
        print("ℹ️  No meal data found for USDA scoring")
        return df
    
    try:
        # Import the new efficient functions
        try:
            from src.nutrilio.usda_nutrition_scoring import (
                extract_unique_ingredients_from_dataframe,
                score_all_ingredients,
                calculate_meal_scores_from_ingredients
            )
        except ImportError:
            try:
                from .usda_nutrition_scoring import (
                    extract_unique_ingredients_from_dataframe,
                    score_all_ingredients,
                    calculate_meal_scores_from_ingredients
                )
            except ImportError:
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                from usda_nutrition_scoring import (
                    extract_unique_ingredients_from_dataframe,
                    score_all_ingredients,
                    calculate_meal_scores_from_ingredients
                )
        
        # Phase 1: Extract unique ingredients
        print(f"📊 Found {len(meal_rows)} meal entries to analyze")
        unique_ingredients = extract_unique_ingredients_from_dataframe(df)
        print(f"📋 Extracted {len(unique_ingredients)} unique ingredients from dataset")
        
        # Phase 2: Score all unique ingredients once
        ingredient_scores = score_all_ingredients(unique_ingredients)
        
        # Phase 3: Calculate meal scores using pre-computed ingredient scores
        df = calculate_meal_scores_from_ingredients(df, ingredient_scores)
        
        return df
        
    except Exception as e:
        print(f"❌ Error in efficient USDA meal scoring: {e}")
        print("🔄 Falling back to original method...")
        return add_usda_meal_scoring_legacy(df, use_usda_scoring=True)


def add_usda_meal_scoring_legacy(df, use_usda_scoring=False):
    """
    Legacy USDA meal scoring method (kept as fallback).
    
    Args:
        df (DataFrame): Input dataframe with meal data
        use_usda_scoring (bool): Whether to enable USDA scoring
        
    Returns:
        DataFrame: Dataframe with USDA scores added (if enabled)
    """
    if not use_usda_scoring:
        return df
    
    print("🥗 Using legacy USDA meal scoring method...")
    
    meal_rows = df[df['Meal'].notna() & (df['Meal'] != '')].copy()
    
    if len(meal_rows) == 0:
        return df
    
    # Add USDA scores
    df['usda_meal_score'] = None
    
    scored_count = 0
    total_meals = len(meal_rows)
    
    for idx, row in meal_rows.iterrows():
        usda_score = score_meal_with_usda(
            row.get('food_list', ''),
            row.get('Amount_text', 'medium'),
            use_usda=True
        )
        
        if usda_score is not None:
            df.loc[idx, 'usda_meal_score'] = usda_score
            scored_count += 1
        
        # Progress indicator for large datasets
        if (scored_count + 1) % 50 == 0:
            print(f"   Processed {scored_count + 1}/{total_meals} meals...")
    
    print(f"✅ Legacy USDA scoring complete: {scored_count}/{total_meals} meals scored ({scored_count/total_meals*100:.1f}%)")
    
    return df


def add_usda_drink_scoring_efficient(df, use_usda_scoring=True):
    """
    Add USDA-based drink scoring using efficient drink-first approach.
    
    Args:
        df (DataFrame): Input dataframe with drink data
        use_usda_scoring (bool): Whether to enable USDA scoring
        
    Returns:
        DataFrame: Dataframe with USDA drink scores added (if enabled)
    """
    if not use_usda_scoring:
        print("ℹ️  USDA drink scoring disabled (use_usda_scoring=False)")
        return df
    
    print("🍹 Starting efficient USDA-based drink scoring...")
    
    # Check if we have the drinks_list column (created by extract_data_count)
    if 'drinks_list' not in df.columns:
        print("ℹ️  No 'drinks_list' column found for USDA drink scoring")
        return df
    
    try:
        # Import the new efficient functions
        try:
            from src.nutrilio.usda_drink_scoring import (
                extract_unique_drinks_from_dataframe,
                score_all_drinks_efficient
            )
        except ImportError:
            try:
                from .usda_drink_scoring import (
                    extract_unique_drinks_from_dataframe,
                    score_all_drinks_efficient
                )
            except ImportError:
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                from usda_drink_scoring import (
                    extract_unique_drinks_from_dataframe,
                    score_all_drinks_efficient
                )
        
        # Phase 1: Extract unique drinks
        unique_drinks = extract_unique_drinks_from_dataframe(df)
        
        if not unique_drinks:
            print("ℹ️  No drinks found in dataset")
            return df
            
        print(f"📋 Extracted {len(unique_drinks)} unique drinks from dataset")
        
        # Phase 2: Score all unique drinks once
        drink_scores = score_all_drinks_efficient(unique_drinks)
        
        # Phase 3: Drink scores are now in the database for Power BI export
        print(f"✅ Drink scoring complete - {len(drink_scores)} drinks processed")
        
        return df
        
    except Exception as e:
        print(f"❌ Error in efficient USDA drink scoring: {e}")
        print("🔄 Continuing without drink scoring...")
        return df


# Meal scoring functions have been removed for faster processing
# Historical scores are preserved in JSON format
# USDA-based scoring available as optional alternative (see add_usda_meal_scoring function)

def extract_data_count(df):
    """Retrieves the number of time each ingredient was eaten"""
    pattern = r'([\w ]+) \((\d+)x\)'
    list_col = []
    for column, indicator in dict_extract_data.items():
        for index, row in df.iterrows():
            if row[column] != row[column]:
                continue
            list_elements = []
            matches = re.findall(pattern, row[column])
            start_point = 0
            if indicator == "food":
                start_point +=1
                meal = matches[0][0].strip()
                df.loc[index, "Meal"] = meal
            for match in matches[start_point:]:
                word = match[0].strip()
                value = int(match[1])
                df.at[index, f"{word}_{indicator}"] = value
                list_col.append(f"{word}_{indicator}")
                list_elements.append(word)
            df.loc[index, f"{indicator}_list"] = str(list_elements)[1:-1]
        df.drop(column, axis = 1, inplace = True)
    return df, list_col

def extract_data(column):
    if column != column:
        return None
    return str(column).split('(')[0].strip()

def check_new_drinks(drinks):
    """
    Check for new drinks that are not in the database.
    
    Args:
        drinks (list): List of drink names to check
        
    Returns:
        list: List of new drinks found
    """
    df_drinks = pd.read_excel('files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx')
    new_drinks = []
    
    for drink in drinks:
        if (drink in list(df_drinks.Drink.unique())) | (drink in new_drinks):
            continue
        else:
            new_drinks.append(drink)
    
    if len(new_drinks) == 0:
        print("✅ No new drinks found")
    else:
        print(f"🍹 Found {len(new_drinks)} new drinks:")
        for drink in new_drinks[:10]:  # Show first 10
            print(f"   • {drink}")
        if len(new_drinks) > 10:
            print(f"   ... and {len(new_drinks) - 10} more")
        print("ℹ️  Note: New drinks need to be manually categorized in the drinks database")
        print("    File: files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx")
    
    return new_drinks

def drinks_category(drinks):
    """Generates a ChatGPT prompt to get info's about new drinks added in the Nutrilio app"""
    df_drinks = pd.read_excel('files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx')
    chat_gpt_prompt = ("\n\n\n" + "Please give me your best assessment of the category for each of the drinks below.\n"
                       "Use these 4 categories: soda/soft drinks, strong alcohol, light alcohol, healthy drinks. \n"
                       "Give me as only output a table with 2 columns: Drink and Category, so that I can easily copy paste it in Excel. \n"
                       "Here are the drinks: \n")
    new_drinks_count = 0
    new_drinks_list = ""
    for drink in drinks:
        if drink in list(df_drinks.Drink.unique()):
            continue
        else:
            new_drinks_count += 1
            new_drinks_list += f"{drink} \n"
    if new_drinks_count == 0:
        return "OK"
    if new_drinks_count > 0:
        print(chat_gpt_prompt + new_drinks_list)
        subprocess.Popen(['open', 'files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx'])
    answer = input("Chat GPT output integrated in work file + file saved & closed? (Y/N) \n")
    while answer != 'Y':
        answer = input("Chat GPT output integrated in work file + file saved & closed? (Y/N) \n")

def generate_pbi_files(df, indicator):
    """Generates multiple files to be ingested in PBI"""
    index_list = []
    for col in df.columns:
        if len(col.split('_')) > 1:
            if '_'.join(col.split('_')[1:]) == indicator:
                index_list.append(df.columns.get_loc(col))
        elif col == "Full Date":
            index_list.append(df.columns.get_loc(col))
    sliced_df = df.iloc[:, index_list]
    sliced_df.columns = [col.split('_')[0] for col in sliced_df.columns]
    melted_df = sliced_df.melt(id_vars='Full Date', value_vars=sliced_df.columns[1:], var_name=indicator, value_name='Value')
    melted_df = melted_df[melted_df['Value'] >= 1]
    melted_df.sort_values("Full Date", ascending = False).to_csv(f"files/processed_files/nutrilio_{indicator}_pbi_processed_file.csv", sep = '|', index = False, encoding='utf-16')
    # Note: Drink scoring is now handled automatically in the main pipeline via USDA scoring

def create_optimized_nutrition_file(df):
    """Create optimized nutrition-only CSV for faster frontend performance"""
    print("Creating optimized nutrition file...")
    
    # Filter only meal entries (exclude Daylio and non-meal data)
    meal_df = df[df['Meal'].notna() & (df['Meal'] != '')].copy()
    print(f"Filtered to {len(meal_df)} meal entries from {len(df)} total rows")
    
    # Select only nutrition-relevant columns
    nutrition_cols = [
        'Full Date', 'Time', 'Meal', 'Amount_text', 'food_list', 'Source'
    ]
    
    # Add USDA score column if it exists
    if 'usda_meal_score' in df.columns:
        nutrition_cols.append('usda_meal_score')
    
    meal_df = meal_df[nutrition_cols].copy()
    
    # Create proper timestamp from Full Date and Time
    meal_df['timestamp'] = pd.to_datetime(meal_df['Full Date'] + ' ' + meal_df['Time'], 
                                         format='%Y-%m-%d %H:%M', errors='coerce')
    meal_df['timestamp'] = meal_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Clean food_list - remove quotes and extra spaces
    meal_df['food_items'] = meal_df['food_list'].str.replace("'", "").str.replace('"', '').str.strip()
    
    # Rename columns for frontend consistency
    meal_df.rename(columns={
        'Meal': 'meal_type',
        'Amount_text': 'amount_text'
    }, inplace=True)
    
    # Select final columns in optimized order
    final_cols = [
        'timestamp', 'meal_type', 'amount_text', 'food_items', 'Source'
    ]
    
    # Add USDA score column if it exists
    if 'usda_meal_score' in meal_df.columns:
        final_cols.insert(2, 'usda_meal_score')  # Insert after meal_type
    
    meal_df = meal_df[final_cols]
    
    # Ensure the directory exists
    os.makedirs("files/processed_files/nutrilio", exist_ok=True)
    
    # Save optimized file
    output_path = "files/processed_files/nutrilio/nutrilio_meals_optimized.csv"
    meal_df.to_csv(output_path, sep='|', index=False, encoding='utf-16')
    
    print(f"✅ Created optimized nutrition file: {output_path}")
    print(f"📊 Optimized file contains {len(meal_df)} meal entries")
    print(f"📊 Reduced from {len(nutrition_cols)} to {len(final_cols)} columns")
    
    return output_path

def create_nutrilio_files():
    df = pd.read_csv(f'files/exports/nutrilio_exports/nutrilio_export.csv', sep = ',')
    #Remove quantity when unnecessary
    for col in col_unnecessary_qty:
        df[col] = df[col].apply(lambda x: extract_data(x))
    #Extract all values & quantities
    df, list_col = extract_data_count(df)
    #Put the newly created columns at the end of the df to simplify checks
    columns_to_concat = []
    for _, val in dict_extract_data.items():
        column = df.pop(f"{val}_list")
        columns_to_concat.append(column)
    df = pd.concat([df.copy()] + columns_to_concat, axis=1)
    # Legacy ingredient checking and meal score migration removed
    # All scoring is now handled by the USDA pipeline
    df['Source'] = 'Nutrilio'
    
    print("📅 Merging with Daylio data...")
    try:
        df_daylio = pd.read_csv('files/processed_files/daylio_processed.csv', sep = '|')
        df = pd.concat([df, df_daylio], ignore_index=True).sort_values('Full Date')
        print(f"✅ Successfully merged with {len(df_daylio)} Daylio records")
    except FileNotFoundError:
        print("ℹ️  Daylio processed file not found, continuing without merging")
        print("   (This is normal if Daylio data hasn't been processed yet)")
    except Exception as e:
        print(f"⚠️  Error reading Daylio data: {e}")
        print("   Continuing without Daylio merge...")
    
    # Legacy ingredient categorization removed - now handled by USDA pipeline
    
    # USDA meal scoring enabled - provides free alternative to OpenAI scoring
    print("🥗 Enabling efficient USDA-based meal scoring...")
    df = add_usda_meal_scoring_efficient(df, use_usda_scoring=True)
    
    # USDA drink scoring enabled - automated drink categorization and health scoring
    print("🍹 Enabling efficient USDA-based drink scoring...")
    df = add_usda_drink_scoring_efficient(df, use_usda_scoring=True)
    
    print("⚙️  Finalizing data processing...")
    df['Work_duration_est'] = df["Work - duration_text"].apply(lambda x: dict_work_duration[x] if x in dict_work_duration.keys() else None)
    df['Work - good day_text'] = df['Work - good day_text'].apply(lambda x: "Average" if x =="Ok" else x)
    
    print("💾 Saving main processed file...")
    df.drop(list_col, axis = 1).to_csv("files/processed_files/nutrilio_processed.csv", sep = '|', index = False, encoding='utf-16')
    
    # Create optimized nutrition file for frontend performance
    optimized_nutrition_file = create_optimized_nutrition_file(df)
    
    print("📊 Generating Power BI files...")
    drive_list = ["files/processed_files/nutrilio_processed.csv",
                  optimized_nutrition_file,
                  "files/work_files/nutrilio_work_files/ingredient_scores_database.json",
                  "files/work_files/nutrilio_work_files/flagged_default_ingredients.json",
                  "files/work_files/nutrilio_work_files/drink_scores_database.json",
                  "files/work_files/nutrilio_work_files/flagged_default_drinks.json"]
    
    for _, value in dict_extract_data.items():
        generate_pbi_files(df, value)
        drive_list.append(f"files/processed_files/nutrilio_{value}_pbi_processed_file.csv")
    
    print(f"✅ Processing complete! Generated {len(drive_list)} files.")
    return drive_list



def download_nutrilio_data():
    """
    Opens Nutrilio export instructions and prompts user to download data.
    Returns True if user confirms download, False otherwise.
    """
    print("🥗 Starting Nutrilio data download...")
    
    print("📝 Instructions:")
    print("   1. Open the Nutrilio app on your phone")
    print("   2. Go to Settings → Export Data")
    print("   3. Select 'Export to CSV'")
    print("   4. Choose email or share the file")
    print("   5. Save the file to your Downloads folder")
    print(f"   6. Rename the file to: Nutrilio-export-{date.today().strftime('%Y-%m-%d')}.csv")
    
    response = prompt_user_download_status("Nutrilio")
    return response


def move_nutrilio_files():
    """
    Moves the downloaded Nutrilio file from Downloads to the correct export folder.
    Returns True if successful, False otherwise.
    """
    print("📁 Moving Nutrilio files...")
    
    move_success = clean_rename_move_file(
        export_folder="files/exports/nutrilio_exports",
        download_folder="/Users/valen/Downloads", 
        file_name=f"Nutrilio-export-{date.today().strftime('%Y-%m-%d')}.csv",
        new_file_name="nutrilio_export.csv"
    )
    
    if move_success:
        print("✅ Successfully moved Nutrilio file to exports folder")
    else:
        print("❌ Failed to move Nutrilio file")
    
    return move_success


def create_nutrilio_processed_files():
    """
    Main processing function that processes the Nutrilio data.
    Returns True if successful, False otherwise.
    """
    print("⚙️ Processing Nutrilio data...")
    
    input_path = "files/exports/nutrilio_exports/nutrilio_export.csv"
    
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"❌ Nutrilio file not found: {input_path}")
            return False
        
        # Process the files using the existing function
        nutrilio_files = create_nutrilio_files()
        
        print(f"✅ Successfully processed {len(nutrilio_files)} Nutrilio files")
        print("📊 Generated files:")
        for file_path in nutrilio_files:
            print(f"   • {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing Nutrilio data: {e}")
        return False


def upload_nutrilio_results():
    """
    Uploads the processed Nutrilio files to Google Drive.
    Returns True if successful, False otherwise.
    """
    print("☁️ Uploading Nutrilio results to Google Drive...")
    
    # Define expected output files
    files_to_upload = [
        "files/processed_files/nutrilio_processed.csv",
        "files/processed_files/nutrilio/nutrilio_meals_optimized.csv",
        "files/work_files/nutrilio_work_files/ingredient_scores_database.json",
        "files/work_files/nutrilio_work_files/flagged_default_ingredients.json",
        "files/work_files/nutrilio_work_files/drink_scores_database.json",
        "files/work_files/nutrilio_work_files/flagged_default_drinks.json",
        "files/processed_files/nutrilio_food_pbi_processed_file.csv",
        "files/processed_files/nutrilio_drinks_pbi_processed_file.csv",
        "files/processed_files/nutrilio_body_sensations_pbi_processed_file.csv",
        "files/processed_files/nutrilio_dreams_pbi_processed_file.csv",
        "files/processed_files/nutrilio_work_content_pbi_processed_file.csv",
        "files/processed_files/nutrilio_self_improvement_pbi_processed_file.csv",
        "files/processed_files/nutrilio_social_activity_pbi_processed_file.csv"
    ]
    
    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No files found to upload")
        return False
    
    print(f"📤 Uploading {len(existing_files)} files...")
    success = upload_multiple_files(existing_files)
    
    if success:
        print("✅ Nutrilio results uploaded successfully!")
    else:
        print("❌ Some files failed to upload")
    
    return success


def full_nutrilio_pipeline(auto_full=False):
    """
    Complete Nutrilio pipeline with user choice options.
    
    Options:
    1. Full pipeline (download → move → process → upload)
    2. Download data only (instructions + move files)
    3. Process existing file only
    4. Process existing file and upload to Drive
    
    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🥗 NUTRILIO DATA PIPELINE")
    print("="*60)
    
    if auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → move → process → upload)")
        print("2. Download data only (instructions + move files)")
        print("3. Process existing file only")
        print("4. Process existing file and upload to Drive")
        
        choice = input("\nEnter your choice (1-4): ").strip()
    
    success = False
    
    if choice == "1":
        print("\n🚀 Starting full Nutrilio pipeline...")
        
        # Step 1: Download
        download_success = download_nutrilio_data()
        
        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_nutrilio_files()
        else:
            print("⚠️ Download not confirmed, but checking for existing files...")
            move_success = move_nutrilio_files()
        
        # Step 3: Process (fallback to option 3 if no new files)
        if move_success:
            process_success = create_nutrilio_processed_files()
        else:
            print("⚠️ No new files found, attempting to process existing files...")
            process_success = create_nutrilio_processed_files()
        
        # Step 4: Upload
        if process_success:
            upload_success = upload_nutrilio_results()
            success = upload_success
        else:
            print("❌ Processing failed, skipping upload")
            success = False
    
    elif choice == "2":
        print("\n📥 Download Nutrilio data only...")
        download_success = download_nutrilio_data()
        if download_success:
            success = move_nutrilio_files()
        else:
            success = False
    
    elif choice == "3":
        print("\n⚙️ Processing existing Nutrilio file only...")
        success = create_nutrilio_processed_files()
    
    elif choice == "4":
        print("\n⚙️ Processing existing file and uploading...")
        process_success = create_nutrilio_processed_files()
        if process_success:
            success = upload_nutrilio_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False
    
    else:
        print("❌ Invalid choice. Please select 1-4.")
        return False
    
    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Nutrilio pipeline completed successfully!")
        # Record successful run
        record_successful_run('nutrilio_nutrilio', 'active')
    else:
        print("❌ Nutrilio pipeline failed")
    print("="*60)
    
    return success


# Legacy function for backward compatibility
def process_nutrilio_export(upload="Y"):
    """
    Legacy function for backward compatibility.
    This maintains the original interface while using the new pipeline.
    """
    if upload == "Y":
        return full_nutrilio_pipeline(auto_full=True)
    else:
        return create_nutrilio_processed_files()


if __name__ == "__main__":
    # Allow running this file directly
    print("🥗 Nutrilio Processing Tool")
    print("This tool downloads, processes, and uploads Nutrilio data.")
    
    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️ Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()
    
    # Run the pipeline
    full_nutrilio_pipeline(auto_full=False)
