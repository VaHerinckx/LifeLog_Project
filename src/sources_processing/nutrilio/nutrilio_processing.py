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
# Drive operations not needed - source processor doesn't upload
from src.utils.web_operations import open_web_urls, prompt_user_download_status
from src.utils.utils_functions import record_successful_run, enforce_snake_case

from src.sources_processing.nutrilio.usda_nutrition_scoring import (
    create_usda_nutrition_scorer,
    extract_unique_ingredients_from_dataframe,
    score_all_ingredients,
    calculate_meal_scores_from_ingredients
)
from src.sources_processing.nutrilio.usda_drink_scoring import (
    extract_unique_drinks_from_dataframe,
    score_all_drinks_efficient
)

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

# Valid meal types for validation
VALID_MEAL_TYPES = ["Breakfast", "Morning snack", "Lunch", "Afternoon snack", "Dinner", "Night snack"]


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
        # Create scorer instance (imports are now at top of file)
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

    meal_rows = df[df['meal'].notna() & (df['meal'] != '')].copy()

    if len(meal_rows) == 0:
        print("ℹ️  No meal data found for USDA scoring")
        return df

    try:
        # Use imports from top of file
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

    meal_rows = df[df['meal'].notna() & (df['meal'] != '')].copy()

    if len(meal_rows) == 0:
        return df

    # Add USDA scores
    df['usda_meal_score'] = None

    scored_count = 0
    total_meals = len(meal_rows)

    for idx, row in meal_rows.iterrows():
        usda_score = score_meal_with_usda(
            row.get('food_list', ''),
            row.get('amount_text', 'medium'),
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
        # Use imports from top of file
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

def extract_data_count(df, invalid_meals_list=None):
    """Retrieves the number of time each ingredient was eaten

    Args:
        df: DataFrame with nutrilio data
        invalid_meals_list: Optional list to collect invalid meal entries

    Returns:
        tuple: (df, list_col, invalid_meals_list)
    """
    if invalid_meals_list is None:
        invalid_meals_list = []

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
                df.loc[index, "meal"] = meal

                # Validate meal type
                if meal not in VALID_MEAL_TYPES:
                    invalid_entry = {
                        'date': row.get('Full Date', row.get('date', 'Unknown')),
                        'time': row.get('Time', 'Unknown'),
                        'meal': meal
                    }
                    invalid_meals_list.append(invalid_entry)
            for match in matches[start_point:]:
                word = match[0].strip()
                value = int(match[1])
                df.at[index, f"{word}_{indicator}"] = value
                list_col.append(f"{word}_{indicator}")
                list_elements.append(word)
            df.loc[index, f"{indicator}_list"] = str(list_elements)[1:-1]
        df.drop(column, axis = 1, inplace = True)
    return df, list_col, invalid_meals_list

def classify_row(row):
    """
    Classify a row as 'daily_summary' or 'meal' based on its content.

    Args:
        row: DataFrame row

    Returns:
        str: 'daily_summary' or 'meal'
    """
    # If row has daily_summary flag set to 'yes', it's a daily summary row
    if pd.notna(row.get('daily_summary')) and str(row['daily_summary']).lower() == 'yes':
        return 'daily_summary'

    # If row has meal type, it's a meal row
    if pd.notna(row.get('meal')):
        return 'meal'

    # Default to daily_summary if unclear
    return 'daily_summary'

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
        elif col == "date":
            index_list.append(df.columns.get_loc(col))
    sliced_df = df.iloc[:, index_list]
    sliced_df.columns = [col.split('_')[0] for col in sliced_df.columns]
    melted_df = sliced_df.melt(id_vars='date', value_vars=sliced_df.columns[1:], var_name=indicator, value_name='Value')
    melted_df = melted_df[melted_df['Value'] >= 1]
    melted_df.sort_values("date", ascending = False).to_csv(f"files/source_processed_files/nutrilio/nutrilio_{indicator}_pbi_processed_file.csv", sep = '|', index = False, encoding='utf-8')
    # Note: Drink scoring is now handled automatically in the main pipeline via USDA scoring

def display_meal_validation_report(invalid_meals_list):
    """
    Display a formatted report of invalid meal entries.

    Args:
        invalid_meals_list: List of dicts with 'date', 'time', 'meal' keys
    """
    if not invalid_meals_list:
        print("✅ All meal types are valid!")
        return

    print("\n" + "="*70)
    print("⚠️  MEAL VALIDATION REPORT")
    print("="*70)
    print(f"\nFound {len(invalid_meals_list)} meal(s) with invalid meal types.")
    print(f"Valid meal types: {', '.join(VALID_MEAL_TYPES)}")
    print("\nPlease add the correct meal type tag in the Nutrilio app for:")
    print("\n" + "-"*70)
    print(f"{'Date':<12} {'Time':<8} {'Invalid Meal Type':<40}")
    print("-"*70)

    # Sort by date descending (most recent first)
    sorted_meals = sorted(invalid_meals_list, key=lambda x: (x['date'], x['time']), reverse=True)

    for entry in sorted_meals:
        date_str = str(entry['date'])[:10] if entry['date'] != 'Unknown' else 'Unknown'
        time_str = str(entry['time'])[:5] if entry['time'] != 'Unknown' else 'Unknown'
        meal_str = str(entry['meal'])[:40]
        print(f"{date_str:<12} {time_str:<8} {meal_str:<40}")

    print("-"*70)
    print(f"\n💡 Action required: Open Nutrilio app and add meal type tags for these {len(invalid_meals_list)} entries")
    print("="*70 + "\n")


def merge_dreams_data(nutrilio_df):
    """
    Merge dreams data from nutrilio_dreams_pbi_processed_file.csv
    into main nutrilio_processed DataFrame.

    If multiple dreams exist for same date, combines them into comma-separated string.

    Args:
        nutrilio_df: Main nutrilio DataFrame

    Returns:
        DataFrame: nutrilio_df with dreams column added (max one row per date)
    """
    dreams_path = 'files/source_processed_files/nutrilio/nutrilio_dreams_pbi_processed_file.csv'

    if os.path.exists(dreams_path):
        print("🌙 Merging dreams data...")
        dreams_df = pd.read_csv(dreams_path, sep='|')
        dreams_df['date'] = pd.to_datetime(dreams_df['date']).dt.strftime('%Y-%m-%d')

        # Aggregate multiple dreams per date into comma-separated string
        dreams_aggregated = dreams_df.groupby('date')['dreams'].apply(
            lambda x: ', '.join(x.astype(str))
        ).reset_index()

        # Count dates with multiple dreams
        multiple_dreams_count = dreams_df.groupby('date').size()
        multiple_dreams_count = (multiple_dreams_count > 1).sum()

        # Ensure date column in nutrilio_df is string format for merging
        nutrilio_df['date'] = nutrilio_df['date'].astype(str)

        # Merge with main nutrilio data (left merge to preserve all nutrilio rows)
        nutrilio_df = nutrilio_df.merge(dreams_aggregated, on='date', how='left')

        print(f"✅ Dreams data merged: {len(dreams_aggregated)} dates with dreams")
        if multiple_dreams_count > 0:
            print(f"   📝 {multiple_dreams_count} dates had multiple dreams (combined with comma)")
    else:
        print(f"⚠️  Dreams file not found: {dreams_path}")
        print("   Adding empty dreams column")
        nutrilio_df['dreams'] = None

    return nutrilio_df


def rename_columns_for_health(df):
    """
    Rename notes_about_today to notes_summary before saving.
    This avoids conflict with the 'daily_summary' column (yes/no indicator).

    Args:
        df: DataFrame with notes_about_today column

    Returns:
        DataFrame: DataFrame with renamed column
    """
    if 'notes_about_today' in df.columns:
        df = df.rename(columns={'notes_about_today': 'notes_summary'})
        print("✏️  Renamed: notes_about_today → notes_summary")

    return df


def create_optimized_nutrition_file(df):
    """Create optimized nutrition-only CSV for faster frontend performance"""
    print("Creating optimized nutrition file...")

    # Filter only meal entries (exclude non-meal data)
    meal_df = df[df['meal'].notna() & (df['meal'] != '')].copy()
    print(f"Filtered to {len(meal_df)} meal entries from {len(df)} total rows")

    # Select only nutrition-relevant columns
    nutrition_cols = [
        'date', 'time', 'meal', 'amount_text', 'food_list', 'source'
    ]

    # Add USDA score column if it exists
    if 'usda_meal_score' in df.columns:
        nutrition_cols.append('usda_meal_score')

    meal_df = meal_df[nutrition_cols].copy()

    # Create proper timestamp from date and time
    meal_df['timestamp'] = pd.to_datetime(meal_df['date'] + ' ' + meal_df['time'],
                                         format='%Y-%m-%d %H:%M', errors='coerce')
    meal_df['timestamp'] = meal_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Clean food_list - remove quotes and extra spaces
    meal_df['food_items'] = meal_df['food_list'].str.replace("'", "").str.replace('"', '').str.strip()

    # Rename columns for frontend consistency
    meal_df.rename(columns={
        'meal': 'meal_type'
    }, inplace=True)

    # Select final columns in optimized order
    final_cols = [
        'timestamp', 'meal_type', 'amount_text', 'food_items', 'source'
    ]

    # Add USDA score column if it exists
    if 'usda_meal_score' in meal_df.columns:
        final_cols.insert(2, 'usda_meal_score')  # Insert after meal_type

    meal_df = meal_df[final_cols]

    # Ensure the directory exists
    os.makedirs("files/source_processed_files/nutrilio", exist_ok=True)

    # Save optimized file
    output_path = "files/source_processed_files/nutrilio/nutrilio_meals_optimized.csv"
    meal_df.to_csv(output_path, sep='|', index=False, encoding='utf-8')

    print(f"✅ Created optimized nutrition file: {output_path}")
    print(f"📊 Optimized file contains {len(meal_df)} meal entries")
    print(f"📊 Reduced from {len(nutrition_cols)} to {len(final_cols)} columns")

    return output_path

def create_nutrilio_files():
    df = pd.read_csv(f'files/exports/nutrilio_exports/nutrilio_export.csv', sep = ',')

    # Create keep columns to preserve original Food and Drinks with amounts
    df['Food_keep'] = df['Food']
    df['Drinks_keep'] = df['Drinks']

    #Remove quantity when unnecessary
    for col in col_unnecessary_qty:
        df[col] = df[col].apply(lambda x: extract_data(x))

    # Initialize invalid meals tracking
    invalid_meals_list = []

    #Extract all values & quantities (with meal validation)
    df, list_col, invalid_meals_list = extract_data_count(df, invalid_meals_list)

    # Rename columns to follow snake_case standards BEFORE enforce_snake_case
    rename_dict = {}
    if 'Full Date' in df.columns:
        rename_dict['Full Date'] = 'date'
    if 'Amount_text' in df.columns:
        rename_dict['Amount_text'] = 'amount_text'
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    # Drop redundant 'Date' column before enforce_snake_case to avoid duplicate 'date' columns
    if 'Date' in df.columns:
        df.drop('Date', axis=1, inplace=True)

    # Enforce snake_case EARLY so list_col contains snake_case column names
    df = enforce_snake_case(df, "processed file")

    # Update list_col to contain snake_case versions of column names
    list_col = [enforce_snake_case(pd.DataFrame(columns=[col]), "temp").columns[0] for col in list_col]

    #Put the newly created columns at the end of the df to simplify checks
    columns_to_concat = []
    for _, val in dict_extract_data.items():
        column = df.pop(f"{val}_list")
        columns_to_concat.append(column)
    df = pd.concat([df.copy()] + columns_to_concat, axis=1)
    # Legacy ingredient checking and meal score migration removed
    # All scoring is now handled by the USDA pipeline
    df['source'] = 'Nutrilio'

    # Legacy ingredient categorization removed - now handled by USDA pipeline

    # USDA meal scoring enabled - provides free alternative to OpenAI scoring
    print("🥗 Enabling efficient USDA-based meal scoring...")
    df = add_usda_meal_scoring_efficient(df, use_usda_scoring=True)

    # USDA drink scoring enabled - automated drink categorization and health scoring
    print("🍹 Enabling efficient USDA-based drink scoring...")
    df = add_usda_drink_scoring_efficient(df, use_usda_scoring=True)

    print("⚙️  Finalizing data processing...")
    df['work_duration_est'] = df["work_-_duration_text"].apply(lambda x: dict_work_duration[x] if x in dict_work_duration.keys() else None)
    df['work_-_good_day_text'] = df['work_-_good_day_text'].apply(lambda x: "Average" if x =="Ok" else x)

    # Generate Power BI files BEFORE dropping list columns (PBI files need these columns)
    print("📊 Generating Power BI files...")
    drive_list = ["files/source_processed_files/nutrilio/nutrilio_processed.csv"]

    for _, value in dict_extract_data.items():
        generate_pbi_files(df, value)
        drive_list.append(f"files/source_processed_files/nutrilio/nutrilio_{value}_pbi_processed_file.csv")

    # Drop list columns AFTER generating PBI files (cleanup for main processed file)
    df = df.drop(list_col, axis=1)

    # Merge dreams data from separate PBI file
    df = merge_dreams_data(df)

    # Rename notes_about_today to daily_summary for health integration
    df = rename_columns_for_health(df)

    # Classify rows as daily_summary or meal
    print("🔍 Classifying rows...")
    df['row_type'] = df.apply(classify_row, axis=1)

    # Separate daily summary and meal rows
    daily_summary_rows = df[df['row_type'] == 'daily_summary'].copy()
    meal_rows = df[df['row_type'] == 'meal'].copy()

    print(f"   Found {len(daily_summary_rows)} daily summary rows")
    print(f"   Found {len(meal_rows)} meal rows")

    # Aggregate daily summary rows by date (keep latest time)
    print("📋 Aggregating daily summary rows by date...")
    if len(daily_summary_rows) > 0:
        # Sort by time descending to keep latest entries first
        daily_summary_rows = daily_summary_rows.sort_values('time', ascending=False)

        # Identify columns to aggregate
        # Non-food metadata columns that should be aggregated
        non_food_columns = [
            'sleep_-_quality', 'sleep_-_quality_text', 'sleep_-_rest_feeling_(points)',
            'fitness_feeling_(points)', 'stress_feeling_(points)',
            'work_-_productivity_(points)', 'work_-_hours_worked',
            'work_-_good_day_text', 'work_duration_est',
            'cycling_distance_(km)', 'reading_-_pages',
            'gaming_-_hours', 'overall_evaluation_(points)',
            'notes_summary', 'dream_description', 'dreams',
            'body_sensations_list', 'work_content_list',
            'self_improvement_list', 'social_activity_list',
            'weight_(kg)'
        ]

        # Build aggregation dictionary
        agg_dict = {}
        for col in non_food_columns:
            if col in daily_summary_rows.columns:
                if col in ['body_sensations_list', 'notes_summary', 'work_content_list', 'self_improvement_list', 'social_activity_list']:
                    # Concatenate list columns with ' | ' separator
                    agg_dict[col] = lambda x: ' | '.join(x.dropna().astype(str)) if len(x.dropna()) > 0 else None
                elif col == 'weight_(kg)':
                    # Average weight values for the day
                    agg_dict[col] = lambda x: x.dropna().mean() if len(x.dropna()) > 0 else None
                else:
                    # Take first non-null value (since sorted by time desc, this is latest)
                    agg_dict[col] = lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None

        # Add standard columns
        agg_dict.update({
            'time': 'first',  # Latest time (since sorted desc)
            'weekday': 'first',
            'daily_summary': 'first',
            'source': 'first'
        })

        # Group by date and aggregate
        aggregated_daily = daily_summary_rows.groupby('date', as_index=False).agg(agg_dict)

        # Check for dates with multiple daily summary entries
        date_counts = daily_summary_rows.groupby('date').size()
        duplicates = date_counts[date_counts > 1]
        if len(duplicates) > 0:
            print(f"   ℹ️  Consolidated {len(duplicates)} dates with multiple daily summary entries:")
            for date, count in duplicates.items():
                times = daily_summary_rows[daily_summary_rows['date'] == date]['time'].tolist()
                print(f"      {date}: {count} entries at times {times}")

        # Clear meal-related columns for daily summary rows
        meal_columns = ['meal', 'food_list', 'drinks_list', 'food_keep', 'drinks_keep',
                       'usda_meal_score', 'usda_drink_score', 'places', 'origin', 'amount']
        for col in meal_columns:
            if col in aggregated_daily.columns:
                aggregated_daily[col] = None

        print(f"   ✅ Aggregated to {len(aggregated_daily)} daily summary rows")
    else:
        aggregated_daily = pd.DataFrame()

    # Check for duplicate meals (same date + same meal type)
    print("🔍 Checking for duplicate meal entries...")
    if len(meal_rows) > 0:
        meal_groups = meal_rows.groupby(['date', 'meal']).size()
        duplicate_meals = meal_groups[meal_groups > 1]
        if len(duplicate_meals) > 0:
            print(f"   ⚠️  WARNING: Found {len(duplicate_meals)} duplicate meal entries:")
            for (date, meal_type), count in duplicate_meals.items():
                times = meal_rows[(meal_rows['date'] == date) & (meal_rows['meal'] == meal_type)]['time'].tolist()
                print(f"      {date} - {meal_type}: {count} entries at times {times}")
            print("   ⚠️  Keeping all entries for investigation. This should not happen normally.")
        else:
            print("   ✅ No duplicate meals found")

    # Combine aggregated daily summary rows and meal rows
    print("🔗 Combining daily summary and meal rows...")
    if len(aggregated_daily) > 0 and len(meal_rows) > 0:
        final_df = pd.concat([aggregated_daily, meal_rows], ignore_index=True)
    elif len(aggregated_daily) > 0:
        final_df = aggregated_daily
    elif len(meal_rows) > 0:
        final_df = meal_rows
    else:
        final_df = df  # Fallback to original if something went wrong

    # Drop temporary row_type column
    if 'row_type' in final_df.columns:
        final_df = final_df.drop(columns=['row_type'])

    # Sort by date (descending) then time (descending) - most recent first
    final_df = final_df.sort_values(['date', 'time'], ascending=[False, False])

    print(f"✅ Final dataframe: {len(final_df)} rows")
    print(f"   Daily summary rows: {len(final_df[final_df['daily_summary'] == 'yes'])}")
    print(f"   Meal rows: {len(final_df[final_df['meal'].notna()])}")

    print("\n💾 Saving split processed files...")

    # SPLIT 1: Health file (daily summary rows only)
    health_df = final_df[final_df['daily_summary'] == 'yes'].copy()

    # Define columns for health file (exclude meal-specific columns)
    health_columns = [
        'date', 'time', 'weekday', 'source', 'daily_summary',
        'sleep_-_quality', 'sleep_-_quality_text', 'sleep_-_rest_feeling_(points)',
        'sleep_-_waking_up', 'sleep_-_night_time',
        'fitness_feeling_(points)', 'stress_feeling_(points)',
        'overall_evaluation_(points)', 'overall_productivity_(points)',
        'work_-_productivity_(points)', 'work_-_productivity_text', 'work_-_good_day', 'work_-_good_day_text',
        'work_duration_est', 'work_-_duration', 'work_-_duration_text',
        'work_-_meetings_(meetings)', 'work_-_location',
        'notes_summary', 'dream_description', 'dreams', 'dreams_list',
        'body_sensations_list', 'work_content_list', 'self_improvement_list', 'social_activity_list',
        'push-ups_(pushups)', 'reading_-_pages', 'gaming_-_hours',
        'podcast', 'podcast_text', 'cycling_distance_(km)', 'cycling_text',
        'distance_in_car', 'distance_in_car_text', 'public_transport', 'public_transport_text',
        'screen_before_sleep', 'weight_(kg)'
    ]

    # Keep only columns that exist
    existing_health_cols = [col for col in health_columns if col in health_df.columns]
    health_df = health_df[existing_health_cols].copy()

    health_file_path = "files/source_processed_files/nutrilio/nutrilio_health_processed.csv"
    os.makedirs(os.path.dirname(health_file_path), exist_ok=True)
    health_df.to_csv(health_file_path, sep='|', index=False, encoding='utf-8')
    print(f"   ✅ Health file: {len(health_df)} daily summary rows → {health_file_path}")

    # SPLIT 2: Nutrition file (meal rows only)
    nutrition_df = final_df[final_df['meal'].notna()].copy()

    # Define columns for nutrition file
    nutrition_columns = [
        'date', 'time', 'weekday', 'source',
        'meal', 'meal_assessment', 'meal_assessment_text',
        'amount', 'amount_text',
        'food_list', 'food_keep',
        'drinks_list', 'drinks_keep', 'drink', 'drink_volume_(ml)',
        'usda_meal_score',
        'places', 'origin'
    ]

    # Keep only columns that exist
    existing_nutrition_cols = [col for col in nutrition_columns if col in nutrition_df.columns]
    nutrition_df = nutrition_df[existing_nutrition_cols].copy()

    nutrition_file_path = "files/source_processed_files/nutrilio/nutrilio_nutrition_processed.csv"
    nutrition_df.to_csv(nutrition_file_path, sep='|', index=False, encoding='utf-8')
    print(f"   ✅ Nutrition file: {len(nutrition_df)} meal rows → {nutrition_file_path}")

    # DEPRECATED: Keep old combined file for backward compatibility (will be removed in future)
    print("\n⚠️  Also saving legacy combined file for backward compatibility...")
    final_df.to_csv("files/source_processed_files/nutrilio/nutrilio_processed.csv", sep='|', index=False, encoding='utf-8')
    print(f"   ⚠️  Legacy file: {len(final_df)} rows → nutrilio_processed.csv (DEPRECATED)")

    # Create optimized nutrition file for frontend performance
    optimized_nutrition_file = create_optimized_nutrition_file(final_df)

    # Add additional files to drive_list
    drive_list.extend([
        optimized_nutrition_file,
        "files/work_files/nutrilio_work_files/ingredient_scores_database.json",
        "files/work_files/nutrilio_work_files/flagged_default_ingredients.json",
        "files/work_files/nutrilio_work_files/drink_scores_database.json",
        "files/work_files/nutrilio_work_files/flagged_default_drinks.json"
    ])

    # Display meal validation report
    display_meal_validation_report(invalid_meals_list)

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

    # Use cross-platform Downloads folder path
    download_folder = os.path.expanduser("~/Downloads")

    move_success = clean_rename_move_file(
        export_folder="files/exports/nutrilio_exports",
        download_folder=download_folder,
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
    print("⚙️  Processing Nutrilio data...")

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


def full_nutrilio_pipeline(auto_full=False, auto_process_only=False):
    """
    Nutrilio SOURCE processor with 2 standard options.

    NOTE: This is a source processor - does NOT upload to Drive.
    Upload is handled by the Health topic coordinator.

    Options:
    1. Download new data and process
    2. Process existing data

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*60)
    print("🥗 NUTRILIO SOURCE PROCESSOR")
    print("="*60)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Download new data and process")
        print("2. Process existing data")

        choice = input("\nEnter your choice (1-2): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Download new data and process...")

        # Step 1: Download
        download_success = download_nutrilio_data()

        # Step 2: Move files (even if download wasn't confirmed, maybe file exists)
        if download_success:
            move_success = move_nutrilio_files()
        else:
            print("⚠️ Download not confirmed, but checking for existing files...")
            move_success = move_nutrilio_files()

        # Step 3: Process (fallback to processing if no new files)
        if move_success:
            success = create_nutrilio_processed_files()
        else:
            print("⚠️ No new files found, attempting to process existing files...")
            success = create_nutrilio_processed_files()

    elif choice == "2":
        print("\n⚙️  Process existing data...")
        success = create_nutrilio_processed_files()

    else:
        print("❌ Invalid choice. Please select 1-2.")
        return False

    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Nutrilio source processing completed successfully!")
        print("📁 Outputs:")
        print("   • Health data: files/source_processed_files/nutrilio/nutrilio_health_processed.csv")
        print("   • Nutrition data: files/source_processed_files/nutrilio/nutrilio_nutrition_processed.csv")
        # Record successful run
        record_successful_run('source_nutrilio', 'active')
    else:
        print("❌ Nutrilio source processing failed")
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
    print("🥗 Nutrilio Source Processing Tool")
    print("This tool downloads and processes Nutrilio data.")
    print("Note: Upload is handled by topic coordinators (Health/Nutrition)")

    # Run the pipeline
    full_nutrilio_pipeline(auto_full=False)
