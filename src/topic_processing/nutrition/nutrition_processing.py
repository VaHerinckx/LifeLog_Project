import pandas as pd
import os
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection
from src.utils.utils_functions import record_successful_run, enforce_snake_case
from src.sources_processing.nutrilio.nutrilio_processing import full_nutrilio_pipeline
from src.topic_processing.website_maintenance.website_maintenance_processing import full_website_maintenance_pipeline


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_nutrilio_nutrition_data():
    """
    Load Nutrilio nutrition data (meal rows only).

    Returns:
        DataFrame: Nutrilio nutrition data with meal-level granularity
    """
    nutrilio_path = 'files/source_processed_files/nutrilio/nutrilio_nutrition_processed.csv'

    if not os.path.exists(nutrilio_path):
        print(f"❌ Nutrilio nutrition file not found: {nutrilio_path}")
        return None

    print(f"🥗 Loading Nutrilio nutrition data...")
    df = pd.read_csv(nutrilio_path, sep='|', encoding='utf-8')

    # Parse date
    df['date'] = pd.to_datetime(df['date'])

    print(f"✅ Loaded Nutrilio: {len(df):,} meal records")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    # Display meal type breakdown
    if 'meal' in df.columns:
        meal_counts = df['meal'].value_counts()
        print(f"   Meal breakdown:")
        for meal_type, count in meal_counts.items():
            print(f"      • {meal_type}: {count:,}")

    return df


# ============================================================================
# WEBSITE FILE GENERATION FUNCTIONS
# ============================================================================

def remove_meal_type_from_food_list(df_web):
    """
    Remove meal type identifiers from the beginning of food_list column.

    Removes the first item if it matches meal types: Breakfast, Morning Snack,
    Lunch, Afternoon Snack, Dinner, Night Snack (case-insensitive).

    Args:
        df_web: DataFrame with food_list column

    Returns:
        DataFrame with cleaned food_list column
    """
    import regex as re

    meal_types = [
        'breakfast', 'morning snack', 'lunch',
        'afternoon snack', 'dinner', 'night snack'
    ]

    for index, row in df_web.iterrows():
        if pd.notna(row.get('food_list')) and row['food_list']:
            # Extract first item before the first pipe
            first_item_match = re.match(r'^([^|]+)(\s*\|(.*))?$', row['food_list'])

            if first_item_match:
                first_item = first_item_match.group(1).strip()
                remaining = first_item_match.group(3)  # Everything after first pipe

                # Extract just the name without quantity
                name_match = re.match(r'(.+?)\s*\(\d+x\)', first_item)
                if name_match:
                    item_name = name_match.group(1).strip().lower()

                    # Check if it's a meal type
                    if item_name in meal_types:
                        # Remove the first item
                        if remaining:
                            df_web.loc[index, 'food_list'] = remaining.strip()
                        else:
                            df_web.loc[index, 'food_list'] = ''

    return df_web


def explode_food_and_drinks(df_web):
    """
    Explode food_list and drinks_list into individual rows with separate columns.

    For each meal, creates multiple rows (one per ingredient/drink, using max count).
    Parses items like "Dinner (1x) | Potatoes (2x)" into individual food/drink and quantity columns.

    Args:
        df_web: DataFrame with food_list and drinks_list columns

    Returns:
        DataFrame with food, food_quantity, drink, drink_quantity columns added
    """
    import regex as re

    result_rows = []
    pattern = r'([^|]+)\s*\((\d+)x\)'

    for _, row in df_web.iterrows():
        # Parse food_list
        food_items = []
        food_quantities = []
        if pd.notna(row.get('food_list')) and row['food_list']:
            matches = re.findall(pattern, row['food_list'])
            for item, qty in matches:
                food_items.append(item.strip())
                food_quantities.append(int(qty))

        # Parse drinks_list
        drink_items = []
        drink_quantities = []
        if pd.notna(row.get('drinks_list')) and row['drinks_list']:
            matches = re.findall(pattern, row['drinks_list'])
            for item, qty in matches:
                drink_items.append(item.strip())
                drink_quantities.append(int(qty))

        # Determine number of rows to create (max of food and drink counts)
        max_items = max(len(food_items), len(drink_items))

        # If no items found, keep original row with empty food/drink columns
        if max_items == 0:
            new_row = row.to_dict()
            new_row['food'] = ''
            new_row['food_quantity'] = None
            new_row['drink'] = ''
            new_row['drink_quantity'] = None
            result_rows.append(new_row)
        else:
            # Create one row per item
            for i in range(max_items):
                new_row = row.to_dict()

                # Add food and quantity (or empty if index exceeds food items)
                if i < len(food_items):
                    new_row['food'] = food_items[i]
                    new_row['food_quantity'] = food_quantities[i]
                else:
                    new_row['food'] = ''
                    new_row['food_quantity'] = None

                # Add drink and quantity (or empty if index exceeds drink items)
                if i < len(drink_items):
                    new_row['drink'] = drink_items[i]
                    new_row['drink_quantity'] = drink_quantities[i]
                else:
                    new_row['drink'] = ''
                    new_row['drink_quantity'] = None

                result_rows.append(new_row)

    return pd.DataFrame(result_rows)


def generate_nutrition_website_file(df):
    """
    Generate website-optimized file for the Nutrition page.

    Args:
        df: Processed dataframe (already in snake_case)

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🌐 Generating website file for Nutrition page...")

    try:
        # Ensure output directory exists
        website_dir = 'files/website_files/nutrition'
        os.makedirs(website_dir, exist_ok=True)

        # Work with copy to avoid modifying original
        df_web = df.copy()

        # Filter to only keep rows where meal is not null
        df_web = df_web[df_web['meal'].notna() & (df_web['meal'] != '')].copy()
        print(f"📊 Filtered to {len(df_web)} meal entries from {len(df)} total rows")

        # Select specific columns for website (use keep columns instead of list columns)
        columns_to_keep = [
            'date', 'weekday', 'time', 'meal', 'food_keep', 'drinks_keep',
            'usda_meal_score', 'places', 'origin', 'amount', 'amount_text',
            'meal_assessment', 'meal_assessment_text'
        ]

        # Keep only columns that exist in the dataframe
        existing_columns = [col for col in columns_to_keep if col in df_web.columns]
        df_web = df_web[existing_columns].copy()

        # Rename keep columns to list columns for website
        df_web.rename(columns={
            'food_keep': 'food_list',
            'drinks_keep': 'drinks_list'
        }, inplace=True)

        # Add meal_id as sequential integer starting from 0 (before explosion)
        df_web.insert(0, 'meal_id', range(len(df_web)))

        # Remove meal type identifiers from food_list
        print(f"🧹 Removing meal type identifiers from food_list...")
        df_web = remove_meal_type_from_food_list(df_web)

        # Explode food_list and drinks_list into individual rows
        print(f"🔄 Exploding food and drinks into individual rows...")
        df_web = explode_food_and_drinks(df_web)
        print(f"📊 Exploded to {len(df_web)} rows (from individual ingredients/drinks)")

        # Enforce snake_case before saving
        df_web = enforce_snake_case(df_web, "nutrition_page_data")

        # Save website file
        website_path = f'{website_dir}/nutrition_page_data.csv'
        df_web.to_csv(website_path, sep='|', index=False, encoding='utf-8')
        print(f"✅ Website file: {len(df_web):,} records → {website_path}")

        return True

    except Exception as e:
        print(f"❌ Error generating website file: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def create_nutrition_files():
    """
    Main processing function that loads nutrition data from sources and generates output files.

    Currently supports:
    - Nutrilio nutrition data (meals)

    Future sources could include:
    - MyFitnessPal
    - Cronometer
    - Other nutrition tracking apps

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("🥗 NUTRITION DATA PROCESSING")
    print("="*70)

    try:
        # STEP 1: Load Nutrilio nutrition data
        print("\n📊 STEP 1: Loading nutrition data from sources...")
        nutrilio_df = load_nutrilio_nutrition_data()

        if nutrilio_df is None or len(nutrilio_df) == 0:
            print("❌ No nutrition data loaded from sources")
            return False

        # STEP 2: Process and merge (currently only one source, but ready for multi-source)
        print("\n🔗 STEP 2: Processing nutrition data...")

        # Start with Nutrilio data as base
        nutrition_df = nutrilio_df.copy()

        # Add any additional processing here
        # For example: calculate daily totals, nutritional summaries, etc.

        # STEP 3: Sort by date and time (most recent first)
        print("\n📋 STEP 3: Sorting data...")
        nutrition_df = nutrition_df.sort_values(['date', 'time'], ascending=[False, False])

        # STEP 4: Enforce snake_case
        print("\n🔤 STEP 4: Enforcing snake_case column names...")
        nutrition_df = enforce_snake_case(nutrition_df, "nutrition_processed")

        # STEP 5: Save processed file
        print("\n💾 STEP 5: Saving processed files...")
        nutrition_dir = 'files/topic_processed_files/nutrition'
        os.makedirs(nutrition_dir, exist_ok=True)

        # Main nutrition file
        nutrition_output = f'{nutrition_dir}/nutrition_processed.csv'
        nutrition_df.to_csv(nutrition_output, sep='|', index=False, encoding='utf-8')

        print(f"✅ Saved nutrition data: {len(nutrition_df):,} records")
        print(f"   Output: {nutrition_output}")

        # STEP 6: Generate website file
        print("\n🌐 STEP 6: Generating website file...")
        website_success = generate_nutrition_website_file(nutrition_df)

        if not website_success:
            print("⚠️  Warning: Website file generation failed, but processed file was saved")

        return True

    except Exception as e:
        print(f"\n❌ Error processing nutrition data: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_nutrition_results():
    """
    Uploads nutrition website file to Google Drive.
    Only uploads the single website file (not processed files).

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n☁️  Uploading nutrition website file to Google Drive...")

    # Only upload website file
    files_to_upload = [
        "files/website_files/nutrition/nutrition_page_data.csv"
    ]

    # Filter to only existing files
    existing_files = [f for f in files_to_upload if os.path.exists(f)]

    if not existing_files:
        print("❌ No website file found to upload")
        print("💡 Make sure generate_nutrition_website_file() ran successfully")
        return False

    print(f"📤 Uploading {len(existing_files)} website file...")
    for f in existing_files:
        print(f"   • {f}")

    success = upload_multiple_files(existing_files)

    if success:
        print("✅ Nutrition website file uploaded successfully!")
    else:
        print("❌ Website file failed to upload")

    return success


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def full_nutrition_pipeline(auto_full=False, auto_process_only=False):
    """
    Complete Nutrition TOPIC COORDINATOR pipeline.

    Options:
    1. Full pipeline (download → process sources → merge → upload)
    2. Process existing data and upload to Drive
    3. Upload existing processed files to Drive

    Args:
        auto_full (bool): If True, automatically runs option 1 without user input
        auto_process_only (bool): If True, automatically runs option 2 without user input

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("\n" + "="*70)
    print("🥗 NUTRITION TOPIC COORDINATOR PIPELINE")
    print("="*70)

    if auto_process_only:
        print("🤖 Auto process mode: Processing existing data and uploading...")
        choice = "2"
    elif auto_full:
        print("🤖 Auto mode: Running full pipeline...")
        choice = "1"
    else:
        print("\nSelect an option:")
        print("1. Full pipeline (download → process sources → merge → upload)")
        print("2. Process existing data and upload to Drive")
        print("3. Upload existing processed files to Drive")

        choice = input("\nEnter your choice (1-3): ").strip()

    success = False

    if choice == "1":
        print("\n🚀 Option 1: Full pipeline...")

        # Step 1: Run Nutrilio source pipeline
        print("\n🥗 Step 1/3: Processing Nutrilio nutrition data...")
        try:
            nutrilio_success = full_nutrilio_pipeline(auto_full=True)
            if not nutrilio_success:
                print("❌ Nutrilio pipeline failed, stopping nutrition coordination pipeline")
                return False
        except Exception as e:
            print(f"❌ Error in Nutrilio pipeline: {e}")
            return False

        # Step 2: Merge nutrition data
        print("\n🔗 Step 2/3: Merging nutrition data...")
        process_success = create_nutrition_files()
        if not process_success:
            print("❌ Nutrition data merge failed, stopping pipeline")
            return False

        # Step 3: Upload results
        print("\n☁️  Step 3/3: Uploading results...")
        success = upload_nutrition_results()

    elif choice == "2":
        print("\n⚙️  Option 2: Process existing data and upload...")
        print("   (Merges already-processed source files)")

        # Merge nutrition data from existing processed files
        process_success = create_nutrition_files()

        if process_success:
            success = upload_nutrition_results()
        else:
            print("❌ Processing failed, skipping upload")
            success = False

    elif choice == "3":
        print("\n☁️  Option 3: Upload existing processed files...")
        success = upload_nutrition_results()

    else:
        print("❌ Invalid choice. Please select 1-3.")
        return False

    # Final status
    print("\n" + "="*70)
    if success:
        print("✅ Nutrition topic coordinator completed successfully!")
        print("📊 Your nutrition dataset is ready for analysis!")
        # Record successful run
        record_successful_run('topic_nutrition', 'active')
        # Update website tracking file
        full_website_maintenance_pipeline(auto_mode=True, quiet=True)
    else:
        print("❌ Nutrition coordination pipeline failed")
    print("="*70)

    return success


if __name__ == "__main__":
    # Allow running this file directly
    print("🥗 Nutrition Topic Coordinator")
    print("This tool coordinates nutrition data from multiple sources.")

    # Test drive connection first
    if not verify_drive_connection():
        print("⚠️  Warning: Google Drive connection issues detected")
        proceed = input("Continue anyway? (Y/N): ").upper() == 'Y'
        if not proceed:
            exit()

    # Run the pipeline
    full_nutrition_pipeline(auto_full=False)
