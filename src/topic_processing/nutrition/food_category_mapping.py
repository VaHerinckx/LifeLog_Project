"""
Food and Drink Category Mapping Utility

Maps specific food and drink items to simplified broad categories.
Loads mappings from CSV files for easy maintenance.

Usage:
    from topic_processing.nutrition.food_category_mapping import get_food_category, get_drink_category
    df['food_category'] = df['food'].apply(get_food_category)
    df['drink_category'] = df['drink'].apply(get_drink_category)

    # Run directly to analyze unmapped items:
    python src/topic_processing/nutrition/food_category_mapping.py
"""

import os
import sys
import pandas as pd

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# Global variables to store loaded mappings
FOOD_MAPPING = {}
DRINK_MAPPING = {}

FOOD_MAPPING_FILE = os.path.join(project_root, 'files', 'work_files', 'nutrilio_work_files', 'food_category_mapping.csv')
DRINK_MAPPING_FILE = os.path.join(project_root, 'files', 'work_files', 'nutrilio_work_files', 'drink_category_mapping.csv')

# Define available categories for reference
FOOD_CATEGORIES = [
    "Meat", "Fish & Seafood", "Dairy", "Eggs",
    "Grains & Starches", "Vegetables", "Fruits",
    "Snacks & Sweets", "Prepared Foods", "Sauces & Condiments",
    "Bakery & Pastries"
]

DRINK_CATEGORIES = [
    "Alcoholic Beverage", "Non-Alcoholic Beer", "Fermented Beverage",
    "Soft Drink", "Diet Soft Drink", "Sports Drink", "Energy Drink",
    "Fruit Juice", "Tea", "Iced Tea", "Hot Beverage",
    "Dairy Beverage", "Protein Beverage", "Specialty Beverage",
    "Herbal Beverage", "Carbonated Water", "Water"
]


def load_food_mapping():
    """
    Load food category mappings from CSV file.

    Returns:
        dict: Food mapping dictionary {food_item: food_category}
    """
    global FOOD_MAPPING

    if FOOD_MAPPING:
        return FOOD_MAPPING

    try:
        if os.path.exists(FOOD_MAPPING_FILE):
            df = pd.read_csv(FOOD_MAPPING_FILE, sep='|', encoding='utf-8')

            if not all(col in df.columns for col in ['food_item', 'food_category']):
                raise ValueError("CSV must have 'food_item' and 'food_category' columns")

            FOOD_MAPPING = dict(zip(
                df['food_item'].str.lower().str.strip(),
                df['food_category']
            ))
        else:
            print(f"⚠️  Warning: Food mapping file not found: {FOOD_MAPPING_FILE}")
            print(f"   Create this file with format: food_item|food_category")

        return FOOD_MAPPING

    except Exception as e:
        print(f"❌ Error loading food mapping: {e}")
        return {}


def load_drink_mapping():
    """
    Load drink category mappings from CSV file.

    Returns:
        dict: Drink mapping dictionary {drink_item: drink_category}
    """
    global DRINK_MAPPING

    if DRINK_MAPPING:
        return DRINK_MAPPING

    try:
        if os.path.exists(DRINK_MAPPING_FILE):
            df = pd.read_csv(DRINK_MAPPING_FILE, sep='|', encoding='utf-8')

            if not all(col in df.columns for col in ['drink_item', 'drink_category']):
                raise ValueError("CSV must have 'drink_item' and 'drink_category' columns")

            DRINK_MAPPING = dict(zip(
                df['drink_item'].str.lower().str.strip(),
                df['drink_category']
            ))
        else:
            print(f"⚠️  Warning: Drink mapping file not found: {DRINK_MAPPING_FILE}")
            print(f"   Create this file with format: drink_item|drink_category")

        return DRINK_MAPPING

    except Exception as e:
        print(f"❌ Error loading drink mapping: {e}")
        return {}


def get_food_category(food_item):
    """
    Map specific food item to food category.

    Uses hierarchical matching:
    1. Direct mapping (exact match)
    2. Partial matching for compound food names
    3. Fallback to 'Other' if unmapped

    Args:
        food_item: Specific food item (e.g., "Grilled chicken")

    Returns:
        Food category (e.g., "Meat") or empty string/Other

    Examples:
        >>> get_food_category('Chicken')
        'Meat'
        >>> get_food_category('Grilled chicken breast')
        'Meat'
        >>> get_food_category(None)
        ''
    """
    if not FOOD_MAPPING:
        load_food_mapping()

    if not food_item or pd.isna(food_item) or food_item == '':
        return ''

    food_lower = str(food_item).lower().strip()

    # Direct mapping (exact match)
    if food_lower in FOOD_MAPPING:
        return FOOD_MAPPING[food_lower]

    # Partial matching for compound names
    # Example: "grilled chicken" contains "chicken" → "Meat"
    for key, value in FOOD_MAPPING.items():
        if key in food_lower:
            return value

    return 'Other'


def get_drink_category(drink_item):
    """
    Map specific drink item to drink category.

    Uses hierarchical matching:
    1. Direct mapping (exact match)
    2. Partial matching for compound drink names
    3. Fallback to 'Other' if unmapped

    Args:
        drink_item: Specific drink item (e.g., "Strong beer")

    Returns:
        Drink category (e.g., "Alcoholic Beverage") or empty string/Other

    Examples:
        >>> get_drink_category('Beer')
        'Alcoholic Beverage'
        >>> get_drink_category('Craft beer')
        'Alcoholic Beverage'
        >>> get_drink_category(None)
        ''
    """
    if not DRINK_MAPPING:
        load_drink_mapping()

    if not drink_item or pd.isna(drink_item) or drink_item == '':
        return ''

    drink_lower = str(drink_item).lower().strip()

    # Direct mapping (exact match)
    if drink_lower in DRINK_MAPPING:
        return DRINK_MAPPING[drink_lower]

    # Partial matching for compound names
    for key, value in DRINK_MAPPING.items():
        if key in drink_lower:
            return value

    return 'Other'


def analyze_unmapped_items(df, top_n=50):
    """
    Analyze unmapped food and drink items and print LLM-ready prompts.

    Displays:
    - Category coverage statistics
    - Top N unmapped items by frequency
    - Copy-paste ready prompt for LLM categorization

    Args:
        df: DataFrame with 'food' and 'drink' columns
        top_n: Number of top unmapped items to display (default: 50)
    """
    # Load mappings if not already loaded
    if not FOOD_MAPPING:
        load_food_mapping()
    if not DRINK_MAPPING:
        load_drink_mapping()

    # Analyze FOOD items
    print("\n" + "="*60)
    print("🍽️  Food Category Analysis")
    print("="*60)

    if 'food' in df.columns:
        # Filter to non-empty food items
        food_df = df[df['food'].notna() & (df['food'] != '')]

        # Add temporary category column
        food_df = food_df.copy()
        food_df['_category'] = food_df['food'].apply(get_food_category)

        # Calculate statistics
        total_items = len(food_df)
        unmapped_items = len(food_df[food_df['_category'] == 'Other'])
        mapped_items = total_items - unmapped_items
        coverage_pct = (mapped_items / total_items) * 100 if total_items > 0 else 0

        unique_categories = sorted(set(FOOD_MAPPING.values())) if FOOD_MAPPING else []

        print(f"Loaded {len(FOOD_MAPPING)} mappings | {len(unique_categories)} categories | {coverage_pct:.1f}% coverage")
        print(f"Mapped: {mapped_items:,} items | Unmapped: {unmapped_items:,} items")
        print(f"\n📋 Current Categories: {', '.join(unique_categories)}")

        if unmapped_items > 0:
            # Get unmapped food items with counts
            unmapped = food_df[food_df['_category'] == 'Other']['food']
            food_counts = unmapped.value_counts().head(top_n)

            print(f"\n🔍 Top {min(top_n, len(food_counts))} Unmapped Foods:")
            print("-"*60)
            for item, count in food_counts.items():
                print(f"  {item:40} {count:>10,}")

            # Print LLM prompt
            print("\n" + "━"*60)
            print("📝 LLM PROMPT FOR FOOD CATEGORIZATION")
            print("━"*60)
            print("\nMap these food items to ONE of these categories:")
            print(f"{', '.join(FOOD_CATEGORIES)}")
            print("\nFormat: food_item|food_category")
            print("-"*30)
            for item, _ in food_counts.items():
                print(f"{item.lower()}|")
            print("\nAppend to: files/work_files/nutrilio_work_files/food_category_mapping.csv")
            print("━"*60)
        else:
            print("\n✅ All food items are mapped!")
    else:
        print("⚠️  No 'food' column found in DataFrame")

    # Analyze DRINK items
    print("\n" + "="*60)
    print("🥤 Drink Category Analysis")
    print("="*60)

    if 'drink' in df.columns:
        # Filter to non-empty drink items
        drink_df = df[df['drink'].notna() & (df['drink'] != '')]

        # Add temporary category column
        drink_df = drink_df.copy()
        drink_df['_category'] = drink_df['drink'].apply(get_drink_category)

        # Calculate statistics
        total_items = len(drink_df)
        unmapped_items = len(drink_df[drink_df['_category'] == 'Other'])
        mapped_items = total_items - unmapped_items
        coverage_pct = (mapped_items / total_items) * 100 if total_items > 0 else 0

        unique_categories = sorted(set(DRINK_MAPPING.values())) if DRINK_MAPPING else []

        print(f"Loaded {len(DRINK_MAPPING)} mappings | {len(unique_categories)} categories | {coverage_pct:.1f}% coverage")
        print(f"Mapped: {mapped_items:,} items | Unmapped: {unmapped_items:,} items")
        print(f"\n📋 Current Categories: {', '.join(unique_categories)}")

        if unmapped_items > 0:
            # Get unmapped drink items with counts
            unmapped = drink_df[drink_df['_category'] == 'Other']['drink']
            drink_counts = unmapped.value_counts().head(top_n)

            print(f"\n🔍 Top {min(top_n, len(drink_counts))} Unmapped Drinks:")
            print("-"*60)
            for item, count in drink_counts.items():
                print(f"  {item:40} {count:>10,}")

            # Print LLM prompt
            print("\n" + "━"*60)
            print("📝 LLM PROMPT FOR DRINK CATEGORIZATION")
            print("━"*60)
            print("\nMap these drink items to ONE of these categories:")
            print(f"{', '.join(DRINK_CATEGORIES)}")
            print("\nFormat: drink_item|drink_category")
            print("-"*30)
            for item, _ in drink_counts.items():
                print(f"{item.lower()}|")
            print("\nAppend to: files/work_files/nutrilio_work_files/drink_category_mapping.csv")
            print("━"*60)
        else:
            print("\n✅ All drink items are mapped!")
    else:
        print("⚠️  No 'drink' column found in DataFrame")


def get_mapping_stats():
    """
    Returns statistics about the current category mappings.

    Returns:
        dict: Statistics including total mappings and categories
    """
    if not FOOD_MAPPING:
        load_food_mapping()
    if not DRINK_MAPPING:
        load_drink_mapping()

    food_categories = sorted(set(FOOD_MAPPING.values())) if FOOD_MAPPING else []
    drink_categories = sorted(set(DRINK_MAPPING.values())) if DRINK_MAPPING else []

    return {
        'food_mappings': len(FOOD_MAPPING),
        'food_categories': food_categories,
        'drink_mappings': len(DRINK_MAPPING),
        'drink_categories': drink_categories
    }


if __name__ == '__main__':
    # Display mapping statistics and analyze unmapped items
    print("\n🚀 Loading food and drink category mappings from CSV...")

    try:
        load_food_mapping()
        load_drink_mapping()
        stats = get_mapping_stats()

        print(f"✅ Food: {stats['food_mappings']} mappings → {len(stats['food_categories'])} categories")
        print(f"   Categories: {', '.join(stats['food_categories'])}")

        print(f"\n✅ Drinks: {stats['drink_mappings']} mappings → {len(stats['drink_categories'])} categories")
        print(f"   Categories: {', '.join(stats['drink_categories'])}")

        # Show example mappings
        print("\n📝 Example food mappings:")
        examples = [
            ('Chicken', get_food_category('Chicken')),
            ('Grilled chicken', get_food_category('Grilled chicken')),
            ('Pasta', get_food_category('Pasta')),
            ('Unknown food', get_food_category('Unknown food'))
        ]
        for item, result in examples:
            print(f"  {item:30} → {result}")

        print("\n📝 Example drink mappings:")
        examples = [
            ('Beer', get_drink_category('Beer')),
            ('Strong beer', get_drink_category('Strong beer')),
            ('Juice', get_drink_category('Juice')),
            ('Unknown drink', get_drink_category('Unknown drink'))
        ]
        for item, result in examples:
            print(f"  {item:30} → {result}")

        # Load nutrition data to analyze unmapped items
        print("\n🔍 Analyzing unmapped items from processed data...")

        processed_file = os.path.join(
            project_root,
            'files',
            'website_files',
            'nutrition',
            'nutrition_page_data.csv'
        )

        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, sep='|', encoding='utf-8')
            analyze_unmapped_items(df, top_n=100)
        else:
            print(f"⚠️  Warning: Processed file not found: {processed_file}")
            print("   Run the Nutrition topic coordinator pipeline first to generate data")
            print("\n💡 To generate data, run:")
            print("   python src/topic_processing/nutrition/nutrition_processing.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
