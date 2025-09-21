#!/usr/bin/env python3
"""
USDA-based Nutrition Scoring Module

This module provides healthiness scoring for ingredients and meals using the
USDA FoodData Central API as the primary data source, with intelligent fallbacks
for missing items (particularly Asian/regional dishes).

Author: LifeLog Project
"""

import requests
import json
import time
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta


class USDANutritionScorer:
    """Handles nutrition scoring using USDA FoodData Central API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the USDA nutrition scorer.
        
        Args:
            api_key (str, optional): USDA API key for higher rate limits
        """
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        # Load API key from environment if not provided
        self.api_key = api_key or os.environ.get('USDA_API_KEY')
        self.cache_file = "files/work_files/nutrilio_work_files/usda_nutrition_cache.json"
        
        # Load caches
        self.nutrition_cache = self._load_cache()
        self.fallback_scores = self._load_fallback_scores()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
    def _load_cache(self) -> Dict:
        """Load nutrition cache from JSON file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading nutrition cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save nutrition cache to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.nutrition_cache, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving nutrition cache: {e}")
    
    def _load_fallback_scores(self) -> Dict:
        """Load hardcoded fallback ingredient scores for items not in USDA."""
        # Hardcoded fallback scores - no external file needed
        return {
            # Asian dishes - estimated based on typical ingredients
            "lu rou fan": {"score": 6.5, "category": "Asian main dish", "notes": "Braised pork rice - moderate protein, high sodium"},
            "pad thai": {"score": 6.0, "category": "Asian noodle dish", "notes": "Stir-fried noodles - moderate nutrition, some vegetables"},
            "xiaolongbao": {"score": 5.5, "category": "Asian dumpling", "notes": "Pork dumplings - high fat, moderate protein"},
            "dim sum": {"score": 6.0, "category": "Asian appetizer", "notes": "Mixed dim sum items - varies by type"},
            "miso soup": {"score": 7.5, "category": "Asian soup", "notes": "Fermented soy soup - probiotics, low calories"},
            "fan tuan": {"score": 6.5, "category": "Asian rice dish", "notes": "Rice with fillings - varies by ingredients"},
            "phó": {"score": 7.0, "category": "Vietnamese soup", "notes": "Pho soup - broth with herbs, moderate nutrition"},
            "pho": {"score": 7.0, "category": "Vietnamese soup", "notes": "Pho soup - alternative spelling"},
            "mantou": {"score": 6.0, "category": "Chinese bread", "notes": "Steamed bread - carbs, moderate nutrition"},
            "lichi": {"score": 8.0, "category": "Asian fruit", "notes": "Lychee fruit - natural sugars, vitamin C"},
            "lychee": {"score": 8.0, "category": "Asian fruit", "notes": "Lychee fruit - alternative spelling"},
            "congzhuabing": {"score": 5.5, "category": "Chinese flatbread", "notes": "Scallion pancake - fried bread, moderate fat"},
            "congyoubing": {"score": 5.5, "category": "Chinese flatbread", "notes": "Scallion oil pancake - fried bread"},
            "loempia": {"score": 4.5, "category": "Indonesian spring roll", "notes": "Fried spring roll - high fat, processed"},
            "luopogao": {"score": 6.5, "category": "Chinese cake", "notes": "Turnip cake - steamed, moderate nutrition"},
            "zongzi": {"score": 6.0, "category": "Chinese rice dumpling", "notes": "Sticky rice dumpling - moderate carbs"},
            "kroepoek": {"score": 3.0, "category": "Indonesian snack", "notes": "Fried prawn crackers - high fat, processed"},
            
            # European dishes
            "croque monsieur": {"score": 4.5, "category": "European sandwich", "notes": "Ham and cheese sandwich - high fat, processed"},
            "béarnaise": {"score": 3.0, "category": "French sauce", "notes": "Butter-based sauce - very high fat"},
            "halloumi": {"score": 6.0, "category": "Mediterranean cheese", "notes": "High protein, moderate fat"},
            "hallumi": {"score": 6.0, "category": "Mediterranean cheese", "notes": "High protein, moderate fat"},
            "tartiflette": {"score": 4.0, "category": "French dish", "notes": "Potato gratin with cheese - high fat, calories"},
            "jambonneau": {"score": 5.5, "category": "French pork", "notes": "Pork knuckle - high protein, moderate fat"},
            "pistolet": {"score": 6.0, "category": "Belgian bread", "notes": "White bread roll - refined carbs"},
            "bolognaise": {"score": 6.5, "category": "Italian sauce", "notes": "Meat sauce - protein with vegetables"},
            "américain": {"score": 4.0, "category": "Belgian spread", "notes": "Raw beef spread - high protein, high fat"},
            "andalouse": {"score": 4.5, "category": "Belgian sauce", "notes": "Mayo-based sauce - high fat"},
            "taboulé": {"score": 8.0, "category": "Middle Eastern salad", "notes": "Parsley salad - fresh herbs, healthy"},
            "yaourt": {"score": 7.0, "category": "Dairy", "notes": "Yogurt - probiotics, protein"},
            
            # Vegetables and ingredients
            "beansprouts": {"score": 8.5, "category": "Vegetable", "notes": "Fresh sprouts - low calorie, nutrients"},
            
            # Food categories for unknown items
            "vegetable_unknown": {"score": 8.5, "category": "Vegetable", "notes": "Default vegetable score"},
            "fruit_unknown": {"score": 8.0, "category": "Fruit", "notes": "Default fruit score"},
            "protein_unknown": {"score": 7.0, "category": "Protein", "notes": "Default protein score"},
            "grain_unknown": {"score": 6.5, "category": "Grain", "notes": "Default grain score"},
            "dairy_unknown": {"score": 6.0, "category": "Dairy", "notes": "Default dairy score"},
            "processed_unknown": {"score": 3.5, "category": "Processed", "notes": "Default processed food score"},
            "sweet_unknown": {"score": 2.5, "category": "Sweet", "notes": "Default sweet/dessert score"}
        }
    
    
    def _rate_limit(self):
        """Enforce rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_usda_food(self, ingredient_name: str) -> Optional[Dict]:
        """
        Search USDA FoodData Central for an ingredient.
        
        Args:
            ingredient_name (str): Name of the ingredient to search
            
        Returns:
            Dict: Nutrition data if found, None otherwise
        """
        # Check cache first
        cache_key = ingredient_name.lower().strip()
        if cache_key in self.nutrition_cache:
            return self.nutrition_cache[cache_key]
        
        # Rate limit API calls
        self._rate_limit()
        
        try:
            # Search for foods
            search_url = f"{self.base_url}/foods/search"
            params = {
                "query": ingredient_name,
                "pageSize": 5  # Get top 5 results
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 429:  # Rate limited
                print(f"⚠️  Rate limited for '{ingredient_name}', using fallback")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Get the best match
            foods = data.get('foods', [])
            if not foods:
                return None
            
            # Select best match (prioritize Foundation Foods and SR Legacy)
            best_food = self._select_best_food_match(foods, ingredient_name)
            
            if best_food:
                # Get detailed nutrition info
                nutrition_data = self._extract_nutrition_data(best_food)
                
                # Cache the result
                self.nutrition_cache[cache_key] = nutrition_data
                self._save_cache()
                
                return nutrition_data
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  API error for '{ingredient_name}': {e}")
        except Exception as e:
            print(f"⚠️  Error processing '{ingredient_name}': {e}")
        
        return None
    
    def _select_best_food_match(self, foods: List[Dict], search_term: str) -> Optional[Dict]:
        """
        Select the best food match from search results.
        
        Args:
            foods (List[Dict]): List of food search results
            search_term (str): Original search term
            
        Returns:
            Dict: Best matching food item
        """
        if not foods:
            return None
        
        # Prioritize by data type
        priority_order = ['Foundation', 'SR Legacy', 'Survey (FNDDS)', 'Branded']
        
        for data_type in priority_order:
            for food in foods:
                if food.get('dataType') == data_type:
                    return food
        
        # If no priority match, return first result
        return foods[0]
    
    def _extract_nutrition_data(self, food_data: Dict) -> Dict:
        """
        Extract relevant nutrition data from USDA food item.
        
        Args:
            food_data (Dict): USDA food data
            
        Returns:
            Dict: Standardized nutrition data
        """
        nutrition = {
            'usda_id': food_data.get('fdcId'),
            'description': food_data.get('description', ''),
            'data_type': food_data.get('dataType', ''),
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbs': 0,
            'fiber': 0,
            'sugar': 0,
            'sodium': 0,
            'nutrients': {}
        }
        
        # Extract nutrients
        nutrients = food_data.get('foodNutrients', [])
        
        # Map USDA nutrient IDs to our standard names
        nutrient_mapping = {
            1008: 'calories',      # Energy
            1003: 'protein',       # Protein
            1004: 'fat',          # Total lipid (fat)
            1005: 'carbs',        # Carbohydrate, by difference
            1079: 'fiber',        # Fiber, total dietary
            2000: 'sugar',        # Total sugars
            1093: 'sodium'        # Sodium
        }
        
        for nutrient in nutrients:
            nutrient_id = nutrient.get('nutrientId')
            value = nutrient.get('value', 0)
            
            if nutrient_id in nutrient_mapping:
                nutrition[nutrient_mapping[nutrient_id]] = value
            
            # Store all nutrients for reference
            nutrition['nutrients'][nutrient_id] = {
                'name': nutrient.get('nutrientName', ''),
                'value': value,
                'unit': nutrient.get('unitName', '')
            }
        
        return nutrition
    
    def calculate_health_score(self, nutrition_data: Dict) -> float:
        """
        Calculate health score from nutrition data.
        
        Args:
            nutrition_data (Dict): Nutrition data from USDA
            
        Returns:
            float: Health score from 1-10 (10 = healthiest)
        """
        if not nutrition_data:
            return 5.0  # Default neutral score
        
        # Base score
        score = 5.0
        
        # Get nutrition values (per 100g)
        calories = nutrition_data.get('calories', 0)
        protein = nutrition_data.get('protein', 0)
        fat = nutrition_data.get('fat', 0)
        carbs = nutrition_data.get('carbs', 0)
        fiber = nutrition_data.get('fiber', 0)
        sugar = nutrition_data.get('sugar', 0)
        sodium = nutrition_data.get('sodium', 0)
        
        # Positive factors (add to score)
        if protein > 10:  # High protein
            score += min(2.0, protein / 20)  # Up to +2 points
        
        if fiber > 3:  # Good fiber content
            score += min(1.5, fiber / 10)  # Up to +1.5 points
        
        if calories < 200:  # Low calorie density
            score += 1.0
        elif calories < 100:
            score += 2.0
        
        # Negative factors (subtract from score)
        if calories > 400:  # High calorie density
            score -= min(2.0, (calories - 400) / 200)
        
        if fat > 20:  # High fat content
            score -= min(2.0, (fat - 20) / 20)
        
        if sugar > 15:  # High sugar content
            score -= min(2.0, (sugar - 15) / 15)
        
        if sodium > 500:  # High sodium content
            score -= min(2.0, (sodium - 500) / 1000)
        
        # Ensure score is within bounds
        score = max(1.0, min(10.0, score))
        
        return round(score, 2)
    
    def get_ingredient_score(self, ingredient_name: str) -> Tuple[float, str]:
        """
        Get health score for a single ingredient.
        
        Args:
            ingredient_name (str): Name of the ingredient
            
        Returns:
            Tuple[float, str]: (score, source) where source is 'usda', 'fallback', or 'category'
        """
        # Clean ingredient name
        ingredient_clean = ingredient_name.lower().strip()
        
        # Try USDA API first
        nutrition_data = self.search_usda_food(ingredient_name)
        
        if nutrition_data:
            score = self.calculate_health_score(nutrition_data)
            return score, 'usda'
        
        # Try fallback scores
        if ingredient_clean in self.fallback_scores:
            return self.fallback_scores[ingredient_clean]['score'], 'fallback'
        
        # Try category-based scoring
        category_score = self._get_category_score(ingredient_name)
        if category_score:
            return category_score, 'category'
        
        # Default neutral score
        return 5.0, 'default'
    
    def _get_category_score(self, ingredient_name: str) -> Optional[float]:
        """
        Get score based on ingredient category classification.
        
        Args:
            ingredient_name (str): Name of the ingredient
            
        Returns:
            float: Category-based score if classification is possible
        """
        ingredient_lower = ingredient_name.lower()
        
        # Vegetable keywords
        vegetable_keywords = ['salad', 'lettuce', 'spinach', 'carrot', 'broccoli', 'tomato', 
                             'pepper', 'onion', 'garlic', 'mushroom', 'zucchini', 'cucumber']
        if any(keyword in ingredient_lower for keyword in vegetable_keywords):
            return self.fallback_scores['vegetable_unknown']['score']
        
        # Fruit keywords
        fruit_keywords = ['apple', 'banana', 'orange', 'berry', 'grape', 'melon', 'peach']
        if any(keyword in ingredient_lower for keyword in fruit_keywords):
            return self.fallback_scores['fruit_unknown']['score']
        
        # Protein keywords
        protein_keywords = ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'egg', 'tofu']
        if any(keyword in ingredient_lower for keyword in protein_keywords):
            return self.fallback_scores['protein_unknown']['score']
        
        # Processed/fried keywords
        processed_keywords = ['fried', 'chips', 'fries', 'processed', 'packaged', 'frozen']
        if any(keyword in ingredient_lower for keyword in processed_keywords):
            return self.fallback_scores['processed_unknown']['score']
        
        # Sweet keywords
        sweet_keywords = ['chocolate', 'candy', 'cookie', 'cake', 'dessert', 'ice cream']
        if any(keyword in ingredient_lower for keyword in sweet_keywords):
            return self.fallback_scores['sweet_unknown']['score']
        
        return None


def extract_unique_ingredients_from_dataframe(df) -> List[str]:
    """
    Extract all unique ingredients from a dataframe containing meal data.
    
    Args:
        df (DataFrame): Dataframe with food_list column containing ingredient data
        
    Returns:
        List[str]: List of unique ingredients found across all meals
    """
    unique_ingredients = set()
    
    # Filter to rows with meal data
    meal_rows = df[df['Meal'].notna() & (df['Meal'] != '') & df['food_list'].notna()]
    
    for _, row in meal_rows.iterrows():
        ingredients_str = row.get('food_list', '')
        if pd.isna(ingredients_str) or not ingredients_str:
            continue
            
        try:
            # Parse ingredients from different formats
            if ',' in ingredients_str:
                ingredients = []
                for ingredient in ingredients_str.split(','):
                    ingredient = ingredient.strip()
                    # Remove surrounding quotes if present
                    if (ingredient.startswith("'") and ingredient.endswith("'")) or \
                       (ingredient.startswith('"') and ingredient.endswith('"')):
                        ingredient = ingredient[1:-1]
                    ingredients.append(ingredient)
            else:
                # Single ingredient - remove quotes if present
                ingredient = ingredients_str.strip()
                if (ingredient.startswith("'") and ingredient.endswith("'")) or \
                   (ingredient.startswith('"') and ingredient.endswith('"')):
                    ingredient = ingredient[1:-1]
                ingredients = [ingredient]
            
            # Add each ingredient to the set
            for ingredient in ingredients:
                cleaned = ingredient.strip().lower()
                if cleaned:
                    unique_ingredients.add(cleaned)
                    
        except Exception as e:
            print(f"⚠️  Error parsing ingredients from: {ingredients_str}")
            continue
    
    return sorted(list(unique_ingredients))


def load_ingredient_scores_database() -> Dict[str, Tuple[float, str]]:
    """
    Load the master ingredient scores database.
    
    Returns:
        Dict[str, Tuple[float, str]]: Dictionary mapping ingredient -> (score, source)
    """
    database_file = "files/work_files/nutrilio_work_files/ingredient_scores_database.json"
    
    if os.path.exists(database_file):
        try:
            with open(database_file, 'r') as f:
                data = json.load(f)
            print(f"📊 Loaded {len(data)} ingredient scores from database")
            return data
        except Exception as e:
            print(f"⚠️  Error loading ingredient database: {e}")
    
    print("📊 No ingredient database found - creating new one")
    return {}


def save_ingredient_scores_database(ingredient_scores: Dict[str, Tuple[float, str]]):
    """
    Save the ingredient scores database.
    
    Args:
        ingredient_scores (Dict): Dictionary mapping ingredient -> (score, source)
    """
    database_file = "files/work_files/nutrilio_work_files/ingredient_scores_database.json"
    
    try:
        os.makedirs(os.path.dirname(database_file), exist_ok=True)
        with open(database_file, 'w') as f:
            json.dump(ingredient_scores, f, indent=2)
        print(f"💾 Saved {len(ingredient_scores)} ingredient scores to database")
    except Exception as e:
        print(f"❌ Error saving ingredient database: {e}")


def score_all_ingredients_efficient(unique_ingredients: List[str], api_key: Optional[str] = None) -> Dict[str, Tuple[float, str]]:
    """
    Efficiently score ingredients using pre-existing database and only scoring new ones.
    
    Args:
        unique_ingredients (List[str]): List of unique ingredient names
        api_key (str, optional): USDA API key for higher rate limits
        
    Returns:
        Dict[str, Tuple[float, str]]: Dictionary mapping ingredient -> (score, source)
    """
    print(f"🧪 Processing {len(unique_ingredients)} unique ingredients...")
    
    # Load existing ingredient scores database
    ingredient_scores = load_ingredient_scores_database()
    
    # Find ingredients that need scoring
    need_scoring = []
    for ingredient in unique_ingredients:
        if ingredient.lower().strip() not in ingredient_scores:
            need_scoring.append(ingredient)
    
    already_scored = len(unique_ingredients) - len(need_scoring)
    print(f"📊 Status: {already_scored} already scored, {len(need_scoring)} need scoring")
    
    if len(need_scoring) == 0:
        print("⚡ All ingredients already in database - no API calls needed!")
    else:
        print(f"🔄 Scoring {len(need_scoring)} new ingredients...")
        
        # Create scorer only for new ingredients
        scorer = USDANutritionScorer(api_key=api_key)
        
        # Counters for new ingredients only
        usda_count = 0
        fallback_count = 0 
        category_count = 0
        default_count = 0
        
        # Score only the new ingredients
        for i, ingredient in enumerate(need_scoring):
            score, source = scorer.get_ingredient_score(ingredient)
            ingredient_scores[ingredient.lower().strip()] = (score, source)
            
            # Count by source
            if source == 'usda':
                usda_count += 1
            elif source == 'fallback':
                fallback_count += 1
            elif source == 'category':
                category_count += 1
            else:  # default
                default_count += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == len(need_scoring) - 1:
                print(f"   Processed {i + 1}/{len(need_scoring)} new ingredients...")
        
        # Save updated database
        save_ingredient_scores_database(ingredient_scores)
        
        # Report on new ingredients
        total_new = len(need_scoring)
        print(f"\n📊 New Ingredient Results:")
        print(f"   🆕 USDA: {usda_count}/{total_new} ({usda_count/total_new*100:.1f}%)")
        print(f"   🌏 Fallback: {fallback_count}/{total_new} ({fallback_count/total_new*100:.1f}%)")
        print(f"   🏷️ Category: {category_count}/{total_new} ({category_count/total_new*100:.1f}%)")
        print(f"   ❓ Default: {default_count}/{total_new} ({default_count/total_new*100:.1f}%)")
    
    # Check for default scores in ALL ingredients (new and existing) and flag them
    flag_default_scores(ingredient_scores, unique_ingredients)
    
    # Create final result mapping with original ingredient names
    final_scores = {}
    for ingredient in unique_ingredients:
        key = ingredient.lower().strip()
        if key in ingredient_scores:
            final_scores[ingredient] = ingredient_scores[key]
        else:
            # Fallback - this shouldn't happen
            final_scores[ingredient] = (5.0, 'default')
    
    print(f"\n✅ Final database contains {len(ingredient_scores)} total ingredients")
    return final_scores


def flag_default_scores(ingredient_scores: Dict[str, Tuple[float, str]], current_ingredients: List[str]):
    """
    Flag ingredients with default scores and save them to a flagged file for review.
    
    Args:
        ingredient_scores (Dict): All ingredient scores
        current_ingredients (List): Current batch of ingredients being processed
    """
    flagged_file = "files/work_files/nutrilio_work_files/flagged_default_ingredients.json"
    
    # Find all default score ingredients from current batch
    current_defaults = []
    for ingredient in current_ingredients:
        key = ingredient.lower().strip()
        if key in ingredient_scores:
            score, source = ingredient_scores[key]
            if source == 'default':
                current_defaults.append({
                    "ingredient": ingredient,
                    "score": score,
                    "source": source,
                    "last_seen": pd.Timestamp.now().isoformat(),
                    "suggestion": f"Consider adding '{ingredient}' to fallback database with appropriate score"
                })
    
    if current_defaults:
        print(f"\n🚨 Found {len(current_defaults)} ingredients with default scores:")
        for item in current_defaults:
            print(f"   ❓ '{item['ingredient']}' → {item['score']} (needs manual scoring)")
        
        # Load existing flagged ingredients
        existing_flagged = {}
        if os.path.exists(flagged_file):
            try:
                with open(flagged_file, 'r') as f:
                    existing_flagged = json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading flagged file: {e}")
        
        # Update flagged ingredients (preserve existing, add new)
        for item in current_defaults:
            key = item['ingredient'].lower().strip()
            existing_flagged[key] = item
        
        # Save updated flagged ingredients
        try:
            os.makedirs(os.path.dirname(flagged_file), exist_ok=True)
            with open(flagged_file, 'w') as f:
                json.dump(existing_flagged, f, indent=2)
            print(f"📝 Updated flagged ingredients file: {flagged_file}")
            print(f"💡 Tip: Review this file to add missing ingredients to the fallback database")
        except Exception as e:
            print(f"❌ Error saving flagged file: {e}")
    else:
        print(f"\n✅ No default score ingredients found in current batch")


# Legacy function kept for compatibility  
def score_all_ingredients(unique_ingredients: List[str], api_key: Optional[str] = None) -> Dict[str, Tuple[float, str]]:
    """Legacy function - use score_all_ingredients_efficient instead."""
    return score_all_ingredients_efficient(unique_ingredients, api_key)


def calculate_meal_scores_from_ingredients(df, ingredient_scores: Dict[str, Tuple[float, str]]) -> pd.DataFrame:
    """
    Calculate meal scores using pre-computed ingredient scores.
    
    Args:
        df (DataFrame): Dataframe with meal data
        ingredient_scores (Dict): Pre-computed ingredient scores
        
    Returns:
        DataFrame: Dataframe with usda_meal_score column added
    """
    print(f"\n🍽️ Calculating meal scores using pre-computed ingredient data...")
    
    # Add USDA scores column
    df['usda_meal_score'] = None
    
    meal_rows = df[df['Meal'].notna() & (df['Meal'] != '')].copy()
    scored_count = 0
    total_meals = len(meal_rows)
    
    for idx, row in meal_rows.iterrows():
        ingredients_str = row.get('food_list', '')
        if pd.isna(ingredients_str) or not ingredients_str:
            continue
            
        try:
            # Parse ingredients using improved logic
            if ',' in ingredients_str:
                ingredients = []
                for ingredient in ingredients_str.split(','):
                    ingredient = ingredient.strip()
                    # Remove surrounding quotes if present
                    if (ingredient.startswith("'") and ingredient.endswith("'")) or \
                       (ingredient.startswith('"') and ingredient.endswith('"')):
                        ingredient = ingredient[1:-1]
                    ingredients.append(ingredient)
            else:
                # Single ingredient - remove quotes if present
                ingredient = ingredients_str.strip()
                if (ingredient.startswith("'") and ingredient.endswith("'")) or \
                   (ingredient.startswith('"') and ingredient.endswith('"')):
                    ingredient = ingredient[1:-1]
                ingredients = [ingredient]
            
            # Get scores for each ingredient
            meal_scores = []
            for ingredient in ingredients:
                ingredient_clean = ingredient.strip().lower()
                if ingredient_clean in ingredient_scores:
                    score, source = ingredient_scores[ingredient_clean]
                    meal_scores.append(score)
            
            # Calculate meal average
            if meal_scores:
                meal_score = sum(meal_scores) / len(meal_scores)
                df.loc[idx, 'usda_meal_score'] = round(meal_score, 2)
                scored_count += 1
                
        except Exception as e:
            continue
    
    print(f"✅ Meal scoring complete: {scored_count}/{total_meals} meals scored ({scored_count/total_meals*100:.1f}%)")
    return df


def create_usda_nutrition_scorer(api_key: Optional[str] = None) -> USDANutritionScorer:
    """
    Factory function to create a USDA nutrition scorer.
    
    Args:
        api_key (str, optional): USDA API key for higher rate limits
        
    Returns:
        USDANutritionScorer: Configured scorer instance
    """
    return USDANutritionScorer(api_key=api_key)


if __name__ == "__main__":
    # Test the USDA nutrition scorer
    scorer = create_usda_nutrition_scorer()
    
    test_ingredients = [
        "chicken breast",
        "broccoli", 
        "lu rou fan",
        "chocolate chip cookie"
    ]
    
    print("🧪 Testing USDA Nutrition Scorer")
    print("=" * 50)
    
    for ingredient in test_ingredients:
        score, source = scorer.get_ingredient_score(ingredient)
        print(f"{ingredient:20} | Score: {score:4.1f} | Source: {source}")