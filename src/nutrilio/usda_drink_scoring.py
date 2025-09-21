#!/usr/bin/env python3
"""
USDA-based Drink Categorization and Scoring Module

This module provides automated drink categorization and health scoring using the
USDA FoodData Central API as the primary data source, with intelligent fallbacks
for specialty drinks not in the USDA database.

Author: LifeLog Project
"""

import requests
import json
import time
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta


class USDDrinkScorer:
    """Handles drink categorization and scoring using USDA FoodData Central API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the USDA drink scorer.
        
        Args:
            api_key (str, optional): USDA API key for higher rate limits
        """
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        # Load API key from environment if not provided
        self.api_key = api_key or os.environ.get('USDA_API_KEY')
        self.cache_file = "files/work_files/nutrilio_work_files/usda_drinks_cache.json"
        
        # Load caches
        self.drinks_cache = self._load_cache()
        self.fallback_drinks = self._load_fallback_drinks()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
    def _load_cache(self) -> Dict:
        """Load drinks cache from JSON file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading drinks cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save drinks cache to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.drinks_cache, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving drinks cache: {e}")
    
    def _load_fallback_drinks(self) -> Dict:
        """Load hardcoded fallback drink scores for items not in USDA."""
        return {
            # Alcoholic beverages
            "sangria": {"score": 3.5, "category": "Alcoholic Beverage", "alcohol_content": "medium", "notes": "Wine-based cocktail with fruit, moderate alcohol"},
            "cocktail": {"score": 2.5, "category": "Alcoholic Beverage", "alcohol_content": "high", "notes": "Mixed alcoholic drink, high sugar and alcohol"},
            "strong beer": {"score": 3.0, "category": "Alcoholic Beverage", "alcohol_content": "high", "notes": "High alcohol content beer"},
            "craft beer": {"score": 4.0, "category": "Alcoholic Beverage", "alcohol_content": "medium", "notes": "Artisanal beer, moderate alcohol"},
            
            # Specialty healthy drinks
            "kombucha": {"score": 7.5, "category": "Fermented Beverage", "alcohol_content": "low", "notes": "Fermented tea, probiotics, low alcohol"},
            "kefir": {"score": 8.0, "category": "Fermented Beverage", "alcohol_content": "none", "notes": "Fermented milk drink, probiotics"},
            "coconut water": {"score": 8.5, "category": "Natural Beverage", "alcohol_content": "none", "notes": "Natural electrolytes, low calories"},
            "aloe vera juice": {"score": 7.0, "category": "Herbal Beverage", "alcohol_content": "none", "notes": "Plant-based drink, potential health benefits"},
            
            # Energy and sports drinks
            "energy drink": {"score": 2.0, "category": "Energy Drink", "alcohol_content": "none", "notes": "High caffeine, high sugar, artificial additives"},
            "sports drink": {"score": 4.0, "category": "Sports Drink", "alcohol_content": "none", "notes": "Electrolytes, moderate sugar for athletes"},
            "protein shake": {"score": 6.5, "category": "Protein Beverage", "alcohol_content": "none", "notes": "High protein, varies by ingredients"},
            
            # Regional/specialty drinks
            "bubble tea": {"score": 3.5, "category": "Specialty Beverage", "alcohol_content": "none", "notes": "Tea with tapioca pearls, high sugar"},
            "matcha latte": {"score": 6.0, "category": "Tea Beverage", "alcohol_content": "none", "notes": "Green tea powder with milk, antioxidants"},
            "golden milk": {"score": 7.5, "category": "Herbal Beverage", "alcohol_content": "none", "notes": "Turmeric milk, anti-inflammatory"},
            "chai latte": {"score": 5.5, "category": "Tea Beverage", "alcohol_content": "none", "notes": "Spiced tea with milk, moderate sugar"},
            
            # Functional beverages
            "probiotic drink": {"score": 7.5, "category": "Functional Beverage", "alcohol_content": "none", "notes": "Live cultures, digestive health"},
            "green juice": {"score": 8.0, "category": "Vegetable Juice", "alcohol_content": "none", "notes": "Cold-pressed vegetables, high nutrients"},
            "wheatgrass shot": {"score": 8.5, "category": "Superfood Beverage", "alcohol_content": "none", "notes": "Concentrated nutrients, detox properties"},
            
            # Sodas and soft drinks
            "diet soda": {"score": 3.0, "category": "Diet Soft Drink", "alcohol_content": "none", "notes": "Artificial sweeteners, no calories"},
            "sparkling water": {"score": 9.0, "category": "Carbonated Water", "alcohol_content": "none", "notes": "Plain carbonated water, very healthy"},
            "flavored water": {"score": 7.0, "category": "Flavored Water", "alcohol_content": "none", "notes": "Water with natural flavoring, low/no calories"},
            
            # Default categories for unknown items
            "alcoholic_unknown": {"score": 3.0, "category": "Alcoholic Beverage", "alcohol_content": "medium", "notes": "Default alcoholic beverage score"},
            "soft_drink_unknown": {"score": 2.5, "category": "Soft Drink", "alcohol_content": "none", "notes": "Default soft drink score"},
            "juice_unknown": {"score": 6.0, "category": "Fruit Juice", "alcohol_content": "none", "notes": "Default fruit juice score"},
            "tea_unknown": {"score": 8.0, "category": "Tea", "alcohol_content": "none", "notes": "Default tea score"},
            "coffee_unknown": {"score": 7.0, "category": "Coffee", "alcohol_content": "none", "notes": "Default coffee score"},
            "water_unknown": {"score": 9.5, "category": "Water", "alcohol_content": "none", "notes": "Default water score"}
        }
    
    def _rate_limit(self):
        """Enforce rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_usda_drink(self, drink_name: str) -> Optional[Dict]:
        """
        Search USDA FoodData Central for a drink/beverage.
        
        Args:
            drink_name (str): Name of the drink to search
            
        Returns:
            Dict: Nutrition data if found, None otherwise
        """
        # Check cache first
        cache_key = drink_name.lower().strip()
        if cache_key in self.drinks_cache:
            return self.drinks_cache[cache_key]
        
        # Rate limit API calls
        self._rate_limit()
        
        try:
            # Search for beverages with drink-specific keywords
            search_terms = [
                drink_name,
                f"{drink_name} beverage",
                f"{drink_name} drink"
            ]
            
            for search_term in search_terms:
                search_url = f"{self.base_url}/foods/search"
                params = {
                    "query": search_term,
                    "pageSize": 10
                }
                
                if self.api_key:
                    params["api_key"] = self.api_key
                
                response = requests.get(search_url, params=params, timeout=10)
                
                if response.status_code == 429:  # Rate limited
                    print(f"⚠️  Rate limited for '{drink_name}', using fallback")
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                # Get beverages from results
                foods = data.get('foods', [])
                beverage_matches = self._filter_beverage_results(foods, drink_name)
                
                if beverage_matches:
                    # Select best match
                    best_match = self._select_best_drink_match(beverage_matches, drink_name)
                    
                    if best_match:
                        # Get detailed nutrition info
                        nutrition_data = self._extract_drink_nutrition_data(best_match)
                        
                        # Cache the result
                        self.drinks_cache[cache_key] = nutrition_data
                        self._save_cache()
                        
                        return nutrition_data
                        
        except requests.exceptions.RequestException as e:
            print(f"⚠️  API error for '{drink_name}': {e}")
        except Exception as e:
            print(f"⚠️  Error processing '{drink_name}': {e}")
        
        return None
    
    def _filter_beverage_results(self, foods: List[Dict], search_term: str) -> List[Dict]:
        """
        Filter search results to focus on beverages/drinks.
        
        Args:
            foods (List[Dict]): List of food search results
            search_term (str): Original search term
            
        Returns:
            List[Dict]: Filtered list of beverage items
        """
        beverage_keywords = [
            'beverage', 'drink', 'juice', 'coffee', 'tea', 'soda', 'water',
            'wine', 'beer', 'alcohol', 'cocktail', 'smoothie', 'shake',
            'milk', 'cream', 'latte', 'espresso', 'cola', 'sprite', 'pepsi'
        ]
        
        beverage_matches = []
        for food in foods:
            description = food.get('description', '').lower()
            
            # Check if description contains beverage keywords
            if any(keyword in description for keyword in beverage_keywords):
                beverage_matches.append(food)
            # Also include if the search term is clearly a drink
            elif any(drink_word in search_term.lower() for drink_word in ['juice', 'coffee', 'tea', 'water', 'soda', 'beer', 'wine']):
                beverage_matches.append(food)
        
        return beverage_matches
    
    def _select_best_drink_match(self, drinks: List[Dict], search_term: str) -> Optional[Dict]:
        """
        Select the best drink match from search results.
        
        Args:
            drinks (List[Dict]): List of drink search results
            search_term (str): Original search term
            
        Returns:
            Dict: Best matching drink item
        """
        if not drinks:
            return None
        
        # Prioritize by data type (same as ingredients)
        priority_order = ['Foundation', 'SR Legacy', 'Survey (FNDDS)', 'Branded']
        
        for data_type in priority_order:
            for drink in drinks:
                if drink.get('dataType') == data_type:
                    # Additional check for relevance
                    description = drink.get('description', '').lower()
                    if search_term.lower() in description:
                        return drink
        
        # If no priority match, return first result
        return drinks[0]
    
    def _extract_drink_nutrition_data(self, drink_data: Dict) -> Dict:
        """
        Extract relevant nutrition data from USDA drink item.
        
        Args:
            drink_data (Dict): USDA drink data
            
        Returns:
            Dict: Standardized drink nutrition data
        """
        nutrition = {
            'usda_id': drink_data.get('fdcId'),
            'description': drink_data.get('description', ''),
            'data_type': drink_data.get('dataType', ''),
            'calories': 0,
            'sugar': 0,
            'sodium': 0,
            'caffeine': 0,
            'alcohol': 0,
            'carbs': 0,
            'protein': 0,
            'nutrients': {}
        }
        
        # Extract nutrients
        nutrients = drink_data.get('foodNutrients', [])
        
        # Map USDA nutrient IDs to our standard names (drink-focused)
        nutrient_mapping = {
            1008: 'calories',      # Energy
            2000: 'sugar',         # Total sugars  
            1093: 'sodium',        # Sodium
            1057: 'caffeine',      # Caffeine
            1018: 'alcohol',       # Alcohol, ethyl
            1005: 'carbs',         # Carbohydrate, by difference
            1003: 'protein'        # Protein
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
    
    def calculate_drink_health_score(self, nutrition_data: Dict) -> float:
        """
        Calculate health score for a drink from nutrition data.
        
        Args:
            nutrition_data (Dict): Nutrition data from USDA
            
        Returns:
            float: Health score from 1-10 (10 = healthiest)
        """
        if not nutrition_data:
            return 5.0  # Default neutral score
        
        # Base score for drinks (start lower than foods)
        score = 6.0
        
        # Get nutrition values (per 100ml typically)
        calories = nutrition_data.get('calories', 0)
        sugar = nutrition_data.get('sugar', 0)
        sodium = nutrition_data.get('sodium', 0)
        caffeine = nutrition_data.get('caffeine', 0)
        alcohol = nutrition_data.get('alcohol', 0)
        
        # Alcohol content (major negative factor)
        if alcohol > 0:
            if alcohol > 10:  # High alcohol content (>10%)
                score -= 4.0
            elif alcohol > 5:  # Medium alcohol content (5-10%)
                score -= 2.5
            elif alcohol > 1:  # Low alcohol content (1-5%)
                score -= 1.0
            else:  # Trace alcohol (<1%)
                score -= 0.5
        
        # Sugar content (major factor for drinks)
        if sugar > 15:  # Very high sugar (>15g per 100ml)
            score -= 3.0
        elif sugar > 10:  # High sugar (10-15g)
            score -= 2.0
        elif sugar > 5:  # Medium sugar (5-10g)
            score -= 1.0
        elif sugar > 2:  # Low sugar (2-5g)
            score -= 0.5
        # No penalty for very low sugar (<2g)
        
        # Calorie content
        if calories > 100:  # High calorie drinks
            score -= 1.5
        elif calories > 50:  # Medium calorie drinks
            score -= 1.0
        elif calories < 10:  # Very low calorie drinks
            score += 1.0
        
        # Sodium content
        if sodium > 200:  # High sodium
            score -= 1.0
        elif sodium > 100:  # Medium sodium
            score -= 0.5
        
        # Caffeine (moderate amounts can be positive)
        if caffeine > 200:  # Very high caffeine
            score -= 1.0
        elif caffeine > 100:  # High caffeine
            score -= 0.5
        elif 30 <= caffeine <= 100:  # Moderate caffeine (potentially positive)
            score += 0.5
        
        # Ensure score is within bounds
        score = max(1.0, min(10.0, score))
        
        return round(score, 2)
    
    def get_drink_score(self, drink_name: str) -> Tuple[float, str, str]:
        """
        Get health score and category for a single drink.
        
        Args:
            drink_name (str): Name of the drink
            
        Returns:
            Tuple[float, str, str]: (score, category, source) where source is 'usda', 'fallback', or 'default'
        """
        # Clean drink name
        drink_clean = drink_name.lower().strip()
        
        # Try USDA API first
        nutrition_data = self.search_usda_drink(drink_name)
        
        if nutrition_data:
            score = self.calculate_drink_health_score(nutrition_data)
            category = self._categorize_from_usda_data(nutrition_data)
            return score, category, 'usda'
        
        # Try fallback drinks
        if drink_clean in self.fallback_drinks:
            drink_info = self.fallback_drinks[drink_clean]
            return drink_info['score'], drink_info['category'], 'fallback'
        
        # Try category-based scoring
        category_info = self._get_category_info(drink_name)
        if category_info:
            return category_info[0], category_info[1], 'category'
        
        # Default neutral score
        return 5.0, 'Unknown Beverage', 'default'
    
    def _categorize_from_usda_data(self, nutrition_data: Dict) -> str:
        """
        Categorize drink based on USDA nutrition data.
        
        Args:
            nutrition_data (Dict): USDA nutrition data
            
        Returns:
            str: Drink category
        """
        description = nutrition_data.get('description', '').lower()
        alcohol = nutrition_data.get('alcohol', 0)
        sugar = nutrition_data.get('sugar', 0)
        caffeine = nutrition_data.get('caffeine', 0)
        
        # Categorize based on content
        if alcohol > 1:
            if alcohol > 10:
                return "Strong Alcoholic Beverage"
            else:
                return "Light Alcoholic Beverage"
        elif 'juice' in description:
            if sugar > 10:
                return "High-Sugar Fruit Juice"
            else:
                return "Fruit Juice"
        elif any(word in description for word in ['coffee', 'espresso', 'latte']):
            return "Coffee Beverage"
        elif any(word in description for word in ['tea', 'chai']):
            return "Tea Beverage"
        elif any(word in description for word in ['soda', 'cola', 'soft drink']):
            return "Soft Drink"
        elif 'water' in description:
            return "Water-Based Beverage"
        elif sugar > 15:
            return "High-Sugar Beverage"
        elif sugar < 2 and caffeine == 0:
            return "Low-Calorie Beverage"
        else:
            return "Mixed Beverage"
    
    def _get_category_info(self, drink_name: str) -> Optional[Tuple[float, str]]:
        """
        Get score and category based on drink name classification.
        
        Args:
            drink_name (str): Name of the drink
            
        Returns:
            Tuple[float, str]: (score, category) if classification is possible
        """
        drink_lower = drink_name.lower()
        
        # Alcoholic keywords
        alcohol_keywords = ['beer', 'wine', 'whiskey', 'vodka', 'rum', 'gin', 'tequila', 'brandy', 'cocktail', 'martini']
        if any(keyword in drink_lower for keyword in alcohol_keywords):
            return self.fallback_drinks['alcoholic_unknown']['score'], self.fallback_drinks['alcoholic_unknown']['category']
        
        # Soft drink keywords  
        soda_keywords = ['soda', 'cola', 'pepsi', 'sprite', 'fanta', 'dr pepper', 'mountain dew']
        if any(keyword in drink_lower for keyword in soda_keywords):
            return self.fallback_drinks['soft_drink_unknown']['score'], self.fallback_drinks['soft_drink_unknown']['category']
        
        # Juice keywords
        juice_keywords = ['juice', 'smoothie', 'nectar']
        if any(keyword in drink_lower for keyword in juice_keywords):
            return self.fallback_drinks['juice_unknown']['score'], self.fallback_drinks['juice_unknown']['category']
        
        # Tea keywords
        tea_keywords = ['tea', 'chai', 'matcha', 'green tea', 'black tea', 'herbal tea']
        if any(keyword in drink_lower for keyword in tea_keywords):
            return self.fallback_drinks['tea_unknown']['score'], self.fallback_drinks['tea_unknown']['category']
        
        # Coffee keywords
        coffee_keywords = ['coffee', 'espresso', 'latte', 'cappuccino', 'americano', 'mocha']
        if any(keyword in drink_lower for keyword in coffee_keywords):
            return self.fallback_drinks['coffee_unknown']['score'], self.fallback_drinks['coffee_unknown']['category']
        
        # Water keywords
        water_keywords = ['water', 'h2o', 'aqua']
        if any(keyword in drink_lower for keyword in water_keywords):
            return self.fallback_drinks['water_unknown']['score'], self.fallback_drinks['water_unknown']['category']
        
        return None


def extract_unique_drinks_from_dataframe(df) -> List[str]:
    """
    Extract all unique drinks from a dataframe containing drinks data.
    
    Args:
        df (DataFrame): Dataframe with drinks column containing drink data
        
    Returns:
        List[str]: List of unique drinks found across all entries
    """
    unique_drinks = set()
    
    # Look specifically for the drinks_list column (created by extract_data_count)
    drinks_list_col = None
    for col in df.columns:
        if col == 'drinks_list':
            drinks_list_col = col
            break
    
    if not drinks_list_col:
        print("⚠️  No 'drinks_list' column found in dataframe")
        print(f"Available columns: {list(df.columns)}")
        return []
    
    print(f"📊 Found drinks column: {drinks_list_col}")
    
    # Process the drinks_list column
    drink_rows = df[df[drinks_list_col].notna() & (df[drinks_list_col] != '')].copy()
    
    for _, row in drink_rows.iterrows():
        drinks_str = row.get(drinks_list_col, '')
        if pd.isna(drinks_str) or not drinks_str:
            continue
        
        # Skip numeric values (these are quantities, not drink names)
        try:
            float(drinks_str)
            print(f"⚠️  Skipping numeric value: {drinks_str}")
            continue
        except (ValueError, TypeError):
            pass  # Good, it's not a number
            
        try:
            # Parse drinks from the drinks_list format
            # Format is typically: 'Beer', 'Wine', 'Strong beer'
            if ',' in drinks_str:
                drinks = []
                for drink in drinks_str.split(','):
                    drink = drink.strip()
                    # Remove surrounding quotes if present
                    if (drink.startswith("'") and drink.endswith("'")) or \
                       (drink.startswith('"') and drink.endswith('"')):
                        drink = drink[1:-1]
                    drinks.append(drink)
            else:
                # Single drink - remove quotes if present
                drink = drinks_str.strip()
                if (drink.startswith("'") and drink.endswith("'")) or \
                   (drink.startswith('"') and drink.endswith('"')):
                    drink = drink[1:-1]
                drinks = [drink]
            
            # Add each drink to the set
            for drink in drinks:
                cleaned = drink.strip().lower()
                if cleaned and cleaned != '':
                    unique_drinks.add(cleaned)
                    
        except Exception as e:
            print(f"⚠️  Error parsing drinks from: {drinks_str}")
            continue
    
    return sorted(list(unique_drinks))


def load_drink_scores_database() -> Dict[str, Tuple[float, str, str]]:
    """
    Load the master drink scores database.
    
    Returns:
        Dict[str, Tuple[float, str, str]]: Dictionary mapping drink -> (score, category, source)
    """
    database_file = "files/work_files/nutrilio_work_files/drink_scores_database.json"
    
    if os.path.exists(database_file):
        try:
            with open(database_file, 'r') as f:
                data = json.load(f)
            print(f"🍹 Loaded {len(data)} drink scores from database")
            return data
        except Exception as e:
            print(f"⚠️  Error loading drink database: {e}")
    
    print("🍹 No drink database found - creating new one")
    return {}


def save_drink_scores_database(drink_scores: Dict[str, Tuple[float, str, str]]):
    """
    Save the drink scores database.
    
    Args:
        drink_scores (Dict): Dictionary mapping drink -> (score, category, source)
    """
    database_file = "files/work_files/nutrilio_work_files/drink_scores_database.json"
    
    try:
        os.makedirs(os.path.dirname(database_file), exist_ok=True)
        with open(database_file, 'w') as f:
            json.dump(drink_scores, f, indent=2)
        print(f"💾 Saved {len(drink_scores)} drink scores to database")
    except Exception as e:
        print(f"❌ Error saving drink database: {e}")


def score_all_drinks_efficient(unique_drinks: List[str], api_key: Optional[str] = None) -> Dict[str, Tuple[float, str, str]]:
    """
    Efficiently score drinks using pre-existing database and only scoring new ones.
    
    Args:
        unique_drinks (List[str]): List of unique drink names
        api_key (str, optional): USDA API key for higher rate limits
        
    Returns:
        Dict[str, Tuple[float, str, str]]: Dictionary mapping drink -> (score, category, source)
    """
    print(f"🍹 Processing {len(unique_drinks)} unique drinks...")
    
    # Load existing drink scores database
    drink_scores = load_drink_scores_database()
    
    # Find drinks that need scoring
    need_scoring = []
    for drink in unique_drinks:
        if drink.lower().strip() not in drink_scores:
            need_scoring.append(drink)
    
    already_scored = len(unique_drinks) - len(need_scoring)
    print(f"📊 Status: {already_scored} already scored, {len(need_scoring)} need scoring")
    
    if len(need_scoring) == 0:
        print("⚡ All drinks already in database - no API calls needed!")
    else:
        print(f"🔄 Scoring {len(need_scoring)} new drinks...")
        
        # Create scorer only for new drinks
        scorer = USDDrinkScorer(api_key=api_key)
        
        # Counters for new drinks only
        usda_count = 0
        fallback_count = 0 
        category_count = 0
        default_count = 0
        
        # Score only the new drinks
        for i, drink in enumerate(need_scoring):
            score, category, source = scorer.get_drink_score(drink)
            drink_scores[drink.lower().strip()] = (score, category, source)
            
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
            if (i + 1) % 5 == 0 or i == len(need_scoring) - 1:
                print(f"   Processed {i + 1}/{len(need_scoring)} new drinks...")
        
        # Save updated database
        save_drink_scores_database(drink_scores)
        
        # Report on new drinks
        total_new = len(need_scoring)
        print(f"\n📊 New Drink Results:")
        print(f"   🆕 USDA: {usda_count}/{total_new} ({usda_count/total_new*100:.1f}%)")
        print(f"   🌏 Fallback: {fallback_count}/{total_new} ({fallback_count/total_new*100:.1f}%)")
        print(f"   🏷️ Category: {category_count}/{total_new} ({category_count/total_new*100:.1f}%)")
        print(f"   ❓ Default: {default_count}/{total_new} ({default_count/total_new*100:.1f}%)")
    
    # Check for default scores in ALL drinks (new and existing) and flag them
    flag_default_drinks(drink_scores, unique_drinks)
    
    # Create final result mapping with original drink names
    final_scores = {}
    for drink in unique_drinks:
        key = drink.lower().strip()
        if key in drink_scores:
            final_scores[drink] = drink_scores[key]
        else:
            # Fallback - this shouldn't happen
            final_scores[drink] = (5.0, 'Unknown Beverage', 'default')
    
    print(f"\n✅ Final database contains {len(drink_scores)} total drinks")
    return final_scores


def flag_default_drinks(drink_scores: Dict[str, Tuple[float, str, str]], current_drinks: List[str]):
    """
    Flag drinks with default scores and save them to a flagged file for review.
    
    Args:
        drink_scores (Dict): All drink scores
        current_drinks (List): Current batch of drinks being processed
    """
    flagged_file = "files/work_files/nutrilio_work_files/flagged_default_drinks.json"
    
    # Find all default score drinks from current batch
    current_defaults = []
    for drink in current_drinks:
        key = drink.lower().strip()
        if key in drink_scores:
            score, category, source = drink_scores[key]
            if source == 'default':
                current_defaults.append({
                    "drink": drink,
                    "score": score,
                    "category": category,
                    "source": source,
                    "last_seen": pd.Timestamp.now().isoformat(),
                    "suggestion": f"Consider adding '{drink}' to fallback database with appropriate score and category"
                })
    
    if current_defaults:
        print(f"\n🚨 Found {len(current_defaults)} drinks with default scores:")
        for item in current_defaults:
            print(f"   ❓ '{item['drink']}' → {item['score']} ({item['category']}) - needs manual scoring")
        
        # Load existing flagged drinks
        existing_flagged = {}
        if os.path.exists(flagged_file):
            try:
                with open(flagged_file, 'r') as f:
                    existing_flagged = json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading flagged file: {e}")
        
        # Update flagged drinks (preserve existing, add new)
        for item in current_defaults:
            key = item['drink'].lower().strip()
            existing_flagged[key] = item
        
        # Save updated flagged drinks
        try:
            os.makedirs(os.path.dirname(flagged_file), exist_ok=True)
            with open(flagged_file, 'w') as f:
                json.dump(existing_flagged, f, indent=2)
            print(f"📝 Updated flagged drinks file: {flagged_file}")
            print(f"💡 Tip: Review this file to add missing drinks to the fallback database")
        except Exception as e:
            print(f"❌ Error saving flagged file: {e}")
    else:
        print(f"\n✅ No default score drinks found in current batch")


def create_usda_drink_scorer(api_key: Optional[str] = None) -> USDDrinkScorer:
    """
    Factory function to create a USDA drink scorer.
    
    Args:
        api_key (str, optional): USDA API key for higher rate limits
        
    Returns:
        USDDrinkScorer: Configured scorer instance
    """
    return USDDrinkScorer(api_key=api_key)


if __name__ == "__main__":
    # Test the USDA drink scorer
    scorer = create_usda_drink_scorer()
    
    test_drinks = [
        "coffee",
        "orange juice", 
        "beer",
        "kombucha",
        "energy drink",
        "water"
    ]
    
    print("🍹 Testing USDA Drink Scorer")
    print("=" * 50)
    
    for drink in test_drinks:
        score, category, source = scorer.get_drink_score(drink)
        print(f"{drink:15} | Score: {score:4.1f} | Category: {category:20} | Source: {source}")