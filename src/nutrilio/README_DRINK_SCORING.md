# USDA-Based Drink Scoring System

This module provides automated drink categorization and health scoring using the USDA FoodData Central API, replacing the manual ChatGPT + Excel workflow.

## Features

- **Automated Processing**: No manual intervention needed for drink categorization
- **USDA Integration**: Primary nutrition data source from FoodData Central API
- **Efficient Database**: Master database prevents re-scoring drinks across pipeline runs
- **Fallback System**: Hardcoded scores for specialty drinks not in USDA database
- **Default Flagging**: Unknown drinks are flagged for manual review
- **Health Scoring**: 1-10 scale based on alcohol, sugar, caffeine, and calorie content

## Usage

### Basic Usage

```python
from src.nutrilio.usda_drink_scoring import create_usda_drink_scorer

# Create scorer instance
scorer = create_usda_drink_scorer()

# Score a single drink
score, category, source = scorer.get_drink_score("kombucha")
print(f"Kombucha: {score}/10 ({category}) - source: {source}")
```

### Integration in Pipeline

The drink scoring is automatically integrated into the main Nutrilio processing pipeline:

```python
# Automatically called in create_nutrilio_files()
df = add_usda_drink_scoring_efficient(df, use_usda_scoring=True)
```

## Scoring Algorithm

### Health Score Calculation (1-10 scale)

**Starting Score**: 6.0 (drinks start lower than foods)

**Negative Factors**:
- **Alcohol Content**: -4.0 (high) to -0.5 (trace)
- **Sugar Content**: -3.0 (>15g) to -0.5 (2-5g)
- **High Calories**: -1.5 (>100 cal) to -1.0 (50-100 cal)
- **High Sodium**: -1.0 (>200mg) to -0.5 (100-200mg)
- **Excessive Caffeine**: -1.0 (>200mg) to -0.5 (>100mg)

**Positive Factors**:
- **Very Low Calories**: +1.0 (<10 cal)
- **Moderate Caffeine**: +0.5 (30-100mg, like tea/coffee)

### Categories

- **Alcoholic Beverages**: Strong vs Light alcohol based on alcohol %
- **Fruit Juices**: High-sugar vs regular based on sugar content
- **Coffee/Tea Beverages**: Caffeine-containing beverages
- **Soft Drinks**: Sodas and carbonated beverages
- **Water-Based**: Plain and flavored waters
- **Specialty**: Energy drinks, sports drinks, fermented beverages

## Database Files

- **`drink_scores_database.json`**: Master database of all scored drinks
- **`flagged_default_drinks.json`**: Drinks needing manual review
- **`usda_drinks_cache.json`**: USDA API response cache

## API Configuration

Set your USDA API key in environment:

```bash
export USDA_API_KEY="your_api_key_here"
```

Get a free API key at: https://fdc.nal.usda.gov/api-key-signup/

## Fallback Drinks

The system includes hardcoded scores for specialty drinks not in USDA:

- **Fermented**: kombucha (7.5), kefir (8.0)
- **Specialty**: bubble tea (3.5), matcha latte (6.0), golden milk (7.5)
- **Energy/Sports**: energy drink (2.0), sports drink (4.0), protein shake (6.5)
- **Functional**: probiotic drink (7.5), green juice (8.0), wheatgrass shot (8.5)
- **Regional**: And many more specialty beverages

## Performance

- **Efficient**: Only scores new drinks, reuses existing scores
- **Fast**: Master database lookup for known drinks
- **Scalable**: Handles unlimited drinks without performance degradation
- **Robust**: Graceful fallbacks when API is unavailable

## Migration from Legacy System

The new system replaces:
- ❌ `check_new_drinks()` function (manual Excel checking)
- ❌ `drinks_category()` function (ChatGPT prompts)
- ❌ `nutrilio_drinks_category.xlsx` file (manual categorization)

With:
- ✅ Automated USDA-based scoring
- ✅ JSON database management
- ✅ Intelligent fallback system
- ✅ Default drink flagging

## Error Handling

- **API Failures**: Graceful fallback to category-based scoring
- **Unknown Drinks**: Default 5.0 score with flagging for review
- **Rate Limiting**: 1-second intervals between API calls
- **Network Issues**: Cache-based fallbacks

## Testing

Run the test suite:

```bash
python src/nutrilio/usda_drink_scoring.py
```

This tests scoring for various drink types and validates the fallback system.