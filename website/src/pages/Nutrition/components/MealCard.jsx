import PropTypes from 'prop-types';
import { Clock, MapPin, Croissant, Coffee, Sandwich, Cookie, UtensilsCrossed, Moon } from 'lucide-react';
import { StarRating } from '../../../components/ui';
import { formatDate } from '../../../utils';
import './MealCard.css';

/**
 * Get the appropriate icon for a meal type
 * @param {string} mealType - The meal type (e.g., 'Breakfast', 'Lunch')
 * @returns {React.Component} Lucide icon component
 */
const getMealIcon = (mealType) => {
  const iconMap = {
    'Breakfast': Croissant,
    'Morning snack': Coffee,
    'Lunch': Sandwich,
    'Afternoon snack': Cookie,
    'Dinner': UtensilsCrossed,
    'Night snack': Moon,
  };
  return iconMap[mealType] || UtensilsCrossed; // fallback to utensils for unknown types
};

/**
 * MealCard - Displays meal information in grid or list view
 *
 * Self-contained component that adapts its layout based on viewMode prop.
 * All styling is contained in MealCard.css.
 *
 * Grid view: Vertical layout with meal info
 * List view: Horizontal layout with meal info
 *
 * @param {Object} meal - Meal data object (grouped by meal_id)
 * @param {string} viewMode - Display mode ('grid' | 'list')
 * @param {Function} onClick - Callback when card is clicked
 */
const MealCard = ({ meal, viewMode = 'grid', onClick }) => {
  const MealIcon = getMealIcon(meal.meal);
  const cardClass = `meal-card meal-card--${viewMode}`;

  // Combine and truncate food and drinks list for preview
  const getCombinedFoodDrinks = (foodList, drinksList, maxLength = 60) => {
    const parts = [];

    if (foodList && foodList.trim()) {
      parts.push(foodList.trim());
    }

    if (drinksList && drinksList.trim()) {
      parts.push(drinksList.trim());
    }

    const combined = parts.join(' | ');

    if (combined.length <= maxLength) return combined;
    return combined.substring(0, maxLength) + '...';
  };

  // Format USDA score for display
  const formatUSDAScore = (score) => {
    return score ? parseFloat(score).toFixed(2) : 'N/A';
  };

  // List view - horizontal layout
  if (viewMode === 'list') {
    return (
      <div className={cardClass} onClick={() => onClick(meal)}>
        <div className="meal-icon-badge"><MealIcon size={24} strokeWidth={2} /></div>

        <div className="meal-info">
          <h3 className="meal-title">{meal.meal}</h3>
          <p className="meal-food-list">{getCombinedFoodDrinks(meal.food_list, meal.drinks_list, 80)}</p>

          <div className="meal-meta">
            {meal.meal_assessment && parseFloat(meal.meal_assessment) > 0 && (
              <div className="rating-container">
                <StarRating rating={parseFloat(meal.meal_assessment)} size={16} />
                <span className="rating-value">{parseFloat(meal.meal_assessment).toFixed(1)}</span>
              </div>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <Clock size={16} />
              <span>{meal.time}</span>
            </div>

            {meal.usda_meal_score && (
              <span className="usda-score-badge">
                USDA: {formatUSDAScore(meal.usda_meal_score)}
              </span>
            )}
          </div>
        </div>

        <div className="meal-tags">
          {meal.places && (
            <span className="meal-tag location-tag">
              <MapPin size={14} />
              {meal.places}
            </span>
          )}
          {meal.origin && (
            <span className="meal-tag origin-tag">{meal.origin}</span>
          )}
        </div>
      </div>
    );
  }

  // Grid view - vertical layout (default)
  return (
    <div className={cardClass} onClick={() => onClick(meal)}>
      <div className="meal-icon-badge"><MealIcon size={24} strokeWidth={2} /></div>

      <div className="meal-info">
        <h3 className="meal-title" title={meal.meal}>{meal.meal}</h3>
        <p className="meal-weekday">{meal.weekday}</p>

        <p className="meal-food-list" title={getCombinedFoodDrinks(meal.food_list, meal.drinks_list, 1000)}>
          {getCombinedFoodDrinks(meal.food_list, meal.drinks_list, 60)}
        </p>

        {meal.meal_assessment && parseFloat(meal.meal_assessment) > 0 && (
          <div className="rating-container">
            <StarRating rating={parseFloat(meal.meal_assessment)} size={16} />
            <span>{parseFloat(meal.meal_assessment).toFixed(1)}</span>
          </div>
        )}

        <div className="meal-meta">
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Clock size={14} />
            <span>{meal.time}</span>
          </div>

          {meal.usda_meal_score && (
            <span className="usda-score-badge">
              USDA: {formatUSDAScore(meal.usda_meal_score)}
            </span>
          )}
        </div>

        {(meal.places || meal.origin) && (
          <div className="meal-tags">
            {meal.places && (
              <span className="meal-tag location-tag">
                <MapPin size={12} />
                {meal.places}
              </span>
            )}
            {meal.origin && (
              <span className="meal-tag origin-tag">{meal.origin}</span>
            )}
          </div>
        )}

        <div className="meal-date">
          <span className="date-label">Date:</span>
          <span className="date-value">{formatDate(meal.date)}</span>
        </div>
      </div>
    </div>
  );
};

MealCard.propTypes = {
  meal: PropTypes.shape({
    meal_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    meal: PropTypes.string.isRequired,
    date: PropTypes.oneOfType([PropTypes.string, PropTypes.instanceOf(Date)]).isRequired,
    weekday: PropTypes.string,
    time: PropTypes.string,
    food_list: PropTypes.string,
    drinks_list: PropTypes.string,
    usda_meal_score: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    places: PropTypes.string,
    origin: PropTypes.string,
    amount: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    amount_text: PropTypes.string,
    meal_assessment: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    meal_assessment_text: PropTypes.string,
  }).isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list']),
  onClick: PropTypes.func.isRequired,
};

export default MealCard;
