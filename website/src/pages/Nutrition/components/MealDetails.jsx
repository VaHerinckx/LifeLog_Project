import React from 'react';
import { X, Clock, MapPin, TrendingUp, Croissant, Coffee, Sandwich, Cookie, UtensilsCrossed, Moon } from 'lucide-react';
import './MealDetails.css';
import { StarRating } from '../../../components/ui';
import { formatDate } from '../../../utils';

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

const MealDetails = ({ meal, onClose }) => {
  if (!meal) return null;

  const MealIcon = getMealIcon(meal.meal);

  // Parse food and drinks lists for display
  const parseFoodList = (foodListStr) => {
    if (!foodListStr) return [];
    // Remove outer quotes and split by delimiter
    return foodListStr
      .replace(/^["']|["']$/g, '')
      .split('|')
      .map(item => item.trim())
      .filter(item => item);
  };

  const foodItems = parseFoodList(meal.food_list);
  const drinkItems = parseFoodList(meal.drinks_list);

  return (
    <div className="meal-details-overlay" onClick={onClose}>
      <div className="meal-details-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="meal-details-content">
          <div className="meal-details-icon">
            <MealIcon className="meal-icon-large" size={64} strokeWidth={1.5} />
          </div>

          <div className="meal-details-info">
            <h2>{meal.meal}</h2>
            <h3>{meal.weekday}, {formatDate(meal.date)}</h3>

            <div className="meal-details-meta">
              <div className="meta-item">
                <Clock size={18} />
                <span>{meal.time}</span>
              </div>

              {meal.places && (
                <div className="meta-item">
                  <MapPin size={18} />
                  <span>{meal.places}</span>
                </div>
              )}

              {meal.usda_meal_score && (
                <div className="meta-item">
                  <TrendingUp size={18} />
                  <span>USDA Score: {parseFloat(meal.usda_meal_score).toFixed(2)}</span>
                </div>
              )}
            </div>

            {/* Food & Drinks Section */}
            <div className="meal-content-section">
              {foodItems.length > 0 && (
                <div className="food-section">
                  <h4>Food</h4>
                  <ul className="food-list">
                    {foodItems.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {drinkItems.length > 0 && (
                <div className="drinks-section">
                  <h4>Drinks</h4>
                  <ul className="food-list">
                    {drinkItems.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Meal Assessment Section */}
            {meal.meal_assessment && parseFloat(meal.meal_assessment) > 0 && (
              <div className="meal-ratings">
                <div className="rating-item">
                  <label>My Rating:</label>
                  <div className="rating-display">
                    <StarRating rating={parseFloat(meal.meal_assessment)} size={18} />
                    <span>{parseFloat(meal.meal_assessment).toFixed(1)}</span>
                    {meal.meal_assessment_text && (
                      <span className="rating-text">({meal.meal_assessment_text})</span>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Meal Details Section (only show if there are values) */}
            {(meal.origin || meal.amount_text) && (
              <div className="meal-details-section">
                <h4>Meal Details</h4>
                <div className="details-grid">
                  {meal.origin && (
                    <div className="detail-item">
                      <span className="detail-label">Origin:</span>
                      <span className="detail-value">{meal.origin}</span>
                    </div>
                  )}
                  {meal.amount_text && (
                    <div className="detail-item">
                      <span className="detail-label">Amount:</span>
                      <span className="detail-value">{meal.amount_text}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Tags */}
            {(meal.places || meal.origin) && (
              <div className="meal-categories">
                {meal.places && (
                  <span className="meal-category location-tag">
                    {meal.places}
                  </span>
                )}
                {meal.origin && (
                  <span className="meal-category origin-tag">
                    {meal.origin}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MealDetails;
