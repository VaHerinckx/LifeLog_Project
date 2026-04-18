/**
 * StarRating Component
 *
 * Displays a star rating visualization.
 * Replaces duplicate StarRating implementations in Reading, Movies, and BookDetails components.
 */

import React from 'react';
import { Star, StarHalf } from 'lucide-react';
import './StarRating.css';

const StarRating = ({ rating, size = 16, className = '' }) => {
  // Calculate full stars, half star, and empty stars
  const fullStars = Math.floor(rating);
  const hasHalfStar = rating % 1 >= 0.5;
  const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

  return (
    <div className={`star-rating ${className}`}>
      {/* Full stars */}
      {[...Array(fullStars)].map((_, i) => (
        <Star
          key={`full-${i}`}
          className="star star-filled"
          size={size}
          fill="#EAC435"
        />
      ))}

      {/* Half star */}
      {hasHalfStar && (
        <StarHalf
          key="half"
          className="star star-half"
          size={size}
        />
      )}

      {/* Empty stars */}
      {[...Array(emptyStars)].map((_, i) => (
        <Star
          key={`empty-${i}`}
          className="star star-empty"
          size={size}
        />
      ))}
    </div>
  );
};

export default StarRating;
