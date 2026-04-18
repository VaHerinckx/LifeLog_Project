import PropTypes from 'prop-types';
import './EvaluationFace.css';

/**
 * EvaluationFace - Professional mood indicator based on evaluation score (1-5)
 *
 * Displays a minimalist face icon that changes expression based on score:
 * 5 - Very happy (big smile, raised eyes)
 * 4 - Happy (smile, normal eyes)
 * 3 - Neutral (straight mouth, normal eyes)
 * 2 - Unhappy (frown, normal eyes)
 * 1 - Very unhappy (big frown, sad eyes)
 *
 * @param {number} rating - Evaluation score (1-5)
 * @param {string} size - Size variant ('sm', 'md', 'lg')
 */
const EvaluationFace = ({ rating, size = 'md' }) => {
  if (!rating || rating === 0) return null;

  const sizeClass = `evaluation-face--${size}`;

  // Round to nearest integer for consistent display
  const score = Math.round(rating);

  // Define face configurations based on score
  const getFaceConfig = () => {
    switch (score) {
      case 5:
        return {
          eyeClass: 'eyes-happy',
          mouthClass: 'mouth-very-happy',
          color: 'var(--color-success, #22c55e)'
        };
      case 4:
        return {
          eyeClass: 'eyes-normal',
          mouthClass: 'mouth-happy',
          color: 'var(--color-info, #3b82f6)'
        };
      case 3:
        return {
          eyeClass: 'eyes-normal',
          mouthClass: 'mouth-neutral',
          color: 'var(--color-warning, #f59e0b)'
        };
      case 2:
        return {
          eyeClass: 'eyes-normal',
          mouthClass: 'mouth-sad',
          color: 'var(--color-error-light, #fb923c)'
        };
      case 1:
        return {
          eyeClass: 'eyes-sad',
          mouthClass: 'mouth-very-sad',
          color: 'var(--color-error, #ef4444)'
        };
      default:
        return {
          eyeClass: 'eyes-normal',
          mouthClass: 'mouth-neutral',
          color: 'var(--color-text-gray)'
        };
    }
  };

  const { eyeClass, mouthClass, color } = getFaceConfig();

  return (
    <div className={`evaluation-face ${sizeClass}`} style={{ '--face-color': color }}>
      <svg viewBox="0 0 100 100" className="face-svg">
        {/* Face circle */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
        />

        {/* Eyes */}
        <g className={`face-eyes ${eyeClass}`}>
          {/* Left eye */}
          {eyeClass === 'eyes-sad' ? (
            // Sad eyes - curved down
            <path
              d="M 30 35 Q 35 40 40 35"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          ) : eyeClass === 'eyes-happy' ? (
            // Happy eyes - curved up
            <path
              d="M 30 40 Q 35 35 40 40"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          ) : (
            // Normal eyes - dots
            <circle cx="35" cy="38" r="3" fill="currentColor" />
          )}

          {/* Right eye */}
          {eyeClass === 'eyes-sad' ? (
            <path
              d="M 60 35 Q 65 40 70 35"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          ) : eyeClass === 'eyes-happy' ? (
            <path
              d="M 60 40 Q 65 35 70 40"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          ) : (
            <circle cx="65" cy="38" r="3" fill="currentColor" />
          )}
        </g>

        {/* Mouth */}
        <g className={`face-mouth ${mouthClass}`}>
          {mouthClass === 'mouth-very-happy' ? (
            // Big smile
            <path
              d="M 25 60 Q 50 80 75 60"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          ) : mouthClass === 'mouth-happy' ? (
            // Regular smile
            <path
              d="M 30 65 Q 50 75 70 65"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          ) : mouthClass === 'mouth-neutral' ? (
            // Straight line
            <line
              x1="30"
              y1="68"
              x2="70"
              y2="68"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          ) : mouthClass === 'mouth-sad' ? (
            // Regular frown
            <path
              d="M 30 72 Q 50 62 70 72"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          ) : (
            // Big frown
            <path
              d="M 25 75 Q 50 55 75 75"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
            />
          )}
        </g>
      </svg>
    </div>
  );
};

EvaluationFace.propTypes = {
  rating: PropTypes.number,
  size: PropTypes.oneOf(['sm', 'md', 'lg'])
};

export default EvaluationFace;
