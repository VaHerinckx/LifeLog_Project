import PropTypes from 'prop-types';
import { Activity, Moon, Clock, StickyNote, Brain } from 'lucide-react';
import { StarRating } from '../../../components/ui';
import { formatDate } from '../../../utils';
import './HealthCard.css';

/**
 * Get the appropriate emoji for an evaluation score (1-5)
 * @param {number} rating - The evaluation score
 * @returns {string} Emoji string
 */
const getEvaluationEmoji = (rating) => {
  if (!rating || rating === 0) return '💪'; // Default health emoji
  const score = Math.round(rating);

  const emojiMap = {
    5: '😄',  // Very happy
    4: '🙂',  // Happy
    3: '😐',  // Neutral
    2: '😕',  // Unhappy
    1: '😞',  // Very unhappy
  };
  return emojiMap[score] || '💪';
};

/**
 * Format time string (HH:MM) to readable format
 */
const formatTime = (timeStr) => {
  if (!timeStr) return null;
  return timeStr;
};

/**
 * HealthCard - Displays daily health information in grid or list view
 *
 * Self-contained component that adapts its layout based on viewMode prop.
 * All styling is contained in HealthCard.css.
 *
 * Grid view: Vertical layout with evaluation emoji in top right
 * List view: Horizontal layout with evaluation emoji in top right
 *
 * @param {Object} day - Health day data object
 * @param {string} viewMode - Display mode ('grid' | 'list')
 * @param {Function} onClick - Callback when card is clicked
 */
const HealthCard = ({ day, viewMode = 'grid', onClick }) => {
  const cardClass = `health-card health-card--${viewMode}`;
  const healthEmoji = getEvaluationEmoji(day.overall_evaluation);

  // Format helper functions
  const formatSteps = (steps) => {
    if (!steps || steps === 0) return 'No data';
    return `${Math.round(steps).toLocaleString()} steps`;
  };

  const formatSleepMinutes = (minutes) => {
    if (!minutes || minutes === 0) return 'No data';
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  };

  // List view - horizontal layout
  if (viewMode === 'list') {
    return (
      <div className={cardClass} onClick={() => onClick(day)}>
        <div className="health-emoji-badge">{healthEmoji}</div>

        <div className="health-info">
          <h3 className="health-date">{formatDate(day.date)}</h3>

          <div className="health-meta">
            <div className="health-metric">
              <Activity size={16} />
              <span>{formatSteps(day.total_steps)}</span>
            </div>

            <div className="health-metric">
              <Moon size={16} />
              <span>{formatSleepMinutes(day.total_sleep_minutes)}</span>
            </div>

            {formatTime(day.sleep_start_time) && formatTime(day.wake_up_time) && (
              <div className="health-metric">
                <Clock size={16} />
                <span>{day.sleep_start_time} - {day.wake_up_time}</span>
              </div>
            )}
          </div>

          {day.overall_evaluation && day.overall_evaluation > 0 && (
            <div className="health-rating">
              <span className="rating-label">Overall:</span>
              <StarRating rating={day.overall_evaluation} maxRating={5} size={16} />
            </div>
          )}
        </div>

        <div className="health-indicators">
          {day.dreams > 0 && (
            <span className="health-indicator" title="Dream recorded">
              <Brain size={14} />
            </span>
          )}
          {day.notes_summary && (
            <span className="health-indicator" title="Notes available">
              <StickyNote size={14} />
            </span>
          )}
        </div>
      </div>
    );
  }

  // Grid view - vertical layout (default)
  return (
    <div className={cardClass} onClick={() => onClick(day)}>
      <div className="health-emoji-badge">{healthEmoji}</div>

      <div className="health-info">
        <h3 className="health-date" title={formatDate(day.date)}>
          {formatDate(day.date)}
        </h3>

        {day.overall_evaluation && day.overall_evaluation > 0 && (
          <div className="health-rating">
            <StarRating rating={day.overall_evaluation} maxRating={5} size={16} />
            <span className="rating-value">{day.overall_evaluation.toFixed(1)}</span>
          </div>
        )}

        <div className="health-meta">
          <div className="health-metric">
            <Activity size={16} />
            <span>{formatSteps(day.total_steps)}</span>
          </div>

          <div className="health-metric">
            <Moon size={16} />
            <span>{formatSleepMinutes(day.total_sleep_minutes)}</span>
          </div>

          {formatTime(day.sleep_start_time) && formatTime(day.wake_up_time) && (
            <div className="health-metric">
              <Clock size={16} />
              <span>{day.sleep_start_time} - {day.wake_up_time}</span>
            </div>
          )}
        </div>

        <div className="health-indicators-grid">
          {day.dreams > 0 && (
            <span className="health-indicator" title="Dream recorded">
              <Brain size={14} /> Dream
            </span>
          )}
          {day.notes_summary && (
            <span className="health-indicator" title="Notes available">
              <StickyNote size={14} /> Notes
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

HealthCard.propTypes = {
  day: PropTypes.shape({
    date: PropTypes.oneOfType([PropTypes.string, PropTypes.instanceOf(Date)]).isRequired,
    total_steps: PropTypes.number,
    total_sleep_minutes: PropTypes.number,
    overall_evaluation: PropTypes.number,
    sleep_start_time: PropTypes.string,
    wake_up_time: PropTypes.string,
    dreams: PropTypes.number,
    notes_summary: PropTypes.string,
  }).isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list']),
  onClick: PropTypes.func.isRequired,
};

export default HealthCard;
