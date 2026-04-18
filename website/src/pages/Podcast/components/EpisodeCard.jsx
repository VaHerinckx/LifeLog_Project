import { memo } from 'react';
import PropTypes from 'prop-types';
import { Calendar, Clock, Headphones } from 'lucide-react';
import { formatDate, formatPodcastDuration, formatCompletion } from '../../../utils';
import './EpisodeCard.css';

/**
 * EpisodeCard - Displays podcast episode information in grid or list view
 *
 * Self-contained component that adapts its layout based on viewMode prop.
 * All styling is contained in EpisodeCard.css.
 *
 * Grid view: Vertical layout with artwork on top, info below
 * List view: Horizontal layout with artwork on left, info on right
 *
 * Memoized to prevent unnecessary re-renders when parent component updates.
 *
 * @param {Object} episode - Episode data object
 * @param {string} viewMode - Display mode ('grid' | 'list')
 * @param {Function} onClick - Callback when card is clicked
 */
const EpisodeCard = memo(({ episode, viewMode = 'grid', onClick }) => {
  const cardClass = `episode-card episode-card--${viewMode}`;

  // List view - horizontal layout
  if (viewMode === 'list') {
    return (
      <div className={cardClass} onClick={() => onClick(episode)}>
        <div className="episode-artwork-container">
          <img
            src={episode.artwork_url || "/api/placeholder/80/80"}
            alt={`${episode.podcast_name} artwork`}
            className="episode-artwork"
            loading="lazy"
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = "/api/placeholder/80/80";
            }}
          />
        </div>

        <div className="episode-info">
          <h3 className="episode-title">{episode.episode_title}</h3>
          <p className="podcast-name">{episode.podcast_name}</p>

          <div className="episode-meta">
            <div className="meta-item">
              <Calendar size={14} />
              <span>Published: {formatDate(episode.published_date)}</span>
            </div>
            <div className="meta-item">
              <Clock size={14} />
              <span>Listened: {formatDate(episode.listened_date)}</span>
            </div>
            <div className="meta-item">
              <Headphones size={14} />
              <span>{formatPodcastDuration(episode.duration_seconds)}</span>
            </div>
            <div className="completion-info">
              <span className="completion-text">{formatCompletion(episode.completion_percent)}% complete</span>
            </div>
          </div>
        </div>

        <div className="episode-badges">
          {episode.language && (
            <span className="episode-badge episode-badge--language">{episode.language}</span>
          )}
          {episode.is_new_podcast === 1 && (
            <span className="episode-badge episode-badge--new">New Podcast</span>
          )}
          {episode.is_recurring_podcast === 1 && (
            <span className="episode-badge episode-badge--recurring">Recurring</span>
          )}
        </div>
      </div>
    );
  }

  // Grid view - vertical layout (default)
  return (
    <div className={cardClass} onClick={() => onClick(episode)}>
      <div className="episode-artwork-container">
        <img
          src={episode.artwork_url || "/api/placeholder/220/220"}
          alt={`${episode.podcast_name} artwork`}
          className="episode-artwork"
          loading="lazy"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = "/api/placeholder/220/220";
          }}
        />
      </div>

      <div className="episode-info">
        <h3 className="episode-title" title={episode.episode_title}>{episode.episode_title}</h3>
        <p className="podcast-name" title={episode.podcast_name}>{episode.podcast_name}</p>

        <div className="episode-meta">
          <div className="meta-item">
            <Calendar size={14} />
            <span>{formatDate(episode.published_date)}</span>
          </div>
          <div className="meta-item meta-item--highlight">
            <Clock size={14} />
            <span>{formatDate(episode.listened_date)}</span>
          </div>
          <div className="meta-item">
            <Headphones size={14} />
            <span>{formatPodcastDuration(episode.duration_seconds)}</span>
          </div>
        </div>

        <div className="completion-container">
          <div className="completion-bar">
            <div
              className="completion-fill"
              style={{ width: `${formatCompletion(episode.completion_percent)}%` }}
            />
          </div>
          <span className="completion-text">{formatCompletion(episode.completion_percent)}%</span>
        </div>

        <div className="episode-badges">
          {episode.language && (
            <span className="episode-badge episode-badge--language">{episode.language}</span>
          )}
          {episode.is_new_podcast === 1 && (
            <span className="episode-badge episode-badge--new">New Podcast</span>
          )}
          {episode.is_recurring_podcast === 1 && (
            <span className="episode-badge episode-badge--recurring">Recurring</span>
          )}
        </div>
      </div>
    </div>
  );
});

EpisodeCard.displayName = 'EpisodeCard';

EpisodeCard.propTypes = {
  episode: PropTypes.shape({
    episode_uuid: PropTypes.string.isRequired,
    episode_title: PropTypes.string.isRequired,
    podcast_name: PropTypes.string.isRequired,
    artwork_url: PropTypes.string,
    duration_seconds: PropTypes.number,
    completion_percent: PropTypes.number,
    published_date: PropTypes.instanceOf(Date),
    listened_date: PropTypes.instanceOf(Date),
    language: PropTypes.string,
    is_new_podcast: PropTypes.number,
    is_recurring_podcast: PropTypes.number,
  }).isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list']),
  onClick: PropTypes.func.isRequired,
};

export default EpisodeCard;
