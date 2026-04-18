import PropTypes from 'prop-types';
import { Tv } from 'lucide-react';
import { formatDate } from '../../../utils';
import { StarRating } from '../../../components/ui';
import './EpisodeCard.css';

/**
 * EpisodeCard - Displays TV episode information in grid or list view
 *
 * Self-contained component that adapts its layout based on viewMode prop.
 * All styling is contained in EpisodeCard.css.
 *
 * Grid view: Vertical layout with poster on top, info below
 * List view: Horizontal layout with poster on left, info on right
 *
 * @param {Object} episode - Episode data object
 * @param {string} viewMode - Display mode ('grid' | 'list')
 * @param {Function} onClick - Callback when card is clicked
 */
const EpisodeCard = ({ episode, viewMode = 'grid', onClick }) => {
  const cardClass = `episode-card episode-card--${viewMode}`;

  // Format season/episode as "S01E06"
  const formatEpisodeNumber = (season, episodeNum) => {
    const s = String(season).padStart(2, '0');
    const e = String(episodeNum).padStart(2, '0');
    return `S${s}E${e}`;
  };

  const episodeCode = formatEpisodeNumber(episode.season, episode.episode_number);
  const progressPercent = episode.progress ? `${Math.round(episode.progress)}%` : '';

  // List view - horizontal layout
  if (viewMode === 'list') {
    return (
      <div className={cardClass} onClick={() => onClick(episode)}>
        <div className="episode-poster-container">
          <img
            src={episode.season_poster_url || "/api/placeholder/80/120"}
            alt={`${episode.show_title} Season ${episode.season} poster`}
            className="episode-poster"
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = "/api/placeholder/80/120";
            }}
          />
        </div>

        <div className="episode-info">
          <h3 className="episode-title">{episode.episode_title}</h3>
          <p className="show-title">{episode.show_title}</p>

          <div className="episode-meta">
            <span className="episode-code">{episodeCode}</span>
            {episode.show_year && <span className="show-year">{episode.show_year}</span>}
            {progressPercent && (
              <span className="progress-badge">{progressPercent}</span>
            )}
          </div>

          {episode.episode_rating && episode.episode_rating > 0 && (
            <div className="rating-container">
              <StarRating rating={episode.episode_rating / 2} size={16} />
              <span className="rating-value">{(episode.episode_rating / 2).toFixed(1)}</span>
            </div>
          )}

          <div className="watch-date">
            <span className="date-value">{formatDate(episode.watched_at)}</span>
          </div>
        </div>
      </div>
    );
  }

  // Grid view - vertical layout (default)
  return (
    <div className={cardClass} onClick={() => onClick(episode)}>
      <div className="episode-poster-container">
        <img
          src={episode.season_poster_url || "/api/placeholder/220/320"}
          alt={`${episode.show_title} Season ${episode.season} poster`}
          className="episode-poster"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = "/api/placeholder/220/320";
          }}
        />
        {progressPercent && (
          <div className="progress-overlay">
            <span className="progress-badge">{progressPercent}</span>
          </div>
        )}
      </div>

      <div className="episode-info">
        <h3 className="episode-title" title={episode.episode_title}>
          {episode.episode_title}
        </h3>
        <p className="show-title" title={episode.show_title}>
          {episode.show_title}
        </p>

        <div className="episode-meta">
          <span className="episode-code">{episodeCode}</span>
          {episode.show_year && <span className="show-year">{episode.show_year}</span>}
        </div>

        {episode.episode_rating && episode.episode_rating > 0 && (
          <div className="rating-container">
            <StarRating rating={episode.episode_rating / 2} size={16} />
            <span className="rating-value">{(episode.episode_rating / 2).toFixed(1)}</span>
          </div>
        )}

        <div className="watch-date">
          <span className="date-label">Watched on:</span>
          <span className="date-value">{formatDate(episode.watched_at)}</span>
        </div>
      </div>
    </div>
  );
};

EpisodeCard.propTypes = {
  episode: PropTypes.shape({
    watch_id: PropTypes.string.isRequired,
    watched_at: PropTypes.oneOfType([PropTypes.string, PropTypes.instanceOf(Date)]).isRequired,
    show_title: PropTypes.string.isRequired,
    show_year: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    season: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    episode_number: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    episode_title: PropTypes.string.isRequired,
    progress: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    season_poster_url: PropTypes.string,
    episode_rating: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  }).isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list']),
  onClick: PropTypes.func.isRequired,
};

export default EpisodeCard;
