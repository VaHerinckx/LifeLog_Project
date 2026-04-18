import PropTypes from 'prop-types';
import { Calendar, RotateCcw, Clock } from 'lucide-react';
import { StarRating } from '../../../components/ui';
import { formatDate } from '../../../utils';
import './MovieCard.css';

/**
 * MovieCard - Displays movie information in grid or list view
 *
 * Self-contained component that adapts its layout based on viewMode prop.
 * All styling is contained in MovieCard.css.
 *
 * Grid view: Vertical layout with poster on top, info below
 * List view: Horizontal layout with poster on left, info on right
 *
 * @param {Object} movie - Movie data object
 * @param {string} viewMode - Display mode ('grid' | 'list')
 * @param {Function} onClick - Callback when card is clicked
 */
const MovieCard = ({ movie, viewMode = 'grid', onClick }) => {
  const cardClass = `movie-card movie-card--${viewMode}`;

  // Use genreArray if available (from deduplicated data), otherwise parse genre field
  const genres = movie.genreArray || (movie.genre ? movie.genre.split(',').map(g => g.trim()).filter(Boolean) : []);
  const displayGenres = genres.slice(0, 2); // Show max 2 genres on card

  // Format runtime
  const formatRuntime = (minutes) => {
    if (!minutes || minutes === 0) return null;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours === 0) return `${mins}m`;
    return `${hours}h ${mins}m`;
  };

  const formattedRuntime = formatRuntime(movie.runtime);

  // List view - horizontal layout
  if (viewMode === 'list') {
    return (
      <div className={cardClass} onClick={() => onClick(movie)}>
        <div className="movie-poster-container">
          <img
            src={movie.poster_url || "/api/placeholder/80/120"}
            alt={`${movie.name} poster`}
            className="movie-poster"
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = "/api/placeholder/80/120";
            }}
          />
        </div>

        <div className="movie-info">
          <div className="movie-title-row">
            <h3 className="movie-title">{movie.name}</h3>
            {movie.isRewatch && (
              <span className="rewatch-badge" title="Rewatch">
                <RotateCcw size={14} />
              </span>
            )}
          </div>

          {/* Metadata badges row */}
          <div className="movie-metadata-badges">
            <span className="movie-year-badge">{movie.year}</span>
            {formattedRuntime && (
              <span className="runtime-badge">
                <Clock size={12} />
                {formattedRuntime}
              </span>
            )}
            {movie.certification && movie.certification !== 'Unknown' && (
              <span className="certification-badge">{movie.certification}</span>
            )}
          </div>

          {/* Director */}
          {movie.director && movie.director !== 'Unknown' && (
            <p className="movie-director">Dir: {movie.director}</p>
          )}

          <div className="movie-meta">
            <div className="rating-and-date">
              {movie.rating > 0 && (
                <div className="rating-container">
                  <StarRating rating={movie.rating} size={16} />
                  <span className="rating-value">{movie.rating.toFixed(1)}</span>
                </div>
              )}

              <div className="watch-date">
                <Calendar size={16} />
                <span>{formatDate(movie.date)}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="movie-genre-tags">
          {displayGenres.map((genre, index) => (
            <span key={index} className="movie-genre">{genre}</span>
          ))}
          {genres.length > 2 && (
            <span className="movie-genre-more">+{genres.length - 2}</span>
          )}
        </div>
      </div>
    );
  }

  // Grid view - vertical layout (default)
  return (
    <div className={cardClass} onClick={() => onClick(movie)}>
      <div className="movie-poster-container">
        {movie.isRewatch && (
          <div className="rewatch-badge-overlay" title="Rewatch">
            <RotateCcw size={16} />
            <span>Rewatch</span>
          </div>
        )}
        <img
          src={movie.poster_url || "/api/placeholder/220/330"}
          alt={`${movie.name} poster`}
          className="movie-poster"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = "/api/placeholder/220/330";
          }}
        />
      </div>

      <div className="movie-info">
        <h3 className="movie-title" title={movie.name}>{movie.name}</h3>

        {/* Metadata badges row */}
        <div className="movie-metadata-badges">
          <span className="movie-year-badge">{movie.year}</span>
          {formattedRuntime && (
            <span className="runtime-badge">
              <Clock size={12} />
              {formattedRuntime}
            </span>
          )}
          {movie.certification && movie.certification !== 'Unknown' && (
            <span className="certification-badge">{movie.certification}</span>
          )}
        </div>

        {/* Director */}
        {movie.director && movie.director !== 'Unknown' && (
          <p className="movie-director" title={movie.director}>Dir: {movie.director}</p>
        )}

        {movie.rating > 0 && (
          <div className="rating-container">
            <StarRating rating={movie.rating} size={16} />
            <span>{movie.rating.toFixed(1)}</span>
          </div>
        )}

        <div className="movie-meta">
          <div className="movie-genre-tags">
            {displayGenres.map((genre, index) => (
              <span key={index} className="movie-genre">{genre}</span>
            ))}
            {genres.length > 2 && (
              <span className="movie-genre-more">+{genres.length - 2}</span>
            )}
          </div>
        </div>

        <div className="watch-date-container">
          <span className="date-label">Watched:</span>
          <span className="date-value">{formatDate(movie.date)}</span>
        </div>
      </div>
    </div>
  );
};

MovieCard.propTypes = {
  movie: PropTypes.shape({
    name: PropTypes.string.isRequired,
    year: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    poster_url: PropTypes.string,
    rating: PropTypes.number,
    genre: PropTypes.string,
    date: PropTypes.instanceOf(Date), // Can be null for unwatched movies
    isRewatch: PropTypes.bool,
    director: PropTypes.string,
    runtime: PropTypes.number,
    certification: PropTypes.string,
  }).isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list']),
  onClick: PropTypes.func.isRequired,
};

export default MovieCard;
