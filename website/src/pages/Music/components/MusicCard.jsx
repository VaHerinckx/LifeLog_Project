import { memo, useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
import { Music } from 'lucide-react';
import { formatDate } from '../../../utils';
import './MusicCard.css';

/**
 * MusicCard - Displays music toggle (listening event) in grid or list view
 *
 * Self-contained component that adapts its layout based on viewMode prop.
 * All styling is contained in MusicCard.css.
 *
 * Grid view: Vertical layout with artwork on top, info below
 * List view: Horizontal layout with artwork on left, info on right
 *
 * @param {Object} toggle - Music toggle data object
 * @param {string} viewMode - Display mode ('grid' | 'list')
 * @param {Function} onClick - Callback when card is clicked
 */
const MusicCard = ({ toggle, viewMode = 'grid', onClick }) => {
  const cardClass = `music-card music-card--${viewMode}`;

  // Memoize completion percentage calculation
  const completionPercent = useMemo(() =>
    toggle.completion ? (toggle.completion * 100).toFixed(0) : 0,
    [toggle.completion]
  );

  // Memoize genres array processing
  const { displayGenres, genresArray } = useMemo(() => {
    const arr = toggle.genres ? toggle.genres.split(', ').filter(g => g.trim()) : [];
    return { genresArray: arr, displayGenres: arr.slice(0, 3) };
  }, [toggle.genres]);

  // Stable click handler to prevent re-renders
  const handleClick = useCallback(() => onClick(toggle), [onClick, toggle]);

  // List view - horizontal layout
  if (viewMode === 'list') {
    return (
      <div className={cardClass} onClick={handleClick}>
        <div className="music-artwork-container">
          {toggle.album_artwork_url ? (
            <img
              src={toggle.album_artwork_url}
              alt={`${toggle.album_name} artwork`}
              className="music-artwork"
              loading="lazy"
              onError={(e) => {
                e.target.onerror = null;
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
            />
          ) : null}
          <div className="music-placeholder-icon" style={{ display: toggle.album_artwork_url ? 'none' : 'flex' }}>
            <Music size={32} />
          </div>
        </div>

        <div className="music-info">
          <h3 className="music-track-name">{toggle.track_name}</h3>
          <p className="music-artist-name">{toggle.artist_name}</p>
          <p className="music-album-name">{toggle.album_name}</p>

          <div className="music-meta">
            <span className="music-date">{formatDate(toggle.timestamp)}</span>
            <span className="music-completion">{completionPercent}% completed</span>
            {toggle.listening_seconds > 0 && (
              <span className="music-duration">
                {Math.floor(toggle.listening_seconds / 60)}m {toggle.listening_seconds % 60}s
              </span>
            )}
          </div>

          {displayGenres.length > 0 && (
            <div className="music-genres">
              {displayGenres.map((genre, index) => (
                <span key={index} className="genre-tag">{genre}</span>
              ))}
              {genresArray.length > 3 && (
                <span className="genre-tag genre-more">+{genresArray.length - 3}</span>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Grid view - vertical layout (default)
  return (
    <div className={cardClass} onClick={handleClick}>
      <div className="music-artwork-container">
        {toggle.album_artwork_url ? (
          <img
            src={toggle.album_artwork_url}
            alt={`${toggle.album_name} artwork`}
            className="music-artwork"
            loading="lazy"
            onError={(e) => {
              e.target.onerror = null;
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'flex';
            }}
          />
        ) : null}
        <div className="music-placeholder-icon" style={{ display: toggle.album_artwork_url ? 'none' : 'flex' }}>
          <Music size={48} />
        </div>
      </div>

      <div className="music-info">
        <h3 className="music-track-name" title={toggle.track_name}>{toggle.track_name}</h3>
        <p className="music-artist-name" title={toggle.artist_name}>{toggle.artist_name}</p>
        <p className="music-album-name" title={toggle.album_name}>{toggle.album_name}</p>

        <div className="music-meta">
          <span className="music-date">{formatDate(toggle.timestamp)}</span>
          <span className="music-completion">{completionPercent}%</span>
        </div>

        {displayGenres.length > 0 && (
          <div className="music-genres">
            {displayGenres.map((genre, index) => (
              <span key={index} className="genre-tag">{genre}</span>
            ))}
            {genresArray.length > 3 && (
              <span className="genre-tag genre-more">+{genresArray.length - 3}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

MusicCard.propTypes = {
  toggle: PropTypes.shape({
    toggle_id: PropTypes.number,
    track_name: PropTypes.string,
    artist_name: PropTypes.string,
    album_name: PropTypes.string,
    album_artwork_url: PropTypes.string,
    timestamp: PropTypes.instanceOf(Date),
    completion: PropTypes.number,
    listening_seconds: PropTypes.number,
    genres: PropTypes.string
  }).isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list']),
  onClick: PropTypes.func.isRequired
};

export default memo(MusicCard);
