import { memo, useMemo } from 'react';
import { X, Music as MusicIcon, Calendar, Clock, User, Disc, TrendingUp, Play } from 'lucide-react';
import './MusicDetails.css';
import { formatDate } from '../../../utils';

// Pure utility functions - defined outside component to avoid recreation
const formatDuration = (ms) => {
  if (!ms) return 'Unknown';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

const formatListeningTime = (seconds) => {
  if (!seconds) return '0s';
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}s`;
};

const MusicDetails = memo(({ toggle, onClose }) => {
  // Memoize genres array processing - must be called before any conditional returns
  const genresArray = useMemo(() =>
    toggle?.genres ? toggle.genres.split(', ').filter(g => g.trim()) : [],
    [toggle?.genres]
  );

  if (!toggle) return null;

  return (
    <div className="music-details-overlay" onClick={onClose}>
      <div className="music-details-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="music-details-content">
          {/* Artwork Section */}
          <div className="music-details-artwork-section">
            {/* Album Artwork */}
            <div className="music-details-artwork">
              {toggle.album_artwork_url ? (
                <img
                  src={toggle.album_artwork_url}
                  alt={`${toggle.album_name} artwork`}
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
              ) : null}
              <div className="music-placeholder-large" style={{ display: toggle.album_artwork_url ? 'none' : 'flex' }}>
                <MusicIcon size={80} />
              </div>
            </div>

            {/* Artist Artwork */}
            {toggle.artist_artwork_url && (
              <div className="music-details-artist-artwork">
                <img
                  src={toggle.artist_artwork_url}
                  alt={`${toggle.artist_name}`}
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.parentElement.style.display = 'none';
                  }}
                />
                <span className="artist-artwork-label">Artist</span>
              </div>
            )}
          </div>

          {/* Track Information */}
          <div className="music-details-info">
            <h2>{toggle.track_name}</h2>
            <h3>{toggle.artist_name}</h3>
            <p className="album-name">{toggle.album_name}</p>

            {/* Track Metadata */}
            <div className="music-details-meta">
              {toggle.album_release_date && (
                <div className="meta-item">
                  <Calendar size={18} />
                  <span>Released: {new Date(toggle.album_release_date).getFullYear()}</span>
                </div>
              )}

              {toggle.track_duration > 0 && (
                <div className="meta-item">
                  <Clock size={18} />
                  <span>Duration: {formatDuration(toggle.track_duration)}</span>
                </div>
              )}

              {toggle.track_popularity > 0 && (
                <div className="meta-item">
                  <TrendingUp size={18} />
                  <span>Track Popularity: {toggle.track_popularity}/100</span>
                </div>
              )}
            </div>

            {/* Artist Information Section */}
            <div className="music-details-section">
              <h4>Artist Information</h4>
              <div className="artist-info">
                <div className="info-row">
                  <User size={16} />
                  <span>{toggle.artist_name}</span>
                </div>
                {toggle.artist_popularity > 0 && (
                  <div className="info-row">
                    <TrendingUp size={16} />
                    <span>Artist Popularity: {toggle.artist_popularity}/100</span>
                  </div>
                )}
                {toggle.followers > 0 && (
                  <div className="info-row">
                    <User size={16} />
                    <span>Followers: {toggle.followers.toLocaleString()}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Listening Statistics Section */}
            <div className="music-details-section">
              <h4>Listening Statistics</h4>
              <div className="listening-stats">
                <div className="stat-row">
                  <Calendar size={16} />
                  <span>Played: {formatDate(toggle.timestamp)}</span>
                </div>
                <div className="stat-row">
                  <Play size={16} />
                  <span>Completion: {(toggle.completion * 100).toFixed(0)}%</span>
                </div>
                {toggle.listening_seconds > 0 && (
                  <div className="stat-row">
                    <Clock size={16} />
                    <span>Listened: {formatListeningTime(toggle.listening_seconds)}</span>
                  </div>
                )}
                {toggle.is_skipped_track === 'Yes' && (
                  <div className="stat-row">
                    <Disc size={16} />
                    <span>Track was skipped</span>
                  </div>
                )}
              </div>

              {/* Discovery Badges */}
              <div className="discovery-badges">
                {toggle.is_new_artist === 'Yes' && (
                  <span className="discovery-badge new-artist">First Artist Listen</span>
                )}
                {toggle.is_new_track === 'Yes' && (
                  <span className="discovery-badge new-track">First Track Listen</span>
                )}
              </div>
            </div>

            {/* Genres Section */}
            {genresArray.length > 0 && (
              <div className="music-genres-section">
                <h4>Genres</h4>
                <div className="genres-list">
                  {genresArray.map((genre, index) => (
                    <span key={index} className="genre-badge">{genre}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

export default MusicDetails;
