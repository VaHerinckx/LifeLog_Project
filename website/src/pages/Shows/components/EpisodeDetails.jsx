import React from 'react';
import { X, Tv, Calendar, ExternalLink, Clock, User, Users } from 'lucide-react';
import './EpisodeDetails.css';
import { formatDate } from '../../../utils';
import { StarRating } from '../../../components/ui';

const EpisodeDetails = ({ episode, onClose }) => {
  if (!episode) return null;

  // Format season/episode as "S01E06"
  const formatEpisodeNumber = (season, episodeNum) => {
    const s = String(season).padStart(2, '0');
    const e = String(episodeNum).padStart(2, '0');
    return `S${s}E${e}`;
  };

  // Format runtime as "Xh Ym"
  const formatRuntime = (minutes) => {
    if (!minutes) return null;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    if (hours > 0 && mins > 0) return `${hours}h ${mins}m`;
    if (hours > 0) return `${hours}h`;
    return `${mins}m`;
  };

  // Parse comma-separated credits into array
  const parseCredits = (creditString) => {
    if (!creditString) return [];
    return creditString.split(',').map(c => c.trim()).filter(Boolean);
  };

  const episodeCode = formatEpisodeNumber(episode.season, episode.episode_number);
  const progressPercent = episode.progress ? `${Math.round(episode.progress)}%` : 'N/A';
  const formattedRuntime = formatRuntime(episode.episode_runtime);
  const castArray = parseCredits(episode.episode_cast);
  const guestStarsArray = parseCredits(episode.episode_guest_stars);

  return (
    <div className="episode-details-overlay" onClick={onClose}>
      <div className="episode-details-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="episode-details-content">
          {/* Fixed Left Column with Images */}
          <div className="episode-details-images">
            <div className="icon-container">
              <Tv size={48} />
            </div>
            {/* Season Poster - always on top, full width */}
            <img
              className="season-poster"
              src={episode.season_poster_url || "/api/placeholder/300/450"}
              alt={`${episode.show_title} Season ${episode.season} poster`}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "/api/placeholder/300/450";
              }}
            />
            {/* Episode Still - below poster */}
            {episode.episode_still_url && (
              <img
                className="episode-still"
                src={episode.episode_still_url}
                alt={`${episode.episode_title} still`}
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.style.display = 'none';
                }}
              />
            )}
          </div>

          {/* Scrollable Right Column with Info */}
          <div className="episode-details-info">
            <h2>{episode.episode_title}</h2>
            <h3>{episode.show_title}</h3>

            {/* Metadata badges */}
            <div className="episode-metadata-row">
              <span className="metadata-badge">
                <Tv size={14} />
                {episodeCode}
              </span>
              {formattedRuntime && (
                <span className="metadata-badge">
                  <Clock size={14} />
                  {formattedRuntime}
                </span>
              )}
              {episode.episode_air_date && (
                <span className="metadata-badge">
                  <Calendar size={14} />
                  {formatDate(episode.episode_air_date)}
                </span>
              )}
            </div>

            {/* Overview Section */}
            {episode.episode_overview && (
              <div className="episode-section">
                <h4>Overview</h4>
                <p className="episode-overview">{episode.episode_overview}</p>
              </div>
            )}

            {/* Ratings Section */}
            {(episode.episode_rating > 0 || episode.season_rating > 0 || episode.show_rating > 0 || episode.episode_vote_average > 0) && (
              <div className="episode-section">
                <h4>Ratings</h4>
                <div className="ratings-grid">
                  {episode.episode_rating > 0 && (
                    <div className="rating-item">
                      <label>My Episode Rating:</label>
                      <div className="rating-display">
                        <StarRating rating={episode.episode_rating / 2} size={18} />
                        <span className="rating-value">{(episode.episode_rating / 2).toFixed(1)}/5</span>
                      </div>
                    </div>
                  )}

                  {episode.season_rating > 0 && (
                    <div className="rating-item">
                      <label>My Season Rating:</label>
                      <div className="rating-display">
                        <StarRating rating={episode.season_rating / 2} size={18} />
                        <span className="rating-value">{(episode.season_rating / 2).toFixed(1)}/5</span>
                      </div>
                    </div>
                  )}

                  {episode.show_rating > 0 && (
                    <div className="rating-item">
                      <label>My Show Rating:</label>
                      <div className="rating-display">
                        <StarRating rating={episode.show_rating / 2} size={18} />
                        <span className="rating-value">{(episode.show_rating / 2).toFixed(1)}/5</span>
                      </div>
                    </div>
                  )}

                  {parseFloat(episode.episode_vote_average) > 0 && (
                    <div className="rating-item">
                      <label>TMDB Rating:</label>
                      <div className="rating-display">
                        <StarRating rating={parseFloat(episode.episode_vote_average) / 2} size={18} />
                        <span className="rating-value">
                          {parseFloat(episode.episode_vote_average).toFixed(1)}/10
                          {parseInt(episode.episode_vote_count, 10) > 0 && (
                            <span className="vote-count"> ({parseInt(episode.episode_vote_count, 10).toLocaleString()} votes)</span>
                          )}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Cast & Crew Section */}
            {(episode.episode_director || episode.episode_writer || castArray.length > 0 || guestStarsArray.length > 0) && (
              <div className="episode-section">
                <h4>Cast & Crew</h4>
                <div className="credits-info">
                  {episode.episode_director && (
                    <div className="credit-row">
                      <User size={16} />
                      <div>
                        <label>Director:</label>
                        <span>{episode.episode_director}</span>
                      </div>
                    </div>
                  )}
                  {episode.episode_writer && (
                    <div className="credit-row">
                      <User size={16} />
                      <div>
                        <label>Writer:</label>
                        <span>{episode.episode_writer}</span>
                      </div>
                    </div>
                  )}
                  {castArray.length > 0 && (
                    <div className="credit-row">
                      <Users size={16} />
                      <div>
                        <label>Cast:</label>
                        <span>{castArray.slice(0, 10).join(', ')}</span>
                      </div>
                    </div>
                  )}
                  {guestStarsArray.length > 0 && (
                    <div className="credit-row">
                      <Users size={16} />
                      <div>
                        <label>Guest Stars:</label>
                        <span>{guestStarsArray.slice(0, 8).join(', ')}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Watch Details Section */}
            <div className="episode-section">
              <h4>Watch Details</h4>
              <div className="episode-details-meta">
                <div className="meta-item">
                  <Calendar size={18} />
                  <span>Watched: {formatDate(episode.watched_at)}</span>
                </div>

                <div className="meta-item">
                  <span className="label">Progress:</span>
                  <span className="progress-value">{progressPercent}</span>
                </div>
              </div>
            </div>

            {/* Show Information Section */}
            <div className="episode-section">
              <h4>Show Information</h4>
              <div className="episode-details-meta">
                {episode.show_year && (
                  <div className="meta-item">
                    <Calendar size={18} />
                    <span>Year: {episode.show_year}</span>
                  </div>
                )}

                <div className="meta-item">
                  <span className="label">Season:</span>
                  <span>{episode.season}</span>
                </div>

                <div className="meta-item">
                  <span className="label">Episode Number:</span>
                  <span>{episode.episode_number}</span>
                </div>
              </div>
            </div>

            {/* External Links Section */}
            <div className="episode-section">
              <h4>Links</h4>
              <div className="external-links">
                {episode.show_imdb_id && (
                  <a
                    href={`https://www.imdb.com/title/${episode.show_imdb_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link"
                  >
                    <ExternalLink size={16} />
                    <span>Show on IMDb</span>
                  </a>
                )}

                {episode.episode_imdb_id && (
                  <a
                    href={`https://www.imdb.com/title/${episode.episode_imdb_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link"
                  >
                    <ExternalLink size={16} />
                    <span>Episode on IMDb</span>
                  </a>
                )}

                {episode.show_trakt_id && (
                  <a
                    href={`https://trakt.tv/shows/${episode.show_trakt_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link"
                  >
                    <ExternalLink size={16} />
                    <span>Show on Trakt</span>
                  </a>
                )}

                {episode.episode_trakt_id && (
                  <a
                    href={`https://trakt.tv/shows/${episode.show_trakt_id}/seasons/${episode.season}/episodes/${episode.episode_number}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link"
                  >
                    <ExternalLink size={16} />
                    <span>Episode on Trakt</span>
                  </a>
                )}

                {episode.show_trakt_id && (
                  <a
                    href={`https://www.themoviedb.org/tv/${episode.show_trakt_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link"
                  >
                    <ExternalLink size={16} />
                    <span>Show on TMDB</span>
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EpisodeDetails;
