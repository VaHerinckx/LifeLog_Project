import React from 'react';
import { X, Calendar, Clock, Headphones, ExternalLink } from 'lucide-react';
import './EpisodeDetails.css';
import { formatDate } from '../../../utils';

const EpisodeDetails = ({ episode, onClose }) => {
  if (!episode) return null;

  // Helper functions
  const formatDuration = (seconds) => {
    if (!seconds || seconds === 0) return 'Unknown';
    const minutes = Math.floor(seconds / 60);
    return `${minutes} min`;
  };

  const formatCompletion = (percent) => {
    if (percent === null || percent === undefined) return 0;
    return Math.round(percent);
  };

  return (
    <div className="episode-details-overlay" onClick={onClose}>
      <div className="episode-details-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="episode-details-content">
          <div className="episode-details-artwork">
            <img
              src={episode.artwork_url || "/api/placeholder/300/300"}
              alt={`${episode.podcast_name} artwork`}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "/api/placeholder/300/300";
              }}
            />
          </div>

          <div className="episode-details-info">
            <h2>{episode.episode_title}</h2>
            <h3>{episode.podcast_name}</h3>
            {episode.artist && <p className="episode-artist">by {episode.artist}</p>}

            <div className="episode-listening-stats">
              <div className="stats-grid">
                <div className="meta-item">
                  <Calendar size={18} />
                  <span>Published: {formatDate(episode.published_date)}</span>
                </div>

                <div className="meta-item">
                  <Clock size={18} />
                  <span>Listened: {formatDate(episode.listened_date)}</span>
                </div>

                <div className="meta-item">
                  <Headphones size={18} />
                  <span>
                    Duration: {formatDuration(episode.duration_seconds)}
                    {episode.listened_seconds && (
                      <> (listened {formatDuration(episode.listened_seconds)}, {formatCompletion(episode.completion_percent)}% complete)</>
                    )}
                  </span>
                </div>
              </div>

              {episode.completion_percent !== null && episode.completion_percent !== undefined && (
                <div className="completion-section">
                  <div className="completion-bar">
                    <div
                      className="completion-fill"
                      style={{ width: `${formatCompletion(episode.completion_percent)}%` }}
                    />
                  </div>
                  <span className="completion-text">{formatCompletion(episode.completion_percent)}% Complete</span>
                </div>
              )}
            </div>

            <div className="episode-categories">
              {episode.genre && (
                <span className="episode-category genre-tag">
                  {episode.genre}
                </span>
              )}

              {episode.language && (
                <span className="episode-category language-tag">
                  {episode.language}
                </span>
              )}

              {episode.is_new_podcast === 1 && (
                <span className="episode-category new-tag">
                  New Podcast
                </span>
              )}

              {episode.is_recurring_podcast === 1 && (
                <span className="episode-category recurring-tag">
                  Recurring
                </span>
              )}
            </div>

            {episode.episode_url && (
              <div className="episode-link-section">
                <a
                  href={episode.episode_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="episode-link-button"
                >
                  <ExternalLink size={18} />
                  <span>Episode Link</span>
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EpisodeDetails;
