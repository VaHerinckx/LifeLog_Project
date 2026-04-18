import React from 'react';
import { X, Calendar, Film, ExternalLink, RotateCcw, Clock, User, Users, Award, DollarSign, Globe, Play, Tag } from 'lucide-react';
import './MovieDetails.css';
import { StarRating } from '../../../components/ui';
import { formatDate } from '../../../utils';

const MovieDetails = ({ movie, onClose }) => {
  if (!movie) return null;

  // Numeric fields are already converted in DataContext
  const { vote_average: voteAverage, vote_count: voteCount, runtime, budget, revenue } = movie;

  // Use genreArray if available (from deduplicated data), otherwise parse genre field
  const genres = movie.genreArray || (movie.genre ? movie.genre.split(',').map(g => g.trim()).filter(Boolean) : []);

  // Parse cast into array
  const castArray = movie.cast && movie.cast !== 'Unknown'
    ? movie.cast.split(',').map(c => c.trim()).filter(Boolean)
    : [];

  // Parse keywords into array
  const keywordsArray = movie.keywords && movie.keywords !== 'None' && movie.keywords !== 'Unknown'
    ? movie.keywords.split(',').map(k => k.trim()).filter(Boolean)
    : [];

  // Format runtime as "Xh Ym"
  const formatRuntime = (minutes) => {
    if (!minutes || minutes === 'Unknown') return null;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours > 0 && mins > 0) return `${hours}h ${mins}m`;
    if (hours > 0) return `${hours}h`;
    return `${mins}m`;
  };

  // Format currency
  const formatCurrency = (amount) => {
    if (!amount || amount === 0) return null;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formattedRuntime = formatRuntime(runtime);
  const formattedBudget = formatCurrency(budget);
  const formattedRevenue = formatCurrency(revenue);

  return (
    <div className="movie-details-overlay" onClick={onClose}>
      <div className="movie-details-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="movie-details-content">
          <div className="movie-details-poster">
            <img
              src={movie.poster_url || "/api/placeholder/300/450"}
              alt={`${movie.name} poster`}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "/api/placeholder/300/450";
              }}
            />
          </div>

          <div className="movie-details-info">
            <h2>{movie.name}</h2>
            {movie.original_title && movie.original_title !== movie.name && movie.original_title !== 'Unknown' && (
              <h4 className="original-title">{movie.original_title}</h4>
            )}
            <h3>{movie.year}</h3>

            {/* Header metadata row */}
            <div className="movie-metadata-row">
              {formattedRuntime && (
                <span className="metadata-badge">
                  <Clock size={14} />
                  {formattedRuntime}
                </span>
              )}
              {movie.certification && movie.certification !== 'Unknown' && (
                <span className="metadata-badge certification">
                  {movie.certification}
                </span>
              )}
              {movie.original_language && movie.original_language !== 'Unknown' && (
                <span className="metadata-badge">
                  {movie.original_language.toUpperCase()}
                </span>
              )}
            </div>

            {/* Tagline */}
            {movie.tagline && movie.tagline !== 'None' && movie.tagline !== 'Unknown' && (
              <div className="movie-tagline">
                <em>"{movie.tagline}"</em>
              </div>
            )}

            {/* Watch info */}
            <div className="movie-details-meta">
              <div className="meta-item">
                <Calendar size={18} />
                <span>Watched: {formatDate(movie.date)}</span>
              </div>

              <div className="meta-item">
                <Film size={18} />
                <span>Released: {movie.year}</span>
              </div>

              {movie.isRewatch && (
                <div className="meta-item rewatch-indicator">
                  <RotateCcw size={18} />
                  <span>Rewatch</span>
                </div>
              )}
            </div>

            {/* Overview */}
            {movie.overview && movie.overview !== 'No overview available' && movie.overview !== 'Unknown' && (
              <div className="movie-section">
                <h4>Overview</h4>
                <p className="movie-overview">{movie.overview}</p>
              </div>
            )}

            {/* Ratings Section */}
            <div className="movie-section">
              <h4>Ratings</h4>
              <div className="ratings-grid">
                {movie.rating > 0 && (
                  <div className="rating-item">
                    <label>My Rating:</label>
                    <div className="rating-display">
                      <StarRating rating={movie.rating} size={18} />
                      <span className="rating-value">{movie.rating.toFixed(1)}</span>
                    </div>
                  </div>
                )}
                {voteAverage && voteAverage > 0 && (
                  <div className="rating-item">
                    <label>TMDB Rating:</label>
                    <div className="rating-display">
                      <StarRating rating={voteAverage / 2} size={18} />
                      <span className="rating-value">
                        {(typeof voteAverage === 'number' ? voteAverage : parseFloat(voteAverage)).toFixed(1)}/10
                        {voteCount && <span className="vote-count"> ({(typeof voteCount === 'number' ? voteCount : parseInt(voteCount, 10)).toLocaleString()} votes)</span>}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Credits Section */}
            {(movie.director && movie.director !== 'Unknown') || (movie.writer && movie.writer !== 'Unknown') || castArray.length > 0 && (
              <div className="movie-section">
                <h4>Cast & Crew</h4>
                <div className="credits-info">
                  {movie.director && movie.director !== 'Unknown' && (
                    <div className="credit-row">
                      <User size={16} />
                      <div>
                        <label>Director:</label>
                        <span>{movie.director}</span>
                      </div>
                    </div>
                  )}
                  {movie.writer && movie.writer !== 'Unknown' && (
                    <div className="credit-row">
                      <User size={16} />
                      <div>
                        <label>Writer:</label>
                        <span>{movie.writer}</span>
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
                </div>
              </div>
            )}

            {/* Genres */}
            {genres.length > 0 && (
              <div className="movie-section">
                <h4>Genres</h4>
                <div className="genre-list">
                  {genres.map((genre, index) => (
                    <span key={index} className="movie-category">
                      {genre}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Production Info */}
            {((movie.production_companies && movie.production_companies !== 'Unknown') ||
              (movie.production_countries && movie.production_countries !== 'Unknown') ||
              formattedBudget || formattedRevenue) && (
              <div className="movie-section">
                <h4>Production</h4>
                <div className="production-info">
                  {movie.production_companies && movie.production_companies !== 'Unknown' && (
                    <div className="production-row">
                      <Globe size={16} />
                      <div>
                        <label>Companies:</label>
                        <span>{movie.production_companies}</span>
                      </div>
                    </div>
                  )}
                  {movie.production_countries && movie.production_countries !== 'Unknown' && (
                    <div className="production-row">
                      <Globe size={16} />
                      <div>
                        <label>Countries:</label>
                        <span>{movie.production_countries}</span>
                      </div>
                    </div>
                  )}
                  {(formattedBudget || formattedRevenue) && (
                    <div className="production-row">
                      <DollarSign size={16} />
                      <div>
                        {formattedBudget && (
                          <>
                            <label>Budget:</label>
                            <span>{formattedBudget} || </span>
                          </>
                        )}
                        {formattedRevenue && (
                          <>
                            <label>Revenue:</label>
                            <span>{formattedRevenue}</span>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Trailer */}
            {movie.trailer_key && movie.trailer_key !== 'No trailer found' && (
              <div className="movie-section">
                <h4>Trailer</h4>
                <div className="trailer-container">
                  <iframe
                    width="100%"
                    height="315"
                    src={`https://www.youtube.com/embed/${movie.trailer_key}`}
                    title="Movie Trailer"
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  ></iframe>
                </div>
              </div>
            )}

            {/* Keywords */}
            {keywordsArray.length > 0 && (
              <div className="movie-section">
                <h4>Keywords</h4>
                <div className="keywords-list">
                  {keywordsArray.map((keyword, index) => (
                    <span key={index} className="keyword-tag">
                      <Tag size={12} />
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* External Links */}
            <div className="movie-section">
              <h4>Links</h4>
              <div className="external-links">
                {movie.letterboxd_uri && (
                  <a
                    href={movie.letterboxd_uri}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link letterboxd-link"
                  >
                    <ExternalLink size={16} />
                    Letterboxd
                  </a>
                )}
                {movie.imdb_id && movie.imdb_id !== 'Unknown' && (
                  <a
                    href={`https://www.imdb.com/title/${movie.imdb_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link imdb-link"
                  >
                    <ExternalLink size={16} />
                    IMDb
                  </a>
                )}
                {movie.tmdb_id && (
                  <a
                    href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link tmdb-link"
                  >
                    <ExternalLink size={16} />
                    TMDB
                  </a>
                )}
                {movie.wikidata_id && movie.wikidata_id !== 'Unknown' && (
                  <a
                    href={`https://www.wikidata.org/wiki/${movie.wikidata_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link wikidata-link"
                  >
                    <ExternalLink size={16} />
                    Wikidata
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

export default MovieDetails;
