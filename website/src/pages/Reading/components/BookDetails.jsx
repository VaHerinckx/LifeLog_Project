import React from 'react';
import { X, BookOpen, Calendar, Clock, User } from 'lucide-react';
import './BookDetails.css';
import { StarRating } from '../../../components/ui';

const BookDetails = ({ book, onClose }) => {
  if (!book) return null;

  return (
    <div className="book-details-overlay" onClick={onClose}>
      <div className="book-details-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="book-details-content">
          <div className="book-details-cover">
            <img
              src={book.cover_url || "/api/placeholder/300/450"}
              alt={`${book.title} cover`}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "/api/placeholder/300/450";
              }}
            />
          </div>

          <div className="book-details-info">
            <h2>{book.title}</h2>

            <div className="book-author-section">
              {book.author_photo_url ? (
                <img
                  src={book.author_photo_url}
                  alt={book.author}
                  className="author-photo"
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              ) : (
                <div className="author-photo-placeholder">
                  <User size={24} />
                </div>
              )}
              <h3>by {book.author}</h3>
            </div>

            <div className="book-details-meta">
              {book.original_publication_year && (
                <div className="meta-item">
                  <Calendar size={18} />
                  <span>Published: {book.original_publication_year}</span>
                </div>
              )}

              <div className="meta-item">
                <BookOpen size={18} />
                <span>{book.number_of_pages.toLocaleString()} pages</span>
              </div>

              {book.reading_duration_final && (
                <div className="meta-item">
                  <Clock size={18} />
                  <span>Read in {book.reading_duration_final} days</span>
                </div>
              )}
            </div>

            <div className="book-ratings">
              <div className="rating-item">
                <label>My Rating:</label>
                <div className="rating-display">
                  <StarRating rating={book.my_rating} size={18} />
                  <span>{book.my_rating.toFixed(1)}</span>
                </div>
              </div>

              <div className="rating-item">
                <label>Community Rating:</label>
                <div className="rating-display">
                  <StarRating rating={book.average_rating} size={18} />
                  <span>{book.average_rating.toFixed(2)}</span>
                </div>
              </div>
            </div>

            <div className="book-categories">
              <span className={`book-category ${book.fiction_yn === 'Fiction' ? 'fiction-tag' : 'non-fiction-tag'}`}>
                {book.fiction_yn}
              </span>

              {book.genre && book.genre !== 'Unknown' && (
                <span className="book-category">
                  {book.genre}
                </span>
              )}
            </div>

            {book.synopsis && (
              <div className="book-synopsis">
                <h4>Synopsis</h4>
                <p>{book.synopsis}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BookDetails;
