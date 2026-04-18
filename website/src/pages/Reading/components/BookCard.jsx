import PropTypes from 'prop-types';
import { BookOpen } from 'lucide-react';
import { StarRating } from '../../../components/ui';
import { formatDate } from '../../../utils';
import './BookCard.css';

/**
 * BookCard - Displays book information in grid or list view
 *
 * Self-contained component that adapts its layout based on viewMode prop.
 * All styling is contained in BookCard.css.
 *
 * Grid view: Vertical layout with cover on top, info below
 * List view: Horizontal layout with cover on left, info on right
 *
 * @param {Object} book - Book data object
 * @param {string} viewMode - Display mode ('grid' | 'list')
 * @param {Function} onClick - Callback when card is clicked
 */
const BookCard = ({ book, viewMode = 'grid', onClick }) => {
  const cardClass = `book-card book-card--${viewMode}`;

  // List view - horizontal layout
  if (viewMode === 'list') {
    return (
      <div className={cardClass} onClick={() => onClick(book)}>
        <div className="book-cover-container">
          <img
            src={book.cover_url || "/api/placeholder/80/120"}
            alt={`${book.title} cover`}
            className="book-cover"
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = "/api/placeholder/80/120";
            }}
          />
        </div>

        <div className="book-info">
          <h3 className="book-title">{book.title}</h3>
          <p className="book-author">by {book.author}</p>

          <div className="book-meta">
            <div className="rating-pages-duration">
              <div className="rating-container">
                <StarRating rating={book.my_rating} size={16} />
                <span className="rating-value">{book.my_rating.toFixed(1)}</span>
              </div>

              {book.number_of_pages > 0 && (
                <div className="pages-info">
                  <BookOpen size={16} />
                  <span>{book.number_of_pages} pages</span>
                </div>
              )}

              {book.reading_duration_final && (
                <span className="reading-duration">
                  Read in {book.reading_duration_final} days
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="book-genre-tags">
          <span className={`book-genre ${book.fiction_yn === 'Fiction' ? 'fiction-tag' : 'non-fiction-tag'}`}>
            {book.fiction_yn}
          </span>
          {book.genre && book.genre !== 'Unknown' && (
            <span className="book-genre">{book.genre}</span>
          )}
        </div>
      </div>
    );
  }

  // Grid view - vertical layout (default)
  return (
    <div className={cardClass} onClick={() => onClick(book)}>
      <div className="book-cover-container">
        <img
          src={book.cover_url || "/api/placeholder/220/320"}
          alt={`${book.title} cover`}
          className="book-cover"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = "/api/placeholder/220/320";
          }}
        />
      </div>

      <div className="book-info">
        <h3 className="book-title" title={book.title}>{book.title}</h3>
        <p className="book-author" title={book.author}>by {book.author}</p>

        <div className="rating-container">
          <StarRating rating={book.my_rating} size={16} />
          <span>{book.my_rating.toFixed(1)}</span>
        </div>

        <div className="book-meta">
          {book.original_publication_year && <span>{book.original_publication_year}</span>}
          <div className="book-genre-tags">
            <span className={`book-genre ${book.fiction_yn === 'Fiction' ? 'fiction-tag' : 'non-fiction-tag'}`}>
              {book.fiction_yn}
            </span>
          </div>
        </div>

        <div className="reading-dates">
          <span className="date-label">Read on:</span>
          <span className="date-value">{formatDate(book.timestamp)}</span>
        </div>

        {book.reading_duration_final && book.timestamp && (
          <div className="reading-duration">
            <span className="duration-value">
              {book.reading_duration_final} days to read
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

BookCard.propTypes = {
  book: PropTypes.shape({
    title: PropTypes.string.isRequired,
    author: PropTypes.string.isRequired,
    cover_url: PropTypes.string,
    my_rating: PropTypes.number.isRequired,
    original_publication_year: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    fiction_yn: PropTypes.string,
    genre: PropTypes.string,
    number_of_pages: PropTypes.number,
    timestamp: PropTypes.instanceOf(Date),
    reading_duration_final: PropTypes.number,
  }).isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list']),
  onClick: PropTypes.func.isRequired,
};

export default BookCard;
