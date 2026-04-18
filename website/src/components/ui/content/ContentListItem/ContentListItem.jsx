import PropTypes from 'prop-types';

/**
 * ContentListItem - Generic horizontal list item component
 *
 * Reusable list item component that works across all content types (books, movies, podcasts, etc.)
 * Displays content in a horizontal layout: [Image] [Title + Subtitle + Metadata] [Tags]
 *
 * Uses CSS classes from components.css and page-specific CSS for styling overrides
 *
 * @param {Object} image - Image configuration { url, alt, fallback, aspectRatio }
 * @param {string} title - Main title text
 * @param {string} subtitle - Subtitle text (author, director, podcast name, etc.)
 * @param {Array} metadata - Array of metadata items to display
 * @param {Array} tags - Array of tag objects to display
 * @param {Function} onClick - Callback when item is clicked
 * @param {string} className - Additional CSS class for page-specific styling
 */
const ContentListItem = ({
  image,
  title,
  subtitle,
  metadata = [],
  tags = [],
  onClick,
  className = '',
}) => {
  const handleImageError = (e) => {
    e.target.onerror = null;
    e.target.src = image.fallback || "/api/placeholder/80/120";
  };

  return (
    <div
      className={`content-list-item ${className}`}
      onClick={onClick}
    >
      {/* Image Section */}
      <div className="content-list-image">
        <img
          src={image.url || image.fallback || "/api/placeholder/80/120"}
          alt={image.alt}
          onError={handleImageError}
        />
      </div>

      {/* Main Info Section */}
      <div className="content-list-info">
        <h3 className="content-list-title">{title}</h3>
        <p className="content-list-subtitle">{subtitle}</p>

        {/* Metadata Row */}
        {metadata.length > 0 && (
          <div className="content-list-meta">
            {metadata.map((item, index) => (
              <div key={index} className="content-list-meta-item">
                {item.component ? (
                  // Render custom component (e.g., StarRating)
                  item.component
                ) : (
                  // Render icon + text
                  <>
                    {item.icon && <span className="content-list-meta-icon">{item.icon}</span>}
                    {item.text && <span className="content-list-meta-text">{item.text}</span>}
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Tags Section */}
      {tags.length > 0 && (
        <div className="content-list-tags">
          {tags.map((tag, index) => (
            <span
              key={index}
              className={`content-list-tag ${tag.className || ''}`}
            >
              {tag.text}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

ContentListItem.propTypes = {
  image: PropTypes.shape({
    url: PropTypes.string,
    alt: PropTypes.string.isRequired,
    fallback: PropTypes.string,
    aspectRatio: PropTypes.oneOf(['portrait', 'square']),
  }).isRequired,
  title: PropTypes.string.isRequired,
  subtitle: PropTypes.string.isRequired,
  metadata: PropTypes.arrayOf(
    PropTypes.shape({
      icon: PropTypes.node,
      text: PropTypes.string,
      component: PropTypes.node,
    })
  ),
  tags: PropTypes.arrayOf(
    PropTypes.shape({
      text: PropTypes.string.isRequired,
      className: PropTypes.string,
    })
  ),
  onClick: PropTypes.func.isRequired,
  className: PropTypes.string,
};

export default ContentListItem;
