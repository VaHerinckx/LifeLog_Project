import PropTypes from 'prop-types';
import LoadingSpinner from '../../common/LoadingSpinner';

/**
 * ContentContainer - Standardized content container with loading and empty states
 *
 * Wraps page content and handles loading, empty states, and content rendering.
 * Uses CSS classes from components.css: .content-container, .loading-state, .empty-state, .empty-state-icon, .empty-state-message
 *
 * @param {boolean} isEmpty - Whether the content is empty
 * @param {boolean} loading - Whether content is currently loading
 * @param {object} emptyState - Empty state configuration: {icon: Component, title: string, message: string}
 * @param {React.ReactNode} children - Content to render when not empty/loading
 * @param {React.Component} loadingIcon - Icon to use for loading spinner (optional)
 */
const ContentContainer = ({
  isEmpty = false,
  loading = false,
  emptyState = null,
  children = null,
  loadingIcon = null
}) => {
  // Show loading state
  if (loading) {
    return (
      <div className="content-container">
        <div className="loading-state">
          <LoadingSpinner centerIcon={loadingIcon} fullPage={false} />
        </div>
      </div>
    );
  }

  // Show empty state
  if (isEmpty) {
    const { icon: Icon, title, message } = emptyState;
    return (
      <div className="content-container">
        <div className="empty-state">
          <div className="empty-state-icon">
            <Icon size={48} />
          </div>
          <h3>{title}</h3>
          <p className="empty-state-message">{message}</p>
        </div>
      </div>
    );
  }

  // Show content
  return <div className="content-container">{children}</div>;
};

ContentContainer.propTypes = {
  isEmpty: PropTypes.bool,
  loading: PropTypes.bool,
  emptyState: PropTypes.shape({
    icon: PropTypes.elementType.isRequired,
    title: PropTypes.string.isRequired,
    message: PropTypes.string.isRequired,
  }),
  children: PropTypes.node,
  loadingIcon: PropTypes.elementType,
};

export default ContentContainer;
