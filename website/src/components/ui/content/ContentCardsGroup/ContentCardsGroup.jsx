import { useState, useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
// Note: Virtual scrolling temporarily disabled due to react-window import issues
// Will be re-enabled after fixing the import
// import { FixedSizeList } from 'react-window';
import Pagination from './Pagination';
import './ContentCardsGroup.css';

/**
 * ContentCardsGroup - Generic container component with pagination and virtual scrolling
 *
 * A high-performance, standardized container that handles:
 * - Layout switching between grid, list, and timeline views
 * - Pagination for large datasets
 * - Virtual scrolling for smooth performance
 * - Responsive grid/list layouts
 *
 * Item-specific rendering is delegated to the parent via renderItem prop.
 *
 * Performance optimizations:
 * - Only renders visible items (virtual scrolling)
 * - Paginated data to limit rendered items
 * - Memoized calculations
 * - Auto-enables features based on dataset size
 *
 * @param {Array} items - Array of items to display
 * @param {string} viewMode - Current view mode ('grid' | 'list' | 'timeline')
 * @param {Function} renderItem - Function to render individual item: (item, index) => JSX
 * @param {string} className - Additional CSS class for customization
 * @param {boolean} enablePagination - Enable pagination (default: auto-enable for >200 items)
 * @param {boolean} enableVirtualScrolling - Enable virtual scrolling (default: auto-enable for >100 items)
 * @param {number} itemsPerPage - Items per page (default: 100)
 * @param {Array} itemsPerPageOptions - Available items per page options (default: [25, 50, 100, 200])
 * @param {Object} cardHeight - Card heights by view mode (default: { grid: 380, list: 100, timeline: 120 })
 */
const ContentCardsGroup = ({
  items,
  viewMode,
  renderItem,
  className = '',
  enablePagination = null, // Auto-enable if null
  enableVirtualScrolling = null, // Auto-enable if null
  itemsPerPage: initialItemsPerPage = 100,
  itemsPerPageOptions = [25, 50, 100, 200],
  cardHeight = { grid: 380, list: 100, timeline: 120 }
}) => {
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(initialItemsPerPage);

  // Auto-enable features based on dataset size
  const shouldEnablePagination = enablePagination !== null
    ? enablePagination
    : items.length > 200;

  const shouldEnableVirtualScrolling = enableVirtualScrolling !== null
    ? enableVirtualScrolling
    : items.length > 100;

  // Calculate pagination
  const totalPages = Math.ceil(items.length / itemsPerPage);

  // Get current page items
  const paginatedItems = useMemo(() => {
    if (!shouldEnablePagination) return items;

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return items.slice(startIndex, endIndex);
  }, [items, currentPage, itemsPerPage, shouldEnablePagination]);

  // Handle page change
  const handlePageChange = useCallback((newPage) => {
    setCurrentPage(newPage);
    // Scroll to top of container
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  // Handle items per page change
  const handleItemsPerPageChange = useCallback((newItemsPerPage) => {
    setItemsPerPage(newItemsPerPage);
    setCurrentPage(1); // Reset to first page
  }, []);

  // Container class
  const containerClass = `card-group-wrapper ${className}`.trim();

  // Get card height for current view mode (for future virtual scrolling implementation)
  const currentCardHeight = cardHeight[viewMode] || 380;

  // Note: Virtual scrolling temporarily disabled - will be re-enabled after fixing react-window import
  // For now, pagination alone provides excellent performance (100 items max per page)

  // Standard rendering for all views with pagination
  const cardGroupClass = `card-group card-group--${viewMode}`;

  return (
    <div className={containerClass}>
      <div className={cardGroupClass}>
        {paginatedItems.map((item, index) => renderItem(item, index))}
      </div>

      {shouldEnablePagination && items.length > itemsPerPage && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          itemsPerPage={itemsPerPage}
          totalItems={items.length}
          onPageChange={handlePageChange}
          onItemsPerPageChange={handleItemsPerPageChange}
          itemsPerPageOptions={itemsPerPageOptions}
        />
      )}
    </div>
  );
};

ContentCardsGroup.propTypes = {
  items: PropTypes.array.isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list', 'timeline']).isRequired,
  renderItem: PropTypes.func.isRequired,
  className: PropTypes.string,
  enablePagination: PropTypes.bool,
  enableVirtualScrolling: PropTypes.bool,
  itemsPerPage: PropTypes.number,
  itemsPerPageOptions: PropTypes.arrayOf(PropTypes.number),
  cardHeight: PropTypes.shape({
    grid: PropTypes.number,
    list: PropTypes.number,
    timeline: PropTypes.number
  })
};

export default ContentCardsGroup;
