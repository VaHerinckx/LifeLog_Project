import { memo } from 'react';
import PropTypes from 'prop-types';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';
import './Pagination.css';

/**
 * Pagination Component
 *
 * Provides pagination controls for navigating through large datasets.
 * Features:
 * - Previous/Next navigation
 * - First/Last page jumps
 * - Current page and total pages display
 * - Items per page selector
 * - Keyboard accessible
 *
 * @param {number} currentPage - Current page number (1-indexed)
 * @param {number} totalPages - Total number of pages
 * @param {number} itemsPerPage - Current items per page setting
 * @param {number} totalItems - Total number of items in dataset
 * @param {Function} onPageChange - Callback when page changes: (newPage) => void
 * @param {Function} onItemsPerPageChange - Callback when items per page changes: (newItemsPerPage) => void
 * @param {Array} itemsPerPageOptions - Available items per page options (default: [25, 50, 100, 200])
 */
const Pagination = memo(({
  currentPage,
  totalPages,
  itemsPerPage,
  totalItems,
  onPageChange,
  onItemsPerPageChange,
  itemsPerPageOptions = [25, 50, 100, 200]
}) => {
  const handlePrevious = () => {
    if (currentPage > 1) {
      onPageChange(currentPage - 1);
    }
  };

  const handleNext = () => {
    if (currentPage < totalPages) {
      onPageChange(currentPage + 1);
    }
  };

  const handleFirst = () => {
    if (currentPage !== 1) {
      onPageChange(1);
    }
  };

  const handleLast = () => {
    if (currentPage !== totalPages) {
      onPageChange(totalPages);
    }
  };

  const handleItemsPerPageChange = (e) => {
    const newItemsPerPage = parseInt(e.target.value);
    onItemsPerPageChange(newItemsPerPage);
  };

  // Calculate range of items being displayed
  const startItem = (currentPage - 1) * itemsPerPage + 1;
  const endItem = Math.min(currentPage * itemsPerPage, totalItems);

  return (
    <div className="pagination">
      <div className="pagination-info">
        <span className="pagination-range">
          Showing {startItem}-{endItem} of {totalItems}
        </span>
      </div>

      <div className="pagination-controls">
        <button
          className="pagination-button"
          onClick={handleFirst}
          disabled={currentPage === 1}
          aria-label="First page"
          title="First page"
        >
          <ChevronsLeft size={16} />
        </button>

        <button
          className="pagination-button"
          onClick={handlePrevious}
          disabled={currentPage === 1}
          aria-label="Previous page"
          title="Previous page"
        >
          <ChevronLeft size={16} />
        </button>

        <span className="pagination-page-info">
          Page {currentPage} of {totalPages}
        </span>

        <button
          className="pagination-button"
          onClick={handleNext}
          disabled={currentPage === totalPages}
          aria-label="Next page"
          title="Next page"
        >
          <ChevronRight size={16} />
        </button>

        <button
          className="pagination-button"
          onClick={handleLast}
          disabled={currentPage === totalPages}
          aria-label="Last page"
          title="Last page"
        >
          <ChevronsRight size={16} />
        </button>
      </div>

      <div className="pagination-settings">
        <label htmlFor="items-per-page" className="pagination-label">
          Items per page:
        </label>
        <select
          id="items-per-page"
          className="pagination-select"
          value={itemsPerPage}
          onChange={handleItemsPerPageChange}
        >
          {itemsPerPageOptions.map(option => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
});

Pagination.displayName = 'Pagination';

Pagination.propTypes = {
  currentPage: PropTypes.number.isRequired,
  totalPages: PropTypes.number.isRequired,
  itemsPerPage: PropTypes.number.isRequired,
  totalItems: PropTypes.number.isRequired,
  onPageChange: PropTypes.func.isRequired,
  onItemsPerPageChange: PropTypes.func.isRequired,
  itemsPerPageOptions: PropTypes.arrayOf(PropTypes.number)
};

export default Pagination;
