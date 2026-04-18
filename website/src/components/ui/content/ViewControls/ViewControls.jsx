import PropTypes from 'prop-types';
import { ArrowUp, ArrowDown } from 'lucide-react';

/**
 * ViewControls - Standardized view mode toggle and sorting controls
 *
 * Used for switching between different view modes (grid, list, timeline) and
 * controlling sort field and direction.
 * Uses CSS classes from components.css: .view-controls, .view-control-btn, .view-control-btn.active, .sort-controls
 *
 * @param {Array} viewModes - Array of view mode objects: [{mode: 'grid', icon: GridIcon}, ...]
 * @param {string} activeMode - Currently active view mode
 * @param {function} onModeChange - Callback function when view mode button is clicked
 * @param {Array} sortOptions - Array of sort options: [{value: 'date', label: 'Date', type: 'date'}, ...]
 * @param {string} sortField - Currently selected sort field
 * @param {string} sortDirection - Current sort direction ('asc' or 'desc')
 * @param {function} onSortChange - Callback function when sort changes (field, direction) => void
 */
const ViewControls = ({
  viewModes,
  activeMode,
  onModeChange,
  sortOptions = null,
  sortField = null,
  sortDirection = 'desc',
  onSortChange = null
}) => {
  return (
    <div className="view-controls">
      <div className="view-mode-buttons">
        {viewModes.map(({ mode, icon: Icon }) => (
          <button
            key={mode}
            className={`view-control-btn ${activeMode === mode ? 'active' : ''}`}
            onClick={() => onModeChange(mode)}
            aria-label={`${mode} view`}
          >
            <Icon size={20} />
          </button>
        ))}
      </div>

      {/* Sort Controls - Only show if sortOptions provided */}
      {sortOptions && sortOptions.length > 0 && onSortChange && (
        <div className="sort-controls">
          <label htmlFor="sort-field" className="sort-label">Sort by:</label>
          <select
            id="sort-field"
            value={sortField}
            onChange={(e) => onSortChange(e.target.value, sortDirection)}
            className="sort-dropdown"
            aria-label="Sort field"
          >
            {sortOptions.map(({ value, label }) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <button
            className="sort-direction-btn"
            onClick={() => onSortChange(sortField, sortDirection === 'asc' ? 'desc' : 'asc')}
            aria-label={`Sort ${sortDirection === 'asc' ? 'descending' : 'ascending'}`}
            title={sortDirection === 'asc' ? 'Ascending (click for descending)' : 'Descending (click for ascending)'}
          >
            {sortDirection === 'asc' ? <ArrowUp size={18} /> : <ArrowDown size={18} />}
          </button>
        </div>
      )}
    </div>
  );
};

ViewControls.propTypes = {
  viewModes: PropTypes.arrayOf(
    PropTypes.shape({
      mode: PropTypes.string.isRequired,
      icon: PropTypes.elementType.isRequired,
    })
  ).isRequired,
  activeMode: PropTypes.string.isRequired,
  onModeChange: PropTypes.func.isRequired,
  sortOptions: PropTypes.arrayOf(
    PropTypes.shape({
      value: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      type: PropTypes.oneOf(['date', 'number', 'string']).isRequired,
    })
  ),
  sortField: PropTypes.string,
  sortDirection: PropTypes.oneOf(['asc', 'desc']),
  onSortChange: PropTypes.func,
};

export default ViewControls;
