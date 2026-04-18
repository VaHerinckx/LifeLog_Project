/**
 * Utils Barrel Export
 *
 * Central export point for all utility functions.
 * Allows clean imports: import { formatDate, applyFilters } from '../utils'
 */

// Date utilities
export {
  formatDate,
  formatDateShort,
  formatDateLong,
  parseDate,
  isValidDate,
  getDateBounds,
  isDateInRange,
  getDateRange,
  getDaysDifference,
  groupByYear,
  groupByMonth
} from './dateUtils';

// Filter utilities
export {
  applyDateRangeFilter,
  applyMultiSelectFilter,
  applyMultiGenreFilter,
  applyTextSearchFilter,
  applyNumericRangeFilter,
  applyFilters,
  getUniqueValues,
  getUniqueGenres,
  getCascadingFilterOptions
} from './filterUtils';

// Statistics utilities
export {
  calculateSum,
  calculateAverage,
  calculateUniqueCount,
  calculateTotal,
  calculatePercentage,
  calculateMedian,
  calculateMinMax,
  calculateRecentCount,
  calculateGroupCounts,
  calculateGroupSums,
  calculateTopN,
  calculateDistribution,
  calculateStats,
  formatNumber,
  formatDuration,
  formatPodcastDuration,
  formatCompletion
} from './statsUtils';

// Sorting utilities
export {
  sortByDateSafely,
  sortByNumberSafely,
  sortByStringSafely
} from './sortingUtils';
