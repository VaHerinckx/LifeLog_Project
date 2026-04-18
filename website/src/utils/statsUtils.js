/**
 * Statistics Utility Functions
 *
 * Centralized statistical calculations used across the LifeLog website.
 * Eliminates duplicate stats calculation code.
 */

import _ from 'lodash';

/**
 * Calculates the sum of a specific field in a dataset
 * @param {Array} data - Array of items
 * @param {string} field - Field name to sum
 * @returns {number} Sum of values
 */
export const calculateSum = (data, field) => {
  if (!data || data.length === 0) return 0;
  return _.sumBy(data, field) || 0;
};

/**
 * Calculates the average of a specific field in a dataset
 * @param {Array} data - Array of items
 * @param {string} field - Field name to average
 * @param {number} decimals - Number of decimal places (default: 1)
 * @param {Function} filter - Optional filter function to apply before averaging
 * @returns {number} Average value
 */
export const calculateAverage = (data, field, decimals = 1, filter = null) => {
  if (!data || data.length === 0) return 0;

  const dataToAverage = filter ? data.filter(filter) : data;
  if (dataToAverage.length === 0) return 0;

  const avg = _.meanBy(dataToAverage, field);
  return parseFloat(avg.toFixed(decimals));
};

/**
 * Calculates the count of unique values for a specific field
 * @param {Array} data - Array of items
 * @param {string} field - Field name to count unique values
 * @returns {number} Count of unique values
 */
export const calculateUniqueCount = (data, field) => {
  if (!data || data.length === 0) return 0;
  const values = data.map(item => item[field]).filter(Boolean);
  return new Set(values).size;
};

/**
 * Calculates the total count of items
 * @param {Array} data - Array of items
 * @returns {number} Total count
 */
export const calculateTotal = (data) => {
  if (!data || data.length === 0) return 0;
  return data.length;
};

/**
 * Calculates percentage
 * @param {number} value - The value to convert
 * @param {number} total - The total value
 * @param {number} decimals - Number of decimal places (default: 1)
 * @returns {number} Percentage value
 */
export const calculatePercentage = (value, total, decimals = 1) => {
  if (!total || total === 0) return 0;
  return parseFloat(((value / total) * 100).toFixed(decimals));
};

/**
 * Calculates median of a specific field
 * @param {Array} data - Array of items
 * @param {string} field - Field name to calculate median
 * @returns {number} Median value
 */
export const calculateMedian = (data, field) => {
  if (!data || data.length === 0) return 0;

  const values = data.map(item => item[field]).filter(v => v !== null && v !== undefined);
  if (values.length === 0) return 0;

  const sorted = values.sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
};

/**
 * Calculates min and max of a specific field
 * @param {Array} data - Array of items
 * @param {string} field - Field name
 * @returns {Object} Object with min and max values
 */
export const calculateMinMax = (data, field) => {
  if (!data || data.length === 0) return { min: 0, max: 0 };

  const values = data.map(item => item[field]).filter(v => v !== null && v !== undefined);
  if (values.length === 0) return { min: 0, max: 0 };

  return {
    min: Math.min(...values),
    max: Math.max(...values)
  };
};

/**
 * Calculates recent items (within a specified time period)
 * @param {Array} data - Array of items
 * @param {string} dateField - Field name containing the date
 * @param {number} days - Number of days to look back (default: 30)
 * @returns {number} Count of recent items
 */
export const calculateRecentCount = (data, dateField, days = 30) => {
  if (!data || data.length === 0) return 0;

  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days);

  return data.filter(item => {
    const itemDate = item[dateField];
    if (!itemDate) return false;

    // Handle both Date objects and date strings
    const date = itemDate instanceof Date ? itemDate : new Date(itemDate);
    return date >= cutoffDate;
  }).length;
};

/**
 * Groups data by a field and counts items in each group
 * @param {Array} data - Array of items
 * @param {string} field - Field name to group by
 * @returns {Object} Object with field values as keys and counts as values
 */
export const calculateGroupCounts = (data, field) => {
  if (!data || data.length === 0) return {};

  return data.reduce((acc, item) => {
    const value = item[field];
    if (value) {
      acc[value] = (acc[value] || 0) + 1;
    }
    return acc;
  }, {});
};

/**
 * Groups data by a field and sums a numeric field for each group
 * @param {Array} data - Array of items
 * @param {string} groupField - Field name to group by
 * @param {string} sumField - Field name to sum
 * @returns {Object} Object with group values as keys and sums as values
 */
export const calculateGroupSums = (data, groupField, sumField) => {
  if (!data || data.length === 0) return {};

  return data.reduce((acc, item) => {
    const groupValue = item[groupField];
    const sumValue = item[sumField] || 0;

    if (groupValue) {
      acc[groupValue] = (acc[groupValue] || 0) + sumValue;
    }
    return acc;
  }, {});
};

/**
 * Calculates top N items by a specific field
 * @param {Array} data - Array of items
 * @param {string} field - Field name to rank by
 * @param {number} n - Number of top items to return (default: 10)
 * @param {string} order - 'desc' for highest first, 'asc' for lowest first (default: 'desc')
 * @returns {Array} Top N items
 */
export const calculateTopN = (data, field, n = 10, order = 'desc') => {
  if (!data || data.length === 0) return [];

  const sorted = _.orderBy(data, [field], [order]);
  return sorted.slice(0, n);
};

/**
 * Calculates distribution (histogram) of values for a numeric field
 * @param {Array} data - Array of items
 * @param {string} field - Field name to analyze
 * @param {number} bins - Number of bins/buckets (default: 10)
 * @returns {Array} Array of bins with count and range
 */
export const calculateDistribution = (data, field, bins = 10) => {
  if (!data || data.length === 0) return [];

  const values = data.map(item => item[field]).filter(v => v !== null && v !== undefined);
  if (values.length === 0) return [];

  const min = Math.min(...values);
  const max = Math.max(...values);
  const binSize = (max - min) / bins;

  const distribution = Array.from({ length: bins }, (_, i) => ({
    min: min + i * binSize,
    max: min + (i + 1) * binSize,
    count: 0
  }));

  values.forEach(value => {
    const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
    distribution[binIndex].count++;
  });

  return distribution;
};

/**
 * Calculates comprehensive stats for a dataset
 * @param {Array} data - Array of items
 * @param {Object} config - Configuration object specifying which stats to calculate
 * @returns {Object} Object containing requested statistics
 *
 * Config format:
 * {
 *   total: true,
 *   unique: { field: 'artist_name' },
 *   sum: { field: 'pages' },
 *   average: { field: 'rating', decimals: 1, filter: (item) => item.rating > 0 },
 *   recent: { dateField: 'timestamp', days: 30 },
 *   percentage: { value: 10, total: 100 }
 * }
 */
export const calculateStats = (data, config) => {
  const stats = {};

  if (!data || data.length === 0) {
    // Return zero values for all requested stats
    Object.keys(config).forEach(key => {
      stats[key] = 0;
    });
    return stats;
  }

  if (config.total) {
    stats.total = calculateTotal(data);
  }

  if (config.unique) {
    stats.unique = calculateUniqueCount(data, config.unique.field);
  }

  if (config.sum) {
    stats.sum = calculateSum(data, config.sum.field);
  }

  if (config.average) {
    stats.average = calculateAverage(
      data,
      config.average.field,
      config.average.decimals,
      config.average.filter
    );
  }

  if (config.recent) {
    stats.recent = calculateRecentCount(
      data,
      config.recent.dateField,
      config.recent.days
    );
  }

  if (config.percentage) {
    stats.percentage = calculatePercentage(
      config.percentage.value,
      config.percentage.total,
      config.percentage.decimals
    );
  }

  if (config.median) {
    stats.median = calculateMedian(data, config.median.field);
  }

  if (config.minMax) {
    stats.minMax = calculateMinMax(data, config.minMax.field);
  }

  return stats;
};

/**
 * Formats a number for display (adds commas, handles decimals)
 * @param {number} value - The number to format
 * @param {number} decimals - Number of decimal places (default: 0)
 * @returns {string} Formatted number string
 */
export const formatNumber = (value, decimals = 0) => {
  if (value === null || value === undefined || isNaN(value)) return '0';
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
};

/**
 * Formats duration in milliseconds to human-readable format
 * @param {number} ms - Duration in milliseconds
 * @param {string} format - 'short' or 'long' (default: 'short')
 * @returns {string} Formatted duration string
 */
export const formatDuration = (ms, format = 'short') => {
  if (!ms || ms === 0) return '0m';

  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (format === 'long') {
    const parts = [];
    if (days > 0) parts.push(`${days} day${days > 1 ? 's' : ''}`);
    if (hours % 24 > 0) parts.push(`${hours % 24} hour${hours % 24 > 1 ? 's' : ''}`);
    if (minutes % 60 > 0) parts.push(`${minutes % 60} minute${minutes % 60 > 1 ? 's' : ''}`);
    return parts.join(', ') || '0 minutes';
  }

  // Short format
  if (days > 0) return `${days}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  return `${minutes}m`;
};

/**
 * Formats duration in seconds to minutes for podcast episodes
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration string (e.g., "45 min")
 */
export const formatPodcastDuration = (seconds) => {
  if (!seconds || seconds === 0) return 'Unknown';
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min`;
};

/**
 * Formats completion percentage for podcast episodes
 * @param {number} percent - Completion percentage (0-100)
 * @returns {number} Rounded completion percentage
 */
export const formatCompletion = (percent) => {
  if (percent === null || percent === undefined) return 0;
  return Math.round(percent);
};
