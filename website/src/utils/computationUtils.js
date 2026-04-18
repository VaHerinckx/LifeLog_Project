/**
 * Computation Utilities for KPI Cards and Charts
 *
 * This module provides standardized computation functions for calculating
 * statistics from data arrays. Used by smart KpiCard components and chart components.
 */

import _ from 'lodash';

/**
 * Resolves the data source and date column for a metric configuration.
 * Supports per-metric data source overrides for charts displaying multiple data sources.
 *
 * @param {Object} metricConfig - The metric configuration object
 * @param {Array} [metricConfig.data] - Optional override data source for this metric
 * @param {string} [metricConfig.dateColumnName] - Optional override date column for this metric
 * @param {Array} componentData - The component-level default data source
 * @param {string} componentDateColumn - The component-level default date column name
 * @returns {Object} Object with resolved { data, dateColumnName }
 *
 * @example
 * // In a chart component:
 * const { data: effectiveData, dateColumnName: effectiveDateColumn } = resolveMetricDataSource(
 *   currentMetricConfig,
 *   props.data,
 *   props.dateColumnName
 * );
 */
export const resolveMetricDataSource = (metricConfig, componentData, componentDateColumn) => ({
  data: metricConfig?.data ?? componentData,
  dateColumnName: metricConfig?.dateColumnName ?? componentDateColumn
});

/**
 * Apply metric filter to data before aggregation
 *
 * Supports two APIs:
 * 1. Legacy: filterField/filterValue - simple field matching
 * 2. New: filterConditions - array of conditions with operators and AND logic
 *
 * @param {Array} data - The data array to filter
 * @param {Object} metricConfig - The metric configuration object
 * @param {string} [metricConfig.filterField] - (Legacy) The field to filter on
 * @param {string|Array} [metricConfig.filterValue] - (Legacy) Value(s) to match
 * @param {Array} [metricConfig.filterConditions] - Array of filter conditions with AND logic
 * @param {string} metricConfig.filterConditions[].field - The field to filter on
 * @param {string} [metricConfig.filterConditions[].operator] - Operator: '=', '==', '!=', '>', '>=', '<', '<=' (default: '=')
 * @param {*} metricConfig.filterConditions[].value - Value to compare (array = OR within this condition)
 * @returns {Array} Filtered data array
 */
export const applyMetricFilter = (data, metricConfig) => {
  // Support new API: filterConditions array
  const conditions = metricConfig?.filterConditions;

  // Backward compatibility: use old API if filterConditions not provided
  if (!conditions && metricConfig?.filterField && metricConfig?.filterValue !== undefined) {
    const { filterField, filterValue } = metricConfig;
    return data.filter(item => {
      const itemValue = item[filterField];
      // Support array of values (OR condition)
      if (Array.isArray(filterValue)) {
        return filterValue.includes(itemValue);
      }
      // Single value match
      return itemValue === filterValue;
    });
  }

  // No filter conditions provided
  if (!conditions || !Array.isArray(conditions) || conditions.length === 0) {
    return data;
  }

  // Apply all conditions with AND logic
  return data.filter(item => {
    return conditions.every(condition => {
      const { field, value, operator = '=' } = condition;
      let itemValue = item[field];

      // Handle Date objects: convert to ISO string (YYYY-MM-DD) for comparison
      if (itemValue instanceof Date) {
        itemValue = itemValue.toISOString().split('T')[0];
      }

      // Handle array values (OR within this condition) - only for equality operators
      if (Array.isArray(value)) {
        if (operator === '!=' || operator === '!==') {
          // For not-equals with array, item must not match ANY value
          return !value.includes(itemValue);
        }
        // For equals (default), item must match at least one value
        return value.includes(itemValue);
      }

      // Handle comparison operators (mathematical symbols)
      // For numeric comparisons, convert string values to numbers
      const isNumericOperator = ['>', '>=', '<', '<='].includes(operator);
      if (isNumericOperator && typeof value === 'number') {
        // Convert itemValue to number for comparison
        const numericItemValue = parseFloat(itemValue);
        // If itemValue can't be parsed as a number, exclude from results
        if (isNaN(numericItemValue)) {
          return false;
        }
        itemValue = numericItemValue;
      }

      switch (operator) {
        case '=':
        case '==':
          return itemValue === value;
        case '!=':
        case '!==':
          return itemValue !== value;
        case '>':
          return itemValue > value;
        case '>=':
          return itemValue >= value;
        case '<':
          return itemValue < value;
        case '<=':
          return itemValue <= value;
        default:
          // Default to equality check
          return itemValue === value;
      }
    });
  });
};

/**
 * Performs a computation on a dataset
 *
 * @param {Array} data - The data array to compute on
 * @param {string} field - The field name to compute (for field-based computations)
 * @param {string} computation - The type of computation to perform
 * @param {Object} options - Additional options for the computation
 * @returns {number|string} The computed value
 */
export const performComputation = (data, field, computation, options = {}) => {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return options.defaultValue ?? 0;
  }

  switch (computation) {
    case 'count':
      return computeCount(data, options);

    case 'count_distinct':
      return computeCountDistinct(data, field, options);

    case 'sum':
      return computeSum(data, field, options);

    case 'average':
    case 'mean':
      return computeAverage(data, field, options);

    case 'median':
      return computeMedian(data, field, options);

    case 'min':
      return computeMin(data, field, options);

    case 'max':
      return computeMax(data, field, options);

    case 'mode':
      return computeMode(data, field, options);

    case 'count_filtered':
      return computeCountFiltered(data, options);

    case 'average_filtered':
      return computeAverageFiltered(data, field, options);

    case 'count_recent':
      return computeCountRecent(data, options);

    case 'count_time_range':
      return computeCountTimeRange(data, options);

    case 'cumsum':
      // Note: cumsum is typically handled at the chart level since it requires
      // computing over sorted time periods, not the entire dataset at once.
      // This fallback just returns a sum if called directly.
      return computeSum(data, field, options);

    default:
      console.warn(`Unknown computation type: ${computation}`);
      return options.defaultValue ?? 0;
  }
};

/**
 * Count total items in the dataset
 */
export const computeCount = (data, options = {}) => {
  return data.length;
};

/**
 * Count distinct values of a field
 */
export const computeCountDistinct = (data, field, options = {}) => {
  if (!field) {
    console.warn('computeCountDistinct requires a field parameter');
    return 0;
  }

  const values = data
    .map(item => {
      const value = _.get(item, field);
      // Convert Date objects to ISO string for proper uniqueness comparison
      if (value instanceof Date) {
        return value.toISOString().split('T')[0];
      }
      return value;
    })
    .filter(v => v !== null && v !== undefined);
  return _.uniq(values).length;
};

/**
 * Sum values of a field
 */
export const computeSum = (data, field, options = {}) => {
  if (!field) {
    console.warn('computeSum requires a field parameter');
    return 0;
  }

  const { convertToHours = false, decimals = null } = options;

  const values = data
    .map(item => {
      const value = _.get(item, field);
      return typeof value === 'number' ? value : parseFloat(value);
    })
    .filter(v => !isNaN(v));

  let sum = _.sum(values);

  // Convert seconds to hours if requested
  if (convertToHours) {
    sum = sum / 3600;
  }

  // Apply decimal rounding if specified
  return decimals !== null ? parseFloat(sum.toFixed(decimals)) : sum;
};

/**
 * Calculate average of a field
 */
export const computeAverage = (data, field, options = {}) => {
  if (!field) {
    console.warn('computeAverage requires a field parameter');
    return 0;
  }

  const { decimals = 1, filterZeros = false } = options;

  let values = data
    .map(item => {
      const value = _.get(item, field);
      return typeof value === 'number' ? value : parseFloat(value);
    })
    .filter(v => !isNaN(v));

  // Optionally filter out zero values
  if (filterZeros) {
    values = values.filter(v => v > 0);
  }

  if (values.length === 0) {
    return options.defaultValue ?? 0;
  }

  const avg = _.mean(values);
  return decimals !== null ? parseFloat(avg.toFixed(decimals)) : avg;
};

/**
 * Calculate median of a field
 */
export const computeMedian = (data, field, options = {}) => {
  if (!field) {
    console.warn('computeMedian requires a field parameter');
    return 0;
  }

  const { decimals = 1 } = options;

  const values = data
    .map(item => {
      const value = _.get(item, field);
      return typeof value === 'number' ? value : parseFloat(value);
    })
    .filter(v => !isNaN(v))
    .sort((a, b) => a - b);

  if (values.length === 0) {
    return options.defaultValue ?? 0;
  }

  const mid = Math.floor(values.length / 2);
  const median = values.length % 2 === 0
    ? (values[mid - 1] + values[mid]) / 2
    : values[mid];

  return decimals !== null ? parseFloat(median.toFixed(decimals)) : median;
};

/**
 * Find minimum value of a field
 */
export const computeMin = (data, field, options = {}) => {
  if (!field) {
    console.warn('computeMin requires a field parameter');
    return 0;
  }

  const values = data
    .map(item => {
      const value = _.get(item, field);
      return typeof value === 'number' ? value : parseFloat(value);
    })
    .filter(v => !isNaN(v));

  if (values.length === 0) {
    return options.defaultValue ?? 0;
  }

  return _.min(values);
};

/**
 * Find maximum value of a field
 */
export const computeMax = (data, field, options = {}) => {
  if (!field) {
    console.warn('computeMax requires a field parameter');
    return 0;
  }

  const values = data
    .map(item => {
      const value = _.get(item, field);
      return typeof value === 'number' ? value : parseFloat(value);
    })
    .filter(v => !isNaN(v));

  if (values.length === 0) {
    return options.defaultValue ?? 0;
  }

  return _.max(values);
};

/**
 * Find the most frequent value (mode) of a field
 */
export const computeMode = (data, field, options = {}) => {
  if (!field) {
    console.warn('computeMode requires a field parameter');
    return options.defaultValue ?? 'N/A';
  }

  // Get all values from the field
  const values = data
    .map(item => _.get(item, field))
    .filter(v => v !== null && v !== undefined && v !== '');

  if (values.length === 0) {
    return options.defaultValue ?? 'N/A';
  }

  // Count frequency of each value
  const frequencyMap = {};
  values.forEach(value => {
    const key = String(value);
    frequencyMap[key] = (frequencyMap[key] || 0) + 1;
  });

  // Find the value with the highest frequency
  let modeValue = null;
  let maxFrequency = 0;

  Object.entries(frequencyMap).forEach(([value, frequency]) => {
    if (frequency > maxFrequency) {
      maxFrequency = frequency;
      modeValue = value;
    }
  });

  return modeValue ?? options.defaultValue ?? 'N/A';
};

/**
 * Count items that match a filter condition
 *
 * @param {Array} data - The dataset
 * @param {Object} options - Must contain filterFn function
 */
export const computeCountFiltered = (data, options = {}) => {
  const { filterFn } = options;

  if (!filterFn || typeof filterFn !== 'function') {
    console.warn('computeCountFiltered requires a filterFn in options');
    return 0;
  }

  return data.filter(filterFn).length;
};

/**
 * Calculate average of a field for filtered items
 *
 * @param {Array} data - The dataset
 * @param {string} field - The field to average
 * @param {Object} options - Must contain filterFn function
 */
export const computeAverageFiltered = (data, field, options = {}) => {
  const { filterFn, decimals = 1 } = options;

  if (!filterFn || typeof filterFn !== 'function') {
    console.warn('computeAverageFiltered requires a filterFn in options');
    return 0;
  }

  if (!field) {
    console.warn('computeAverageFiltered requires a field parameter');
    return 0;
  }

  const filteredData = data.filter(filterFn);

  if (filteredData.length === 0) {
    return options.defaultValue ?? 0;
  }

  return computeAverage(filteredData, field, { decimals });
};

/**
 * Count items within a recent time period (e.g., last month, last week)
 *
 * @param {Array} data - The dataset
 * @param {Object} options - Must contain dateField and timeframe
 * @param {string} options.dateField - The field containing the date/timestamp
 * @param {string} options.timeframe - The timeframe ('day', 'week', 'month', 'quarter', 'year')
 * @param {number} [options.amount=1] - Number of timeframe units (e.g., 2 months)
 */
export const computeCountRecent = (data, options = {}) => {
  const { dateField = 'timestamp', timeframe = 'month', amount = 1 } = options;

  if (!dateField) {
    console.warn('computeCountRecent requires a dateField in options');
    return 0;
  }

  const now = new Date();
  const startDate = new Date(now);

  // Calculate start date based on timeframe
  switch (timeframe.toLowerCase()) {
    case 'day':
    case 'days':
      startDate.setDate(now.getDate() - amount);
      break;
    case 'week':
    case 'weeks':
      startDate.setDate(now.getDate() - (amount * 7));
      break;
    case 'month':
    case 'months':
      startDate.setMonth(now.getMonth() - amount);
      break;
    case 'quarter':
    case 'quarters':
      startDate.setMonth(now.getMonth() - (amount * 3));
      break;
    case 'year':
    case 'years':
      startDate.setFullYear(now.getFullYear() - amount);
      break;
    default:
      console.warn(`Unknown timeframe: ${timeframe}`);
      return 0;
  }

  return data.filter(item => {
    const itemDate = new Date(_.get(item, dateField));
    return !isNaN(itemDate.getTime()) && itemDate >= startDate;
  }).length;
};

/**
 * Count items within a specific time range
 *
 * @param {Array} data - The dataset
 * @param {Object} options - Must contain dateField and startDate/endDate
 * @param {string} options.dateField - The field containing the date/timestamp
 * @param {Date|string} [options.startDate] - Start date (inclusive)
 * @param {Date|string} [options.endDate] - End date (inclusive)
 */
export const computeCountTimeRange = (data, options = {}) => {
  const { dateField = 'timestamp', startDate, endDate } = options;

  if (!dateField) {
    console.warn('computeCountTimeRange requires a dateField in options');
    return 0;
  }

  const start = startDate ? new Date(startDate) : null;
  const end = endDate ? new Date(endDate) : null;

  return data.filter(item => {
    const itemDate = new Date(_.get(item, dateField));
    if (isNaN(itemDate.getTime())) return false;

    if (start && itemDate < start) return false;
    if (end && itemDate > end) return false;

    return true;
  }).length;
};

/**
 * Format a computed value for display
 *
 * @param {number|string} value - The value to format
 * @param {Object} options - Formatting options
 * @returns {string} Formatted value
 */
export const formatComputedValue = (value, options = {}) => {
  const {
    type = 'number',
    decimals = 0,
    locale = 'en-US',
    prefix = '',
    suffix = ''
  } = options;

  let formatted;

  switch (type) {
    case 'number':
      formatted = typeof value === 'number'
        ? value.toLocaleString(locale, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
          })
        : value;
      break;

    case 'currency':
      formatted = typeof value === 'number'
        ? value.toLocaleString(locale, {
            style: 'currency',
            currency: options.currency || 'USD'
          })
        : value;
      break;

    case 'percent':
      formatted = typeof value === 'number'
        ? `${(value * 100).toFixed(decimals)}%`
        : value;
      break;

    default:
      formatted = value;
  }

  return `${prefix}${formatted}${suffix}`;
};
