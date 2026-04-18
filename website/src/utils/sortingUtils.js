import _ from 'lodash';

/**
 * Safely sort items by date field with defensive handling
 * Handles invalid dates, missing values, and timezone issues
 *
 * @param {Array} items - Array of items to sort
 * @param {string} dateField - Name of the date field to sort by (default: 'timestamp')
 * @param {boolean} ascending - Sort order (default: false for descending/most recent first)
 * @returns {Array} Sorted array
 */
export const sortByDateSafely = (items, dateField = 'timestamp', ascending = false) => {
  if (!items || !Array.isArray(items) || items.length === 0) {
    return [];
  }

  const sorted = _.sortBy(items, item => {
    const date = item[dateField];

    // Handle missing or null values
    if (!date) {
      return ascending ? Infinity : -Infinity;
    }

    // If already a Date object, get time
    if (date instanceof Date) {
      const time = date.getTime();
      return !isNaN(time) ? time : (ascending ? Infinity : -Infinity);
    }

    // If string, try to parse
    if (typeof date === 'string') {
      const parsedDate = new Date(date);
      const time = parsedDate.getTime();
      return !isNaN(time) ? time : (ascending ? Infinity : -Infinity);
    }

    // If number (timestamp), use directly
    if (typeof date === 'number') {
      return !isNaN(date) ? date : (ascending ? Infinity : -Infinity);
    }

    // Unknown type, send to end
    return ascending ? Infinity : -Infinity;
  });

  return ascending ? sorted : sorted.reverse();
};

/**
 * Sort items by numeric field with defensive handling
 *
 * @param {Array} items - Array of items to sort
 * @param {string} numericField - Name of the numeric field to sort by
 * @param {boolean} ascending - Sort order (default: false for descending)
 * @returns {Array} Sorted array
 */
export const sortByNumberSafely = (items, numericField, ascending = false) => {
  if (!items || !Array.isArray(items) || items.length === 0) {
    return [];
  }

  const sorted = _.sortBy(items, item => {
    const value = item[numericField];

    // Handle missing or null values
    if (value === null || value === undefined) {
      return ascending ? Infinity : -Infinity;
    }

    // Convert to number
    const num = typeof value === 'number' ? value : parseFloat(value);

    return !isNaN(num) ? num : (ascending ? Infinity : -Infinity);
  });

  return ascending ? sorted : sorted.reverse();
};

/**
 * Sort items by string field (case-insensitive)
 *
 * @param {Array} items - Array of items to sort
 * @param {string} stringField - Name of the string field to sort by
 * @param {boolean} ascending - Sort order (default: true for A-Z)
 * @returns {Array} Sorted array
 */
export const sortByStringSafely = (items, stringField, ascending = true) => {
  if (!items || !Array.isArray(items) || items.length === 0) {
    return [];
  }

  const sorted = _.sortBy(items, item => {
    const value = item[stringField];

    if (!value) {
      return ascending ? '' : '\uffff'; // Send to end
    }

    return String(value).toLowerCase();
  });

  return ascending ? sorted : sorted.reverse();
};
