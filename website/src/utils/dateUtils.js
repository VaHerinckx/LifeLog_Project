/**
 * Date Utility Functions
 *
 * Centralized date formatting and manipulation utilities
 * used across the LifeLog website.
 */

/**
 * Formats a date object to a localized string
 * @param {Date} date - The date to format
 * @param {Object} options - Optional formatting options
 * @returns {string} Formatted date string or 'Unknown date' if invalid
 */
export const formatDate = (date, options = { year: 'numeric', month: 'short', day: 'numeric' }) => {
  if (!date || !(date instanceof Date) || isNaN(date)) {
    return 'Unknown date';
  }
  return date.toLocaleDateString(undefined, options);
};

/**
 * Formats a date to a short format (e.g., "Jan 2024")
 * @param {Date} date - The date to format
 * @returns {string} Formatted short date string
 */
export const formatDateShort = (date) => {
  return formatDate(date, { year: 'numeric', month: 'short' });
};

/**
 * Formats a date to a long format (e.g., "January 15, 2024")
 * @param {Date} date - The date to format
 * @returns {string} Formatted long date string
 */
export const formatDateLong = (date) => {
  return formatDate(date, { year: 'numeric', month: 'long', day: 'numeric' });
};

/**
 * Parses a date string or object into a Date object
 * @param {string|Date} dateInput - The date to parse
 * @returns {Date|null} Parsed Date object or null if invalid
 */
export const parseDate = (dateInput) => {
  if (!dateInput) return null;

  if (dateInput instanceof Date) {
    return isNaN(dateInput) ? null : dateInput;
  }

  const parsed = new Date(dateInput);
  return isNaN(parsed) ? null : parsed;
};

/**
 * Validates if a date is valid
 * @param {Date} date - The date to validate
 * @returns {boolean} True if valid, false otherwise
 */
export const isValidDate = (date) => {
  if (!date || !(date instanceof Date)) return false;
  if (isNaN(date.getTime())) return false;
  // Check if date is after 1970 (common data quality check)
  if (date.getFullYear() <= 1970) return false;
  return true;
};

/**
 * Gets the start and end of day for a given date
 * @param {Date} date - The date to process
 * @returns {Object} Object with startOfDay and endOfDay Date objects
 */
export const getDateBounds = (date) => {
  if (!date) return null;

  const startOfDay = new Date(date);
  startOfDay.setHours(0, 0, 0, 0);

  const endOfDay = new Date(date);
  endOfDay.setHours(23, 59, 59, 999);

  return { startOfDay, endOfDay };
};

/**
 * Checks if a date falls within a date range
 * @param {Date} date - The date to check
 * @param {Date|null} startDate - The start of the range (inclusive)
 * @param {Date|null} endDate - The end of the range (inclusive)
 * @returns {boolean} True if date is within range
 */
export const isDateInRange = (date, startDate, endDate) => {
  if (!date || !(date instanceof Date) || isNaN(date)) {
    return false;
  }

  if (startDate) {
    const start = new Date(startDate);
    start.setHours(0, 0, 0, 0);
    if (date < start) return false;
  }

  if (endDate) {
    const end = new Date(endDate);
    end.setHours(23, 59, 59, 999);
    if (date > end) return false;
  }

  return true;
};

/**
 * Gets a date range for common periods (last week, last month, last year, etc.)
 * @param {string} period - The period type ('week', 'month', 'year', 'all')
 * @returns {Object} Object with startDate and endDate
 */
export const getDateRange = (period) => {
  const now = new Date();
  const endDate = new Date(now);
  let startDate = new Date(now);

  switch(period) {
    case 'week':
      startDate.setDate(now.getDate() - 7);
      break;
    case 'month':
      startDate.setMonth(now.getMonth() - 1);
      break;
    case 'year':
      startDate.setFullYear(now.getFullYear() - 1);
      break;
    case 'all':
      return { startDate: null, endDate: null };
    default:
      return { startDate: null, endDate: null };
  }

  return { startDate, endDate };
};

/**
 * Calculates the difference between two dates in days
 * @param {Date} date1 - First date
 * @param {Date} date2 - Second date
 * @returns {number} Difference in days
 */
export const getDaysDifference = (date1, date2) => {
  if (!isValidDate(date1) || !isValidDate(date2)) return 0;

  const msPerDay = 1000 * 60 * 60 * 24;
  const diffMs = Math.abs(date2 - date1);
  return Math.floor(diffMs / msPerDay);
};

/**
 * Groups dates by year
 * @param {Array} items - Array of items with date properties
 * @param {string} dateField - The field name containing the date
 * @returns {Object} Object with years as keys and items as values
 */
export const groupByYear = (items, dateField = 'date') => {
  return items.reduce((acc, item) => {
    const date = item[dateField];
    if (isValidDate(date)) {
      const year = date.getFullYear();
      if (!acc[year]) acc[year] = [];
      acc[year].push(item);
    }
    return acc;
  }, {});
};

/**
 * Groups dates by month
 * @param {Array} items - Array of items with date properties
 * @param {string} dateField - The field name containing the date
 * @returns {Object} Object with month keys (YYYY-MM) and items as values
 */
export const groupByMonth = (items, dateField = 'date') => {
  return items.reduce((acc, item) => {
    const date = item[dateField];
    if (isValidDate(date)) {
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      if (!acc[monthKey]) acc[monthKey] = [];
      acc[monthKey].push(item);
    }
    return acc;
  }, {});
};
