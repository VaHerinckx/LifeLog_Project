/**
 * Filter Utility Functions
 *
 * Centralized filtering logic used across the LifeLog website.
 * Eliminates duplicate date range and multi-select filtering code.
 */

import { isValidDate } from './dateUtils';

/**
 * Applies a date range filter to an array of items
 * @param {Array} data - Array of items to filter
 * @param {string} dateField - The field name containing the date
 * @param {Object} dateRange - Object with optional startDate and endDate
 * @param {boolean} strictValidation - If true, excludes dates <= 1970 (default: true)
 * @returns {Array} Filtered array
 */
export const applyDateRangeFilter = (data, dateField, dateRange, strictValidation = true) => {
  if (!dateRange || (!dateRange.startDate && !dateRange.endDate)) {
    return data;
  }

  return data.filter(item => {
    // Get the date value from the item (supports nested fields via dot notation)
    let itemDate;
    if (dateField.includes('.')) {
      // Support for nested fields like 'originalEntry.date'
      const parts = dateField.split('.');
      let value = item;
      for (const part of parts) {
        value = value?.[part];
        if (!value) break;
      }
      itemDate = value;
    } else {
      itemDate = item[dateField];
    }

    // Handle both Date objects and date strings
    if (typeof itemDate === 'string') {
      itemDate = new Date(itemDate);
    }

    // Validate the date
    if (!itemDate || !(itemDate instanceof Date) || isNaN(itemDate.getTime())) {
      return false;
    }

    // Optional strict validation (excludes dates <= 1970)
    if (strictValidation && itemDate.getFullYear() <= 1970) {
      return false;
    }

    // Check start date
    if (dateRange.startDate) {
      const startDate = new Date(dateRange.startDate);
      startDate.setHours(0, 0, 0, 0);
      if (itemDate < startDate) return false;
    }

    // Check end date
    if (dateRange.endDate) {
      const endDate = new Date(dateRange.endDate);
      endDate.setHours(23, 59, 59, 999);
      if (itemDate > endDate) return false;
    }

    return true;
  });
};

/**
 * Applies a multi-select filter to an array of items
 * @param {Array} data - Array of items to filter
 * @param {string} dataField - The field name to filter by
 * @param {Array} selectedValues - Array of selected values
 * @param {string|null} delimiter - Optional delimiter for splitting field values
 * @param {string} matchMode - Match mode: 'exact' | 'any' | 'all' (default: 'exact')
 * @returns {Array} Filtered array
 */
export const applyMultiSelectFilter = (data, dataField, selectedValues, delimiter = null, matchMode = 'exact') => {
  if (!selectedValues || !Array.isArray(selectedValues) || selectedValues.length === 0) {
    return data;
  }

  return data.filter(item => {
    const itemValue = dataField.includes('.')
      ? getNestedValue(item, dataField)
      : item[dataField];

    // Handle delimited values
    if (delimiter) {
      return matchDelimitedValue(itemValue, selectedValues, delimiter, matchMode);
    }

    // Standard exact match
    return selectedValues.includes(itemValue);
  });
};

/**
 * Matches a delimited value against selected values
 * @param {string} itemValue - The delimited value from the item (e.g., "rock, pop, jazz")
 * @param {Array} selectedValues - Array of selected values
 * @param {string} delimiter - Delimiter used to split the value
 * @param {string} matchMode - Match mode: 'exact' | 'any' | 'all'
 * @returns {boolean} True if the value matches based on the match mode
 */
export const matchDelimitedValue = (itemValue, selectedValues, delimiter, matchMode = 'any') => {
  if (!itemValue) return false;

  // Split the delimited value and trim each part
  const itemValues = itemValue
    .split(delimiter)
    .map(v => v.trim())
    .filter(v => v !== '');

  if (itemValues.length === 0) return false;

  // Apply match mode
  switch (matchMode) {
    case 'exact':
      // Exact match - the delimited string must match one of the selected values exactly
      return selectedValues.includes(itemValue);

    case 'any':
      // Any match - at least one of the item's delimited values must be in selected values
      return itemValues.some(v => selectedValues.includes(v));

    case 'all':
      // All match - all selected values must be present in the item's delimited values
      return selectedValues.every(sv => itemValues.includes(sv));

    default:
      return itemValues.some(v => selectedValues.includes(v));
  }
};

/**
 * Extracts unique values from delimited strings in a dataset
 * @param {Array} data - Array of items
 * @param {string} field - Field name containing delimited values
 * @param {string} delimiter - Delimiter used to split the values
 * @param {boolean} sort - Whether to sort the results (default: true)
 * @returns {Array} Array of unique individual values
 */
export const extractDelimitedUniqueValues = (data, field, delimiter, sort = true) => {
  const allValues = new Set();

  data.forEach(item => {
    const itemValue = field.includes('.')
      ? getNestedValue(item, field)
      : item[field];

    if (itemValue) {
      // Split by delimiter and add each trimmed value to the set
      itemValue
        .split(delimiter)
        .map(v => v.trim())
        .filter(v => v !== '')
        .forEach(v => allValues.add(v));
    }
  });

  const unique = Array.from(allValues);
  return sort ? unique.sort() : unique;
};

/**
 * Applies a multi-genre filter (supports multiple genre fields per item)
 * Used for items that can have multiple genres (genre_1, genre_2, etc.)
 * @param {Array} data - Array of items to filter
 * @param {Array} genreFields - Array of genre field names (e.g., ['genre_1', 'genre_2'])
 * @param {Array} selectedGenres - Array of selected genre values
 * @returns {Array} Filtered array
 */
export const applyMultiGenreFilter = (data, genreFields, selectedGenres) => {
  if (!selectedGenres || !Array.isArray(selectedGenres) || selectedGenres.length === 0) {
    return data;
  }

  return data.filter(item => {
    // Collect all non-empty genre values for this item
    const itemGenres = genreFields
      .map(field => item[field])
      .filter(Boolean)
      .filter(genre => genre !== 'Unknown' && genre.trim() !== '');

    // Return true if any of the item's genres match the selected genres
    return itemGenres.some(genre => selectedGenres.includes(genre));
  });
};

/**
 * Applies a text search filter across multiple fields
 * @param {Array} data - Array of items to filter
 * @param {Array} searchFields - Array of field names to search in
 * @param {string} searchTerm - The search term
 * @param {boolean} caseSensitive - Whether search should be case-sensitive (default: false)
 * @returns {Array} Filtered array
 */
export const applyTextSearchFilter = (data, searchFields, searchTerm, caseSensitive = false) => {
  if (!searchTerm || searchTerm.trim() === '') {
    return data;
  }

  const term = caseSensitive ? searchTerm : searchTerm.toLowerCase();

  return data.filter(item => {
    return searchFields.some(field => {
      const value = item[field];
      if (!value) return false;

      const stringValue = String(value);
      const searchValue = caseSensitive ? stringValue : stringValue.toLowerCase();

      return searchValue.includes(term);
    });
  });
};

/**
 * Applies a numeric range filter
 * @param {Array} data - Array of items to filter
 * @param {string} field - Field name containing the numeric value
 * @param {number|null} min - Minimum value (inclusive)
 * @param {number|null} max - Maximum value (inclusive)
 * @returns {Array} Filtered array
 */
export const applyNumericRangeFilter = (data, field, min, max) => {
  if (min === null && max === null) {
    return data;
  }

  return data.filter(item => {
    const value = parseFloat(item[field]);
    if (isNaN(value)) return false;

    if (min !== null && value < min) return false;
    if (max !== null && value > max) return false;

    return true;
  });
};

/**
 * Applies multiple filters to data in sequence
 * @param {Array} data - Array of items to filter
 * @param {Object} filters - Object containing filter configurations
 * @param {Object} filterConfigs - Optional filter configuration mappings
 * @returns {Array} Filtered array
 */
export const applyFilters = (data, filters, filterConfigs = {}) => {
  let filtered = [...data];

  // Apply each filter
  Object.keys(filters).forEach(filterKey => {
    const filterValue = filters[filterKey];
    const config = filterConfigs[filterKey];

    if (!filterValue || (Array.isArray(filterValue) && filterValue.length === 0)) {
      return; // Skip empty filters
    }

    // Apply filter based on type
    if (config?.type === 'daterange') {
      filtered = applyDateRangeFilter(filtered, config.dataField, filterValue);
    } else if (config?.type === 'multiselect') {
      // Pass delimiter and matchMode to applyMultiSelectFilter
      filtered = applyMultiSelectFilter(
        filtered,
        config.dataField,
        filterValue,
        config.delimiter || null,
        config.matchMode || 'exact'
      );
    } else if (config?.type === 'multigenre') {
      filtered = applyMultiGenreFilter(filtered, config.genreFields, filterValue);
    } else if (config?.type === 'hierarchical') {
      filtered = applyHierarchicalFilter(filtered, config.dataField, config.childField, filterValue);
    } else if (config?.type === 'textsearch') {
      filtered = applyTextSearchFilter(filtered, config.searchFields, filterValue);
    } else if (config?.type === 'numberrange' || config?.type === 'numericrange') {
      filtered = applyNumericRangeFilter(filtered, config.dataField, filterValue.min, filterValue.max);
    }
  });

  return filtered;
};

/**
 * Helper: Gets a nested value from an object using dot notation
 * @param {Object} obj - The object to traverse
 * @param {string} path - The dot-notation path (e.g., 'user.address.city')
 * @returns {*} The value at the path, or undefined
 */
const getNestedValue = (obj, path) => {
  const parts = path.split('.');
  let value = obj;
  for (const part of parts) {
    value = value?.[part];
    if (value === undefined) break;
  }
  return value;
};

/**
 * Extracts unique values from a dataset for a given field
 * Useful for populating filter dropdowns
 * @param {Array} data - Array of items
 * @param {string} field - Field name to extract values from
 * @param {boolean} sort - Whether to sort the results (default: true)
 * @returns {Array} Array of unique values
 */
export const getUniqueValues = (data, field, sort = true) => {
  const values = data
    .map(item => field.includes('.') ? getNestedValue(item, field) : item[field])
    .filter(value => value !== null && value !== undefined && value !== '');

  const unique = [...new Set(values)];

  return sort ? unique.sort() : unique;
};

/**
 * Extracts unique values from multiple genre fields
 * @param {Array} data - Array of items
 * @param {Array} genreFields - Array of genre field names
 * @param {boolean} sort - Whether to sort the results (default: true)
 * @returns {Array} Array of unique genre values
 */
export const getUniqueGenres = (data, genreFields, sort = true) => {
  const allGenres = data.flatMap(item =>
    genreFields
      .map(field => item[field])
      .filter(Boolean)
      .filter(genre => genre !== 'Unknown' && genre.trim() !== '')
  );

  const unique = [...new Set(allGenres)];

  return sort ? unique.sort() : unique;
};

/**
 * Gets available filter options based on currently applied filters
 * (Cascading filter logic - used by FilteringPanel)
 * @param {Array} data - Full dataset
 * @param {Object} currentFilters - Currently applied filters
 * @param {string} currentFilterKey - The filter being calculated
 * @param {Array} filterConfigs - Array of all filter configurations
 * @returns {Array} Available options for the current filter
 */
export const getCascadingFilterOptions = (data, currentFilters, currentFilterKey, filterConfigs) => {
  // Apply all filters except the current one
  let filteredData = [...data];

  filterConfigs.forEach(config => {
    if (config.key === currentFilterKey) return; // Skip current filter

    const filterValue = currentFilters[config.key];
    if (!filterValue || (Array.isArray(filterValue) && filterValue.length === 0)) {
      return;
    }

    // Apply the filter based on type
    if (config.type === 'multiselect') {
      filteredData = applyMultiSelectFilter(filteredData, config.dataField, filterValue);
    } else if (config.type === 'daterange') {
      filteredData = applyDateRangeFilter(filteredData, config.dataField, filterValue);
    } else if (config.type === 'multigenre') {
      filteredData = applyMultiGenreFilter(filteredData, config.genreFields, filterValue);
    } else if (config.type === 'hierarchical') {
      filteredData = applyHierarchicalFilter(filteredData, config.dataField, config.childField, filterValue);
    }
  });

  // Extract unique values from filtered data
  const currentConfig = filterConfigs.find(c => c.key === currentFilterKey);
  if (!currentConfig) return [];

  if (currentConfig.type === 'multigenre') {
    return getUniqueGenres(filteredData, currentConfig.genreFields);
  } else {
    return getUniqueValues(filteredData, currentConfig.dataField || currentConfig.optionsSource);
  }
};

/**
 * Builds a hierarchical structure from parent-child relationships in data
 * @param {Array} data - Array of items with parent and child fields
 * @param {string} parentField - Field name for parent category
 * @param {string} childField - Field name for child category
 * @returns {Map} Map of parent values to arrays of child values
 *
 * Example output:
 * Map {
 *   'Food & drinks' => ['Lunch', 'Dinner', 'Groceries', ...],
 *   'Transportation' => ['Subway', 'Train', 'Taxi', ...],
 *   'Initial capital' => []  // No children
 * }
 */
export const buildHierarchy = (data, parentField, childField) => {
  const hierarchy = new Map();

  data.forEach(item => {
    const parent = item[parentField];
    const child = item[childField];

    if (!parent) return;

    // Initialize parent entry if it doesn't exist
    if (!hierarchy.has(parent)) {
      hierarchy.set(parent, new Set());
    }

    // Add child if it exists and isn't empty
    if (child && child.trim() !== '') {
      hierarchy.get(parent).add(child);
    }
  });

  // Convert Sets to sorted arrays
  const result = new Map();
  hierarchy.forEach((children, parent) => {
    result.set(parent, Array.from(children).sort());
  });

  return result;
};

/**
 * Applies a hierarchical filter (parent-child relationship)
 * @param {Array} data - Array of items to filter
 * @param {string} parentField - Field name for parent category
 * @param {string} childField - Field name for child category
 * @param {Array|string} selectedValues - Selected parent and/or child values (array for multi, string for single)
 * @returns {Array} Filtered array
 *
 * Logic: An item matches if its parent OR child matches any selected value
 * Example: If 'Food & drinks' is selected, all food items match
 *          If 'Lunch' is selected, only lunch items match
 */
export const applyHierarchicalFilter = (data, parentField, childField, selectedValues) => {
  // Handle both single-select (string) and multi-select (array)
  const valuesArray = Array.isArray(selectedValues) ? selectedValues : [selectedValues];

  if (!valuesArray || valuesArray.length === 0) {
    return data;
  }

  return data.filter(item => {
    const parentValue = item[parentField];
    const childValue = item[childField];

    // Match if parent OR child is in selected values
    return valuesArray.some(selected => {
      return selected === parentValue || selected === childValue;
    });
  });
};

/**
 * Gets hierarchy with transaction counts for each parent and child
 * @param {Array} data - Array of items
 * @param {string} parentField - Field name for parent category
 * @param {string} childField - Field name for child category
 * @returns {Map} Map of parent values to objects containing children and counts
 *
 * Example output:
 * Map {
 *   'Food & drinks' => {
 *     count: 1923,
 *     children: Map {
 *       'Lunch' => 543,
 *       'Dinner' => 421,
 *       ...
 *     }
 *   }
 * }
 */
export const buildHierarchyWithCounts = (data, parentField, childField) => {
  const hierarchy = new Map();

  // Count occurrences
  data.forEach(item => {
    const parent = item[parentField];
    const child = item[childField];

    if (!parent) return;

    // Initialize parent entry if it doesn't exist
    if (!hierarchy.has(parent)) {
      hierarchy.set(parent, {
        count: 0,
        children: new Map()
      });
    }

    const parentData = hierarchy.get(parent);
    parentData.count++;

    // Count child if it exists
    if (child && child.trim() !== '') {
      const currentChildCount = parentData.children.get(child) || 0;
      parentData.children.set(child, currentChildCount + 1);
    }
  });

  // Sort children by count (descending) within each parent
  hierarchy.forEach((parentData) => {
    const sortedChildren = new Map(
      Array.from(parentData.children.entries())
        .sort((a, b) => b[1] - a[1]) // Sort by count descending
    );
    parentData.children = sortedChildren;
  });

  return hierarchy;
};
