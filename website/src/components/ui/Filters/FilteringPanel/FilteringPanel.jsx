// src/components/ui/Filters/FilteringPanel/FilteringPanel.jsx
import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import _ from 'lodash';
import { SlidersHorizontal } from 'lucide-react';
import Filter from '../Filter/Filter';
import { applyFilters, matchDelimitedValue, buildHierarchyWithCounts } from '../../../../utils/filterUtils';
import './FilteringPanel.css';

/**
 * Helper: Check if an item matches a filter value
 * Used for index-based filtering optimization
 */
const itemMatchesFilter = (item, config, filterValue) => {
  const dataField = config.dataField || config.optionsSource;
  if (!dataField) return true;

  const itemValue = dataField.includes('.')
    ? _.get(item, dataField)
    : item[dataField];

  if (config.type === 'multiselect') {
    if (!Array.isArray(filterValue) || filterValue.length === 0) return true;
    if (config.delimiter) {
      return matchDelimitedValue(itemValue, filterValue, config.delimiter, config.matchMode || 'any');
    }
    return filterValue.includes(itemValue);
  }

  if (config.type === 'singleselect') {
    if (!filterValue || filterValue === 'all' || filterValue === config.defaultValue) return true;
    return itemValue === filterValue;
  }

  if (config.type === 'daterange') {
    if (!filterValue || (!filterValue.startDate && !filterValue.endDate)) return true;
    if (!itemValue) return false;
    const itemDate = new Date(itemValue);
    if (isNaN(itemDate.getTime())) return false;
    if (filterValue.startDate && itemDate < new Date(filterValue.startDate)) return false;
    if (filterValue.endDate && itemDate > new Date(filterValue.endDate)) return false;
    return true;
  }

  if (config.type === 'numberrange') {
    if (!filterValue || (filterValue.min === null && filterValue.max === null)) return true;
    if (itemValue === null || itemValue === undefined) return false;
    const numValue = parseFloat(itemValue);
    if (isNaN(numValue)) return false;
    if (filterValue.min !== null && numValue < filterValue.min) return false;
    if (filterValue.max !== null && numValue > filterValue.max) return false;
    return true;
  }

  if (config.type === 'hierarchical') {
    const values = config.selectionMode === 'single' ? [filterValue] : filterValue;
    if (!values || values.length === 0 || values[0] === null) return true;
    const parentValue = item[config.dataField];
    const childValue = item[config.childField];
    return values.some(selected => selected === parentValue || selected === childValue);
  }

  return true;
};

/**
 * Helper: Extract a bridge field value with normalization
 * Handles dates by normalizing to YYYY-MM-DD for cross-source comparison
 */
const extractBridgeValue = (item, field) => {
  const value = field.includes('.') ? _.get(item, field) : item[field];
  if (!value) return null;

  // Normalize dates to YYYY-MM-DD for comparison
  if (value instanceof Date) {
    return value.toISOString().split('T')[0];
  }

  // Check if it's a date string and normalize
  const dateValue = new Date(value);
  if (!isNaN(dateValue.getTime())) {
    return dateValue.toISOString().split('T')[0];
  }

  return value;
};

/**
 * A comprehensive filtering panel that manages filters and automatically
 * updates filter options based on data
 *
 * @param {Object} props
 * @param {Array|Object} props.data - The raw data to filter (array for single source, object for multiple sources)
 * @param {React.ReactNode} props.children - Filter components as children
 * @param {string} [props.fullDataset] - Full dataset for date boundary calculation (music data)
 * @param {Function} props.onFiltersChange - Callback when filters change
 * @param {boolean} [props.loading] - Loading state
 * @param {string} [props.renderMode] - 'dropdown' or 'bubble' (default: 'dropdown')
 * @param {Object} [props.bubbleConfig] - Configuration for bubble mode (maxVisible, searchThreshold)
 */
const FilteringPanel = ({
  data = [],
  fullDataset = '',
  children,
  onFiltersChange,
  loading = false,
  renderMode = 'dropdown',
  bubbleConfig = {}
}) => {
  // Ensure data is never null or undefined
  const safeData = data ?? [];

  // Determine if we're working with multiple data sources
  const isMultiSource = !Array.isArray(safeData) && typeof safeData === 'object' && safeData !== null;

  // For backward compatibility, convert single array to object format
  // Memoize to prevent unnecessary re-renders
  const dataSources = useMemo(() => {
    return isMultiSource ? safeData : { default: safeData };
  }, [isMultiSource, safeData]);

  // Extract filter configs from children
  const filterConfigs = useMemo(() => {
    const configs = [];

    React.Children.forEach(children, (child) => {
      if (!React.isValidElement(child)) return;

      // Extract props from Filter child to build config
      const { type, label, icon, placeholder, searchPlaceholder, searchable, allLabel, defaultValue, delimiter, matchMode, sortType, selectionMode, childField, fieldMap, ...rest } = child.props;

      // Generate a key from the field or label
      const key = rest.field || label?.toLowerCase().replace(/\s+/g, '_') || `filter_${configs.length}`;

      configs.push({
        key,
        type: type || 'singleselect',
        label,
        icon,
        placeholder,
        searchPlaceholder,
        searchable,
        allLabel,
        defaultValue,
        dataField: rest.field,
        childField: childField || null,
        selectionMode: selectionMode || 'multi',
        optionsSource: rest.field,
        dataSources: rest.dataSources,
        options: rest.options,
        delimiter: delimiter || null,
        matchMode: matchMode || 'exact',
        sortType: sortType || 'alpha',
        fieldMap: fieldMap || null  // Per-source field mapping for multi-source filtering
      });
    });

    return configs;
  }, [children]);

  // Compute bridge relationships between data sources
  // A "bridge" is a filter that spans multiple sources with fieldMap - it defines how sources relate
  const bridgeRelationships = useMemo(() => {
    const relationships = new Map();

    filterConfigs.forEach(config => {
      // Bridge filters span multiple sources with fieldMap
      if (config.dataSources?.length > 1 && config.fieldMap) {
        const sources = Object.keys(config.fieldMap);
        for (let i = 0; i < sources.length; i++) {
          for (let j = i + 1; j < sources.length; j++) {
            const key = `${sources[i]}:${sources[j]}`;
            relationships.set(key, {
              fieldA: config.fieldMap[sources[i]],
              fieldB: config.fieldMap[sources[j]]
            });
            // Reverse lookup
            relationships.set(`${sources[j]}:${sources[i]}`, {
              fieldA: config.fieldMap[sources[j]],
              fieldB: config.fieldMap[sources[i]]
            });
          }
        }
      }
    });

    return relationships;
  }, [filterConfigs]);

  // Get primary data source (memoized)
  const primarySource = useMemo(() => {
    return Object.values(dataSources).find(source => Array.isArray(source) && source.length > 0) || [];
  }, [dataSources]);

  // OPTIMIZATION: Pre-compute unique values and value-to-indices mapping on data load
  // This runs ONCE when data changes, not on every filter change
  const precomputedOptions = useMemo(() => {
    if (primarySource.length === 0 && Object.values(dataSources).every(s => !s || s.length === 0)) {
      return { cache: {}, dateBoundaries: {}, numberBoundaries: {} };
    }

    const cache = {};
    const dateBoundaries = {};
    const numberBoundaries = {};

    filterConfigs.forEach(config => {
      const fieldName = config.dataField || config.optionsSource;
      if (!fieldName) return;

      // Determine which data source to use for extracting options
      // If filter specifies dataSources, use the first one; otherwise use primarySource
      let sourceData = primarySource;
      if (config.dataSources && config.dataSources.length > 0) {
        const specifiedSource = dataSources[config.dataSources[0]];
        if (Array.isArray(specifiedSource) && specifiedSource.length > 0) {
          sourceData = specifiedSource;
        }
      }

      if (config.optionsSource === 'static' && config.options) {
        // Static options - just store them
        cache[config.key] = { allValues: config.options, valueToItemIndices: null };
        return;
      }

      if (config.type === 'daterange') {
        // Pre-compute date boundaries from full dataset
        let minDate = null;
        let maxDate = null;
        sourceData.forEach(item => {
          const value = fieldName.includes('.') ? _.get(item, fieldName) : item[fieldName];
          if (value) {
            const date = new Date(value);
            if (!isNaN(date.getTime()) && date.getFullYear() > 1900) {
              if (!minDate || date < minDate) minDate = date;
              if (!maxDate || date > maxDate) maxDate = date;
            }
          }
        });
        if (minDate && maxDate) {
          dateBoundaries[config.key] = { minDate, maxDate };
        }
        return;
      }

      if (config.type === 'numberrange') {
        // Pre-compute number boundaries from full dataset
        let minNum = null;
        let maxNum = null;
        sourceData.forEach(item => {
          const value = fieldName.includes('.') ? _.get(item, fieldName) : item[fieldName];
          if (value !== null && value !== undefined && value !== '') {
            const num = parseFloat(value);
            if (!isNaN(num)) {
              if (minNum === null || num < minNum) minNum = num;
              if (maxNum === null || num > maxNum) maxNum = num;
            }
          }
        });
        if (minNum !== null && maxNum !== null) {
          numberBoundaries[config.key] = { minNumber: minNum, maxNumber: maxNum };
        }
        return;
      }

      if (config.type === 'hierarchical') {
        // For hierarchical, we'll compute on demand (uses buildHierarchyWithCounts)
        return;
      }

      // For multiselect/singleselect: Build unique values + value-to-indices map
      const uniqueSet = new Set();
      const valueToItemIndices = new Map();

      sourceData.forEach((item, index) => {
        const value = fieldName.includes('.') ? _.get(item, fieldName) : item[fieldName];

        if (value && value !== 'Unknown' && value.toString().trim() !== '') {
          if (config.delimiter) {
            // Handle delimited values (e.g., genres: "rock, pop, jazz")
            value.split(config.delimiter).forEach(v => {
              const trimmed = v.trim();
              if (trimmed) {
                uniqueSet.add(trimmed);
                if (!valueToItemIndices.has(trimmed)) valueToItemIndices.set(trimmed, []);
                valueToItemIndices.get(trimmed).push(index);
              }
            });
          } else {
            uniqueSet.add(value);
            if (!valueToItemIndices.has(value)) valueToItemIndices.set(value, []);
            valueToItemIndices.get(value).push(index);
          }
        }
      });

      // Sort allValues based on sortType
      let allValues = Array.from(uniqueSet);
      if (config.sortType === 'numeric') {
        allValues.sort((a, b) => Number(a) - Number(b));
      } else if (config.sortType === 'numericDesc') {
        allValues.sort((a, b) => Number(b) - Number(a));
      } else {
        allValues.sort();
      }

      cache[config.key] = { allValues, valueToItemIndices };
    });

    return { cache, dateBoundaries, numberBoundaries };
  }, [primarySource, dataSources, filterConfigs]); // Only recompute when DATA changes, not on filter changes

  // Initialize filters state with proper defaults
  const [filters, setFilters] = useState(() => {
    const initialState = {};

    filterConfigs.forEach(config => {
      if (config.type === 'multiselect') {
        // For multiselect, always start with empty array
        initialState[config.key] = [];
      } else if (config.type === 'hierarchical') {
        // For hierarchical, single-select starts with null, multi-select with empty array
        initialState[config.key] = config.selectionMode === 'single' ? null : [];
      } else if (config.type === 'singleselect') {
        // For singleselect, use provided initial value or first option
        initialState[config.key] = config.defaultValue || 'all';
      } else if (config.type === 'daterange') {
        // For daterange, start with no selection
        initialState[config.key] = { startDate: null, endDate: null };
      } else if (config.type === 'numberrange') {
        // For numberrange, start with no selection
        initialState[config.key] = { min: null, max: null };
      }
    });

    return initialState;
  });

  // Debounced filters state for cascading options calculation
  // This prevents expensive recalculation on every filter click
  const [debouncedFilters, setDebouncedFilters] = useState(filters);

  // Mobile modal state management
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 992);

  // Debounce filter updates for options calculation (150ms delay)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedFilters(filters);
    }, 150);

    return () => clearTimeout(timer);
  }, [filters]);

  // Count active filters (non-default values)
  const getActiveFilterCount = useCallback((filterState) => {
    return filterConfigs.reduce((count, config) => {
      const value = filterState[config.key];

      // Multiselect: count if non-empty array
      if (config.type === 'multiselect' && Array.isArray(value) && value.length > 0) {
        return count + 1;
      }

      // Singleselect: count if not default value
      if (config.type === 'singleselect' && value !== config.defaultValue && value !== 'all') {
        return count + 1;
      }

      // Date range: count if either bound is set
      if (config.type === 'daterange' && (value?.startDate || value?.endDate)) {
        return count + 1;
      }

      // Number range: count if either bound is set
      if (config.type === 'numberrange' && (value?.min !== null || value?.max !== null)) {
        return count + 1;
      }

      // Hierarchical: count if non-empty selection
      if (config.type === 'hierarchical' && Array.isArray(value) && value.length > 0) {
        return count + 1;
      }

      return count;
    }, 0);
  }, [filterConfigs]);

  // Modal handler functions
  const openModal = useCallback(() => {
    setIsModalOpen(true);
  }, []);

  const closeModal = useCallback(() => {
    setIsModalOpen(false);
  }, []);

  const clearAllFilters = useCallback(() => {
    const resetState = {};
    filterConfigs.forEach(config => {
      if (config.type === 'multiselect') {
        resetState[config.key] = [];
      } else if (config.type === 'hierarchical') {
        resetState[config.key] = config.selectionMode === 'single' ? null : [];
      } else if (config.type === 'singleselect') {
        resetState[config.key] = config.defaultValue || 'all';
      } else if (config.type === 'daterange') {
        resetState[config.key] = { startDate: null, endDate: null };
      } else if (config.type === 'numberrange') {
        resetState[config.key] = { min: null, max: null };
      }
    });
    setFilters(resetState);
  }, [filterConfigs]);

  const activeFilterCount = useMemo(() => getActiveFilterCount(filters), [filters, getActiveFilterCount]);

  // Window resize listener
  useEffect(() => {
    const handleResize = () => {
      const newIsMobile = window.innerWidth <= 992;
      setIsMobile(newIsMobile);

      // Close modal when transitioning mobile → desktop
      if (!newIsMobile && isModalOpen) {
        setIsModalOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isModalOpen]);

  // Body scroll lock when modal is open
  useEffect(() => {
    if (isMobile && isModalOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isMobile, isModalOpen]);

  // ESC key handler for closing modal
  useEffect(() => {
    if (!isModalOpen) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        closeModal();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isModalOpen, closeModal]);

  // OPTIMIZATION: Compute indices for each "exclude one filter" scenario
  // This allows efficient cascading filter options calculation
  const excludeOneFilterIndices = useMemo(() => {
    if (primarySource.length === 0) return new Map();

    const cache = new Map();

    filterConfigs.forEach(excludeConfig => {
      // Compute which items pass ALL filters EXCEPT this one
      const passingIndices = new Set();

      for (let idx = 0; idx < primarySource.length; idx++) {
        const item = primarySource[idx];
        let passes = true;

        for (const otherConfig of filterConfigs) {
          if (otherConfig.key === excludeConfig.key) continue; // Skip the excluded filter

          const filterValue = debouncedFilters[otherConfig.key];
          if (!itemMatchesFilter(item, otherConfig, filterValue)) {
            passes = false;
            break;
          }
        }

        if (passes) {
          passingIndices.add(idx);
        }
      }

      cache.set(excludeConfig.key, passingIndices);
    });

    return cache;
  }, [primarySource, filterConfigs, debouncedFilters]);

  // OPTIMIZED: Calculate filtered options using precomputed data and exclusion indices
  const filterOptions = useMemo(() => {
    // Check if any data source has data
    const hasAnyData = primarySource.length > 0 || Object.values(dataSources).some(s => Array.isArray(s) && s.length > 0);
    if (!hasAnyData) return { options: {}, dateBoundaries: {}, numberBoundaries: {} };

    const options = {};
    // Use precomputed boundaries (static, computed once on data load)
    const dateBoundaries = { ...precomputedOptions.dateBoundaries };
    const numberBoundaries = { ...precomputedOptions.numberBoundaries };

    filterConfigs.forEach(config => {
      // Static options - use as-is
      if (config.optionsSource === 'static' && config.options) {
        options[config.key] = config.options;
        return;
      }

      // Skip daterange and numberrange (boundaries already precomputed)
      if (config.type === 'daterange' || config.type === 'numberrange') {
        return;
      }

      // Hierarchical filters need special handling (compute on demand with filtered data)
      if (config.type === 'hierarchical' && config.childField) {
        const passingIndices = excludeOneFilterIndices.get(config.key) || new Set();
        const filteredData = [];
        passingIndices.forEach(idx => filteredData.push(primarySource[idx]));
        const hierarchyWithCounts = buildHierarchyWithCounts(
          filteredData,
          config.dataField || config.optionsSource,
          config.childField
        );
        options[config.key] = hierarchyWithCounts;
        return;
      }

      // For multiselect/singleselect: Use precomputed values + exclusion indices
      const cached = precomputedOptions.cache[config.key];
      if (!cached || !cached.valueToItemIndices) {
        options[config.key] = cached?.allValues || [];
        return;
      }

      // Check if this filter uses a different data source than primarySource
      // If so, cascading doesn't apply - just return all precomputed values
      if (config.dataSources && config.dataSources.length > 0) {
        const specifiedSource = dataSources[config.dataSources[0]];
        if (specifiedSource && specifiedSource !== primarySource) {
          // This filter has its own data source, no cascading needed
          options[config.key] = cached.allValues;
          return;
        }
      }

      // Get indices that pass all OTHER filters
      const passingIndices = excludeOneFilterIndices.get(config.key);
      if (!passingIndices || passingIndices.size === 0) {
        // No items pass other filters - show empty options
        options[config.key] = [];
        return;
      }

      // If all items pass (no active filters), return all precomputed values
      if (passingIndices.size === primarySource.length) {
        options[config.key] = cached.allValues;
        return;
      }

      // Filter allValues to only those that have at least one item in passingIndices
      const availableValues = cached.allValues.filter(value => {
        const itemIndices = cached.valueToItemIndices.get(value);
        if (!itemIndices) return false;
        // Check if ANY item with this value passes the other filters
        for (const idx of itemIndices) {
          if (passingIndices.has(idx)) return true;
        }
        return false;
      });

      options[config.key] = availableValues;
    });

    return { options, dateBoundaries, numberBoundaries };
  }, [primarySource, dataSources, filterConfigs, excludeOneFilterIndices, precomputedOptions]);

  // Use refs to store callback and configs to prevent infinite loops
  const onFiltersChangeRef = useRef(onFiltersChange);
  const filterConfigsRef = useRef(filterConfigs);
  const dataSourcesRef = useRef(dataSources);
  const bridgeRelationshipsRef = useRef(bridgeRelationships);

  useEffect(() => {
    onFiltersChangeRef.current = onFiltersChange;
  }, [onFiltersChange]);

  useEffect(() => {
    filterConfigsRef.current = filterConfigs;
  }, [filterConfigs]);

  useEffect(() => {
    dataSourcesRef.current = dataSources;
  }, [dataSources]);

  useEffect(() => {
    bridgeRelationshipsRef.current = bridgeRelationships;
  }, [bridgeRelationships]);

  // Handle individual filter changes
  const handleFilterChange = (filterKey, newValue) => {
    setFilters(prevFilters => {
      const updatedFilters = {
        ...prevFilters,
        [filterKey]: newValue
      };
      return updatedFilters;
    });
  };

  // Apply filters to all data sources and notify parent
  // Uses two-phase approach: direct filtering, then cross-source cascading via bridge fields
  useEffect(() => {
    if (!onFiltersChangeRef.current) return;

    // Use refs to get current values
    const currentFilterConfigs = filterConfigsRef.current;
    const currentDataSources = dataSourcesRef.current;
    const currentBridgeRelationships = bridgeRelationshipsRef.current;

    // Always use current filters state (same behavior on mobile and desktop)
    const filtersToApply = filters;

    // If single-source mode (backward compatibility), just return filters
    if (!isMultiSource) {
      onFiltersChangeRef.current(filtersToApply);
      return;
    }

    // Create filter config map for faster lookup
    const filterConfigsMap = currentFilterConfigs.reduce((acc, config) => {
      acc[config.key] = config;
      return acc;
    }, {});

    // PHASE 1: Apply direct filters to each data source
    const directFilteredSources = {};
    const sourceHadUniqueFilter = {}; // Track which sources had source-specific (non-bridge) filters

    Object.keys(currentDataSources).forEach(sourceName => {
      const sourceData = currentDataSources[sourceName];
      if (!Array.isArray(sourceData)) return;

      let filtered = [...sourceData];
      let hadUniqueFilter = false;

      // Apply each filter
      Object.keys(filtersToApply).forEach(filterKey => {
        const filterValue = filtersToApply[filterKey];
        const config = filterConfigsMap[filterKey];

        // Skip empty filter values
        if (!filterValue || (Array.isArray(filterValue) && filterValue.length === 0)) return;
        if (config?.type === 'daterange' && !filterValue.startDate && !filterValue.endDate) return;
        if (config?.type === 'numberrange' && filterValue.min === null && filterValue.max === null) return;

        // Skip if filter doesn't apply to this data source
        if (config?.dataSources && config.dataSources.length > 0) {
          if (!config.dataSources.includes(sourceName)) {
            return; // Skip this filter for this data source
          }

          // Track if this source had a source-specific (non-bridge) filter applied
          if (config.dataSources.length === 1) {
            hadUniqueFilter = true;
          }
        }

        // Create a source-specific config with resolved field from fieldMap
        const sourceConfig = { ...config };
        if (config.fieldMap && config.fieldMap[sourceName]) {
          sourceConfig.dataField = config.fieldMap[sourceName];
        }

        // Apply filter using utility
        const singleFilterObj = { [filterKey]: filterValue };
        const singleConfigMap = { [filterKey]: sourceConfig };
        filtered = applyFilters(filtered, singleFilterObj, singleConfigMap);
      });

      directFilteredSources[sourceName] = filtered;
      sourceHadUniqueFilter[sourceName] = hadUniqueFilter;
    });

    // PHASE 2: Apply cross-source cascading through bridge fields
    // When a source has a unique filter, cascade to connected sources via bridge fields
    const cascadedSources = { ...directFilteredSources };

    Object.keys(directFilteredSources).forEach(sourceName => {
      // Only cascade if this source had a source-specific (unique) filter applied
      if (!sourceHadUniqueFilter[sourceName]) return;

      // Find all other sources connected to this one via bridges
      Object.keys(currentDataSources).forEach(otherSource => {
        if (otherSource === sourceName) return;

        // Look up bridge relationship between these sources
        const bridgeKey = `${sourceName}:${otherSource}`;
        const bridge = currentBridgeRelationships.get(bridgeKey);

        if (!bridge) return; // No bridge between these sources

        // Extract bridge field values from the filtered source
        const bridgeValues = new Set();
        directFilteredSources[sourceName].forEach(item => {
          const val = extractBridgeValue(item, bridge.fieldA);
          if (val) bridgeValues.add(val);
        });

        // Filter the other source to only items matching the bridge values
        if (bridgeValues.size > 0) {
          cascadedSources[otherSource] = cascadedSources[otherSource].filter(item => {
            const val = extractBridgeValue(item, bridge.fieldB);
            return bridgeValues.has(val);
          });
        } else {
          // No items in source after filtering - cascade empty result
          cascadedSources[otherSource] = [];
        }
      });
    });

    onFiltersChangeRef.current(cascadedSources, filtersToApply);
  }, [filters, isMultiSource]); // Using refs for filterConfigs, dataSources, bridgeRelationships

  // Helper function to render filter items (used in both desktop and mobile)
  const renderFilterItems = () => {
    return React.Children.map(children, (child) => {
      if (!React.isValidElement(child)) return null;

      // Find the config for this child
      const childKey = child.props.field || child.props.label?.toLowerCase().replace(/\s+/g, '_');
      const config = filterConfigs.find(c => c.key === childKey);

      if (!config) return null;

      // Use custom options if provided, otherwise use auto-calculated options
      const customOptions = config.options || child.props.options;
      const calculatedOptions = filterOptions.options?.[config.key] || [];

      // For hierarchical filters, calculatedOptions is a Map, not an array
      const availableOptions = config.type === 'hierarchical'
        ? null  // Don't use options for hierarchical
        : (customOptions || calculatedOptions);

      const boundaries = filterOptions.dateBoundaries?.[config.key];
      const numBoundaries = filterOptions.numberBoundaries?.[config.key];
      const currentValue = filters[config.key];

      // For hierarchical filters, check if hierarchy is available
      if (config.type === 'hierarchical' && !calculatedOptions) {
        return null;
      }

      // Note: We no longer hide daterange/numberrange filters when no boundaries exist
      // Instead, we render them with empty/disabled state for UI stability

      // Clone the child and inject the computed props
      const clonedProps = {
        options: availableOptions,
        value: currentValue,
        onChange: (value) => handleFilterChange(config.key, value),
        minDate: boundaries?.minDate,
        maxDate: boundaries?.maxDate,
        minNumber: numBoundaries?.minNumber,
        maxNumber: numBoundaries?.maxNumber,
        suffix: child.props.suffix || '',
        // Pass bubble mode props
        renderMode: renderMode,
        maxVisibleBubbles: bubbleConfig.maxVisible || 10,
        searchThreshold: bubbleConfig.searchThreshold || 15,
        showBubbleCounts: bubbleConfig.showCounts || false
      };

      // Add hierarchical-specific props if this is a hierarchical filter
      if (config.type === 'hierarchical') {
        clonedProps.hierarchyWithCounts = calculatedOptions;
        clonedProps.selectionMode = config.selectionMode;
      }

      return (
        <div key={config.key} className="filter-item">
          {React.cloneElement(child, clonedProps)}
        </div>
      );
    });
  };

  // Loading state
  if (loading) {
    return (
      <div className="filtering-panel loading">
        <div className="filtering-panel-content">
          <div className="filters-grid">
            {[1, 2, 3].map(index => (
              <div key={index} className="filter-skeleton">
                <div className="skeleton-label"></div>
                <div className="skeleton-filter"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Always render filter button and modal (modal-first approach for all screen sizes)
  return (
    <>
      {/* Filter Toggle Button */}
      <button
        className={`filter-toggle-button ${activeFilterCount > 0 ? 'active' : ''}`}
        onClick={openModal}
        aria-label={`Open filters${activeFilterCount > 0 ? ` (${activeFilterCount} active)` : ''}`}
      >
        <SlidersHorizontal size={20} />
        <span>Filters</span>
        {activeFilterCount > 0 && (
          <span className="filter-badge">{activeFilterCount}</span>
        )}
      </button>

      {/* Filter Modal (full-screen on mobile, centered on desktop) */}
      {isModalOpen && (
        <div className="filter-modal-overlay" onClick={closeModal}>
          <div
            className="filter-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="filter-modal-title"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Sticky Header */}
            <div className="filter-modal-header">
              <button
                className="filter-modal-header-button"
                onClick={clearAllFilters}
                aria-label="Clear all filters"
              >
                Clear All
              </button>
              <button
                className="filter-modal-header-button"
                onClick={closeModal}
                aria-label="Close filters"
              >
                X
              </button>
            </div>

            {/* Scrollable Content */}
            <div className="filter-modal-content">
              <div className="filter-modal-filters">
                {renderFilterItems()}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default FilteringPanel;
