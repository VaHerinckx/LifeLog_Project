// src/components/ui/Filters/Filter/Filter.jsx
import React, { useState, useRef, useEffect } from 'react';
import { Search, ChevronDown, ChevronRight, X, Check, Calendar } from 'lucide-react';
import './Filter.css';

/**
 * A versatile filter component that supports single-select, multi-select, hierarchical, date range, and number range modes
 * with PowerBI-style radio button (circle) and checkbox (square) styling, or Airbnb-style bubble interface
 *
 * @param {Object} props
 * @param {string} props.type - 'singleselect', 'multiselect', 'hierarchical', 'daterange', or 'numberrange'
 * @param {Array} props.options - Array of option strings (not used for daterange, numberrange, or hierarchical)
 * @param {string|Array|Object} props.value - Selected value(s), date range object, or number range object
 * @param {Function} props.onChange - Callback when selection changes
 * @param {string} props.label - Label for the filter
 * @param {React.ReactNode} [props.icon] - Optional icon
 * @param {string} [props.placeholder] - Placeholder text
 * @param {string} [props.searchPlaceholder] - Search input placeholder
 * @param {boolean} [props.searchable] - Whether to show search input
 * @param {string} [props.allLabel] - Label for "All" option (default: "All") - only for multiselect
 * @param {string} [props.defaultValue] - Default value for single select (required for singleselect)
 * @param {Date} [props.minDate] - Minimum date for daterange
 * @param {Date} [props.maxDate] - Maximum date for daterange
 * @param {number} [props.minNumber] - Minimum number for numberrange
 * @param {number} [props.maxNumber] - Maximum number for numberrange
 * @param {string} [props.suffix] - Suffix for number display (e.g., " min", " years")
 * @param {Map} [props.hierarchy] - Hierarchy Map for hierarchical type (parent -> [children])
 * @param {Map} [props.hierarchyWithCounts] - Hierarchy with counts for hierarchical type
 * @param {string} [props.selectionMode] - 'single' or 'multi' for hierarchical type
 * @param {boolean} [props.showCounts] - Whether to show counts in hierarchical type (default: true)
 * @param {string} [props.renderMode] - 'dropdown' or 'bubble' (default: 'dropdown')
 * @param {number} [props.maxVisibleBubbles] - Maximum bubbles to show before "Show more" (default: 10)
 * @param {number} [props.searchThreshold] - Show search bar when options exceed this (default: 15)
 * @param {boolean} [props.showBubbleCounts] - Show counts in bubble labels (default: false)
 * @param {number} [props.bubbleExpandIncrement] - Number of items to show/hide when clicking show more/less (default: 10)
 */
const Filter = ({
  type = 'singleselect',
  options = [],
  value,
  onChange,
  label,
  icon = null,
  placeholder = "Select option",
  searchPlaceholder = "Search...",
  searchable = true,
  allLabel = "All",
  defaultValue = null,
  minDate = null,
  maxDate = null,
  minNumber = null,
  maxNumber = null,
  suffix = '',
  hierarchy = null,
  hierarchyWithCounts = null,
  selectionMode = 'multi',
  showCounts = true,
  // Bubble mode props
  renderMode = 'dropdown',
  maxVisibleBubbles = 10,
  searchThreshold = 15,
  showBubbleCounts = false,
  bubbleExpandIncrement = 10
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState(new Set());
  const dropdownRef = useRef(null);
  const searchInputRef = useRef(null);
  const sliderRef = useRef(null);
  const onChangeRef = useRef(onChange);
  const tempRangeRef = useRef(null);
  const justFinishedDraggingRef = useRef(false);

  // Date range specific state
  const [dragging, setDragging] = useState(null);
  const [tempRange, setTempRange] = useState(null); // Store temporary range during drag

  // Bubble mode specific state
  const [additionalItemsShown, setAdditionalItemsShown] = useState(0); // Tracks how many extra items beyond maxVisibleBubbles are shown
  const [expandedParents, setExpandedParents] = useState(new Set()); // For hierarchical parent expansion in bubble mode

  // Normalize value to always be an array for easier processing (except daterange, numberrange, and hierarchical single-select)
  const selectedValues = type === 'daterange'
    ? value || { startDate: null, endDate: null }
    : type === 'numberrange'
    ? value || { min: null, max: null }
    : type === 'hierarchical' && selectionMode === 'single'
    ? (value ? [value] : [])
    : type === 'multiselect' || (type === 'hierarchical' && selectionMode === 'multi')
    ? (Array.isArray(value) ? value : [])
    : (value ? [value] : []);

  const selectedValuesRef = useRef(selectedValues);

  // Keep refs up to date
  useEffect(() => {
    onChangeRef.current = onChange;
    selectedValuesRef.current = selectedValues;
  }, [onChange, selectedValues]);

  // Use temp range during drag for immediate visual feedback
  const displayValues = type === 'daterange' && tempRange ? tempRange : selectedValues;

  // Validation for single select
  if (type === 'singleselect' && !defaultValue) {
    console.error('Filter: defaultValue is required for singleselect type');
  }

  // Date range helper functions
  const formatDate = (date) => {
    if (!date) return '';
    return date.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const formatDateForInput = (date) => {
    if (!date) return '';
    return date.toISOString().split('T')[0];
  };

  const positionToDate = (position) => {
    if (!minDate || !maxDate) return null;
    const totalMs = maxDate.getTime() - minDate.getTime();
    const offsetMs = totalMs * (position / 100);
    return new Date(minDate.getTime() + offsetMs);
  };

  const dateToPosition = (date) => {
    if (!date || !minDate || !maxDate) return 0;
    const totalMs = maxDate.getTime() - minDate.getTime();
    if (totalMs === 0) return 0;
    const dateMs = date.getTime() - minDate.getTime();
    return (dateMs / totalMs) * 100;
  };

  // Number range helper functions
  const formatNumber = (num) => {
    if (num === null || num === undefined) return '';
    return num + suffix;
  };

  const positionToNumber = (position) => {
    if (minNumber === null || maxNumber === null) return null;
    const totalRange = maxNumber - minNumber;
    const offsetValue = totalRange * (position / 100);
    return Math.round(minNumber + offsetValue);
  };

  const numberToPosition = (num) => {
    if (num === null || minNumber === null || maxNumber === null) return 0;
    const totalRange = maxNumber - minNumber;
    if (totalRange === 0) return 0;
    const numOffset = num - minNumber;
    return (numOffset / totalRange) * 100;
  };

  // Filter options based on search term (skip for hierarchical type)
  const filteredOptions = (options || []).filter(option => {
    // Ensure option is a string before calling toLowerCase
    const optionStr = typeof option === 'string' ? option : String(option);
    return optionStr.toLowerCase().includes(searchTerm.toLowerCase());
  });

  // Filter hierarchy based on search term (for hierarchical type)
  const getFilteredHierarchy = () => {
    if (type !== 'hierarchical' || !hierarchyWithCounts) return null;
    if (!searchTerm) return hierarchyWithCounts;

    const filtered = new Map();
    const lowerSearch = searchTerm.toLowerCase();

    hierarchyWithCounts.forEach((parentData, parent) => {
      const parentMatches = parent.toLowerCase().includes(lowerSearch);
      const matchingChildren = new Map();

      parentData.children.forEach((count, child) => {
        if (child.toLowerCase().includes(lowerSearch)) {
          matchingChildren.set(child, count);
        }
      });

      // Include parent if it matches OR has matching children
      if (parentMatches || matchingChildren.size > 0) {
        filtered.set(parent, {
          count: parentData.count,
          children: parentMatches ? parentData.children : matchingChildren
        });
      }
    });

    return filtered;
  };

  const filteredHierarchy = type === 'hierarchical' ? getFilteredHierarchy() : null;

  // Handle clicking outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (isOpen && searchable && searchInputRef.current && type !== 'daterange') {
      searchInputRef.current.focus();
    }
  }, [isOpen, searchable, type]);

  // Reset additional items shown when search term changes
  useEffect(() => {
    setAdditionalItemsShown(0);
  }, [searchTerm]);

  // Handle mouse/touch events for date range and number range sliders
  useEffect(() => {
    if (type !== 'daterange' && type !== 'numberrange') return;

    const handleMouseMove = (e) => {
      if (!dragging || !sliderRef.current) return;

      const rect = sliderRef.current.getBoundingClientRect();
      const position = ((e.clientX - rect.left) / rect.width) * 100;
      const clampedPosition = Math.max(0, Math.min(100, position));

      if (type === 'daterange') {
        const date = positionToDate(clampedPosition);
        if (!date) return;

        setTempRange((current) => {
          const baseRange = current || selectedValuesRef.current;
          let newRange = { ...baseRange };

          if (dragging === 'start') {
            if (date <= new Date(baseRange.endDate || maxDate)) {
              newRange.startDate = formatDateForInput(date);
            }
          } else if (dragging === 'end') {
            if (date >= new Date(baseRange.startDate || minDate)) {
              newRange.endDate = formatDateForInput(date);
            }
          }

          tempRangeRef.current = newRange;
          return newRange;
        });
      } else if (type === 'numberrange') {
        const num = positionToNumber(clampedPosition);
        if (num === null) return;

        setTempRange((current) => {
          const baseRange = current || selectedValuesRef.current;
          let newRange = { ...baseRange };

          if (dragging === 'start') {
            if (num <= (baseRange.max !== null ? baseRange.max : maxNumber)) {
              newRange.min = num;
            }
          } else if (dragging === 'end') {
            if (num >= (baseRange.min !== null ? baseRange.min : minNumber)) {
              newRange.max = num;
            }
          }

          tempRangeRef.current = newRange;
          return newRange;
        });
      }
    };

    const handleEnd = () => {
      // IMMEDIATELY remove event listeners to prevent stale events
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleEnd);
      document.removeEventListener('touchmove', handleMouseMove);
      document.removeEventListener('touchend', handleEnd);

      // Mark that we just finished dragging to prevent click event
      justFinishedDraggingRef.current = true;
      setTimeout(() => {
        justFinishedDraggingRef.current = false;
      }, 10);

      // Commit the temp range to parent on mouse up
      if (tempRangeRef.current) {
        onChangeRef.current(tempRangeRef.current);
      }
      setDragging(null);
      setTempRange(null);
      tempRangeRef.current = null;
      document.body.style.userSelect = '';
    };

    if (dragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleEnd);
      document.addEventListener('touchmove', handleMouseMove);
      document.addEventListener('touchend', handleEnd);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleEnd);
      document.removeEventListener('touchmove', handleMouseMove);
      document.removeEventListener('touchend', handleEnd);
    };
  }, [dragging, minDate, maxDate, minNumber, maxNumber, type]);

  // Toggle dropdown
  const toggleDropdown = () => {
    setIsOpen(!isOpen);
    setSearchTerm('');
  };

  // Handle single select option selection
  const handleSingleSelect = (option) => {
    onChange(option);
    setIsOpen(false);
  };

  // Handle multi-select option selection
  const handleMultiSelect = (option) => {
    let newSelected;
    if (selectedValues.includes(option)) {
      newSelected = selectedValues.filter(item => item !== option);
    } else {
      newSelected = [...selectedValues, option];
    }
    onChange(newSelected);
  };

  // Handle hierarchical selection (single or multi mode)
  const handleHierarchicalSelect = (option) => {
    if (selectionMode === 'single') {
      onChange(option);
      setIsOpen(false);
    } else {
      // Multi-select mode
      let newSelected;
      if (selectedValues.includes(option)) {
        newSelected = selectedValues.filter(item => item !== option);
      } else {
        newSelected = [...selectedValues, option];
      }
      onChange(newSelected);
    }
  };

  // Toggle category expansion in hierarchical view
  const toggleCategoryExpansion = (category, e) => {
    e.stopPropagation();
    setExpandedCategories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(category)) {
        newSet.delete(category);
      } else {
        newSet.add(category);
      }
      return newSet;
    });
  };

  // Handle "All" option for multiselect only
  const handleSelectAll = () => {
    if (type === 'multiselect') {
      if (selectedValues.length === filteredOptions.length) {
        onChange([]);
      } else {
        onChange([...filteredOptions]);
      }
    }
  };

  // Handle "Select All" for hierarchical filters (multi-select mode only)
  const handleHierarchicalSelectAll = () => {
    if (type === 'hierarchical' && selectionMode === 'multi' && hierarchyWithCounts) {
      const allValues = [];
      hierarchyWithCounts.forEach((parentData, parent) => {
        allValues.push(parent);
        parentData.children.forEach((count, child) => {
          allValues.push(child);
        });
      });

      if (selectedValues.length === allValues.length) {
        onChange([]);
      } else {
        onChange(allValues);
      }
    }
  };

  // Handle date range input changes
  const handleDateInputChange = (e, dateType) => {
    const newDate = e.target.value;
    const newRange = {
      ...selectedValues,
      [dateType]: newDate
    };
    onChange(newRange);
  };

  // Handle number range input changes
  const handleNumberInputChange = (e, rangeType) => {
    const newNum = e.target.value === '' ? null : parseInt(e.target.value, 10);
    const newRange = {
      ...selectedValues,
      [rangeType]: newNum
    };
    onChange(newRange);
  };

  // Handle date range slider interactions
  const handleSliderMouseDown = (e, handle) => {
    e.preventDefault();
    console.log('🎯 MouseDown - handle:', handle, 'selectedValues:', JSON.stringify(selectedValues));
    console.log('🎯 MouseDown - tempRange:', JSON.stringify(tempRange));
    console.log('🎯 MouseDown - selectedValuesRef.current:', JSON.stringify(selectedValuesRef.current));
    setDragging(handle);
    document.body.style.userSelect = 'none';
  };

  const handleTrackClick = (e) => {
    // Ignore clicks that happen right after dragging
    if (justFinishedDraggingRef.current) {
      console.log('🚫 Ignoring track click - just finished dragging');
      return;
    }

    if (!sliderRef.current) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickPosition = (clickX / rect.width) * 100;

    if (type === 'daterange') {
      if (!minDate || !maxDate) return;

      const clickDate = positionToDate(clickPosition);
      if (!clickDate) return;

      const startPos = dateToPosition(new Date(selectedValues.startDate || minDate));
      const endPos = dateToPosition(new Date(selectedValues.endDate || maxDate));
      const distToStart = Math.abs(clickPosition - startPos);
      const distToEnd = Math.abs(clickPosition - endPos);

      let newRange = { ...selectedValues };

      if (distToStart <= distToEnd) {
        newRange.startDate = formatDateForInput(clickDate);
      } else {
        newRange.endDate = formatDateForInput(clickDate);
      }

      // Ensure start date is not after end date
      if (new Date(newRange.startDate) > new Date(newRange.endDate)) {
        if (distToStart <= distToEnd) {
          newRange.startDate = newRange.endDate;
        } else {
          newRange.endDate = newRange.startDate;
        }
      }

      onChange(newRange);
    } else if (type === 'numberrange') {
      if (minNumber === null || maxNumber === null) return;

      const clickNum = positionToNumber(clickPosition);
      if (clickNum === null) return;

      const startPos = numberToPosition(selectedValues.min !== null ? selectedValues.min : minNumber);
      const endPos = numberToPosition(selectedValues.max !== null ? selectedValues.max : maxNumber);
      const distToStart = Math.abs(clickPosition - startPos);
      const distToEnd = Math.abs(clickPosition - endPos);

      let newRange = { ...selectedValues };

      if (distToStart <= distToEnd) {
        newRange.min = clickNum;
      } else {
        newRange.max = clickNum;
      }

      // Ensure min is not greater than max
      if (newRange.min !== null && newRange.max !== null && newRange.min > newRange.max) {
        if (distToStart <= distToEnd) {
          newRange.min = newRange.max;
        } else {
          newRange.max = newRange.min;
        }
      }

      onChange(newRange);
    }
  };

  // Clear all selections
  const clearAll = (e) => {
    e.stopPropagation();
    if (type === 'multiselect') {
      onChange([]);
    } else if (type === 'daterange') {
      onChange({ startDate: null, endDate: null });
    } else if (type === 'numberrange') {
      onChange({ min: null, max: null });
    }
  };

  // Get display text for the selected value(s)
  const getDisplayText = () => {
    if (type === 'daterange') {
      if (!selectedValues.startDate && !selectedValues.endDate) {
        return placeholder;
      }
      const start = selectedValues.startDate ? formatDate(new Date(selectedValues.startDate)) : 'Start';
      const end = selectedValues.endDate ? formatDate(new Date(selectedValues.endDate)) : 'End';
      return `${start} - ${end}`;
    } else if (type === 'numberrange') {
      if (selectedValues.min === null && selectedValues.max === null) {
        return placeholder;
      }
      const start = selectedValues.min !== null ? formatNumber(selectedValues.min) : formatNumber(minNumber);
      const end = selectedValues.max !== null ? formatNumber(selectedValues.max) : formatNumber(maxNumber);
      return `${start} - ${end}`;
    } else if (type === 'hierarchical') {
      if (selectionMode === 'single') {
        return value || placeholder;
      } else {
        if (selectedValues.length === 0) {
          return allLabel;
        } else if (selectedValues.length === 1) {
          return selectedValues[0];
        } else {
          return `${selectedValues.length} items selected`;
        }
      }
    } else if (type === 'singleselect') {
      if (!value) {
        return defaultValue || placeholder;
      }
      return value;
    } else {
      if (selectedValues.length === 0) {
        return allLabel;
      } else if (selectedValues.length === 1) {
        return selectedValues[0];
      } else {
        return `${selectedValues.length} items selected`;
      }
    }
  };

  // Check if all filtered options are selected (for multiselect)
  const areAllFilteredSelected = () => {
    return filteredOptions.length > 0 &&
           filteredOptions.every(option => selectedValues.includes(option));
  };

  // Check if all hierarchical items are selected
  const areAllHierarchicalSelected = () => {
    if (type !== 'hierarchical' || !hierarchyWithCounts) return false;

    const allValues = [];
    hierarchyWithCounts.forEach((parentData, parent) => {
      allValues.push(parent);
      parentData.children.forEach((count, child) => {
        allValues.push(child);
      });
    });

    return allValues.length > 0 && allValues.every(val => selectedValues.includes(val));
  };

  // Check if we should show clear button
  const shouldShowClear = () => {
    if (type === 'multiselect') {
      return selectedValues.length > 0;
    } else if (type === 'hierarchical' && selectionMode === 'multi') {
      return selectedValues.length > 0;
    } else if (type === 'daterange') {
      return selectedValues.startDate || selectedValues.endDate;
    } else if (type === 'numberrange') {
      return selectedValues.min !== null || selectedValues.max !== null;
    }
    return false;
  };

  // ===================
  // BUBBLE MODE RENDERING
  // ===================

  /**
   * Render range filters (daterange, numberrange) inline in bubble mode
   * Always visible, no dropdown wrapper
   */
  const renderRangeInline = () => {
    if (type === 'daterange') {
      return minDate && maxDate ? (
        <div className="filter-range-inline">
          <div className="range-inputs">
            <div className="range-input-group">
              <label htmlFor="start-date">Start:</label>
              <input
                type="date"
                id="start-date"
                value={selectedValues.startDate || formatDateForInput(minDate)}
                onChange={(e) => handleDateInputChange(e, 'startDate')}
                min={formatDateForInput(minDate)}
                max={formatDateForInput(maxDate)}
              />
            </div>
            <div className="range-input-group">
              <label htmlFor="end-date">End:</label>
              <input
                type="date"
                id="end-date"
                value={selectedValues.endDate || formatDateForInput(maxDate)}
                onChange={(e) => handleDateInputChange(e, 'endDate')}
                min={formatDateForInput(minDate)}
                max={formatDateForInput(maxDate)}
              />
            </div>
          </div>

          <div className="range-slider" ref={sliderRef} onClick={handleTrackClick}>
            <div className="slider-track">
              <div
                className="slider-fill"
                style={{
                  left: `${dateToPosition(new Date(displayValues.startDate || minDate))}%`,
                  width: `${dateToPosition(new Date(displayValues.endDate || maxDate)) - dateToPosition(new Date(displayValues.startDate || minDate))}%`
                }}
              />
            </div>

            <div
              className="slider-handle slider-handle-start"
              style={{ left: `${dateToPosition(new Date(displayValues.startDate || minDate))}%` }}
              onMouseDown={(e) => handleSliderMouseDown(e, 'start')}
              title={formatDate(new Date(displayValues.startDate || minDate))}
            />

            <div
              className="slider-handle slider-handle-end"
              style={{ left: `${dateToPosition(new Date(displayValues.endDate || maxDate))}%` }}
              onMouseDown={(e) => handleSliderMouseDown(e, 'end')}
              title={formatDate(new Date(displayValues.endDate || maxDate))}
            />
          </div>
        </div>
      ) : (
        <div className="filter-no-results">Date range data not available</div>
      );
    } else if (type === 'numberrange') {
      return minNumber !== null && maxNumber !== null ? (
        <div className="filter-range-inline">
          <div className="range-inputs">
            <div className="range-input-group">
              <label htmlFor="min-number">Min:</label>
              <input
                type="number"
                id="min-number"
                value={selectedValues.min !== null ? selectedValues.min : minNumber}
                onChange={(e) => handleNumberInputChange(e, 'min')}
                min={minNumber}
                max={maxNumber}
              />
            </div>
            <div className="range-input-group">
              <label htmlFor="max-number">Max:</label>
              <input
                type="number"
                id="max-number"
                value={selectedValues.max !== null ? selectedValues.max : maxNumber}
                onChange={(e) => handleNumberInputChange(e, 'max')}
                min={minNumber}
                max={maxNumber}
              />
            </div>
          </div>

          <div className="range-slider" ref={sliderRef} onClick={handleTrackClick}>
            <div className="slider-track">
              <div
                className="slider-fill"
                style={{
                  left: `${numberToPosition(displayValues.min !== null ? displayValues.min : minNumber)}%`,
                  width: `${numberToPosition(displayValues.max !== null ? displayValues.max : maxNumber) - numberToPosition(displayValues.min !== null ? displayValues.min : minNumber)}%`
                }}
              />
            </div>

            <div
              className="slider-handle slider-handle-start"
              style={{ left: `${numberToPosition(displayValues.min !== null ? displayValues.min : minNumber)}%` }}
              onMouseDown={(e) => handleSliderMouseDown(e, 'start')}
              title={`${formatNumber(displayValues.min !== null ? displayValues.min : minNumber)}${suffix}`}
            />

            <div
              className="slider-handle slider-handle-end"
              style={{ left: `${numberToPosition(displayValues.max !== null ? displayValues.max : maxNumber)}%` }}
              onMouseDown={(e) => handleSliderMouseDown(e, 'end')}
              title={`${formatNumber(displayValues.max !== null ? displayValues.max : maxNumber)}${suffix}`}
            />
          </div>
        </div>
      ) : (
        <div className="filter-no-results">Number range data not available</div>
      );
    }
    return null;
  };

  /**
   * Get visible bubbles based on search term and additional items shown
   * Returns array of options to render
   */
  const getVisibleBubbles = () => {
    // Filter by search term first
    const searchFiltered = searchTerm
      ? filteredOptions
      : (options || []);

    // When searching, show all matches (ignore maxVisibleBubbles)
    if (searchTerm) {
      return searchFiltered;
    }

    // Apply show more/less limit only when not searching
    const visibleCount = maxVisibleBubbles + additionalItemsShown;
    return searchFiltered.slice(0, visibleCount);
  };

  /**
   * Show more items (increment by bubbleExpandIncrement)
   */
  const handleShowMore = () => {
    setAdditionalItemsShown(prev => {
      const newTotal = prev + bubbleExpandIncrement;
      const totalOptions = (searchTerm ? filteredOptions : options).length;
      const maxAdditional = totalOptions - maxVisibleBubbles;
      return Math.min(newTotal, maxAdditional);
    });
  };

  /**
   * Show less items (decrement by bubbleExpandIncrement)
   */
  const handleShowLess = () => {
    setAdditionalItemsShown(prev => {
      const newTotal = prev - bubbleExpandIncrement;
      return Math.max(newTotal, 0);
    });
  };

  /**
   * Toggle hierarchical parent expansion in bubble mode
   */
  const toggleParentExpansion = (parent) => {
    setExpandedParents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(parent)) {
        newSet.delete(parent);
      } else {
        newSet.add(parent);
      }
      return newSet;
    });
  };

  /**
   * Render bubble interface for singleselect and multiselect
   */
  const renderBubbles = () => {
    const visibleBubbles = getVisibleBubbles();
    const allOptions = searchTerm ? filteredOptions : (options || []);
    const currentVisibleCount = maxVisibleBubbles + additionalItemsShown;
    const remainingCount = allOptions.length - currentVisibleCount;
    // Always base threshold on original options count, not filtered count
    const shouldShowSearchBar = (options || []).length > searchThreshold;

    return (
      <div className="filter-bubble-mode">
        {/* Search bar (if threshold exceeded) */}
        {shouldShowSearchBar && (
          <div className="filter-bubble-search">
            <Search size={16} />
            <input
              type="text"
              placeholder={searchPlaceholder}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="filter-bubble-search-input"
            />
          </div>
        )}

        {/* Bubble container */}
        <div className="filter-bubble-container">
          {/* Select All bubble for multiselect */}
          {type === 'multiselect' && (
            <button
              className={`filter-bubble select-all-bubble ${areAllFilteredSelected() ? 'selected' : ''}`}
              onClick={handleSelectAll}
            >
              {areAllFilteredSelected() ? 'Deselect All' : 'Select All'}
            </button>
          )}

          {/* Option bubbles */}
          {visibleBubbles.length > 0 ? (
            visibleBubbles.map((option, index) => {
              const isSelected = selectedValues.includes(option);

              return (
                <button
                  key={option}
                  className={`filter-bubble ${isSelected ? 'selected' : ''} entering`}
                  style={{ '--bubble-index': index }}
                  onClick={() =>
                    type === 'singleselect'
                      ? handleSingleSelect(option)
                      : handleMultiSelect(option)
                  }
                >
                  {option}
                  {type === 'multiselect' && isSelected && (
                    <Check size={14} />
                  )}
                </button>
              );
            })
          ) : (
            <div className="filter-no-results">
              {searchTerm ? 'No options match your search' : 'No options available'}
            </div>
          )}

          {/* Show more/less bubbles */}
          {!searchTerm && (
            <>
              {/* Show More button - appears when there are items beyond current visible count */}
              {allOptions.length > (maxVisibleBubbles + additionalItemsShown) && (
                <button
                  className="filter-bubble show-more"
                  onClick={handleShowMore}
                >
                  Show {Math.min(bubbleExpandIncrement, allOptions.length - maxVisibleBubbles - additionalItemsShown)} more
                </button>
              )}

              {/* Show Less button - appears when showing more than initial maxVisibleBubbles */}
              {additionalItemsShown > 0 && (
                <button
                  className="filter-bubble show-less"
                  onClick={handleShowLess}
                >
                  Show {Math.min(bubbleExpandIncrement, additionalItemsShown)} less
                </button>
              )}
            </>
          )}
        </div>
      </div>
    );
  };

  /**
   * Render hierarchical bubbles (two-level parent-child)
   */
  const renderHierarchicalBubbles = () => {
    const hierarchy = filteredHierarchy || hierarchyWithCounts;
    if (!hierarchy) return null;

    const allOptions = Array.from(hierarchy.keys());
    const shouldShowSearchBar = allOptions.length > searchThreshold;

    return (
      <div className="filter-bubble-mode">
        {/* Search bar (if threshold exceeded) */}
        {shouldShowSearchBar && (
          <div className="filter-bubble-search">
            <Search size={16} />
            <input
              type="text"
              placeholder={searchPlaceholder}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="filter-bubble-search-input"
            />
          </div>
        )}

        {/* Bubble container */}
        <div className="filter-bubble-container">
          {/* Select All bubble for multiselect mode */}
          {selectionMode === 'multi' && (
            <button
              className={`filter-bubble select-all-bubble ${areAllHierarchicalSelected() ? 'selected' : ''}`}
              onClick={handleHierarchicalSelectAll}
            >
              {areAllHierarchicalSelected() ? 'Deselect All' : 'Select All'}
            </button>
          )}

          {/* Hierarchical parent-child bubbles */}
          {hierarchy.size > 0 ? (
            Array.from(hierarchy.entries()).map(([parent, parentData]) => {
              const isParentSelected = selectedValues.includes(parent);
              const isParentExpanded = expandedParents.has(parent);
              const hasChildren = parentData.children && parentData.children.size > 0;

              return (
                <div key={parent} className="hierarchical-bubble-group">
                  <div className="hierarchical-parent-row">
                    {/* Parent bubble */}
                    <button
                      className={`filter-bubble hierarchical-parent ${isParentSelected ? 'selected' : ''}`}
                      onClick={() => handleHierarchicalSelect(parent)}
                    >
                      {parent}
                      {showBubbleCounts && <span className="bubble-count">({parentData.count})</span>}
                      {selectionMode === 'multi' && isParentSelected && (
                        <Check size={14} />
                      )}
                    </button>

                    {/* Chevron outside the bubble */}
                    {hasChildren && (
                      <button
                        className={`bubble-chevron-external ${isParentExpanded ? 'expanded' : ''}`}
                        onClick={() => toggleParentExpansion(parent)}
                        aria-label={isParentExpanded ? 'Collapse' : 'Expand'}
                      >
                        <ChevronDown size={16} />
                      </button>
                    )}
                  </div>

                  {/* Child bubbles (shown when parent is expanded or searching) */}
                  {hasChildren && (isParentExpanded || searchTerm) && (
                    <div className="filter-bubble-children">
                      {Array.from(parentData.children.entries()).map(([child, count]) => {
                        const isChildSelected = selectedValues.includes(child);

                        return (
                          <button
                            key={child}
                            className={`filter-bubble ${isChildSelected ? 'selected' : ''}`}
                            onClick={() => handleHierarchicalSelect(child)}
                          >
                            {child}
                            {showBubbleCounts && <span className="bubble-count">({count})</span>}
                            {selectionMode === 'multi' && isChildSelected && (
                              <Check size={14} />
                            )}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <div className="filter-no-results">
              {searchTerm ? 'No options match your search' : 'No options available'}
            </div>
          )}
        </div>
      </div>
    );
  };

  /**
   * Main bubble interface renderer
   * Decides which bubble type to render based on filter type
   */
  const renderBubbleInterface = () => {
    // Range filters (daterange, numberrange) render inline slider
    if (type === 'daterange' || type === 'numberrange') {
      return renderRangeInline();
    }

    // Hierarchical filters use two-level bubble rendering
    if (type === 'hierarchical') {
      return renderHierarchicalBubbles();
    }

    // Singleselect and multiselect use standard bubble rendering
    return renderBubbles();
  };

  // ===================
  // MAIN RENDER
  // ===================

  // Bubble mode: All filter types supported
  if (renderMode === 'bubble') {
    return (
      <div className="filter-container bubble-mode">
        {label && <label className="filter-label">{label}</label>}
        {renderBubbleInterface()}
      </div>
    );
  }

  // Standard dropdown mode rendering
  return (
    <div className="filter-container" ref={dropdownRef}>
      {label && <label className="filter-label">{label}</label>}

      <div
        className={`filter-header ${isOpen ? 'open' : ''}`}
        onClick={toggleDropdown}
      >
        {icon && <span className="filter-icon">{icon}</span>}

        <div className="filter-selection-display">
          <span className={selectedValues.length === 0 && type !== 'daterange' ? 'filter-placeholder' : 'filter-selected'}>
            {getDisplayText()}
          </span>
        </div>

        {shouldShowClear() && (
          <button
            className="filter-clear-btn"
            onClick={clearAll}
            title="Clear selection"
          >
            <X size={16} />
          </button>
        )}

        <span className="filter-dropdown-arrow">
          <ChevronDown size={18} />
        </span>
      </div>

      {isOpen && (
        <div className="filter-dropdown">
          {/* Date Range Content */}
          {type === 'daterange' && (
            <>
              {minDate && maxDate ? (
                <div className="filter-range-content">
                  <div className="range-inputs">
                    <div className="range-input-group">
                      <label htmlFor="start-date">Start:</label>
                      <input
                        type="date"
                        id="start-date"
                        value={selectedValues.startDate || formatDateForInput(minDate)}
                        onChange={(e) => handleDateInputChange(e, 'startDate')}
                        min={formatDateForInput(minDate)}
                        max={formatDateForInput(maxDate)}
                      />
                    </div>
                    <div className="range-input-group">
                      <label htmlFor="end-date">End:</label>
                      <input
                        type="date"
                        id="end-date"
                        value={selectedValues.endDate || formatDateForInput(maxDate)}
                        onChange={(e) => handleDateInputChange(e, 'endDate')}
                        min={formatDateForInput(minDate)}
                        max={formatDateForInput(maxDate)}
                      />
                    </div>
                  </div>

                  <div className="range-slider" ref={sliderRef} onClick={handleTrackClick}>
                    <div className="slider-track">
                      {/* Slider fill */}
                      <div
                        className="slider-fill"
                        style={{
                          left: `${dateToPosition(new Date(displayValues.startDate || minDate))}%`,
                          width: `${dateToPosition(new Date(displayValues.endDate || maxDate)) - dateToPosition(new Date(displayValues.startDate || minDate))}%`
                        }}
                      />
                    </div>

                    {/* Start handle */}
                    <div
                      className="slider-handle slider-handle-start"
                      style={{ left: `${dateToPosition(new Date(displayValues.startDate || minDate))}%` }}
                      onMouseDown={(e) => handleSliderMouseDown(e, 'start')}
                      title={formatDate(new Date(displayValues.startDate || minDate))}
                    />

                    {/* End handle */}
                    <div
                      className="slider-handle slider-handle-end"
                      style={{ left: `${dateToPosition(new Date(displayValues.endDate || maxDate))}%` }}
                      onMouseDown={(e) => handleSliderMouseDown(e, 'end')}
                      title={formatDate(new Date(displayValues.endDate || maxDate))}
                    />
                  </div>
                </div>
              ) : (
                <div className="filter-no-results">No dates available with current filters</div>
              )}
            </>
          )}

          {/* Number Range Content */}
          {type === 'numberrange' && (
            <>
              {minNumber !== null && maxNumber !== null ? (
                <div className="filter-range-content">
                  <div className="range-inputs">
                    <div className="range-input-group">
                      <label htmlFor="min-number">Min:</label>
                      <input
                        type="number"
                        id="min-number"
                        value={selectedValues.min !== null ? selectedValues.min : minNumber}
                        onChange={(e) => handleNumberInputChange(e, 'min')}
                        min={minNumber}
                        max={maxNumber}
                      />
                    </div>
                    <div className="range-input-group">
                      <label htmlFor="max-number">Max:</label>
                      <input
                        type="number"
                        id="max-number"
                        value={selectedValues.max !== null ? selectedValues.max : maxNumber}
                        onChange={(e) => handleNumberInputChange(e, 'max')}
                        min={minNumber}
                        max={maxNumber}
                      />
                    </div>
                  </div>

                  <div className="range-slider" ref={sliderRef} onClick={handleTrackClick}>
                    <div className="slider-track">
                      {/* Slider fill */}
                      <div
                        className="slider-fill"
                        style={{
                          left: `${numberToPosition(displayValues.min !== null ? displayValues.min : minNumber)}%`,
                          width: `${numberToPosition(displayValues.max !== null ? displayValues.max : maxNumber) - numberToPosition(displayValues.min !== null ? displayValues.min : minNumber)}%`
                        }}
                      />
                    </div>

                    {/* Start handle */}
                    <div
                      className="slider-handle slider-handle-start"
                      style={{ left: `${numberToPosition(displayValues.min !== null ? displayValues.min : minNumber)}%` }}
                      onMouseDown={(e) => handleSliderMouseDown(e, 'start')}
                      title={formatNumber(displayValues.min !== null ? displayValues.min : minNumber)}
                    />

                    {/* End handle */}
                    <div
                      className="slider-handle slider-handle-end"
                      style={{ left: `${numberToPosition(displayValues.max !== null ? displayValues.max : maxNumber)}%` }}
                      onMouseDown={(e) => handleSliderMouseDown(e, 'end')}
                      title={formatNumber(displayValues.max !== null ? displayValues.max : maxNumber)}
                    />
                  </div>
                </div>
              ) : (
                <div className="filter-no-results">No number range available with current filters</div>
              )}
            </>
          )}

          {/* Search for non-daterange and non-numberrange types */}
          {type !== 'daterange' && type !== 'numberrange' && searchable && (
            <div className="filter-search-container">
              <Search size={16} className="filter-search-icon" />
              <input
                ref={searchInputRef}
                type="text"
                className="filter-search-input"
                placeholder={searchPlaceholder}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onClick={(e) => e.stopPropagation()}
              />
            </div>
          )}

          {/* Select All for multiselect only */}
          {type === 'multiselect' && (
            <div className="filter-select-all-container">
              <div
                className="filter-option checkbox-option"
                onClick={handleSelectAll}
              >
                <span className="filter-control checkbox-control">
                  <span className={`checkbox ${areAllFilteredSelected() ? 'selected' : ''}`}>
                    {areAllFilteredSelected() && <Check size={14} />}
                  </span>
                </span>
                <span className="filter-option-text">
                  {areAllFilteredSelected() ? "Deselect All" : "Select All"}
                </span>
              </div>
            </div>
          )}

          {/* Select All for hierarchical multiselect */}
          {type === 'hierarchical' && selectionMode === 'multi' && (
            <div className="filter-select-all-container">
              <div
                className="filter-option checkbox-option"
                onClick={handleHierarchicalSelectAll}
              >
                <span className="filter-control checkbox-control">
                  <span className={`checkbox ${areAllHierarchicalSelected() ? 'selected' : ''}`}>
                    {areAllHierarchicalSelected() && <Check size={14} />}
                  </span>
                </span>
                <span className="filter-option-text">
                  {areAllHierarchicalSelected() ? "Deselect All" : "Select All"}
                </span>
              </div>
            </div>
          )}

          {/* Hierarchical options list */}
          {type === 'hierarchical' && filteredHierarchy && (
            <div className="filter-options-list hierarchical-options-list">
              {filteredHierarchy.size > 0 ? (
                Array.from(filteredHierarchy.entries()).map(([parent, parentData]) => {
                  const hasChildren = parentData.children.size > 0;
                  const isExpanded = expandedCategories.has(parent);
                  const controlType = selectionMode === 'single' ? 'radio' : 'checkbox';

                  return (
                    <div key={parent} className="hierarchical-parent-group">
                      {/* Parent option */}
                      <div
                        className={`filter-option hierarchical-parent ${controlType}-option ${
                          selectedValues.includes(parent) ? 'selected' : ''
                        }`}
                      >
                        {/* Expand/collapse icon (only if has children) */}
                        {hasChildren && (
                          <span
                            className={`hierarchical-expand-icon ${isExpanded ? 'expanded' : ''}`}
                            onClick={(e) => toggleCategoryExpansion(parent, e)}
                          >
                            <ChevronRight size={16} />
                          </span>
                        )}

                        {/* Parent checkbox/radio and label */}
                        <div
                          className="hierarchical-parent-content"
                          onClick={() => handleHierarchicalSelect(parent)}
                        >
                          <span className={`filter-control ${controlType}-control`}>
                            {selectionMode === 'single' ? (
                              <span className={`radio-button ${selectedValues.includes(parent) ? 'selected' : ''}`}>
                                {selectedValues.includes(parent) && <span className="radio-dot" />}
                              </span>
                            ) : (
                              <span className={`checkbox ${selectedValues.includes(parent) ? 'selected' : ''}`}>
                                {selectedValues.includes(parent) && <Check size={14} />}
                              </span>
                            )}
                          </span>
                          <span className="filter-option-text">
                            {parent}
                            {showCounts && <span className="hierarchical-count"> ({parentData.count})</span>}
                          </span>
                        </div>
                      </div>

                      {/* Children options (shown when expanded or when searching) */}
                      {hasChildren && (isExpanded || searchTerm) && (
                        <div className="hierarchical-children">
                          {Array.from(parentData.children.entries()).map(([child, count]) => (
                            <div
                              key={child}
                              className={`filter-option hierarchical-child ${controlType}-option ${
                                selectedValues.includes(child) ? 'selected' : ''
                              }`}
                              onClick={() => handleHierarchicalSelect(child)}
                            >
                              <span className={`filter-control ${controlType}-control`}>
                                {selectionMode === 'single' ? (
                                  <span className={`radio-button ${selectedValues.includes(child) ? 'selected' : ''}`}>
                                    {selectedValues.includes(child) && <span className="radio-dot" />}
                                  </span>
                                ) : (
                                  <span className={`checkbox ${selectedValues.includes(child) ? 'selected' : ''}`}>
                                    {selectedValues.includes(child) && <Check size={14} />}
                                  </span>
                                )}
                              </span>
                              <span className="filter-option-text">
                                {child}
                                {showCounts && <span className="hierarchical-count"> ({count})</span>}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })
              ) : (
                <div className="filter-no-results">
                  {searchTerm ? 'No options match your search' : 'No options available with current filters'}
                </div>
              )}
            </div>
          )}

          {/* Options list for non-daterange, non-numberrange, and non-hierarchical types */}
          {type !== 'daterange' && type !== 'numberrange' && type !== 'hierarchical' && (
            <div className="filter-options-list">
              {filteredOptions.length > 0 ? (
                filteredOptions.map((option, index) => (
                  <div
                    key={index}
                    className={`filter-option ${type === 'singleselect' ? 'radio-option' : 'checkbox-option'} ${
                      selectedValues.includes(option) ? 'selected' : ''
                    }`}
                    onClick={() =>
                      type === 'singleselect'
                        ? handleSingleSelect(option)
                        : handleMultiSelect(option)
                    }
                  >
                    <span className={`filter-control ${type === 'singleselect' ? 'radio-control' : 'checkbox-control'}`}>
                      {type === 'singleselect' ? (
                        <span className={`radio-button ${selectedValues.includes(option) ? 'selected' : ''}`}>
                          {selectedValues.includes(option) && <span className="radio-dot" />}
                        </span>
                      ) : (
                        <span className={`checkbox ${selectedValues.includes(option) ? 'selected' : ''}`}>
                          {selectedValues.includes(option) && <Check size={14} />}
                        </span>
                      )}
                    </span>
                    <span className="filter-option-text">{option}</span>
                  </div>
                ))
              ) : (
                <div className="filter-no-results">
                  {searchTerm ? 'No options match your search' : 'No options available with current filters'}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Filter;
