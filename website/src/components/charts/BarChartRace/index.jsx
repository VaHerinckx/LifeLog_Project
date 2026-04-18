// src/components/charts/BarChartRace/index.jsx
import { useEffect, useState, useRef, useMemo } from 'react';
import PropTypes from 'prop-types';
import { race } from 'racing-bars';
import { Play, Pause, SkipBack, X } from 'lucide-react';
import _ from 'lodash';
import { groupByMonth, parseDate, isValidDate } from '../../../utils/dateUtils';
import './BarChartRace.css';

const BarChartRace = ({
  data,
  dateColumnName,
  dimensionOptions = [],
  metricOptions = [],
  defaultDimension,
  defaultMetric,
  title = 'Bar Chart Race',
  topN = 10,
  timePeriod = 'monthly',
  cumulative = true,
  autoPlay = false,
  frameDuration = 500
}) => {
  // State for user-selected controls
  const [selectedDimension, setSelectedDimension] = useState(defaultDimension || dimensionOptions[0]?.value);
  const [selectedMetric, setSelectedMetric] = useState(defaultMetric || metricOptions[0]?.value);
  const [selectedTimePeriod, setSelectedTimePeriod] = useState(timePeriod);
  const [topNValue, setTopNValue] = useState(topN);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [isPlaying, setIsPlaying] = useState(autoPlay);

  // Refs
  const chartContainerRef = useRef(null);
  const focusChartContainerRef = useRef(null);
  const raceInstanceRef = useRef(null);
  const focusRaceInstanceRef = useRef(null);

  // Escape key handler for focus mode
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isFocusMode) setIsFocusMode(false);
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isFocusMode]);

  // Find current dimension and metric configs
  const currentDimensionConfig = dimensionOptions.find(d => d.value === selectedDimension);
  const currentMetricConfig = metricOptions.find(m => m.value === selectedMetric);

  /**
   * Transform data to RacingBars format
   * Returns array of {date, name, value} objects
   */
  const transformDataForRacing = useMemo(() => {
    if (!Array.isArray(data) || !currentDimensionConfig || !currentMetricConfig) {
      return [];
    }

    const dimensionField = currentDimensionConfig.field;
    const metricField = currentMetricConfig.field;
    const metricAggregation = currentMetricConfig.aggregation;

    // Filter out invalid data
    const validData = data.filter(item => {
      const dateValue = item[dateColumnName];
      const dimensionValue = item[dimensionField];
      return (
        isValidDate(parseDate(dateValue)) &&
        dimensionValue &&
        dimensionValue !== 'Unknown' &&
        dimensionValue.toString().trim() !== ''
      );
    });

    if (validData.length === 0) return [];

    // Group by time period
    let periodGroups = {};

    if (selectedTimePeriod === 'monthly') {
      periodGroups = groupByMonth(validData, dateColumnName);
    } else if (selectedTimePeriod === 'yearly') {
      periodGroups = validData.reduce((acc, item) => {
        const date = parseDate(item[dateColumnName]);
        if (isValidDate(date)) {
          const year = date.getFullYear().toString();
          if (!acc[year]) acc[year] = [];
          acc[year].push(item);
        }
        return acc;
      }, {});
    } else if (selectedTimePeriod === 'quarterly') {
      periodGroups = validData.reduce((acc, item) => {
        const date = parseDate(item[dateColumnName]);
        if (isValidDate(date)) {
          const year = date.getFullYear();
          const quarter = Math.floor(date.getMonth() / 3) + 1;
          const key = `${year}-Q${quarter}`;
          if (!acc[key]) acc[key] = [];
          acc[key].push(item);
        }
        return acc;
      }, {});
    }

    // For each period, aggregate by dimension
    const racingData = [];
    const sortedPeriods = Object.keys(periodGroups).sort();

    sortedPeriods.forEach(periodKey => {
      const periodRecords = periodGroups[periodKey];

      // Group by dimension field
      const dimensionGroups = _.groupBy(periodRecords, dimensionField);

      // Calculate metric for each dimension group
      Object.entries(dimensionGroups).forEach(([dimValue, records]) => {
        let value = 0;

        switch (metricAggregation) {
          case 'count':
            value = records.length;
            break;
          case 'count_distinct':
            if (metricField) {
              const distinctValues = new Set(
                records
                  .map(r => r[metricField])
                  .filter(v => v !== null && v !== undefined && v !== '')
              );
              value = distinctValues.size;
            } else {
              value = records.length;
            }
            break;
          case 'sum':
          case 'cumsum':
            if (metricField) {
              value = _.sumBy(records, r => parseFloat(r[metricField]) || 0);
            }
            break;
          case 'average':
            if (metricField) {
              const sum = _.sumBy(records, r => parseFloat(r[metricField]) || 0);
              value = records.length > 0 ? sum / records.length : 0;
            }
            break;
          default:
            value = records.length;
        }

        racingData.push({
          date: periodKey,
          name: dimValue,
          value: value
        });
      });
    });

    // If cumulative, compute running totals
    if (cumulative) {
      const runningTotals = {};
      const sortedData = _.sortBy(racingData, 'date');

      return sortedData.map(item => {
        const key = item.name;
        runningTotals[key] = (runningTotals[key] || 0) + item.value;
        return {
          ...item,
          value: runningTotals[key]
        };
      });
    }

    return racingData;
  }, [data, dateColumnName, currentDimensionConfig, currentMetricConfig, selectedTimePeriod, cumulative]);

  /**
   * Initialize/update the racing chart
   */
  useEffect(() => {
    if (transformDataForRacing.length === 0 || !chartContainerRef.current) return;

    // Clean up existing instance
    if (raceInstanceRef.current) {
      try {
        raceInstanceRef.current.destroy();
      } catch (e) {
        // Ignore cleanup errors
      }
    }

    // Clear container
    chartContainerRef.current.innerHTML = '';

    // Create new instance
    try {
      const instance = race(transformDataForRacing, chartContainerRef.current, {
        title: '',
        subTitle: '',
        caption: '',
        marginTop: 20,
        marginRight: 60,
        marginBottom: 20,
        marginLeft: 160,
        topN: topNValue,
        barHeight: 40,
        tickDuration: frameDuration,
        loop: false,
        autorun: autoPlay,
        dataShape: 'long',
        dateCounter: selectedTimePeriod === 'yearly' ? 'YYYY' : selectedTimePeriod === 'quarterly' ? '[Q]Q YYYY' : 'MMM YYYY',
        theme: {
          background: 'transparent',
          titleColor: 'var(--color-text-primary)',
          subTitleColor: 'var(--color-text-secondary)',
          dateCounterColor: 'var(--color-text-secondary)',
          textColor: 'var(--color-text-primary)',
          axisColor: 'var(--border-color-medium)',
          labelsColor: 'var(--color-text-primary)',
          colors: [
            'var(--base-color-primary)',
            'var(--base-color-timeline-1)',
            'var(--base-color-timeline-2)',
            'var(--base-color-timeline-3)',
            'var(--base-color-timeline-4)',
            'var(--base-color-timeline-5)',
            'var(--base-color-accent)',
            'var(--base-color-success)'
          ]
        }
      });

      raceInstanceRef.current = instance;
      setIsPlaying(autoPlay);
    } catch (error) {
      console.error('Error creating racing chart:', error);
    }

    return () => {
      if (raceInstanceRef.current) {
        try {
          raceInstanceRef.current.destroy();
        } catch (e) {
          // Ignore cleanup errors
        }
      }
    };
  }, [transformDataForRacing, topNValue, frameDuration, selectedTimePeriod, autoPlay]);

  /**
   * Handle focus mode chart initialization
   */
  useEffect(() => {
    if (!isFocusMode || transformDataForRacing.length === 0 || !focusChartContainerRef.current) return;

    // Clean up existing focus instance
    if (focusRaceInstanceRef.current) {
      try {
        focusRaceInstanceRef.current.destroy();
      } catch (e) {
        // Ignore cleanup errors
      }
    }

    // Clear container
    focusChartContainerRef.current.innerHTML = '';

    // Create new instance for focus mode
    try {
      const instance = race(transformDataForRacing, focusChartContainerRef.current, {
        title: '',
        subTitle: '',
        caption: '',
        marginTop: 40,
        marginRight: 80,
        marginBottom: 40,
        marginLeft: 200,
        topN: topNValue,
        barHeight: 50,
        tickDuration: frameDuration,
        loop: false,
        autorun: false,
        dataShape: 'long',
        dateCounter: selectedTimePeriod === 'yearly' ? 'YYYY' : selectedTimePeriod === 'quarterly' ? '[Q]Q YYYY' : 'MMM YYYY',
        theme: {
          background: 'transparent',
          titleColor: 'var(--color-text-primary)',
          subTitleColor: 'var(--color-text-secondary)',
          dateCounterColor: 'var(--color-text-secondary)',
          textColor: 'var(--color-text-primary)',
          axisColor: 'var(--border-color-medium)',
          labelsColor: 'var(--color-text-primary)',
          colors: [
            'var(--base-color-primary)',
            'var(--base-color-timeline-1)',
            'var(--base-color-timeline-2)',
            'var(--base-color-timeline-3)',
            'var(--base-color-timeline-4)',
            'var(--base-color-timeline-5)',
            'var(--base-color-accent)',
            'var(--base-color-success)'
          ]
        }
      });

      focusRaceInstanceRef.current = instance;
    } catch (error) {
      console.error('Error creating focus racing chart:', error);
    }

    return () => {
      if (focusRaceInstanceRef.current) {
        try {
          focusRaceInstanceRef.current.destroy();
        } catch (e) {
          // Ignore cleanup errors
        }
      }
    };
  }, [isFocusMode, transformDataForRacing, topNValue, frameDuration, selectedTimePeriod]);

  // Playback control handlers
  const handlePlayPause = () => {
    const instance = isFocusMode ? focusRaceInstanceRef.current : raceInstanceRef.current;
    if (!instance) return;

    try {
      if (isPlaying) {
        instance.pause();
      } else {
        instance.play();
      }
      setIsPlaying(!isPlaying);
    } catch (error) {
      console.error('Error toggling playback:', error);
    }
  };

  const handleRestart = () => {
    const instance = isFocusMode ? focusRaceInstanceRef.current : raceInstanceRef.current;
    if (!instance) return;

    try {
      instance.replay();
      setIsPlaying(true);
    } catch (error) {
      console.error('Error restarting:', error);
    }
  };

  // Render controls (shared between normal and focus mode)
  const renderControls = (inFocusMode = false) => (
    <div className="chart-controls" onClick={(e) => e.stopPropagation()}>
      {/* Dimension Selector */}
      {dimensionOptions.length > 1 && (
        <div className="chart-filter">
          <label htmlFor={inFocusMode ? "focus-dimension-select" : "dimension-select"}>Group by:</label>
          <select
            id={inFocusMode ? "focus-dimension-select" : "dimension-select"}
            value={selectedDimension}
            onChange={(e) => setSelectedDimension(e.target.value)}
            className="filter-select"
          >
            {dimensionOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      )}

      {/* Metric Selector */}
      {metricOptions.length > 1 && (
        <div className="chart-filter">
          <label htmlFor={inFocusMode ? "focus-metric-select" : "metric-select"}>Measure:</label>
          <select
            id={inFocusMode ? "focus-metric-select" : "metric-select"}
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="filter-select"
          >
            {metricOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      )}

      {/* Time Period Selector */}
      <div className="chart-filter">
        <label htmlFor={inFocusMode ? "focus-period-select" : "period-select"}>Period:</label>
        <select
          id={inFocusMode ? "focus-period-select" : "period-select"}
          value={selectedTimePeriod}
          onChange={(e) => setSelectedTimePeriod(e.target.value)}
          className="filter-select"
        >
          <option value="monthly">Monthly</option>
          <option value="quarterly">Quarterly</option>
          <option value="yearly">Yearly</option>
        </select>
      </div>

      {/* Top N Selector */}
      <div className="chart-filter">
        <label htmlFor={inFocusMode ? "focus-topn-select" : "topn-select"}>Show Top:</label>
        <select
          id={inFocusMode ? "focus-topn-select" : "topn-select"}
          value={topNValue}
          onChange={(e) => setTopNValue(Number(e.target.value))}
          className="filter-select"
        >
          <option value="5">5</option>
          <option value="10">10</option>
          <option value="15">15</option>
          <option value="20">20</option>
        </select>
      </div>

      {/* Playback Controls */}
      <div className="racing-controls">
        <button
          onClick={handleRestart}
          className="racing-btn"
          title="Restart"
          aria-label="Restart animation"
        >
          <SkipBack size={18} />
        </button>
        <button
          onClick={handlePlayPause}
          className="racing-btn racing-btn-primary"
          title={isPlaying ? 'Pause' : 'Play'}
          aria-label={isPlaying ? 'Pause animation' : 'Play animation'}
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
        </button>
      </div>
    </div>
  );

  if (transformDataForRacing.length === 0) {
    return (
      <div className="bar-chart-race-container">
        <div className="chart-header">
          <h3 className="chart-title">{title}</h3>
        </div>
        <div className="racing-chart-wrapper">
          <div className="racing-empty-state">
            <p>No data available for the selected filters</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Normal view */}
      <div className="bar-chart-race-container" onClick={() => setIsFocusMode(true)}>
        <div className="chart-header">
          <h3 className="chart-title">{title}</h3>
          {renderControls(false)}
        </div>
        <div className="racing-chart-wrapper" ref={chartContainerRef} />
      </div>

      {/* Focus mode overlay */}
      {isFocusMode && (
        <div className="racing-focus-overlay" onClick={() => setIsFocusMode(false)}>
          <div className="racing-focus-content" onClick={(e) => e.stopPropagation()}>
            <button
              className="focus-close-button"
              onClick={() => setIsFocusMode(false)}
              aria-label="Close focus mode"
            >
              <X size={24} />
            </button>
            <div className="racing-focus-controls-bar">
              {renderControls(true)}
            </div>
            <div className="racing-focus-chart-container" ref={focusChartContainerRef} />
          </div>
        </div>
      )}
    </>
  );
};

BarChartRace.propTypes = {
  data: PropTypes.array.isRequired,
  dateColumnName: PropTypes.string.isRequired,
  dimensionOptions: PropTypes.arrayOf(
    PropTypes.shape({
      value: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      field: PropTypes.string.isRequired
    })
  ).isRequired,
  metricOptions: PropTypes.arrayOf(
    PropTypes.shape({
      value: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      aggregation: PropTypes.oneOf(['count', 'count_distinct', 'sum', 'average', 'cumsum']).isRequired,
      field: PropTypes.string,
      suffix: PropTypes.string,
      prefix: PropTypes.string,
      decimals: PropTypes.number
    })
  ).isRequired,
  defaultDimension: PropTypes.string,
  defaultMetric: PropTypes.string,
  title: PropTypes.string,
  topN: PropTypes.number,
  timePeriod: PropTypes.oneOf(['monthly', 'quarterly', 'yearly']),
  cumulative: PropTypes.bool,
  autoPlay: PropTypes.bool,
  frameDuration: PropTypes.number
};

export default BarChartRace;
