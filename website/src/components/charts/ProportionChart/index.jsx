// src/components/charts/ProportionChart/index.jsx
import { useEffect, useState, useMemo } from 'react';
import PropTypes from 'prop-types';
import { PieChart, Pie, Cell, Treemap, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { X, PieChart as PieChartIcon, LayoutGrid, ArrowLeft } from 'lucide-react';
import _ from 'lodash';
import { applyMetricFilter, resolveMetricDataSource, performComputation, formatComputedValue } from '../../../utils/computationUtils';
import './ProportionChart.css';

// Color palette for categories (from design tokens)
const CATEGORY_COLORS = [
  '#3423A6',  // Primary purple
  '#3B82F6',  // Blue (timeline-1)
  '#14B8A6',  // Teal (timeline-2)
  '#F97316',  // Orange (timeline-3)
  '#A78BFA',  // Lavender (timeline-4)
  '#10B981',  // Green (timeline-5)
  '#FB4B4E',  // Accent red
  '#9CA3AF',  // Gray (for "Other")
];

// Extended colors for drill-down view
const EXTENDED_COLORS = [
  ...CATEGORY_COLORS,
  '#6366F1',  // Indigo
  '#EC4899',  // Pink
  '#8B5CF6',  // Violet
  '#06B6D4',  // Cyan
  '#84CC16',  // Lime
  '#F59E0B',  // Amber
  '#EF4444',  // Red
  '#6B7280',  // Gray
];

const formatNumber = (num, decimals = 0) => {
  if (decimals > 0) {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(num);
  }
  return new Intl.NumberFormat().format(Math.round(num));
};

const formatCompactValue = (value) => {
  const absValue = Math.abs(value);
  if (absValue >= 1000000) {
    const millions = value / 1000000;
    return millions % 1 === 0 ? `${millions}M` : `${millions.toFixed(1)}M`;
  } else if (absValue >= 10000) {
    const thousands = value / 1000;
    return `${Math.round(thousands)}K`;
  }
  return null;
};

const ProportionChart = ({
  data,
  dimensionOptions = [],
  metricOptions = [],
  defaultDimension,
  defaultMetric,
  title,
  chartType: initialChartType = 'donut',
  enableChartTypeToggle = true,
  maxCategories = 8,
  showLegend = true,
  showPercentages = true
}) => {
  // State
  const [selectedDimension, setSelectedDimension] = useState(defaultDimension || dimensionOptions[0]?.value);
  const [selectedMetric, setSelectedMetric] = useState(defaultMetric || metricOptions[0]?.value);
  const [chartType, setChartType] = useState(initialChartType);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [drillDownData, setDrillDownData] = useState(null); // null = main view, array = drill-down

  // Escape key handler for focus mode
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        if (drillDownData) {
          setDrillDownData(null);
        } else if (isFocusMode) {
          setIsFocusMode(false);
        }
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isFocusMode, drillDownData]);

  // Find current configs
  const currentDimensionConfig = dimensionOptions.find(d => d.value === selectedDimension);
  const currentMetricConfig = metricOptions.find(m => m.value === selectedMetric);

  // Process data into chart format
  const { chartData, otherItems, totalValue } = useMemo(() => {
    if (!Array.isArray(data) || !currentDimensionConfig || !currentMetricConfig) {
      return { chartData: [], otherItems: [], totalValue: 0 };
    }

    // Resolve data source
    const { data: effectiveData } = resolveMetricDataSource(
      currentMetricConfig,
      data,
      null
    );

    if (!Array.isArray(effectiveData) || effectiveData.length === 0) {
      return { chartData: [], otherItems: [], totalValue: 0 };
    }

    const dimensionField = currentDimensionConfig.field;
    const metricField = currentMetricConfig.field;
    const metricAggregation = currentMetricConfig.aggregation;
    const delimiter = currentDimensionConfig.delimiter;

    // Filter out invalid/unknown values
    let filteredData = effectiveData.filter(item => {
      const value = item[dimensionField];
      return value && value !== 'Unknown' && value.toString().trim() !== '';
    });

    // Expand delimited values if specified
    if (delimiter) {
      const expandedData = [];
      filteredData.forEach(item => {
        const value = item[dimensionField];
        if (value && typeof value === 'string') {
          const values = value.split(delimiter).map(v => v.trim()).filter(v => v !== '' && v !== 'Unknown');
          values.forEach(val => {
            expandedData.push({
              ...item,
              [dimensionField]: val
            });
          });
        }
      });
      filteredData = expandedData;
    }

    // Apply metric filter
    filteredData = applyMetricFilter(filteredData, currentMetricConfig);

    // Group by dimension
    const grouped = _(filteredData).groupBy(dimensionField);

    // Calculate metric for each group
    let processedData = grouped.map((items, groupKey) => {
      let metricValue;

      switch (metricAggregation) {
        case 'count':
          metricValue = items.length;
          break;
        case 'count_distinct': {
          const distinctValues = new Set(
            items
              .map(item => {
                const val = item[metricField];
                if (val instanceof Date) {
                  return val.toISOString().split('T')[0];
                }
                return val;
              })
              .filter(val => val !== null && val !== undefined && val !== '')
          );
          metricValue = distinctValues.size;
          break;
        }
        case 'sum':
        case 'cumsum':
          metricValue = _.sumBy(items, item => {
            const val = parseFloat(item[metricField]);
            return isNaN(val) ? 0 : val;
          });
          break;
        case 'average': {
          const validItems = items.filter(item => {
            const val = parseFloat(item[metricField]);
            return !isNaN(val) && val !== null && val !== undefined;
          });
          if (validItems.length === 0) {
            metricValue = 0;
          } else {
            const sum = _.sumBy(validItems, item => parseFloat(item[metricField]));
            metricValue = sum / validItems.length;
          }
          break;
        }
        default:
          metricValue = items.length;
      }

      return {
        name: groupKey,
        value: metricValue,
        count: items.length
      };
    }).value();

    // Sort by value descending
    processedData = _(processedData).orderBy(['value'], ['desc']).value();

    // Calculate total
    const total = _.sumBy(processedData, 'value');

    // Split into main items and "Other"
    let mainItems = processedData;
    let otherGroupItems = [];

    if (processedData.length > maxCategories) {
      mainItems = processedData.slice(0, maxCategories - 1);
      otherGroupItems = processedData.slice(maxCategories - 1);

      const otherValue = _.sumBy(otherGroupItems, 'value');
      const otherCount = _.sumBy(otherGroupItems, 'count');

      mainItems.push({
        name: 'Other',
        value: otherValue,
        count: otherCount,
        isOther: true,
        itemCount: otherGroupItems.length
      });
    }

    // Add percentage and color to each item
    mainItems = mainItems.map((item, index) => ({
      ...item,
      percentage: total > 0 ? (item.value / total * 100) : 0,
      color: item.isOther ? CATEGORY_COLORS[CATEGORY_COLORS.length - 1] : CATEGORY_COLORS[index % CATEGORY_COLORS.length]
    }));

    return {
      chartData: mainItems,
      otherItems: otherGroupItems.map((item, index) => ({
        ...item,
        percentage: total > 0 ? (item.value / total * 100) : 0,
        color: EXTENDED_COLORS[index % EXTENDED_COLORS.length]
      })),
      totalValue: total
    };
  }, [data, currentDimensionConfig, currentMetricConfig, maxCategories]);

  // Format value for display
  const formatValue = (value) => {
    const decimals = currentMetricConfig?.decimals || 0;
    const prefix = currentMetricConfig?.prefix || '';
    const suffix = currentMetricConfig?.suffix || '';
    const useCompact = currentMetricConfig?.compactNumbers || false;

    const formatted = useCompact
      ? (formatCompactValue(value) || formatNumber(value, decimals))
      : formatNumber(value, decimals);

    return `${prefix}${formatted}${suffix}`;
  };

  // Handle slice/cell click
  const handleSliceClick = (entry) => {
    if (entry.isOther && otherItems.length > 0) {
      setDrillDownData(otherItems);
    }
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null;

    const item = payload[0].payload;
    return (
      <div className="proportion-tooltip">
        <p className="tooltip-name">{item.name}</p>
        <p className="tooltip-value">{formatValue(item.value)}</p>
        {showPercentages && (
          <p className="tooltip-percentage">{item.percentage.toFixed(1)}%</p>
        )}
        {item.isOther && (
          <p className="tooltip-hint">Click to see {item.itemCount} items</p>
        )}
      </div>
    );
  };

  // Custom label for pie chart
  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
    if (!percent || percent < 0.05) return null; // Don't show label for slices < 5%

    const RADIAN = Math.PI / 180;
    const radius = outerRadius * 1.15;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    const displayName = name || 'Unknown';

    return (
      <text
        x={x}
        y={y}
        fill="var(--color-text-primary)"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        fontSize="12"
      >
        {displayName.length > 15 ? displayName.substring(0, 15) + '...' : displayName}
        {showPercentages && ` (${(percent * 100).toFixed(0)}%)`}
      </text>
    );
  };

  // Treemap custom content - Recharts passes: root, depth, x, y, width, height, index, name, value
  const TreemapContent = (props) => {
    const { x, y, width, height, name, value, index } = props;

    // Skip rendering if essential props are missing
    if (x === undefined || y === undefined || width === undefined || height === undefined) {
      return null;
    }

    // Get custom properties from chartData by matching the name
    const dataItem = (drillDownData || chartData).find(item => item.name === name) || {};
    const { isOther, itemCount, color: itemColor } = dataItem;

    // Use color from data item, or fallback to index-based color
    const color = itemColor || CATEGORY_COLORS[index % CATEGORY_COLORS.length];

    const shouldRenderText = width > 40 && height > 30;
    const displayName = name || 'Unknown';
    const percentage = totalValue > 0 && value ? (value / totalValue * 100).toFixed(1) : 0;

    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          fill={color}
          stroke="var(--color-surface)"
          strokeWidth={2}
          style={{ cursor: isOther ? 'pointer' : 'default' }}
          onClick={() => isOther && handleSliceClick({ isOther, itemCount })}
        />
        {shouldRenderText && (
          <foreignObject x={x} y={y} width={width} height={height}>
            <div
              xmlns="http://www.w3.org/1999/xhtml"
              className="treemap-cell-content"
              style={{ cursor: isOther ? 'pointer' : 'default' }}
              onClick={() => isOther && handleSliceClick({ isOther, itemCount })}
            >
              <span className="treemap-cell-name">
                {displayName.length > 20 ? displayName.substring(0, 20) + '...' : displayName}
              </span>
              {showPercentages && <span className="treemap-cell-percentage">{percentage}%</span>}
              {isOther && <span className="treemap-cell-hint">Click to expand</span>}
            </div>
          </foreignObject>
        )}
      </g>
    );
  };

  // Render controls
  const renderControls = (inFocusMode = false) => (
    <div className="chart-controls" onClick={(e) => e.stopPropagation()}>
      {/* Chart Type Toggle */}
      {enableChartTypeToggle && (
        <div className="chart-type-toggle">
          <button
            className={`toggle-btn ${chartType === 'donut' ? 'active' : ''}`}
            onClick={() => setChartType('donut')}
            aria-label="Donut chart"
            title="Donut chart"
          >
            <PieChartIcon size={16} />
          </button>
          <button
            className={`toggle-btn ${chartType === 'treemap' ? 'active' : ''}`}
            onClick={() => setChartType('treemap')}
            aria-label="Treemap"
            title="Treemap"
          >
            <LayoutGrid size={16} />
          </button>
        </div>
      )}

      {/* Dimension Selector */}
      {dimensionOptions.length > 1 && (
        <div className="chart-filter">
          <label htmlFor={inFocusMode ? "focus-dimension-select" : "dimension-select"}>Split by:</label>
          <select
            id={inFocusMode ? "focus-dimension-select" : "dimension-select"}
            className="filter-select"
            value={selectedDimension}
            onChange={(e) => {
              setSelectedDimension(e.target.value);
              setDrillDownData(null);
            }}
          >
            {dimensionOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Metric Selector */}
      {metricOptions.length > 1 && (
        <div className="chart-filter">
          <label htmlFor={inFocusMode ? "focus-metric-select" : "metric-select"}>Size by:</label>
          <select
            id={inFocusMode ? "focus-metric-select" : "metric-select"}
            className="filter-select"
            value={selectedMetric}
            onChange={(e) => {
              setSelectedMetric(e.target.value);
              setDrillDownData(null);
            }}
          >
            {metricOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );

  // Render donut chart
  const renderDonutChart = (dataToRender, isDrillDown = false) => (
    <div className="donut-chart-wrapper">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={dataToRender}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius="55%"
            outerRadius="75%"
            paddingAngle={2}
            label={renderCustomLabel}
            labelLine={true}
            onClick={(entry) => handleSliceClick(entry)}
            style={{ cursor: 'pointer' }}
          >
            {dataToRender.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.color}
                stroke="var(--color-surface)"
                strokeWidth={2}
                style={{ cursor: entry.isOther ? 'pointer' : 'default' }}
              />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          {showLegend && !isDrillDown && (
            <Legend
              layout="horizontal"
              verticalAlign="bottom"
              align="center"
              formatter={(value) => (
                <span style={{ color: 'var(--color-text-primary)' }}>
                  {value.length > 20 ? value.substring(0, 20) + '...' : value}
                </span>
              )}
            />
          )}
        </PieChart>
      </ResponsiveContainer>
      {/* Center label showing total */}
      <div className="donut-center-label">
        <span className="donut-total-value">{formatValue(totalValue)}</span>
        <span className="donut-total-label">Total</span>
      </div>
    </div>
  );

  // Render treemap chart
  const renderTreemapChart = (dataToRender) => (
    <ResponsiveContainer width="100%" height="100%">
      <Treemap
        data={dataToRender}
        dataKey="value"
        aspectRatio={4 / 3}
        content={<TreemapContent />}
      >
        <Tooltip content={<CustomTooltip />} />
      </Treemap>
    </ResponsiveContainer>
  );

  // Render main chart content
  const renderChart = () => {
    const dataToRender = drillDownData || chartData;

    if (dataToRender.length === 0) {
      return (
        <div className="no-chart-data">
          <p>No data available for the selected filters.</p>
        </div>
      );
    }

    // In drill-down mode, always show treemap for better visibility of many items
    if (drillDownData) {
      return renderTreemapChart(drillDownData);
    }

    return chartType === 'donut'
      ? renderDonutChart(chartData)
      : renderTreemapChart(chartData);
  };

  // Render drill-down header
  const renderDrillDownHeader = () => {
    if (!drillDownData) return null;

    return (
      <div className="drilldown-header" onClick={(e) => e.stopPropagation()}>
        <button
          className="drilldown-back-btn"
          onClick={() => setDrillDownData(null)}
          aria-label="Back to main view"
        >
          <ArrowLeft size={18} />
          <span>Back</span>
        </button>
        <span className="drilldown-title">
          Other ({otherItems.length} items)
        </span>
      </div>
    );
  };

  return (
    <>
      {/* Normal view */}
      <div className="proportion-chart-container" onClick={() => setIsFocusMode(true)}>
        <div className="chart-header">
          <h3 className="chart-title">{title || 'Distribution'}</h3>
          {!drillDownData && renderControls(false)}
        </div>
        {renderDrillDownHeader()}
        <div className="proportion-chart-content">
          {renderChart()}
        </div>
      </div>

      {/* Focus mode overlay */}
      {isFocusMode && (
        <div className="proportion-focus-overlay" onClick={() => {
          setIsFocusMode(false);
          setDrillDownData(null);
        }}>
          <div className="proportion-focus-content" onClick={(e) => e.stopPropagation()}>
            <button
              className="focus-close-button"
              onClick={() => {
                setIsFocusMode(false);
                setDrillDownData(null);
              }}
              aria-label="Close focus mode"
            >
              <X size={24} />
            </button>
            <div className="proportion-focus-controls-bar">
              {!drillDownData && renderControls(true)}
              {renderDrillDownHeader()}
            </div>
            <div className="proportion-focus-chart-container">
              {renderChart()}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

ProportionChart.propTypes = {
  data: PropTypes.array.isRequired,
  dimensionOptions: PropTypes.arrayOf(
    PropTypes.shape({
      value: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      field: PropTypes.string.isRequired,
      delimiter: PropTypes.string
    })
  ).isRequired,
  metricOptions: PropTypes.arrayOf(
    PropTypes.shape({
      value: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      aggregation: PropTypes.oneOf(['count', 'count_distinct', 'sum', 'average', 'cumsum']).isRequired,
      field: PropTypes.string,
      prefix: PropTypes.string,
      suffix: PropTypes.string,
      decimals: PropTypes.number,
      compactNumbers: PropTypes.bool,
      filterConditions: PropTypes.arrayOf(PropTypes.shape({
        field: PropTypes.string.isRequired,
        operator: PropTypes.oneOf(['=', '==', '!=', '!==', '>', '>=', '<', '<=']),
        value: PropTypes.oneOfType([
          PropTypes.string,
          PropTypes.number,
          PropTypes.bool,
          PropTypes.array
        ]).isRequired
      })),
      filterField: PropTypes.string,
      filterValue: PropTypes.oneOfType([PropTypes.string, PropTypes.number, PropTypes.bool, PropTypes.array]),
      data: PropTypes.array
    })
  ).isRequired,
  defaultDimension: PropTypes.string,
  defaultMetric: PropTypes.string,
  title: PropTypes.string,
  chartType: PropTypes.oneOf(['donut', 'treemap']),
  enableChartTypeToggle: PropTypes.bool,
  maxCategories: PropTypes.number,
  showLegend: PropTypes.bool,
  showPercentages: PropTypes.bool
};

export default ProportionChart;
