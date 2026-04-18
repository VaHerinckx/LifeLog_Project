// AnalysisTab.jsx - Reusable analysis tab component
// Eliminates boilerplate for displaying charts with standardized structure
// Each chart specifies its own data source directly

import React from 'react';
import PropTypes from 'prop-types';
import './AnalysisTab.css';

/**
 * AnalysisTab - Standardized component for analysis views
 *
 * @param {Function} renderCharts - Function that returns chart JSX elements (no parameters)
 *
 * @example
 * <AnalysisTab
 *   renderCharts={() => (
 *     <>
 *       <TimeSeriesBarChart data={filteredData} ... />
 *       <IntensityHeatmap data={filteredHourlyData} ... />
 *     </>
 *   )}
 * />
 */
const AnalysisTab = ({ renderCharts }) => {
  // Validate required props
  if (!renderCharts || typeof renderCharts !== 'function') {
    return null;
  }

  // Render charts and wrap each child in analysis-chart-section
  const renderWrappedCharts = () => {
    const charts = renderCharts();

    // Handle single element
    if (!charts) return null;

    // Convert to array - handle Fragments properly
    let chartArray;

    // Check if charts is a Fragment (has props.children)
    if (React.isValidElement(charts) && charts.type === React.Fragment) {
      // Extract children from the Fragment
      chartArray = React.Children.toArray(charts.props.children);
    } else {
      // Otherwise treat as regular element or array
      chartArray = React.Children.toArray(charts);
    }

    // Wrap each chart in analysis-chart-section
    return chartArray.map((chart, index) => (
      <div key={`chart-section-${index}`} className="analysis-chart-section">
        {chart}
      </div>
    ));
  };

  return (
    <div className="analysis-tab-container">
      <div className="analysis-charts-grid">
        {renderWrappedCharts()}
      </div>
    </div>
  );
};

AnalysisTab.propTypes = {
  renderCharts: PropTypes.func.isRequired
};

export default AnalysisTab;
