/* eslint-disable react/prop-types */
// src/components/ui/KPICardsPanel/KPICardsPanel.jsx
import React from 'react';
import './KPICardsPanel.css';

/**
 * A reusable component for displaying a grid of KPI cards with consistent layout and styling
 *
 * Uses children-based API - pass KpiCard components as children
 *
 * @param {Object} props
 * @param {React.ReactNode} props.children - KpiCard components
 * @param {Object} [props.dataSources] - Object mapping data source names to data arrays
 * @param {boolean} [props.loading] - Whether the cards are in a loading state
 * @param {string} [props.className] - Additional CSS classes to apply to the container
 */
const KPICardsPanel = ({
  children,
  dataSources = {},
  loading = false,
  className = ''
}) => {
  // Show loading state if requested
  if (loading) {
    return (
      <div className={`cards-panel ${className}`}>
        <div className="cards-panel-grid">
          {/* Show placeholder cards while loading */}
          {[1, 2, 3, 4].map((index) => (
            <div key={index} className="cards-panel-skeleton">
              <div className="skeleton-icon"></div>
              <div className="skeleton-value"></div>
              <div className="skeleton-label"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={`cards-panel ${className}`}>
      <div className="cards-panel-grid">
        {React.Children.map(children, (child) => {
          if (!React.isValidElement(child)) return null;

          // Clone child and inject dataSource data if needed
          if (child.props.dataSource) {
            const dataSourceName = child.props.dataSource;
            const data = dataSources[dataSourceName];

            return React.cloneElement(child, {
              data: data
            });
          }

          // Return child as-is if no dataSource (legacy mode)
          return child;
        })}
      </div>
    </div>
  );
};

export default KPICardsPanel;
