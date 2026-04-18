import React from 'react';
import { AlertCircle } from 'lucide-react';
import './PageWrapper.css';

/**
 * PageWrapper component
 * Standardized wrapper for all pages with built-in error handling
 *
 * @param {Object} props
 * @param {React.ReactNode} props.children - Page content to render
 * @param {string|null} props.error - Error message to display (if any)
 * @param {string} props.errorTitle - Title for error state (default: page title)
 * @param {boolean} props.wide - Use wide page layout (default: false)
 * @param {string} props.className - Additional CSS classes
 */
const PageWrapper = ({
  children,
  error = null,
  errorTitle = "Error Loading Data",
  wide = false,
  className = ""
}) => {
  // Error state
  if (error) {
    return (
      <div className="page-container">
        <div className="page-content">
          <div className="page-error">
            <AlertCircle className="page-error-icon" />
            <h1 className="page-error-title">{errorTitle}</h1>
            <div className="page-error-message">
              {error}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Normal state
  return (
    <div className={`page-container ${wide ? 'page-container--wide' : ''} ${className}`}>
      <div className="page-content">
        {children}
      </div>
    </div>
  );
};

export default PageWrapper;
