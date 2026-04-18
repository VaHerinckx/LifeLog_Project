import PropTypes from 'prop-types';
import { BarChart } from 'lucide-react';

/**
 * TabNavigation - Standardized tab navigation component
 *
 * Used for switching between Content and Analysis views on main pages.
 * Uses CSS classes from components.css: .page-tabs, .page-tab, .page-tab.active
 *
 * @param {string} contentLabel - Label for the content tab (e.g., "Books", "Movies", "Tracks")
 * @param {React.Component} contentIcon - Icon component for the content tab (e.g., BookOpen, Film)
 * @param {string} activeTab - Currently active tab: "content" or "analysis"
 * @param {function} onTabChange - Callback function when tab is clicked
 */
const TabNavigation = ({ contentLabel, contentIcon: ContentIcon, activeTab, onTabChange }) => {
  return (
    <div className="page-tabs">
      <button
        className={`page-tab ${activeTab === 'content' ? 'active' : ''}`}
        onClick={() => onTabChange('content')}
      >
        <ContentIcon size={18} style={{ marginRight: '8px' }} />
        {contentLabel}
      </button>
      <button
        className={`page-tab ${activeTab === 'analysis' ? 'active' : ''}`}
        onClick={() => onTabChange('analysis')}
      >
        <BarChart size={18} style={{ marginRight: '8px' }} />
        Analysis
      </button>
    </div>
  );
};

TabNavigation.propTypes = {
  contentLabel: PropTypes.string.isRequired,
  contentIcon: PropTypes.elementType.isRequired,
  activeTab: PropTypes.oneOf(['content', 'analysis']).isRequired,
  onTabChange: PropTypes.func.isRequired,
};

export default TabNavigation;
