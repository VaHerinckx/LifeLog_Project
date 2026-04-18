import PropTypes from 'prop-types';

/**
 * PageHeader - Standardized page header component
 *
 * Used at the top of all main pages to display title and description.
 * Uses CSS classes from components.css: .page-title, .page-description
 *
 * @param {string} title - Main page title (displayed in red accent color)
 * @param {string} description - Page description/subtitle
 */
const PageHeader = ({ title, description }) => {
  return (
    <>
      <h1 className="page-title">{title}</h1>
      {description && <p className="page-description">{description}</p>}
    </>
  );
};

PageHeader.propTypes = {
  title: PropTypes.string.isRequired,
  description: PropTypes.string,
};

export default PageHeader;
