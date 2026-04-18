import { useEffect } from 'react';

/**
 * Custom hook to set the document title dynamically
 * @param {string} pageTitle - The page-specific title (e.g., "Reading", "Music")
 * @param {string} baseTitle - The base title to prepend (default: "Lifelog")
 */
export const usePageTitle = (pageTitle, baseTitle = 'Lifelog') => {
  useEffect(() => {
    // Set the document title
    document.title = pageTitle ? `${baseTitle} | ${pageTitle}` : baseTitle;

    // Cleanup function to reset title when component unmounts
    return () => {
      document.title = baseTitle;
    };
  }, [pageTitle, baseTitle]);
};
