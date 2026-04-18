import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [userRole, setUserRole] = useState('admin'); // Default to admin for smooth UX
  const [allowedPages, setAllowedPages] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const API_BASE = import.meta.env.PROD ? '' : 'http://localhost:3001';
        const response = await fetch(`${API_BASE}/api/user-info`, {
          credentials: 'include'
        });

        if (response.ok) {
          const data = await response.json();
          setUserRole(data.role);
          setAllowedPages(data.allowedPages);
        }
      } catch (error) {
        console.error('Failed to fetch user info:', error);
        // Keep default admin role on error
      } finally {
        setLoading(false);
      }
    };

    fetchUserInfo();
  }, []);

  const isPageAllowed = (path) => {
    // Admin has access to everything
    if (userRole === 'admin' || !allowedPages) {
      return true;
    }

    // Check if path is in allowed pages
    return allowedPages.some(page =>
      path === page || path.startsWith(page + '/')
    );
  };

  const value = {
    userRole,
    allowedPages,
    loading,
    isPageAllowed,
    isGuest: userRole === 'guest',
    isAdmin: userRole === 'admin'
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
