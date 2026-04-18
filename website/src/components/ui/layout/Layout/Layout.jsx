// components/Reusable_components/Layout.jsx
import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import NavigationBar from '../NavigationBar';
import './Layout.css';

const Layout = () => {
  const location = useLocation();

  return (
    <div className="layout">
      <NavigationBar />
      <main className="layout-content">
        <Outlet key={location.pathname} />
      </main>
    </div>
  );
};

export default Layout;
