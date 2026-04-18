// src/config/config.js
export const DRIVE_FILES = {
  PODCASTS: {
    FILE_ID: import.meta.env.VITE_PODCASTS_FILE_ID
  },
  READING: {
    FILE_ID: import.meta.env.VITE_READING_FILE_ID
  },
  READING_BOOKS: {
    FILE_ID: import.meta.env.VITE_READING_BOOKS_FILE_ID
  },
  READING_SESSIONS: {
    FILE_ID: import.meta.env.VITE_READING_SESSIONS_FILE_ID
  },
  MUSIC: {
    FILE_ID: import.meta.env.VITE_MUSIC_FILE_ID
  },
  MOVIES: {
    FILE_ID: import.meta.env.VITE_MOVIES_FILE_ID
  },
  TRAKT: {
    FILE_ID: import.meta.env.VITE_TRAKT_FILE_ID
  },
  HEALTH_DAILY: {
    FILE_ID: import.meta.env.VITE_HEALTH_DAILY_FILE_ID
  },
  HEALTH_HOURLY: {
    FILE_ID: import.meta.env.VITE_HEALTH_HOURLY_FILE_ID
  },
  NUTRITION: {
    FILE_ID: import.meta.env.VITE_NUTRITION_FILE_ID
  },
  FINANCES: {
    FILE_ID: import.meta.env.VITE_FINANCES_FILE_ID
  },
  SHOWS: {
    FILE_ID: import.meta.env.VITE_SHOWS_FILE_ID
  },
  TRACKING: {
    FILE_ID: import.meta.env.VITE_TRACKING_FILE_ID
  }
};

// API base URL - empty in production (same origin), localhost in development
const API_BASE = import.meta.env.PROD ? '' : 'http://localhost:3001';

export const getDriveDownloadUrl = (fileId) => {
  // Add cache buster to force fresh fetch from Google Drive
  const cacheBuster = Date.now();
  return `${API_BASE}/api/google-drive/${fileId}?_cb=${cacheBuster}`;
};
