// src/context/DataContext.jsx
import React, { createContext, useContext, useState, useCallback, useRef, useMemo } from 'react';
import Papa from 'papaparse';
import { DRIVE_FILES, getDriveDownloadUrl } from '../config/config';

const DataContext = createContext();

// Helper function to clean UTF-8 strings
const decodeUTF8 = (str) => {
  if (!str) return str;

  // Just remove control characters and trim
  // Modern browsers handle UTF-8 correctly, no need for aggressive decoding
  return str
    .replace(/[\u0000-\u0019]/g, '') // Remove control characters
    .trim();
};

// In the clean string function in DataContext.jsx
const cleanString = (str) => {
  if (!str) return str;
  // Remove null bytes and trim
  return decodeUTF8(str.replace(/\u0000/g, '').trim());
};

// Add a new helper function for date validation and cleaning
const cleanDate = (dateStr) => {
  if (!dateStr) return null;

  try {
    // Parse the date string
    const date = new Date(dateStr);

    // Check if date is valid and within reasonable range (e.g., between 1990 and 2040)
    if (isNaN(date.getTime()) ||
        date.getFullYear() < 1990 ||
        date.getFullYear() > 2040) {
      return null;
    }

    return date.toISOString();
  } catch (e) {
    return null;
  }
};


// Helper function to clean data
const cleanData = (data, dataType = null) => {
  // Skip expensive date conversions for large datasets like music
  const skipDateConversion = dataType === 'music';

  return data.map(item => {
    const cleanedItem = {};
    Object.entries(item).forEach(([key, value]) => {
      const cleanKey = cleanString(key).trim();

      // Special handling for date fields (skip for large datasets)
      // Exclude segment_start_time and segment_end_time which are HH:MM time strings, not dates
      if (!skipDateConversion &&
          (cleanKey.includes('date') ||
          cleanKey.includes('timestamp') ||
          cleanKey.includes('finish') ||
          cleanKey.includes('start')) &&
          !cleanKey.includes('segment_start_time') &&
          !cleanKey.includes('segment_end_time')) {
        cleanedItem[cleanKey] = cleanDate(value);
      } else {
        // Normal string cleaning for non-date fields
        cleanedItem[cleanKey] = typeof value === 'string' ? cleanString(value).trim() : value;
      }
    });
    return cleanedItem;
  });
};


export const DataProvider = ({ children }) => {
  const [data, setData] = useState({
    nutrition: null,
    reading: null,
    readingBooks: null,
    readingSessions: null,
    movies: null,
    shows: null,
    podcasts: null,
    music: null,
    finance: null,
    healthDaily: null,
    healthHourly: null,
    tracking: null
  });
  const [loading, setLoading] = useState({});
  const [error, setError] = useState({});
  const [loadingProgress, setLoadingProgress] = useState({});

  // Track which data types have been logged to avoid duplicates
  const loggedDataTypes = useRef(new Set());

  // Use a ref to track data without causing re-renders
  const dataRef = useRef(data);
  dataRef.current = data;

  const fetchData = useCallback(async (dataType) => {
    if (dataRef.current[dataType]) {
      return dataRef.current[dataType];
    }

    setLoading(prev => ({ ...prev, [dataType]: true }));

    try {
      let fileId;
      switch (dataType) {
        case 'reading':
          fileId = DRIVE_FILES.READING.FILE_ID;
          break;
        case 'readingBooks':
          fileId = DRIVE_FILES.READING_BOOKS.FILE_ID;
          break;
        case 'readingSessions':
          fileId = DRIVE_FILES.READING_SESSIONS.FILE_ID;
          break;
        case 'movies':
          fileId = DRIVE_FILES.MOVIES.FILE_ID;
          break;
        case 'shows':
          fileId = DRIVE_FILES.SHOWS.FILE_ID;
          break;
        case 'nutrition':
          fileId = DRIVE_FILES.NUTRITION.FILE_ID;
          break;
        case 'podcasts':
          fileId = DRIVE_FILES.PODCASTS.FILE_ID;
          break;
        case 'music':
          fileId = DRIVE_FILES.MUSIC.FILE_ID;
          break;
        case 'finance':
          fileId = DRIVE_FILES.FINANCES.FILE_ID;
          break;
        case 'healthDaily':
          fileId = DRIVE_FILES.HEALTH_DAILY.FILE_ID;
          break;
        case 'healthHourly':
          fileId = DRIVE_FILES.HEALTH_HOURLY.FILE_ID;
          break;
        case 'tracking':
          fileId = DRIVE_FILES.TRACKING.FILE_ID;
          break;
        default:
          throw new Error(`Unknown data type: ${dataType}`);
      }


      // Check if fileId is configured (not undefined or placeholder)
      if (!fileId || fileId === 'undefined') {
        const errorMsg = `File ID not configured for ${dataType}`;
        console.warn(errorMsg);
        setError(prev => ({ ...prev, [dataType]: errorMsg }));
        setLoading(prev => ({ ...prev, [dataType]: false }));
        throw new Error(errorMsg);
      }

      const response = await fetch(getDriveDownloadUrl(fileId));
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const csvText = await response.text();

      // Use chunked loading for large datasets (music)
      const useChunkedLoading = dataType === 'music';
      const chunkSize = 10000; // Process 10K rows at a time

      return new Promise((resolve, reject) => {
        let accumulatedData = [];
        let rowsProcessed = 0;
        let totalRows = 0;

        // Estimate total rows for progress (rough estimate from file size)
        if (useChunkedLoading) {
          const lines = csvText.split('\n').length;
          totalRows = Math.max(lines - 1, 0); // Subtract header row
          setLoadingProgress(prev => ({ ...prev, [dataType]: { current: 0, total: totalRows } }));
        }

        Papa.parse(csvText, {
          delimiter: "|",
          header: true,
          skipEmptyLines: true,
          encoding: '', // Auto-detect encoding (handles both UTF-8 and UTF-16)
          transform: (value) => cleanString(value), // Clean each value as it's parsed
          chunk: useChunkedLoading ? (results, parser) => {
            // Process chunk
            accumulatedData = accumulatedData.concat(results.data);
            rowsProcessed += results.data.length;

            // Update progress
            const progress = Math.min(Math.round((rowsProcessed / totalRows) * 100), 100);
            setLoadingProgress(prev => ({ ...prev, [dataType]: { current: rowsProcessed, total: totalRows, percent: progress } }));

            // Small pause every 5 chunks to allow UI updates
            if (rowsProcessed % (chunkSize * 5) === 0) {
              parser.pause();
              setTimeout(() => parser.resume(), 10);
            }
          } : undefined,
          complete: (results) => {
            // Use accumulated data for chunked loading, otherwise use results.data
            const rawData = useChunkedLoading ? accumulatedData : results.data;

            // Clear loading progress when complete
            if (useChunkedLoading) {
              setLoadingProgress(prev => ({ ...prev, [dataType]: { current: totalRows, total: totalRows, percent: 100 } }));
              console.log(`🎵 Loaded ${rawData.length.toLocaleString()} music records`);
            }

            // Simplified logging for reading data types (only log once per data type)
            if ((dataType === 'readingBooks' || dataType === 'readingSessions') && !loggedDataTypes.current.has(dataType)) {
              loggedDataTypes.current.add(dataType);

              console.log(`📚 ${dataType} columns:`, results.meta.fields);

              // Find latest date - look for date-related columns
              const dateColumns = results.meta.fields.filter(f =>
                f && (f.toLowerCase().includes('date') ||
                      f.toLowerCase().includes('finish') ||
                      f.toLowerCase().includes('timestamp'))
              );

              if (dateColumns.length > 0 && rawData.length > 0) {
                // Try each date column and find the latest date across all of them
                let allDates = [];

                dateColumns.forEach(dateColumn => {
                  const dates = rawData
                    .map(row => row[dateColumn])
                    .filter(d => d && d.toString().trim())
                    .map(d => new Date(d))
                    .filter(d => !isNaN(d.getTime()));

                  allDates = allDates.concat(dates);
                });

                if (allDates.length > 0) {
                  allDates.sort((a, b) => b - a);
                  console.log(`📚 ${dataType} latest date:`, allDates[0].toISOString().split('T')[0]);
                }
              }
            }

            let cleanedData = cleanData(rawData, dataType);

            // Type conversion for reading books
            if (dataType === 'readingBooks') {
              cleanedData = cleanedData.map(book => ({
                ...book,
                my_rating: book.my_rating ? parseFloat(book.my_rating) : 0,
                average_rating: book.average_rating ? parseFloat(book.average_rating) : 0,
                number_of_pages: book.number_of_pages ? parseInt(book.number_of_pages) : 0,
                original_publication_year: book.original_publication_year ? parseInt(book.original_publication_year) : null,
                reading_duration_final: book.reading_duration_final ? parseInt(book.reading_duration_final) : null,
                pages_per_day : book.pages_per_day ? parseFloat(book.pages_per_day) : 0
              }));
            }

            // Type conversion for reading sessions
            if (dataType === 'readingSessions') {
              cleanedData = cleanedData.map(session => ({
                ...session,
                page_split: session.page_split ? parseInt(session.page_split) : 0,
                my_rating: session.my_rating ? parseFloat(session.my_rating) : 0
              }));
            }

            // Type conversion for podcasts
            if (dataType === 'podcasts') {
              cleanedData = cleanedData.map(episode => ({
                ...episode,
                duration_seconds: episode.duration_seconds ? parseInt(episode.duration_seconds) : 0,
                listened_seconds: episode.listened_seconds ? parseInt(episode.listened_seconds) : 0,
                listened_minutes: episode.listened_minutes ? parseFloat(episode.listened_minutes) : 0,
                listened_hours: episode.listened_hours ? parseFloat(episode.listened_hours) : 0,
                completion_percent: episode.completion_percent ? parseFloat(episode.completion_percent) : 0,
                // Boolean columns - convert to Yes/No for user-friendly filtering
                is_new_podcast: parseInt(episode.is_new_podcast) === 1 ? 'Yes' : 'No',
                is_new_recurring_podcast: parseInt(episode.is_new_recurring_podcast) === 1 ? 'Yes' : 'No',
                is_recurring_podcast: parseInt(episode.is_recurring_podcast) === 1 ? 'Yes' : 'No',
              }));
            }

            // Type conversion for movies
            if (dataType === 'movies') {
              cleanedData = cleanedData.map(movie => ({
                ...movie,
                rating: movie.rating ? parseFloat(movie.rating) : 0,
                runtime: movie.runtime ? parseInt(movie.runtime) : null,
                vote_average: movie.vote_average ? parseFloat(movie.vote_average) : null,
                vote_count: movie.vote_count ? parseInt(movie.vote_count) : null,
                popularity: movie.popularity ? parseFloat(movie.popularity) : null,
                budget: movie.budget ? parseInt(movie.budget) : null,
                revenue: movie.revenue ? parseInt(movie.revenue) : null,
                tmdb_id: movie.tmdb_id ? parseInt(movie.tmdb_id) : null,
                year: movie.year ? parseInt(movie.year) : null
              }));
            }

            if (dataType === 'shows') {
              cleanedData = cleanedData.map(episode => ({
                ...episode,
                episode_runtime: episode.episode_runtime ? parseInt(episode.episode_runtime) : 0,
                episode_runtime_hours: episode.episode_runtime_hours ? parseFloat(episode.episode_runtime_hours) : 0,

              }));
            }

            // Type conversion for music
            if (dataType === 'music') {
              cleanedData = cleanedData.map(toggle => {
                const trackDuration = toggle.track_duration ? parseInt(toggle.track_duration) : 0;
                const completion = toggle.completion ? parseFloat(toggle.completion) : 0;
                const listeningSeconds = Math.round((trackDuration / 1000) * (completion / 100));

                // Extract year from timestamp (keep as string for filtering)
                const listeningYear = toggle.timestamp ? toggle.timestamp.substring(0, 4) : null;

                // Use first genre as simplified genre
                const simplifiedGenre = toggle.genre_1 || null;

                return {
                  ...toggle,
                  toggle_id: toggle.toggle_id ? parseInt(toggle.toggle_id) : 0,
                  followers: toggle.followers ? parseInt(toggle.followers) : 0,
                  artist_popularity: toggle.artist_popularity ? parseInt(toggle.artist_popularity) : 0,
                  track_popularity: toggle.track_popularity ? parseInt(toggle.track_popularity) : 0,
                  track_duration: trackDuration,
                  completion: completion,
                  listening_seconds: listeningSeconds,
                  listening_hours: listeningSeconds / 3600,
                  listening_year: listeningYear,
                  simplified_genre: simplifiedGenre,
                  // Boolean columns - convert to Yes/No for user-friendly filtering
                  is_skipped_track: parseInt(toggle.is_skipped_track) === 1 ? 'Yes' : 'No',
                  is_new_artist: parseInt(toggle.is_new_artist) === 1 ? 'Yes' : 'No',
                  is_new_track: parseInt(toggle.is_new_track) === 1 ? 'Yes' : 'No',
                  is_recurring_artist: parseInt(toggle.is_recurring_artist) === 1 ? 'Yes' : 'No',
                  is_recurring_track: parseInt(toggle.is_recurring_track) === 1 ? 'Yes' : 'No',
                  is_new_recurring_artist: parseInt(toggle.is_new_recurring_artist) === 1 ? 'Yes' : 'No',
                  is_new_recurring_track: parseInt(toggle.is_new_recurring_track) === 1 ? 'Yes' : 'No'
                };
              });
            }

            // Type conversion for healthDaily (daily summary data)
            if (dataType === 'healthDaily') {
              cleanedData = cleanedData.map(day => ({
                ...day,
                // Subjective metrics
                sleep_quality: day.sleep_quality ? parseFloat(day.sleep_quality) : null,
                dreams: day.dreams ? parseInt(day.dreams) : null,
                sleep_rest_feeling: day.sleep_rest_feeling ? parseFloat(day.sleep_rest_feeling) : null,
                fitness_feeling: day.fitness_feeling ? parseFloat(day.fitness_feeling) : null,
                overall_evaluation: day.overall_evaluation ? parseFloat(day.overall_evaluation) : null,
                // Sleep times (numeric for averaging)
                sleep_start_time_minutes: day.sleep_start_time_minutes ? parseFloat(day.sleep_start_time_minutes) : null,
                wake_up_time_minutes: day.wake_up_time_minutes ? parseFloat(day.wake_up_time_minutes) : null,
                // Daily totals
                total_steps: day.total_steps ? parseInt(day.total_steps) : 0,
                total_apple_distance_meters: day.total_apple_distance_meters ? parseFloat(day.total_apple_distance_meters) : 0,
                total_flights_climbed: day.total_flights_climbed ? parseInt(day.total_flights_climbed) : 0,
                total_active_energy_kcal: day.total_active_energy_kcal ? parseFloat(day.total_active_energy_kcal) : 0,
                total_resting_energy_kcal: day.total_resting_energy_kcal ? parseFloat(day.total_resting_energy_kcal) : 0,
                total_sleep_minutes: day.total_sleep_minutes ? parseFloat(day.total_sleep_minutes) : 0,
                total_deep_sleep_minutes: day.total_deep_sleep_minutes ? parseFloat(day.total_deep_sleep_minutes) : 0,
                total_rem_sleep_minutes: day.total_rem_sleep_minutes ? parseFloat(day.total_rem_sleep_minutes) : 0,
                total_core_sleep_minutes: day.total_core_sleep_minutes ? parseFloat(day.total_core_sleep_minutes) : 0,
                total_awake_minutes: day.total_awake_minutes ? parseFloat(day.total_awake_minutes) : 0,
                total_screen_time_minutes: day.total_screen_time_minutes ? parseFloat(day.total_screen_time_minutes) : 0,
                total_phone_pickups: day.total_phone_pickups ? parseInt(day.total_phone_pickups) : 0,
                total_screen_before_sleep_minutes: day.total_screen_before_sleep_minutes ? parseFloat(day.total_screen_before_sleep_minutes) : 0
              }));
            }

            // Type conversion for healthHourly (hourly segment data)
            if (dataType === 'healthHourly') {
              cleanedData = cleanedData.map(segment => ({
                ...segment,
                // Time identifiers
                hour: segment.hour ? parseInt(segment.hour) : 0,
                weekday: segment.weekday ? parseInt(segment.weekday) : 0,
                segment_duration_minutes: segment.segment_duration_minutes ? parseFloat(segment.segment_duration_minutes) : 0,
                segment_duration_hours: segment.segment_duration_hours ? parseFloat(segment.segment_duration_hours) : 0,
                segment_duration_days: segment.segment_duration_days ? parseFloat(segment.segment_duration_days) : 0,
                // Movement metrics
                steps: segment.steps ? parseInt(segment.steps) : 0,
                apple_distance_meters: segment.apple_distance_meters ? parseFloat(segment.apple_distance_meters) : 0,
                distance_meters: segment.distance_meters ? parseFloat(segment.distance_meters) : 0,
                flights_climbed: segment.flights_climbed ? parseInt(segment.flights_climbed) : 0,
                // Energy metrics
                active_energy_kcal: segment.active_energy_kcal ? parseFloat(segment.active_energy_kcal) : 0,
                resting_energy_kcal: segment.resting_energy_kcal ? parseFloat(segment.resting_energy_kcal) : 0,
                // Averages
                avg_step_length_cm: segment.avg_step_length_cm ? parseFloat(segment.avg_step_length_cm) : null,
                avg_walking_speed_kmh: segment.avg_walking_speed_kmh ? parseFloat(segment.avg_walking_speed_kmh) : null,
                avg_heart_rate: segment.avg_heart_rate ? parseFloat(segment.avg_heart_rate) : null,
                avg_audio_exposure: segment.avg_audio_exposure ? parseFloat(segment.avg_audio_exposure) : null,
                // Body metrics
                body_weight_kg: segment.body_weight_kg ? parseFloat(segment.body_weight_kg) : null,
                body_fat_percent: segment.body_fat_percent ? parseFloat(segment.body_fat_percent) : null,
                // Screen time
                screen_time_minutes: segment.screen_time_minutes ? parseFloat(segment.screen_time_minutes) : 0,
                phone_pickups: segment.phone_pickups ? parseInt(segment.phone_pickups) : 0,
                screen_time_minutes_before_sleep: segment.screen_time_minutes_before_sleep ? parseFloat(segment.screen_time_minutes_before_sleep) : 0,
                // Sleep metrics
                sleep_minutes: segment.sleep_minutes ? parseFloat(segment.sleep_minutes) : 0,
                deep_sleep_minutes: segment.deep_sleep_minutes ? parseFloat(segment.deep_sleep_minutes) : 0,
                rem_sleep_minutes: segment.rem_sleep_minutes ? parseFloat(segment.rem_sleep_minutes) : 0,
                core_sleep_minutes: segment.core_sleep_minutes ? parseFloat(segment.core_sleep_minutes) : 0,
                awake_minutes: segment.awake_minutes ? parseFloat(segment.awake_minutes) : 0,
                // Boolean conversion
                is_home: segment.is_home === 'True' || segment.is_home === true || segment.is_home === 'true'
              }));
            }

            setData(prev => ({ ...prev, [dataType]: cleanedData }));
            setLoading(prev => ({ ...prev, [dataType]: false }));
            resolve(cleanedData);
          },
          error: (error) => {
            console.error(`${dataType} Papa parse error:`, error);
            setError(prev => ({ ...prev, [dataType]: error.message }));
            setLoading(prev => ({ ...prev, [dataType]: false }));
            reject(error);
          }
        });
      });
    } catch (err) {
      setError(prev => ({ ...prev, [dataType]: err.message }));
      setLoading(prev => ({ ...prev, [dataType]: false }));
      throw err;
    }
  }, []); // Remove data dependency to prevent infinite loops


  const value = useMemo(() => ({
    data,
    loading,
    error,
    loadingProgress,
    fetchData
  }), [data, loading, error, loadingProgress, fetchData]);

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
};

export const useData = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};
