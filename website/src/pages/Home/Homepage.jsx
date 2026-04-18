import { Link } from 'react-router-dom';
import { useState, useEffect, useRef, useMemo } from 'react';
import { BookOpen, Film, Music, UtensilsCrossed, Mic, Tv, DollarSign, Activity, Moon, Clock } from 'lucide-react';
import { usePageTitle } from '../../hooks/usePageTitle';
import { useData } from '../../context/DataContext';
import { useAuth } from '../../context/AuthContext';
import KPICardsPanel from '../../components/ui/content/KPICardsPanel/KPICardsPanel';
import KpiCard from '../../components/charts/KpiCard';
import './Homepage.css';

// --- Metric helpers (module-level, no side effects) ---

const calcAvg = (data, field, filterFn) => {
  if (!data?.length) return null;
  const filtered = filterFn ? data.filter(filterFn) : data;
  if (!filtered.length) return null;
  const sum = filtered.reduce((acc, row) => acc + (Number(row[field]) || 0), 0);
  return sum / filtered.length;
};

const calcSum = (data, field) => {
  if (!data?.length) return null;
  return data.reduce((acc, row) => acc + (Number(row[field]) || 0), 0);
};

const calcCountDistinct = (data, field) => {
  if (!data?.length) return null;
  const unique = new Set(data.map(row => row[field]).filter(Boolean));
  return unique.size;
};

const calcMode = (data, field) => {
  if (!data?.length) return null;
  const counts = {};
  data.forEach(row => {
    const val = row[field];
    if (val) counts[val] = (counts[val] || 0) + 1;
  });
  let maxCount = 0;
  let mode = null;
  Object.entries(counts).forEach(([val, count]) => {
    if (count > maxCount) { maxCount = count; mode = val; }
  });
  return mode;
};

const formatNumber = (val, decimals = 0) => {
  if (val == null || isNaN(val)) return '—';
  const abs = Math.abs(val);
  if (abs >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
  if (abs >= 10000) return `${Math.round(val / 1000)}K`;
  return Number(val).toLocaleString('en-US', { maximumFractionDigits: decimals });
};

// --- Date formatting for today's display ---
const formatTodayDate = () => {
  const now = new Date();
  return now.toLocaleDateString('en-GB', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric'
  });
};

// --- Category definitions ---
const CATEGORIES = [
  {
    id: 'reading',
    title: 'Reading',
    icon: BookOpen,
    description: 'Your reading journey',
    path: '/reading',
    accentColor: '#10B981',
    dataKey: 'readingBooks',
    trackingKey: 'topic_reading',
    trackingMessage: 'Last book finished'
  },
  {
    id: 'movies',
    title: 'Movies',
    icon: Film,
    description: 'Viewing habits and preferences',
    path: '/movies',
    accentColor: '#F59E0B',
    dataKey: 'movies',
    trackingKey: 'topic_movies',
    trackingMessage: 'Last watched'
  },
  {
    id: 'music',
    title: 'Music',
    icon: Music,
    description: 'Listening history and trends',
    path: '/music',
    accentColor: '#3B82F6',
    dataKey: 'music',
    trackingKey: 'topic_music',
    trackingMessage: 'Last listened'
  },
  {
    id: 'nutrition',
    title: 'Nutrition',
    icon: UtensilsCrossed,
    description: 'Meals and nutritional patterns',
    path: '/nutrition',
    accentColor: '#EF4444',
    dataKey: 'nutrition',
    trackingKey: 'topic_nutrition',
    trackingMessage: 'Last meal logged'
  },
  {
    id: 'podcasts',
    title: 'Podcasts',
    icon: Mic,
    description: 'Podcast listening insights',
    path: '/podcasts',
    accentColor: '#A78BFA',
    dataKey: 'podcasts',
    trackingKey: 'topic_podcasts',
    trackingMessage: 'Last episode'
  },
  {
    id: 'shows',
    title: 'TV Shows',
    icon: Tv,
    description: 'TV watching history',
    path: '/shows',
    accentColor: '#14B8A6',
    dataKey: 'shows',
    trackingKey: 'topic_shows',
    trackingMessage: 'Last watched'
  },
  {
    id: 'finance',
    title: 'Finance',
    icon: DollarSign,
    description: 'Income, expenses, and patterns',
    path: '/finance',
    accentColor: '#F97316',
    dataKey: 'finance',
    trackingKey: 'topic_finance',
    trackingMessage: 'Last transaction'
  },
  {
    id: 'health',
    title: 'Health',
    icon: Activity,
    description: 'Activity, sleep, and wellness',
    path: '/health',
    accentColor: '#EC4899',
    dataKey: 'healthDaily',
    trackingKey: 'topic_health',
    trackingMessage: 'Last activity'
  }
];

const Homepage = () => {
  usePageTitle('Dashboard');
  const { data, loading, fetchData } = useData();
  const { isPageAllowed } = useAuth();
  const [categoryStatuses, setCategoryStatuses] = useState({});
  const [isLoadingTracking, setIsLoadingTracking] = useState(true);
  const hasCalculated = useRef(false);

  // Fetch all data sources on mount
  useEffect(() => {
    if (typeof fetchData === 'function') {
      fetchData('readingBooks');
      fetchData('movies');
      fetchData('music');
      fetchData('podcasts');
      fetchData('nutrition');
      fetchData('healthDaily');
      fetchData('shows');
      fetchData('finance');
      fetchData('tracking');
    }
  }, [fetchData]);

  // Calculate tracking statuses once when tracking data loads
  useEffect(() => {
    if (!data?.tracking || hasCalculated.current) return;

    const formatDate = (dateString) => {
      if (!dateString) return null;
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return null;
      const day = String(date.getDate()).padStart(2, '0');
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const year = date.getFullYear();
      return `${day}/${month}/${year}`;
    };

    const getTrackingInfo = (sourceName) => {
      if (!data.tracking?.length) return { latestData: null };
      const record = data.tracking.find(r => r.source_name === sourceName);
      if (!record) return { latestData: null };
      return { latestData: formatDate(record.latest_data_date) };
    };

    const statuses = {};
    CATEGORIES.forEach(cat => {
      statuses[cat.id] = getTrackingInfo(cat.trackingKey);
    });

    setCategoryStatuses(statuses);
    setIsLoadingTracking(false);
    hasCalculated.current = true;
  }, [data?.tracking]);

  // --- Quick Stats: current-year filtered data ---
  const currentYear = new Date().getFullYear();
  const yearStr = String(currentYear);

  const currentYearBooks = useMemo(() => {
    if (!data?.readingBooks) return [];
    return data.readingBooks.filter(b => {
      const d = b.timestamp instanceof Date ? b.timestamp : new Date(b.timestamp);
      return d.getFullYear() === currentYear;
    });
  }, [data?.readingBooks, currentYear]);

  const currentYearMovies = useMemo(() => {
    if (!data?.movies) return [];
    return data.movies.filter(m => {
      const d = m.timestamp instanceof Date ? m.timestamp : new Date(m.timestamp);
      return d.getFullYear() === currentYear;
    });
  }, [data?.movies, currentYear]);

  const currentYearMusic = useMemo(() => {
    if (!data?.music) return [];
    return data.music.filter(m => m.timestamp && String(m.timestamp).startsWith(yearStr));
  }, [data?.music, yearStr]);

  const currentYearPodcasts = useMemo(() => {
    if (!data?.podcasts) return [];
    return data.podcasts.filter(p => {
      const d = p.timestamp instanceof Date ? p.timestamp : new Date(p.timestamp);
      return d.getFullYear() === currentYear;
    });
  }, [data?.podcasts, currentYear]);

  const quickStatsSources = useMemo(() => ({
    readingBooks: currentYearBooks,
    movies: currentYearMovies,
    music: currentYearMusic,
    podcasts: currentYearPodcasts,
    healthDaily: data?.healthDaily || []
  }), [currentYearBooks, currentYearMovies, currentYearMusic, currentYearPodcasts, data?.healthDaily]);

  const isQuickStatsLoading = loading?.readingBooks || loading?.movies || loading?.music || loading?.podcasts || loading?.healthDaily;

  // --- Per-category metrics ---
  const categoryMetrics = useMemo(() => {
    const metrics = {};

    if (data?.readingBooks?.length) {
      metrics.reading = {
        m1: { value: formatNumber(data.readingBooks.length), label: 'books' },
        m2: { value: formatNumber(calcAvg(data.readingBooks, 'pages_per_day', r => r.pages_per_day > 0)), label: 'pages/day' }
      };
    }

    if (data?.movies?.length) {
      metrics.movies = {
        m1: { value: formatNumber(data.movies.length), label: 'movies' },
        m2: { value: formatNumber(calcAvg(data.movies, 'rating', r => r.rating > 0), 1), label: 'avg rating' }
      };
    }

    if (data?.music?.length) {
      metrics.music = {
        m1: { value: formatNumber(calcCountDistinct(data.music, 'artist_name')), label: 'artists' },
        m2: { value: formatNumber(calcSum(data.music, 'listening_hours')), label: 'hours' }
      };
    }

    if (data?.nutrition?.length) {
      metrics.nutrition = {
        m1: { value: formatNumber(data.nutrition.length), label: 'meals logged' },
        m2: null
      };
    }

    if (data?.podcasts?.length) {
      metrics.podcasts = {
        m1: { value: formatNumber(data.podcasts.length), label: 'episodes' },
        m2: { value: formatNumber(calcSum(data.podcasts, 'listened_hours')), label: 'hours' }
      };
    }

    if (data?.shows?.length) {
      metrics.shows = {
        m1: { value: formatNumber(data.shows.length), label: 'episodes' },
        m2: { value: formatNumber(calcSum(data.shows, 'episode_runtime_hours')), label: 'hours' }
      };
    }

    if (data?.finance?.length) {
      metrics.finance = {
        m1: { value: formatNumber(data.finance.length), label: 'transactions' },
        m2: { value: calcMode(data.finance, 'category') || '—', label: 'top category' }
      };
    }

    if (data?.healthDaily?.length) {
      const avgSteps = calcAvg(data.healthDaily, 'total_steps', r => r.total_steps > 0);
      const avgSleepMin = calcAvg(data.healthDaily, 'total_sleep_minutes', r => r.total_sleep_minutes > 0);
      metrics.health = {
        m1: { value: formatNumber(avgSteps), label: 'avg steps' },
        m2: { value: avgSleepMin != null ? `${(avgSleepMin / 60).toFixed(1)}h` : '—', label: 'avg sleep' }
      };
    }

    return metrics;
  }, [data?.readingBooks, data?.movies, data?.music, data?.nutrition, data?.podcasts, data?.shows, data?.finance, data?.healthDaily]);

  // Filter categories by auth
  const visibleCategories = CATEGORIES.filter(cat => isPageAllowed(cat.path));

  const todayDate = useMemo(() => formatTodayDate(), []);

  return (
    <div className="homepage">
      <main className="homepage-main">
        <div className="homepage-container">
          {/* Section 1: Header */}
          <header className="homepage-header">
            <h1 className="homepage-title">LifeLog</h1>
            <p className="homepage-date">{todayDate}</p>
          </header>

          {/* Section 2: Quick Stats */}
          <section className="homepage-stats">
            <h2 className="section-label">This Year at a Glance</h2>
            <KPICardsPanel
              dataSources={quickStatsSources}
              loading={isQuickStatsLoading}
            >
              <KpiCard
                dataSource="readingBooks"
                metricOptions={{ label: 'Books Read', aggregation: 'count' }}
                icon={<BookOpen size={20} />}
              />
              <KpiCard
                dataSource="movies"
                metricOptions={{ label: 'Movies Watched', aggregation: 'count' }}
                icon={<Film size={20} />}
              />
              <KpiCard
                dataSource="music"
                metricOptions={{ label: 'Listening Hours', aggregation: 'sum', field: 'listening_hours', decimals: 0, compactNumbers: true }}
                icon={<Music size={20} />}
              />
              <KpiCard
                dataSource="podcasts"
                metricOptions={{ label: 'Podcast Hours', aggregation: 'sum', field: 'listened_hours', decimals: 0 }}
                icon={<Mic size={20} />}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{ label: 'Avg. Daily Steps', aggregation: 'average', field: 'total_steps', decimals: 0, compactNumbers: true, filterConditions: [{ field: 'total_steps', operator: '>', value: 0 }] }}
                icon={<Activity size={20} />}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{ label: 'Avg. Sleep', aggregation: 'average', field: 'total_sleep_minutes', decimals: 0, filterConditions: [{ field: 'total_sleep_minutes', operator: '>', value: 0 }] }}
                formatValue={(mins) => `${(mins / 60).toFixed(1)}h`}
                icon={<Moon size={20} />}
              />
            </KPICardsPanel>
          </section>

          {/* Section 3: Category Grid */}
          <section className="homepage-categories">
            <h2 className="section-label">Categories</h2>
            <div className="dashboard-grid">
              {visibleCategories.map((category) => {
                const IconComponent = category.icon;
                const metrics = categoryMetrics[category.id];
                const status = categoryStatuses[category.id];
                const isCatLoading = loading?.[category.dataKey] || !data?.[category.dataKey];

                return (
                  <Link
                    key={category.id}
                    to={category.path}
                    className={`dashboard-card${isCatLoading ? ' dashboard-card--loading' : ''}`}
                    style={{ '--card-accent': category.accentColor }}
                  >
                    <div className="dashboard-card-header">
                      <div className="dashboard-card-icon" style={{ backgroundColor: category.accentColor }}>
                        <IconComponent size={20} />
                      </div>
                      <div className="dashboard-card-title-group">
                        <h3 className="dashboard-card-title">{category.title}</h3>
                        <p className="dashboard-card-description">{category.description}</p>
                      </div>
                    </div>

                    <div className="dashboard-card-metrics">
                      {isCatLoading ? (
                        <>
                          <div className="dashboard-card-metric">
                            <span className="metric-value metric-skeleton">&nbsp;</span>
                            <span className="metric-label metric-skeleton">&nbsp;</span>
                          </div>
                          <div className="dashboard-card-metric">
                            <span className="metric-value metric-skeleton">&nbsp;</span>
                            <span className="metric-label metric-skeleton">&nbsp;</span>
                          </div>
                        </>
                      ) : (
                        <>
                          {metrics?.m1 && (
                            <div className="dashboard-card-metric">
                              <span className="metric-value">{metrics.m1.value}</span>
                              <span className="metric-label">{metrics.m1.label}</span>
                            </div>
                          )}
                          {metrics?.m2 && (
                            <div className="dashboard-card-metric">
                              <span className="metric-value">{metrics.m2.value}</span>
                              <span className="metric-label">{metrics.m2.label}</span>
                            </div>
                          )}
                        </>
                      )}
                    </div>

                    {!isLoadingTracking && status?.latestData && (
                      <div className="dashboard-card-footer">
                        <span className="dashboard-card-status">
                          <Clock size={12} />
                          {category.trackingMessage}: {status.latestData}
                        </span>
                      </div>
                    )}
                  </Link>
                );
              })}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default Homepage;
