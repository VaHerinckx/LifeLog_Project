import { useState, useEffect, useMemo } from 'react';
import { Film, Grid, List, Calendar, Tag, Star, User, Clock, Award } from 'lucide-react';
import { useData } from '../../context/DataContext';
import { usePageTitle } from '../../hooks/usePageTitle';

// Import components
import MovieDetails from './components/MovieDetails';
import MovieCard from './components/MovieCard';

// Import standardized UI components
import {
  FilteringPanel,
  Filter,
  PageWrapper,
  PageHeader,
  TabNavigation,
  ContentTab,
  AnalysisTab,
  KPICardsPanel,
  ContentCardsGroup
} from '../../components/ui';
import KpiCard from '../../components/charts/KpiCard';

// Import chart components for analysis tab
import TimeSeriesBarChart from '../../components/charts/TimeSeriesBarChart';
import IntensityHeatmap from '../../components/charts/IntensityHeatmap';
import TopChart from '../../components/charts/TopChart';
import ProportionChart from '../../components/charts/ProportionChart';

const MoviesPage = () => {
  usePageTitle('Movies');
  const { data, loading, error, fetchData } = useData();
  const [movies, setMovies] = useState([]);
  const [filteredMovies, setFilteredMovies] = useState([]);
  const [isProcessing, setIsProcessing] = useState(true);

  const [viewMode, setViewMode] = useState('grid');
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [activeTab, setActiveTab] = useState('content');

  // Fetch movies data when component mounts
  useEffect(() => {
    if (typeof fetchData === 'function') {
      fetchData('movies');
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Process movies data when it's loaded
  useEffect(() => {
    if (data?.movies) {
      setIsProcessing(true);

      // Log data info
      if (data.movies.length > 0) {
        const columns = Object.keys(data.movies[0]);
        console.log('📊 Movies data loaded');
        console.log('   Columns:', columns.join(', '));
      }

      // Group by movie_id to get unique movies (since genres are now in separate rows)
      const moviesByIdMap = new Map();

      data.movies.forEach((row) => {
        const movieId = row.movie_id;
        if (!moviesByIdMap.has(movieId)) {
          // First time seeing this movie - create the movie object
          // Only convert date if it's not null/empty
          const parsedDate = row.date && row.date.trim() !== '' ? new Date(row.date) : null;

          moviesByIdMap.set(movieId, {
            ...row,
            id: movieId,
            date: parsedDate,
            rating: row.rating ? parseFloat(row.rating) : 0,
            year: row.year ? parseInt(row.year, 10) : null,
            runtime: row.runtime ? parseInt(row.runtime, 10) : null,
            genres: [row.genre] // Start collecting genres
          });
        } else {
          // We've seen this movie before - just add this genre to the list
          const existingMovie = moviesByIdMap.get(movieId);
          if (row.genre && !existingMovie.genres.includes(row.genre)) {
            existingMovie.genres.push(row.genre);
          }
        }
      });

      // Convert map to array and detect rewatches
      const seenMovies = new Set();
      const processedMovies = Array.from(moviesByIdMap.values()).map((movie) => {
        const movieKey = `${movie.name}-${movie.year}`;
        const isRewatch = seenMovies.has(movieKey);
        seenMovies.add(movieKey);

        // Parse cast field into array of individual actors
        const castArray = movie.cast
          ? movie.cast.split(',').map(actor => actor.trim()).filter(Boolean)
          : [];

        return {
          ...movie,
          isRewatch,
          // Keep original genre field for display, add genres array for filtering
          genreArray: movie.genres,
          // Add cast array for individual actor filtering
          castArray: castArray
        };
      });

      // Sort by date (most recent first) and log last movie date
      const sortedMovies = processedMovies.sort((a, b) => {
        if (!a.date) return 1;
        if (!b.date) return -1;
        return b.date - a.date;
      });

      setMovies(sortedMovies);
      setFilteredMovies(sortedMovies);
      setIsProcessing(false);
      }
  }, [data?.movies]);

  // Apply filters when FilteringPanel filters change
  // FilteringPanel returns pre-filtered data per source
  const handleFiltersChange = (filteredDataSources) => {
    // ContentTab now handles sorting internally
    setFilteredMovies(filteredDataSources.movies || []);
  };

  const handleMovieClick = (movie) => {
    setSelectedMovie(movie);
  };

  const handleCloseDetails = () => {
    setSelectedMovie(null);
  };

  // Memoize data object to prevent FilteringPanel re-renders
  const filterPanelData = useMemo(() => ({
    movies: movies
  }), [movies]);

  return (
    <PageWrapper
      error={error?.movies}
      errorTitle="Movies & TV Tracker"
    >
      {/* Page Header */}
      <PageHeader
        title="Movies & TV Tracker"
        description="Track your viewing habits and discover insights about your movie preferences"
      />

      {!loading?.movies && !isProcessing && (
        <>
          {/* FilteringPanel with Filter children */}
          <FilteringPanel
            data={filterPanelData}
            onFiltersChange={handleFiltersChange}
            renderMode="bubble"
            bubbleConfig={{
              maxVisible: 10,
              searchThreshold: 15,
              showCounts: false
            }}
          >
            <Filter
              type="daterange"
              label="Watch Date"
              field="date"
              icon={<Calendar />}
              dataSources={['movies']}
            />
            <Filter
                type="multiselect"
                label="Watch Year"
                field="watch_year"
                icon={<Clock />}
                placeholder="Select years"
                dataSources={['movies']}
            />
            <Filter
              type="numberrange"
              label="Release Year"
              field="year"
              icon={<Film />}
              placeholder="Select year range"
              dataSources={['movies']}
            />
            <Filter
              type="multiselect"
              label="Genres"
              field="genre"
              icon={<Tag />}
              placeholder="Select genres"
              dataSources={['movies']}
            />


            <Filter
              type="multiselect"
              label="Director"
              field="director"
              icon={<User />}
              placeholder="Select directors"
              dataSources={['movies']}
            />
            <Filter
              type="multiselect"
              label="Cast"
              field="cast"
              icon={<User />}
              placeholder="Search cast members"
              searchPlaceholder="Search actors..."
              delimiter=","
              matchMode="contains"
              dataSources={['movies']}
            />
            <Filter
              type="multiselect"
              label="Language"
              field="original_language"
              icon={<Award />}
              placeholder="Select Languages"
              dataSources={['movies']}
            />
            <Filter
              type="multiselect"
              label="Rating"
              field="rating"
              icon={<Star />}
              placeholder="Select ratings"
              dataSources={['movies']}
            />
          </FilteringPanel>

          {/* Statistics Cards with KpiCard children */}
          <KPICardsPanel
            dataSources={{
              movies: filteredMovies
            }}
            loading={loading?.movies}
          >
            <KpiCard
              dataSource="movies"
              metricOptions={{
                label: 'Movies Watched',
                aggregation: 'count'
              }}
              icon={<Film />}
            />
            <KpiCard
              dataSource="movies"
              metricOptions={{
                label: 'Average Rating',
                aggregation: 'average',
                field: 'rating',
                decimals: 1
              }}
              icon={<Star />}
            />
            <KpiCard
              dataSource="movies"
              metricOptions={{
                label: 'Avg Runtime (min)',
                aggregation: 'average',
                field: 'runtime',
                decimals: 0,
                suffix: ' min'
              }}
              icon={<Clock />}
            />
            <KpiCard
              dataSource="movies"
              metricOptions={{
                label: 'Total Runtime (h)',
                aggregation: 'sum',
                field: 'runtime_hour',
                decimals: 1,
                suffix: ' h'
              }}
              icon={<Clock />}
            />
            <KpiCard
              dataSource="movies"
              metricOptions={{
                label: 'Directors',
                aggregation: 'count_distinct',
                field: 'director',
                decimals: 0
              }}
              icon={<Star />}
            />
          </KPICardsPanel>

          {/* Tab Navigation */}
          <TabNavigation
            contentLabel="Movies"
            contentIcon={Film}
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        </>
      )}

      {/* Movies Tab Content */}
      {activeTab === 'content' && (
        <ContentTab
          loading={loading?.movies || isProcessing}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          viewModes={[
            { mode: 'grid', icon: Grid },
            { mode: 'list', icon: List }
          ]}
          items={filteredMovies}
          loadingIcon={Film}
          emptyState={{
            icon: Film,
            title: "No movies found",
            message: "No movies match your current filters. Try adjusting your criteria."
          }}
          sortOptions={[
            { value: 'date', label: 'Watch Date', type: 'date' },
            { value: 'name', label: 'Title', type: 'string' },
            { value: 'year', label: 'Release Year', type: 'number' },
            { value: 'rating', label: 'Rating', type: 'number' },
            { value: 'runtime', label: 'Runtime', type: 'number' }
          ]}
          defaultSortField="date"
          defaultSortDirection="desc"
          renderGrid={(movies) => (
            <ContentCardsGroup
              items={movies}
              viewMode="grid"
              renderItem={(movie) => (
                <MovieCard
                  key={movie.id}
                  movie={movie}
                  viewMode="grid"
                  onClick={handleMovieClick}
                />
              )}
            />
          )}
          renderList={(movies) => (
            <ContentCardsGroup
              items={movies}
              viewMode="list"
              renderItem={(movie) => (
                <MovieCard
                  key={`list-${movie.id}`}
                  movie={movie}
                  viewMode="list"
                  onClick={handleMovieClick}
                />
              )}
            />
          )}
        />
      )}

      {/* Analysis Tab Content */}
      {activeTab === 'analysis' && (
        <AnalysisTab
          renderCharts={() => (
            <>
              <TimeSeriesBarChart
                data={filteredMovies}
                dateColumnName="date"
                metricOptions={[
                  { value: 'movies', label: 'Movies', aggregation: 'count_distinct', field: 'movie_id', decimals: 0 },
                  { value: 'runtime_hours', label: 'Runtime', aggregation: 'sum', field: 'runtime_hour', decimals: 1 },
                  { value: 'avgRating', label: 'Avg Rating', aggregation: 'average', field: 'rating', suffix: '★', decimals: 1 }
                ]}
                defaultMetric="count"
                title="Movies Over Time"
              />
              <IntensityHeatmap
                  data={filteredMovies}
                  dateColumnName="date"
                  valueColumnName="runtime"
                  title="Reading Activity by Day and Time"
                  treatMidnightAsUnknown={true}
                />
              <TopChart
                data={filteredMovies}
                dimensionOptions={[
                  { value: 'genre', label: 'Genre', field: 'genre', labelFields: ['genre'] },
                  { value: 'release_year', label: 'Release Year', field: 'year', labelFields: ['year'] },
                  { value: 'original_language', label: 'Language', field: 'original_language', labelFields: ['original_language'] },
                  { value: 'name', label: 'Movie', field: 'name', labelFields: ['name'] },
                  { value: 'director', label: 'Director', field: 'director', labelFields: ['director'] },
                  { value: 'cast', label: 'Cast', field: 'cast', labelFields: ['cast'], delimiter: ',' },
                ]}
                metricOptions={[
                  { value: 'movies', label: 'Movies', aggregation: 'count_distinct', field: 'movie_id', suffix: ' movies', decimals: 0 },
                  { value: 'runtime', label: 'Runtime', aggregation: 'sum', field: 'runtime_hour', suffix: ' hours', decimals: 0},
                  { value: 'avgRating', label: 'Avg Rating', aggregation: 'average', field: 'rating', suffix: '★', decimals: 1 },
                ]}
                defaultDimension="genre"
                defaultMetric="movies"
                title="Top Movies Analysis"
                topN={10}
                imageField="poster_url"
                enableTopNControl={true}
                topNOptions={[5, 10, 15, 20, 25, 30]}
                enableSortToggle={true}
                scrollable={true}
                barHeight={50}
              />
              <ProportionChart
                data={filteredMovies}
                dimensionOptions={[
                  { value: 'genre', label: 'Genre', field: 'genre' },
                  { value: 'original_language', label: 'Language', field: 'original_language' },
                  { value: 'director', label: 'Director', field: 'director' },
                  { value: 'year', label: 'Release Year', field: 'year' }
                ]}
                metricOptions={[
                  { value: 'runtime', label: 'Watch Time', aggregation: 'sum', field: 'runtime_hour', suffix: ' hrs', decimals: 1 },
                  { value: 'movies', label: 'Movies', aggregation: 'count', field: 'movie_id' }
                ]}
                defaultDimension="genre"
                defaultMetric="runtime"
                title="Viewing Distribution"
                maxCategories={8}
                showPercentages={true}
              />
            </>
          )}
        />
      )}

      {/* Movie Details Modal */}
      {selectedMovie && (
        <MovieDetails movie={selectedMovie} onClose={handleCloseDetails} />
      )}
    </PageWrapper>
  );
};

export default MoviesPage;
