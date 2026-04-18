import { useState, useEffect, useMemo } from 'react';
import { Grid, List, Calendar, Star, User, Tv, Clock } from 'lucide-react';
import { useData } from '../../context/DataContext';
import { usePageTitle } from '../../hooks/usePageTitle';

// Import components
import EpisodeDetails from './components/EpisodeDetails';
import EpisodeCard from './components/EpisodeCard';

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

// Import utilities
import { sortByDateSafely } from '../../utils/sortingUtils';

const ShowsPage = () => {
  usePageTitle('TV Shows');
  const { data, loading, error, fetchData } = useData();
  const [episodes, setEpisodes] = useState([]);
  const [filteredEpisodes, setFilteredEpisodes] = useState([]);

  const [viewMode, setViewMode] = useState('grid');
  const [selectedEpisode, setSelectedEpisode] = useState(null);
  const [activeTab, setActiveTab] = useState('content');

  // Fetch shows data when component mounts
  useEffect(() => {
    if (typeof fetchData === 'function') {
      fetchData('shows');
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Process episodes data when it's loaded
  useEffect(() => {
    if (data?.shows) {
      // Convert timestamp strings to Date objects for JavaScript date operations
      const processedEpisodes = data.shows.map(episode => ({
        ...episode,
        watched_at: episode.watched_at ? new Date(episode.watched_at) : null
      }));

      // Sort by most recent first
      const sortedEpisodes = sortByDateSafely(processedEpisodes, 'watched_at');

      setEpisodes(sortedEpisodes);
      setFilteredEpisodes(sortedEpisodes);
      // Reset content ready state when new data arrives
      }
  }, [data?.shows]);

  // Apply filters when FilteringPanel filters change
  const handleFiltersChange = (filteredDataSources) => {
    // Re-sort filtered data (most recent first)
    const sortedEpisodes = sortByDateSafely(filteredDataSources.shows || [], 'watched_at');

    setFilteredEpisodes(sortedEpisodes);
  };

  const handleEpisodeClick = (episode) => {
    setSelectedEpisode(episode);
  };

  const handleCloseDetails = () => {
    setSelectedEpisode(null);
  };

  // Memoize data object to prevent FilteringPanel re-renders
  const filterPanelData = useMemo(() => ({
    shows: episodes
  }), [episodes]);

  return (
    <PageWrapper
      error={error?.shows}
      errorTitle="TV Shows Tracker"
    >
      {/* Page Header */}
      <PageHeader
        title="TV Shows Tracker"
        description="Track your TV show watching history with detailed episode information and viewing patterns"
      />

        {!(loading?.shows) && (
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
                field="watched_at"
                icon={<Calendar />}
                dataSources={['shows']}
              />
              <Filter
                type="multiselect"
                label="Watch Year"
                field="watch_year"
                icon={<Clock />}
                placeholder="Select years"
                dataSources={['shows']}
              />
              <Filter
                type="multiselect"
                label="Show Title"
                field="show_title"
                icon={<Tv />}
                placeholder="Select shows"
                dataSources={['shows']}
              />
              <Filter
                type="multiselect"
                label="Show Rating"
                field="show_rating"
                icon={<Star />}
                placeholder="Select rating"
                dataSources={['shows']}
                sortType="numeric"
              />
              <Filter
                type="multiselect"
                label="Season Rating"
                field="season_rating"
                icon={<Star />}
                placeholder="Select rating"
                dataSources={['shows']}
                sortType="numeric"
              />
              <Filter
                type="multiselect"
                label="Episode Rating"
                field="episode_rating"
                icon={<Star />}
                placeholder="Select rating"
                dataSources={['shows']}
                sortType="numeric"
              />
              <Filter
                type="multiselect"
                label="Cast"
                field="episode_cast"
                icon={<User />}
                placeholder="Search cast members"
                searchPlaceholder="Search actors..."
                delimiter=","
                matchMode="contains"
                dataSources={['shows']}
              />
              <Filter
                type="multiselect"
                label="Director"
                field="episode_director"
                icon={<User />}
                placeholder="Search director"
                searchPlaceholder="Search director..."
                delimiter=","
                matchMode="contains"
                dataSources={['shows']}
              />
              <Filter
                type="multiselect"
                label="Writer"
                field="episode_writer"
                icon={<User />}
                placeholder="Search writer"
                searchPlaceholder="Search writer..."
                delimiter=","
                matchMode="contains"
                dataSources={['shows']}
              />
            </FilteringPanel>

            {/* Statistics Cards with KpiCard children */}
            <KPICardsPanel
              dataSources={{
                shows: filteredEpisodes
              }}
              loading={loading?.shows}
            >
              <KpiCard
                dataSource="shows"
                metricOptions={{
                  label: 'Episodes Watched',
                  aggregation: 'count'
                }}
                icon={<Tv />}
              />
              <KpiCard
                dataSource="shows"
                metricOptions={{
                  label: 'Unique Shows',
                  aggregation: 'count_distinct',
                  field: 'show_title'
                }}
                icon={<Tv />}
              />
              <KpiCard
                dataSource="shows"
                metricOptions={{
                  label: 'Seasons Watched',
                  aggregation: 'count_distinct',
                  field: 'season_show_id'
                }}
                icon={<Tv />}
              />
              <KpiCard
                dataSource="shows"
                metricOptions={{
                  label: 'Runtime (h)',
                  aggregation: 'sum',
                  field: 'episode_runtime_hours',
                  decimals: 0
                }}
                icon={<Tv />}
              />
            </KPICardsPanel>

            {/* Tab Navigation */}
            <TabNavigation
              contentLabel="Episodes"
              contentIcon={Tv}
              activeTab={activeTab}
              onTabChange={setActiveTab}
            />
          </>
        )}

        {/* Episodes Tab Content */}
        {activeTab === 'content' && (
          <ContentTab
            loading={loading?.shows}
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            viewModes={[
              { mode: 'grid', icon: Grid },
              { mode: 'list', icon: List }
            ]}
            items={filteredEpisodes}
            loadingIcon={Tv}
            emptyState={{
              icon: Tv,
              title: "No episodes found",
              message: "No episodes match your current filters. Try adjusting your criteria."
            }}
            renderGrid={(episodes) => (
              <ContentCardsGroup
                items={episodes}
                viewMode="grid"
                renderItem={(episode) => (
                  <EpisodeCard
                    key={episode.watch_id}
                    episode={episode}
                    viewMode="grid"
                    onClick={handleEpisodeClick}
                  />
                )}
              />
            )}
            renderList={(episodes) => (
              <ContentCardsGroup
                items={episodes}
                viewMode="list"
                renderItem={(episode) => (
                  <EpisodeCard
                    key={`list-${episode.watch_id}`}
                    episode={episode}
                    viewMode="list"
                    onClick={handleEpisodeClick}
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
                  data={filteredEpisodes}
                  dateColumnName="watched_at"
                  metricOptions={[
                    { value: 'episodes', label: 'Episodes Watched', aggregation: 'count', field: 'watch_id', decimals: 0 },
                    { value: 'runtime', label: 'Runtime', aggregation: 'sum', field: 'episode_runtime_hours', decimals: 1 },
                  ]}
                  defaultMetric="episodes"
                  title="Shows Watch Over Time"
                />
                <IntensityHeatmap
                  data={filteredEpisodes}
                  dateColumnName="watched_at"
                  metricOptions={[
                    { value: 'episodes', label: 'Episodes', field: 'watch_id', aggregation: 'count_distinct', suffix: ' episodes' },
                    { value: 'episode_runtime_hours', label: 'Watch Time', field: 'episode_runtime_hours', aggregation: 'sum', suffix: ' hours', decimals: 1 }
                  ]}
                  rowAxis="time_period"    // Time periods as rows
                  columnAxis="weekday"     // Days as columns
                  defaultMetric="episodes"
                  showAxisSwap={true}  // Hide the swap button
                  title="TV Activity"
                />
                <TopChart
                  data={filteredEpisodes}
                  dimensionOptions={[
                    { value: 'show_title', label: 'Show', field: 'show_title', labelFields: ['show_title'] },
                    { value: 'show_year', label: 'Release Year', field: 'show_year', labelFields: ['show_year'] },
                    { value: 'episode_cast', label: 'Actor', field: 'episode_cast', labelFields: ['episode_cast'], delimiter: ',' },
                    { value: 'episode_director', label: 'Director', field: 'episode_director', labelFields: ['episode_director'], delimiter: ',' },
                    { value: 'episode_writer', label: 'Writer', field: 'episode_writer', labelFields: ['episode_writer'], delimiter: ',' }
                  ]}
                  metricOptions={[
                    { value: 'watch_id', label: 'Episodes', aggregation: 'count_distinct', field: 'watch_id', suffix: ' episodes', decimals: 0 },
                    { value: 'episode_runtime_hours', label: 'Runtime', aggregation: 'sum', field: 'episode_runtime_hours', suffix: ' hours', decimals: 1 },
                    { value: 'episode_rating', label: 'Rating', aggregation: 'average', field: 'episode_rating', suffix: '★', decimals: 1 }
                  ]}
                  defaultDimension="show_title"
                  defaultMetric="watch_id"
                  title="Top Shows Analysis"
                  topN={10}
                  imageField="cover_url"
                  enableTopNControl={true}
                  topNOptions={[5, 10, 15, 20, 25, 30]}
                  enableSortToggle={true}
                  scrollable={true}
                  barHeight={50}
                />
              </>
            )}
          />
        )}

      {/* Episode Details Modal */}
      {selectedEpisode && (
        <EpisodeDetails episode={selectedEpisode} onClose={handleCloseDetails} />
      )}
    </PageWrapper>
  );
};

export default ShowsPage;
