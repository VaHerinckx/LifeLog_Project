import { useState, useEffect, useMemo, useCallback } from 'react';
import { Mic, Headphones, List, Grid, Clock, Calendar, Tag, Globe, Sparkles, Repeat, TrendingUp } from 'lucide-react';
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
import ProportionChart from '../../components/charts/ProportionChart';

const PodcastPage = () => {
  usePageTitle('Podcasts');
  const { data, loading, error, fetchData } = useData();
  const [podcasts, setPodcasts] = useState([]);
  const [filteredPodcasts, setFilteredPodcasts] = useState([]);
  const [isProcessing, setIsProcessing] = useState(true);

  const [viewMode, setViewMode] = useState('grid');
  const [selectedEpisode, setSelectedEpisode] = useState(null);
  const [activeTab, setActiveTab] = useState('content');

  // Fetch podcast data when component mounts
  useEffect(() => {
    console.log('🎙️ PodcastPage useEffect - fetching data');

    if (typeof fetchData === 'function') {
      fetchData('podcasts');
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Process podcast data when it's loaded
  useEffect(() => {
    if (data?.podcasts) {
      setIsProcessing(true);

      // Convert timestamp strings to Date objects for JavaScript date operations
      const processedPodcasts = data.podcasts.map(episode => ({
        ...episode,
        listened_date: episode.listened_date ? new Date(episode.listened_date) : null,
        published_date: episode.published_date ? new Date(episode.published_date) : null
      }));

      // Data already sorted by Python, but set in state
      setPodcasts(processedPodcasts);
      setFilteredPodcasts(processedPodcasts);
      setIsProcessing(false);
      }
  }, [data?.podcasts]);

  // Apply filters when FilteringPanel filters change
  // FilteringPanel now returns pre-filtered data per source!
  const handleFiltersChange = (filteredDataSources) => {
    // ContentTab now handles sorting internally
    setFilteredPodcasts(filteredDataSources.podcasts || []);
  };

  const handleEpisodeClick = useCallback((episode) => {
    setSelectedEpisode(episode);
  }, []);

  const handleCloseDetails = useCallback(() => {
    setSelectedEpisode(null);
  }, []);

  // Memoize data object to prevent FilteringPanel re-renders
  const filterPanelData = useMemo(() => ({
    podcasts: podcasts
  }), [podcasts]);

  return (
    <PageWrapper
      error={error?.podcasts}
      errorTitle="Podcast Tracker"
    >
      {/* Page Header */}
      <PageHeader
        title="Podcast Tracker"
        description="Track your podcast listening habits and discover insights about your episodes"
      />

        {!loading?.podcasts && !isProcessing && (
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
                label="Listened Date"
                field="listened_date"
                icon={<Calendar />}
                dataSources={['podcasts']}
              />
              <Filter
                type="multiselect"
                label="Listened Year"
                field="listened_year"
                icon={<Clock />}
                placeholder="Select years"
                dataSources={['podcasts']}
              />
              <Filter
                type="multiselect"
                label="Podcast"
                field="podcast_name"
                icon={<Tag />}
                placeholder="Select podcast"
                dataSources={['podcasts']}
              />
              <Filter
                type="multiselect"
                label="Host"
                field="artist"
                icon={<Tag />}
                placeholder="Select host"
                dataSources={['podcasts']}
              />
              <Filter
                type="multiselect"
                label="Genres"
                field="genre"
                icon={<Tag />}
                placeholder="Select genres"
                dataSources={['podcasts']}
              />
              <Filter
                type="multiselect"
                label="Languages"
                field="language"
                icon={<Globe />}
                placeholder="Select languages"
                dataSources={['podcasts']}
              />
              <Filter
                type="multiselect"
                label="New Recurring Podcast"
                field="is_new_recurring_podcast"
                icon={<Globe />}
                placeholder="Select option"
                dataSources={['podcasts']}
              />
              <Filter
                type="multiselect"
                label="Recurring Podcast"
                field="is_recurring_podcast"
                icon={<Globe />}
                placeholder="Select option"
                dataSources={['podcasts']}
              />
            </FilteringPanel>

            {/* Statistics Cards with KpiCard children */}
            <KPICardsPanel
              dataSources={{
                podcasts: filteredPodcasts
              }}
              loading={loading?.podcasts}
            >
              <KpiCard
                dataSource="podcasts"
                metricOptions={{
                  label: 'Podcasts',
                  aggregation: 'count_distinct',
                  field: 'podcast_id'
                }}
                icon={<Mic />}
              />
              <KpiCard
                dataSource="podcasts"
                metricOptions={{
                  label: 'Episodes',
                  aggregation: 'count'
                }}
                icon={<Headphones />}
              />
              <KpiCard
                dataSource="podcasts"
                metricOptions={{
                  label: 'Hours Listened',
                  aggregation: 'sum',
                  field: 'listened_hours',
                  decimals: 0
                }}
                icon={<Clock />}
              />
              <KpiCard
                dataSource="podcasts"
                metricOptions={{
                  label: 'Avg Completion',
                  aggregation: 'average',
                  field: 'completion_percent',
                  decimals: 0,
                  suffix: '%'
                }}
                icon={<TrendingUp />}
              />
              <KpiCard
                dataSource="podcasts"
                metricOptions={{
                  label: 'Recurring Podcasts',
                  aggregation: 'count_distinct',
                  field: 'podcast_id',
                  filterConditions: [{ field: 'is_recurring_podcast', value: 'Yes' }]
                }}
                icon={<Repeat />}
              />
              <KpiCard
                dataSource="podcasts"
                metricOptions={{
                  label: 'New Recurring Podcasts',
                  aggregation: 'count_distinct',
                  field: 'podcast_id',
                  filterConditions: [{ field: 'is_new_recurring_podcast', value: 'Yes' }]
                }}
                icon={<Sparkles />}
              />
            </KPICardsPanel>

            {/* Tab Navigation */}
            <TabNavigation
              contentLabel="Episodes"
              contentIcon={Headphones}
              activeTab={activeTab}
              onTabChange={setActiveTab}
            />
          </>
        )}

        {/* Episodes Tab Content */}
        {activeTab === 'content' && (
          <ContentTab
            loading={loading?.podcasts || isProcessing}
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            viewModes={[
              { mode: 'grid', icon: Grid },
              { mode: 'list', icon: List }
            ]}
            items={filteredPodcasts}
            loadingIcon={Mic}
            emptyState={{
              icon: Headphones,
              title: "No episodes found",
              message: "No episodes match your current filters. Try adjusting your criteria."
            }}
            sortOptions={[
              { value: 'listened_date', label: 'Listen Date', type: 'date' },
              { value: 'published_date', label: 'Publish Date', type: 'date' },
              { value: 'episode_title', label: 'Episode Title', type: 'string' },
              { value: 'podcast_title', label: 'Podcast Title', type: 'string' },
              { value: 'duration', label: 'Duration', type: 'number' }
            ]}
            defaultSortField="listened_date"
            defaultSortDirection="desc"
            renderGrid={(episodes) => (
              <ContentCardsGroup
                items={episodes}
                viewMode="grid"
                renderItem={(episode) => (
                  <EpisodeCard
                    key={episode.episode_uuid}
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
                    key={`list-${episode.episode_uuid}`}
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
                  data={filteredPodcasts}
                  dateColumnName="listened_date"
                  metricOptions={[
                    { value: 'podcast_count', label: 'Podcasts', aggregation: 'count_distinct', field: 'podcast_id', decimals: 0 },
                    { value: 'episode_count', label: 'Episodes', aggregation: 'count_distinct', field: 'episode_uuid', decimals: 0 },
                    { value: 'listening_time', label: 'Hours', aggregation: 'sum', field: 'listened_hours', decimals: 1 },
                    { value: 'avg_completion', label: 'Average Completion Rate', aggregation: 'average', field: 'completion_percent', decimals: 1 },
                  ]}
                  defaultMetric="podcast_count"
                  title="Podcast Listening Activity"
                />
                <IntensityHeatmap
                  data={filteredPodcasts}
                  dateColumnName="listened_date"
                  valueColumnName="listened_hours"
                  title="Podcast Activity by Day and Time"
                  treatMidnightAsUnknown={true}
                />
                <TopChart
                  data={filteredPodcasts}
                  dimensionOptions={[
                    { value: 'podcast_name', label: 'Podcast', field: 'podcast_name', labelFields: ['podcast'] },
                    { value: 'genre', label: 'Genre', field: 'genre', labelFields: ['genre'] },
                    { value: 'artist', label: 'Host', field: 'artist', labelFields: ['artist'] },
                    { value: 'language', label: 'Language', field: 'language', labelFields: ['language'] },
                    { value: 'episode_title', label: 'Episode', field: 'episode_title', labelFields: ['episode_title'] },
                    { value: 'recurring', label: 'Recurring Podcasts', field: 'is_recurring_podcast', labelFields: ['is_recurring_podcast'] }
                  ]}
                  metricOptions={[
                    { value: 'listened_hours', label: 'Hours', aggregation: 'sum', field: 'listened_hours', suffix: ' hours', decimals: 0 },
                    { value: 'episodes', label: 'Episodes', aggregation: 'count_distinct', field: 'episode_uuid', suffix: ' episodes', decimals: 0 },
                  ]}
                  defaultDimension="podcast_name"
                  defaultMetric="listened_hours"
                  title="Top Podcast Analysis"
                  topN={10}
                  imageField="artwork_url"
                  enableTopNControl={true}
                  topNOptions={[5, 10, 15, 20, 25, 30]}
                  enableSortToggle={true}
                  scrollable={true}
                  barHeight={50}
                />
                <ProportionChart
                  data={filteredPodcasts}
                  dimensionOptions={[
                    { value: 'genre', label: 'Genre', field: 'genre' },
                    { value: 'podcast_name', label: 'Podcast', field: 'podcast_name' },
                    { value: 'artist', label: 'Host', field: 'artist' },
                    { value: 'language', label: 'Language', field: 'language' },
                    { value: 'is_recurring_podcast', label: 'Recurring', field: 'is_recurring_podcast' }
                  ]}
                  metricOptions={[
                    { value: 'listened_hours', label: 'Listening Hours', aggregation: 'sum', field: 'listened_hours', suffix: ' hrs', decimals: 1 },
                    { value: 'episodes', label: 'Episodes', aggregation: 'count', field: 'episode_uuid' }
                  ]}
                  defaultDimension="genre"
                  defaultMetric="listened_hours"
                  title="Listening Distribution"
                  maxCategories={8}
                  showPercentages={true}
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

export default PodcastPage;
