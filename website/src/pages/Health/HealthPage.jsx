import { useState, useEffect, useMemo } from 'react';
import { Activity, List, Grid, Calendar, Moon, Heart, Flag, Building2, Store, Clock } from 'lucide-react';
import { useData } from '../../context/DataContext';
import { usePageTitle } from '../../hooks/usePageTitle';

// Import components
import HealthDetails from './components/HealthDetails';
import HealthCard from './components/HealthCard';

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

const HealthPage = () => {
  usePageTitle('Health');
  const { data, loading, error, fetchData } = useData();
  const [healthDays, setHealthDays] = useState([]);
  const [filteredHealthDays, setFilteredHealthDays] = useState([]);
  const [healthHourly, setHealthHourly] = useState([]);
  const [filteredHealthHourly, setFilteredHealthHourly] = useState([]);
  const [isProcessing, setIsProcessing] = useState(true);

  const [viewMode, setViewMode] = useState('grid');
  const [selectedDay, setSelectedDay] = useState(null);
  const [activeTab, setActiveTab] = useState('content');

  // Fetch health data when component mounts
  useEffect(() => {
    if (typeof fetchData === 'function') {
      fetchData('healthDaily');
      fetchData('healthHourly');
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Process health daily data when it's loaded
  useEffect(() => {
    if (data?.healthDaily) {
      setIsProcessing(true);

      // Convert date strings to Date objects for JavaScript date operations
      // Filter out entries with invalid dates
      const processedDays = data.healthDaily
        .map(day => ({
          ...day,
          date: day.date ? new Date(day.date) : null
        }))
        .filter(day => day.date !== null && !isNaN(day.date.getTime()));

      // ContentTab now handles sorting internally
      setHealthDays(processedDays);
      setFilteredHealthDays(processedDays);
      setIsProcessing(false);
      }
  }, [data?.healthDaily]);

  // Process hourly health data when it's loaded
  useEffect(() => {
    if (data?.healthHourly) {
      // Convert date strings to Date objects and create datetime for heatmap compatibility
      const processedHours = data.healthHourly
        .map(segment => {
          const dateObj = segment.date ? new Date(segment.date) : null;
          // Create datetime by combining date and hour
          let datetime = null;
          if (dateObj && !isNaN(dateObj.getTime())) {
            datetime = new Date(dateObj);
            datetime.setHours(segment.hour || 0, 0, 0, 0);
          }
          return {
            ...segment,
            date: dateObj,
            datetime: datetime
          };
        })
        .filter(segment => segment.date !== null && !isNaN(segment.date.getTime()));

      setHealthHourly(processedHours);
      setFilteredHealthHourly(processedHours);
    }
  }, [data?.healthHourly]);

  // Apply filters when FilteringPanel filters change
  const handleFiltersChange = (filteredDataSources) => {
    // ContentTab now handles sorting internally
    setFilteredHealthDays(filteredDataSources.healthDaily || []);
    setFilteredHealthHourly(filteredDataSources.healthHourly || []);
  };

  const handleCardClick = (day) => {
    setSelectedDay(day);
  };

  const handleCloseDetails = () => {
    setSelectedDay(null);
  };

  // Memoize data object to prevent FilteringPanel re-renders
  // Include all data sources so they can all be filtered consistently
  const filterPanelData = useMemo(() => ({
    healthDaily: healthDays,
    healthHourly: healthHourly
  }), [healthDays, healthHourly]);

  return (
    <PageWrapper
      error={error?.healthDaily}
      errorTitle="Health Tracker"
    >
      {/* Page Header */}
      <PageHeader
        title="Health"
        description="Track your daily activity, sleep patterns, and overall wellness"
      />

        {!loading?.healthDaily && !isProcessing && (
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
                label="Date"
                field="date"
                fieldMap={{
                  healthDaily: 'date',
                  healthHourly: 'datetime'
                }}
                icon={<Calendar />}
                dataSources={['healthDaily', 'healthHourly']}
              />
              <Filter
                type="multiselect"
                label="Year"
                field="health_year"
                fieldMap={{
                  healthDaily: 'health_year',
                  healthHourly: 'health_year'
                }}
                icon={<Clock />}
                dataSources={['healthDaily', 'healthHourly']}
              />
              <Filter
                type="multiselect"
                label="Sleep Quality"
                field="sleep_quality_text"
                icon={<Moon />}
                placeholder="Select quality"
                dataSources={['healthDaily']}
              />
              <Filter
                type="multiselect"
                label="Overall Evaluation"
                field="overall_evaluation"
                icon={<Heart />}
                placeholder="Select rating"
                dataSources={['healthDaily']}
              />
              <Filter
                type="multiselect"
                label="Country"
                field="country"
                icon={<Flag />}
                placeholder="Select country"
                dataSources={['healthHourly']}
              />
              <Filter
                type="multiselect"
                label="City"
                field="city"
                icon={<Building2 />}
                placeholder="Select city"
                dataSources={['healthHourly']}
              />
              <Filter
                type="multiselect"
                label="Place"
                field="place_name"
                icon={<Store />}
                placeholder="Select visited place(s)"
                dataSources={['healthHourly']}
              />
            </FilteringPanel>

            {/* Statistics Cards with KpiCard children */}
            <KPICardsPanel
              dataSources={{
                healthDaily: filteredHealthDays
              }}
              loading={loading?.healthDaily}
            >
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{
                  label: 'Days Tracked',
                  aggregation: 'count'
                }}
                icon={<Calendar />}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{
                  label: 'Avg. Daily Steps',
                  aggregation: 'average',
                  field: 'total_steps',
                  decimals: 0,
                  filterConditions: [{ field: 'total_steps', operator: '>', value: 0 }]
                }}
                icon={<Activity />}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{
                  label: 'Avg. Active Energy (kcal)',
                  aggregation: 'average',
                  field: 'total_active_energy_kcal',
                  decimals: 0,
                  filterConditions: [{ field: 'total_active_energy_kcal', operator: '>', value: 0 }]
                }}
                icon={<Heart />}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{
                  label: 'Avg. Sleep (min)',
                  aggregation: 'average',
                  field: 'total_sleep_minutes',
                  decimals: 0,
                  filterConditions: [{ field: 'total_sleep_minutes', operator: '>', value: 0 }]
                }}
                icon={<Moon />}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{
                  label: 'Avg. Sleep Start',
                  aggregation: 'average',
                  field: 'sleep_start_time_minutes',
                  decimals: 0,
                  filterConditions: [{ field: 'sleep_start_time_minutes', operator: '>', value: 0 }]
                }}
                icon={<Moon />}
                formatValue={(mins) => {
                  const hours = Math.floor(mins / 60);
                  const minutes = Math.round(mins % 60);
                  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
                }}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{
                  label: 'Avg. Overall Feeling',
                  aggregation: 'average',
                  field: 'overall_evaluation',
                  decimals: 1,
                  filterConditions: [{ field: 'overall_evaluation', operator: '>', value: 0 }]
                }}
                icon={<Heart />}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{
                  label: 'Avg. Sleep Quality',
                  aggregation: 'average',
                  field: 'sleep_quality',
                  decimals: 1,
                  filterConditions: [{ field: 'sleep_quality', operator: '>', value: 0 }]
                }}
                icon={<Moon />}
              />
              <KpiCard
                dataSource="healthDaily"
                metricOptions={{
                  label: 'Avg. Fitness Feeling',
                  aggregation: 'average',
                  field: 'fitness_feeling',
                  decimals: 1,
                  filterConditions: [{ field: 'fitness_feeling', operator: '>', value: 0 }]
                }}
                icon={<Activity />}
              />
            </KPICardsPanel>

            {/* Tab Navigation */}
            <TabNavigation
              contentLabel="Daily Logs"
              contentIcon={Activity}
              activeTab={activeTab}
              onTabChange={setActiveTab}
            />
          </>
        )}

        {/* Daily Logs Tab Content */}
        {activeTab === 'content' && (
          <ContentTab
            loading={loading?.healthDaily || isProcessing}
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            viewModes={[
              { mode: 'grid', icon: Grid },
              { mode: 'list', icon: List }
            ]}
            items={filteredHealthDays}
            loadingIcon={Activity}
            emptyState={{
              icon: Activity,
              title: "No health data found",
              message: "No health data matches your current filters. Try adjusting your criteria."
            }}
            sortOptions={[
              { value: 'date', label: 'Date', type: 'date' },
              { value: 'total_steps', label: 'Steps', type: 'number' },
              { value: 'sleep_quality', label: 'Sleep Quality', type: 'number' },
              { value: 'total_sleep_minutes', label: 'Sleep Duration', type: 'number' },
              { value: 'fitness_feeling', label: 'Fitness Feeling', type: 'number' }
            ]}
            defaultSortField="date"
            defaultSortDirection="desc"
            renderGrid={(days) => (
              <ContentCardsGroup
                items={days}
                viewMode="grid"
                renderItem={(day) => (
                  <HealthCard
                    key={day.date.toISOString()}
                    day={day}
                    viewMode="grid"
                    onClick={handleCardClick}
                  />
                )}
              />
            )}
            renderList={(days) => (
              <ContentCardsGroup
                items={days}
                viewMode="list"
                renderItem={(day) => (
                  <HealthCard
                    key={`list-${day.date.toISOString()}`}
                    day={day}
                    viewMode="list"
                    onClick={handleCardClick}
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
                  data={filteredHealthDays}
                  dateColumnName="date"
                  metricOptions={[
                    { value: 'steps', label: 'Daily Steps', aggregation: 'average', field: 'total_steps', decimals: 0 },
                    { value: 'sleep', label: 'Sleep Duration (min)', aggregation: 'average', field: 'total_sleep_minutes', decimals: 0 },
                    { value: 'sleep quality', label: 'Sleep Quality (1-5)', aggregation: 'average', field: 'sleep_quality', decimals: 1 },
                    { value: 'sleep rest feeling', label: 'Rest Feeling (1-5)', aggregation: 'average', field: 'sleep_rest_feeling', decimals: 1 },
                    { value: 'stresslevel', label: 'Stress Level', aggregation: 'average', field: 'daily_stress_avg', decimals: 0 },
                    { value: 'screentime', label: 'Screen Time (min)', aggregation: 'average', field: 'total_screen_time_minutes', decimals: 0 },
                    { value: 'screentime_before_sleep', label: 'Screen Time before sleep (min)', aggregation: 'average', field: 'total_screen_time_minutes_before_sleep', decimals: 0 },
                    { value: 'energy', label: 'Active Energy (kcal)', aggregation: 'average', field: 'total_active_energy_kcal', decimals: 0 },
                    { value: 'fitness', label: 'Fitness Feeling (1-5)', aggregation: 'average', field: 'fitness_feeling', decimals: 1},
                    { value: 'evaluation', label: 'Day Score (1-5)', aggregation: 'average', field: 'overall_evaluation', decimals: 1},
                    { value: 'visits', label: 'Unique Places Visited', aggregation: 'count_distinct', field: 'place_name', decimals: 0, data: filteredHealthHourly, dateColumnName: 'datetime', filterConditions: [{ field: 'date', operator: '>=', value: '2025-03-01'}] },
                  ]}
                  defaultMetric="steps"
                  title="Health Metrics by Period"
                />
                <IntensityHeatmap
                  data={filteredHealthHourly}
                  dateColumnName="datetime"
                  treatMidnightAsUnknown={false}
                  metricOptions={[
                    { value: 'steps', label: 'Steps', field: 'steps', aggregation: 'sum', decimals: 0, compactNumbers: false },
                    { value: 'active_energy', label: 'Active Energy (kcal)', field: 'active_energy_kcal', aggregation: 'sum', decimals: 0, compactNumbers: true },
                    { value: 'screen_time', label: 'Screen Time (min)', field: 'screen_time_minutes', aggregation: 'sum', decimals: 0 },
                    { value: 'heart_rate', label: 'Avg Heart Rate (bpm)', field: 'avg_heart_rate', aggregation: 'average', decimals: 0 },
                    { value: 'sleep', label: 'Sleep (min)', field: 'sleep_minutes', aggregation: 'sum', decimals: 0, compactNumbers: true },
                    { value: 'distance', label: 'Distance (m)', field: 'distance_meters', aggregation: 'sum', decimals: 0, compactNumbers: true },
                    { value: 'pickups', label: 'Phone Pickups', field: 'phone_pickups', aggregation: 'sum', decimals: 0 }
                  ]}
                  defaultMetric="steps"
                  rowAxis="time_period"
                  columnAxis="weekday"
                  showAxisSwap={true}
                  title="Health Activity by Time of Day"
                />
                <TopChart
                  data={filteredHealthHourly}
                  dimensionOptions={[
                    { value: 'place_name', label: 'Place', field: 'place_name', labelFields: ['place_name'] },
                    { value: 'time_period', label: 'Time of Day', field: 'time_period', labelFields: ['time_period'] },
                    { value: 'hour', label: 'Hour', field: 'hour', labelFields: ['hour'] },
                  ]}
                  metricOptions={[
                    { value: 'steps', label: 'Steps', field: 'steps', aggregation: 'sum', decimals: 0, compactNumbers: false },
                    { value: 'duration', label: 'Duration (hours)', field: 'segment_duration_hours', aggregation: 'sum', decimals: 0, compactNumbers: false },
                    { value: 'active_energy', label: 'Active Energy (kcal)', field: 'active_energy_kcal', aggregation: 'sum', decimals: 0, compactNumbers: true },
                    { value: 'screen_time', label: 'Screen Time (min)', field: 'screen_time_minutes', aggregation: 'sum', decimals: 0 },
                    { value: 'screentime_before_sleep', label: 'Screen Time before sleep (min)', field: 'screen_time_minutes_before_sleep', aggregation: 'sum', decimals: 0},
                    { value: 'heart_rate', label: 'Avg Heart Rate (bpm)', field: 'avg_heart_rate', aggregation: 'average', decimals: 0 },
                    { value: 'sleep', label: 'Sleep (min)', field: 'sleep_minutes', aggregation: 'sum', decimals: 0, compactNumbers: true },
                    { value: 'distance', label: 'Distance (m)', field: 'distance_meters', aggregation: 'sum', decimals: 0, compactNumbers: true },
                    { value: 'pickups', label: 'Phone Pickups', field: 'phone_pickups', aggregation: 'sum', decimals: 0 },
                    { value: 'visits', label: 'Days visited', field: 'date', aggregation: 'count_distinct', decimals: 0 },
                  ]}
                  defaultDimension="place_name"
                  defaultMetric="steps"
                  title="Top Health Analysis"
                  topN={10}
                  enableTopNControl={true}
                  topNOptions={[5, 10, 15, 20, 25, 30]}
                  enableSortToggle={true}
                  scrollable={true}
                  barHeight={50}
                />
                <ProportionChart
                  data={filteredHealthHourly}
                  dimensionOptions={[
                    { value: 'time_period', label: 'Time of Day', field: 'time_period' },
                    { value: 'country', label: 'Country', field: 'country' },
                    { value: 'city', label: 'City', field: 'city' },
                    { value: 'place_name', label: 'Place', field: 'place_name' }
                  ]}
                  metricOptions={[
                    { value: 'steps', label: 'Steps', aggregation: 'sum', field: 'steps', decimals: 0 },
                    { value: 'duration', label: 'Hours', aggregation: 'sum', field: 'segment_duration_hours', suffix: ' hrs', decimals: 1 },
                    { value: 'active_energy', label: 'Active Energy', aggregation: 'sum', field: 'active_energy_kcal', suffix: ' kcal', decimals: 0 }
                  ]}
                  defaultDimension="time_period"
                  defaultMetric="steps"
                  title="Activity Distribution"
                  maxCategories={8}
                  showPercentages={true}
                />
              </>
            )}
          />
        )}

      {/* Health Details Modal */}
      {selectedDay && (
        <HealthDetails
          day={selectedDay}
          hourlyData={healthHourly.filter(h => {
            if (!h.date || !selectedDay.date) return false;
            // Compare dates - handle both Date objects and strings
            const hDate = h.date instanceof Date ? h.date : new Date(h.date);
            const sDate = selectedDay.date instanceof Date ? selectedDay.date : new Date(selectedDay.date);
            return hDate.toDateString() === sDate.toDateString();
          })}
          onClose={handleCloseDetails}
        />
      )}
    </PageWrapper>
  );
};

export default HealthPage;
