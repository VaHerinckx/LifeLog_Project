import { useState, useEffect, useMemo, useCallback } from 'react';
import { Utensils, List, Grid, Calendar, MapPin, Tag, Coffee, Clock } from 'lucide-react';
import { useData } from '../../context/DataContext';
import { usePageTitle } from '../../hooks/usePageTitle';

// Import components
import MealDetails from './components/MealDetails';
import MealCard from './components/MealCard';

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
import KpiCard from '../../components/charts/KpiCard/index';

// Import chart components for analysis tab
import TimeSeriesBarChart from '../../components/charts/TimeSeriesBarChart';
import TopChart from '../../components/charts/TopChart';
import ProportionChart from '../../components/charts/ProportionChart';

// Helper function to parse time string to minutes for efficient sorting
const parseTimeToMinutes = (timeString) => {
  if (!timeString) return 0;
  const [hours, minutes] = timeString.split(':').map(Number);
  return (hours || 0) * 60 + (minutes || 0);
};

// Helper function to group nutrition items by meal_id
const groupItemsByMealId = (items) => {
  const mealMap = new Map();

  items.forEach(item => {
    const mealId = item.meal_id;

    if (!mealMap.has(mealId)) {
      // First occurrence of this meal_id - use this item's meal-level data
      mealMap.set(mealId, {
        meal_id: mealId,
        date: item.date,
        weekday: item.weekday,
        time: item.time,
        timeInMinutes: item.timeInMinutes,
        meal: item.meal,
        food_list: item.food_list,
        drinks_list: item.drinks_list,
        usda_meal_score: item.usda_meal_score,
        places: item.places,
        origin: item.origin,
        amount: item.amount,
        amount_text: item.amount_text,
        meal_assessment: item.meal_assessment,
        meal_assessment_text: item.meal_assessment_text,
      });
    }
  });

  return Array.from(mealMap.values());
};

const NutritionPage = () => {
  usePageTitle('Nutrition');
  console.log('🥗 NutritionPage component mounting/rendering');

  const { data, loading, error, fetchData } = useData();
  const [nutritionItems, setNutritionItems] = useState([]); // Item-level data
  const [filteredMeals, setFilteredMeals] = useState([]);
  const [filteredItems, setFilteredItems] = useState([]); // For ingredient-level KPIs
  const [isProcessing, setIsProcessing] = useState(true);

  const [viewMode, setViewMode] = useState('grid');
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [activeTab, setActiveTab] = useState('content');

  // Fetch nutrition data when component mounts
  useEffect(() => {
    console.log('🥗 NutritionPage useEffect - fetching data');

    if (typeof fetchData === 'function') {
      fetchData('nutrition');
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Process nutrition items: convert dates, parse numbers, pre-compute time values
  const processedItems = useMemo(() => {
    if (!data?.nutrition) return [];

    console.log('🥗 Raw nutrition data:', data.nutrition.slice(0, 3));

    const processed = data.nutrition.map(item => ({
      ...item,
      date: item.date ? new Date(item.date) : null,
      meal_id: String(item.meal_id), // Ensure meal_id is string for grouping
      food_quantity: parseFloat(item.food_quantity) || 0,
      drink_quantity: parseFloat(item.drink_quantity) || 0,
      usda_meal_score: parseFloat(item.usda_meal_score) || 0,
      meal_assessment: parseFloat(item.meal_assessment) || 0,
      amount: parseFloat(item.amount) || 0,
      timeInMinutes: parseTimeToMinutes(item.time), // Pre-compute for efficient sorting
    }));

    console.log('🥗 Processed items:', processed.slice(0, 3));
    return processed;
  }, [data?.nutrition]);

  // Group items into meals and sort (computed once, cached)
  const meals = useMemo(() => {
    if (!processedItems.length) return [];

    const groupedMeals = groupItemsByMealId(processedItems);

    // Sort meals by date and time (most recent first)
    groupedMeals.sort((a, b) => {
      // First compare dates
      const dateCompare = b.date - a.date;
      if (dateCompare !== 0) return dateCompare;

      // If dates are the same, compare pre-computed time values
      return b.timeInMinutes - a.timeInMinutes;
    });

    console.log('🥗 Grouped meals:', groupedMeals.length, 'meals from', processedItems.length, 'items');
    console.log('🥗 First 3 meals (sorted):', groupedMeals.slice(0, 3));

    return groupedMeals;
  }, [processedItems]);

  // Update state when meals/items change
  useEffect(() => {
    if (data?.nutrition) {
      setIsProcessing(true);
      setNutritionItems(processedItems);
      setFilteredMeals(meals);
      setFilteredItems(processedItems);
      setIsProcessing(false);
    }
  }, [processedItems, meals, data?.nutrition]);

  // Apply filters when FilteringPanel filters change
  // Wrapped in useCallback to prevent unnecessary re-renders
  const handleFiltersChange = useCallback((filteredDataSources) => {
    console.log('🥗 Filters changed:', filteredDataSources);

    const filteredNutritionItems = filteredDataSources.nutrition || [];

    // Group filtered items by meal_id and sort
    const groupedFilteredMeals = groupItemsByMealId(filteredNutritionItems);

    // Sort filtered meals by date and time (most recent first)
    groupedFilteredMeals.sort((a, b) => {
      const dateCompare = b.date - a.date;
      if (dateCompare !== 0) return dateCompare;
      return b.timeInMinutes - a.timeInMinutes;
    });

    console.log('🥗 Filtered meals:', groupedFilteredMeals.length);

    setFilteredMeals(groupedFilteredMeals);
    setFilteredItems(filteredNutritionItems); // Store filtered items for KPI calculations
  }, []);

  const handleMealClick = useCallback((meal) => {
    setSelectedMeal(meal);
  }, []);

  const handleCloseDetails = useCallback(() => {
    setSelectedMeal(null);
  }, []);

  // Memoize data object to prevent FilteringPanel re-renders
  const filterPanelData = useMemo(() => ({
    nutrition: nutritionItems // Pass item-level data for filtering
  }), [nutritionItems]);

  return (
    <PageWrapper
      error={error?.nutrition}
      errorTitle="Nutrition Tracker"
    >
      {/* Page Header */}
      <PageHeader
        title="Nutrition Tracker"
        description="Track meals, food choices, and nutritional patterns across breakfast, lunch, dinner, and snacks"
      />

        {!loading?.nutrition && !isProcessing && (
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
                label="Meal Date"
                field="date"
                icon={<Calendar />}
                dataSources={['nutrition']}
              />
              <Filter
                type="multiselect"
                label="Meal Year"
                field="nutrition_year"
                icon={<Clock />}
                placeholder="Select years"
                dataSources={['nutrition']}
              />
              <Filter
                type="multiselect"
                label="Meal Type"
                field="meal"
                icon={<Utensils />}
                placeholder="Select meal types"
                dataSources={['nutrition']}
              />
              <Filter
                type="multiselect"
                label="Location"
                field="places"
                icon={<MapPin />}
                placeholder="Select locations"
                dataSources={['nutrition']}
              />
              <Filter
                type="multiselect"
                label="Origin"
                field="origin"
                icon={<Tag />}
                placeholder="Select meal origins"
                dataSources={['nutrition']}
              />
              <Filter
                type="hierarchical"
                selectionMode="multi"
                label="Food Category"
                field="food_category"
                childField="food"
                icon={<Utensils />}
                placeholder="Select food categories"
                dataSources={['nutrition']}
                showCounts={false}
              />
              <Filter
                type="hierarchical"
                selectionMode="multi"
                label="Drink Category"
                field="drink_category"
                childField="drink"
                icon={<Coffee />}
                placeholder="Select drink categories"
                dataSources={['nutrition']}
                showCounts={false}
              />
            </FilteringPanel>

            {/* Statistics Cards with KpiCard children */}
            <KPICardsPanel
              dataSources={{
                nutrition: filteredMeals, // Use meal-level data for meal-based KPIs
                nutritionItems: filteredItems // Use item-level data for ingredient KPIs
              }}
              loading={loading?.nutrition}
            >
              <KpiCard
                dataSource="nutrition"
                metricOptions={{
                  label: 'Total Meals/Occurrences',
                  aggregation: 'count_distinct',
                  field: 'meal_id'
                }}
                icon={<Utensils />}
              />
              <KpiCard
                dataSource="nutritionItems"
                metricOptions={{
                  label: 'Total Food Items',
                  aggregation: 'sum',
                  field: 'food_quantity',
                  decimals: 0
                }}
                icon={<Utensils />}
              />
              <KpiCard
                dataSource="nutritionItems"
                metricOptions={{
                  label: 'Total Drink Items',
                  aggregation: 'sum',
                  field: 'drink_quantity',
                  decimals: 0
                }}
                icon={<Utensils />}
              />
              <KpiCard
                dataSource="nutrition"
                metricOptions={{
                  label: 'Avg Meal Score',
                  aggregation: 'average',
                  field: 'usda_meal_score',
                  decimals: 2
                }}
                icon={<Tag />}
              />
              <KpiCard
                dataSource="nutrition"
                metricOptions={{
                  label: 'Avg Meal Taste',
                  aggregation: 'average',
                  field: 'meal_assessment',
                  decimals: 1
                }}
                icon={<Tag />}
              />
            </KPICardsPanel>

            {/* Tab Navigation */}
            <TabNavigation
              contentLabel="Meals"
              contentIcon={Utensils}
              activeTab={activeTab}
              onTabChange={setActiveTab}
            />
          </>
        )}

        {/* Meals Tab Content */}
        {activeTab === 'content' && (
          <ContentTab
            loading={loading?.nutrition || isProcessing}
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            viewModes={[
              { mode: 'grid', icon: Grid },
              { mode: 'list', icon: List }
            ]}
            items={filteredMeals}
            loadingIcon={Utensils}
            emptyState={{
              icon: Utensils,
              title: "No meals found",
              message: "No meals match your current filters. Try adjusting your criteria."
            }}
            sortOptions={[
              { value: 'date', label: 'Date', type: 'date' },
              { value: 'usda_meal_score', label: 'USDA Score', type: 'number' }
            ]}
            defaultSortField="date"
            defaultSortDirection="desc"
            renderGrid={(meals) => (
              <ContentCardsGroup
                items={meals}
                viewMode="grid"
                renderItem={(meal) => (
                  <MealCard
                    key={meal.meal_id}
                    meal={meal}
                    viewMode="grid"
                    onClick={handleMealClick}
                  />
                )}
              />
            )}
            renderList={(meals) => (
              <ContentCardsGroup
                items={meals}
                viewMode="list"
                renderItem={(meal) => (
                  <MealCard
                    key={`list-${meal.meal_id}`}
                    meal={meal}
                    viewMode="list"
                    onClick={handleMealClick}
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
                  data={filteredItems}
                  dateColumnName="date"
                  metricOptions={[
                    { value: 'meals', label: 'Number of Meals', aggregation: 'count_distinct', field: 'meal_id', decimals: 0 },
                    { value: 'ingredient occurrences', label: 'Number of Food Occurrences', aggregation: 'sum', field: 'food_quantity', decimals: 0 },
                    { value: 'drink occurrences', label: 'Number of Drink Occurrences', aggregation: 'sum', field: 'drink_quantity', decimals: 0 },
                    { value: 'meal score', label: 'Avg Meal Score', aggregation: 'average', field: 'usda_meal_score', decimals: 1 },
                    { value: 'meal taste', label: 'Avg Meal Taste', aggregation: 'average', field: 'meal_assessment', decimals: 1 },
                    { value: 'amount', label: 'Avg Amount', aggregation: 'average', field: 'amount', decimals: 1 }
                  ]}
                  defaultMetric="meals"
                  title="Meals Over Time"
                />
                <TopChart
                    data={filteredItems}
                    dimensionOptions={[
                      { value: 'food_list', label: 'Meal', field: 'food_list', labelFields: ['food_list'] },
                      { value: 'food_category', label: 'Food Category', field: 'food_category', labelFields: ['food_category'] },
                      { value: 'food', label: 'Food', field: 'food', labelFields: ['food'] },
                      { value: 'drink_category', label: 'Drink Category', field: 'drink_category', labelFields: ['drink_category'] },
                      { value: 'drink', label: 'Drink', field: 'drink', labelFields: ['drink'] },
                      { value: 'places', label: 'Location', field: 'places', labelFields: ['places'] },
                      { value: 'origin', label: 'Origin', field: 'origin', labelFields: ['origin'] },
                    ]}
                    metricOptions={[
                      { value: 'occurences', label: 'Occurences', aggregation: 'count_distinct', field: 'meal_id', suffix: ' times', decimals: 0 },
                      { value: 'f_quantity', label: 'Food Quantity', aggregation: 'sum', field: 'food_quantity', suffix: ' times', decimals: 0 },
                      { value: 'd_quantity', label: 'Drink Quantity', aggregation: 'sum', field: 'drink_quantity', suffix: ' times', decimals: 0 },
                      { value: 'meal_score', label: 'Meal Score', aggregation: 'average', field: 'usda_meal_score', decimals: 1 },
                      { value: 'meal_taste', label: 'Meal Taste', aggregation: 'average', field: 'meal_assessment', decimals: 1 }
                    ]}
                    defaultDimension="food"
                    defaultMetric="occurences"
                    title="Top Food Analysis"
                    topN={10}
                    imageField="cover_url"
                    enableTopNControl={true}
                    topNOptions={[5, 10, 15, 20, 25, 30]}
                    enableSortToggle={true}
                    scrollable={true}
                    barHeight={50}
                />
                <ProportionChart
                  data={filteredMeals}
                  dimensionOptions={[
                    { value: 'meal', label: 'Meal Type', field: 'meal' },
                    { value: 'places', label: 'Location', field: 'places' },
                    { value: 'origin', label: 'Origin', field: 'origin' },
                    { value: 'weekday', label: 'Weekday', field: 'weekday' }
                  ]}
                  metricOptions={[
                    { value: 'meals', label: 'Meals', aggregation: 'count', field: 'meal_id' },
                    { value: 'meal_score', label: 'Avg Meal Score', aggregation: 'average', field: 'usda_meal_score', decimals: 1 }
                  ]}
                  defaultDimension="meal"
                  defaultMetric="meals"
                  title="Meal Distribution"
                  maxCategories={8}
                  showPercentages={true}
                />
              </>
            )}
          />
        )}

      {/* Meal Details Modal */}
      {selectedMeal && (
        <MealDetails meal={selectedMeal} onClose={handleCloseDetails} />
      )}
    </PageWrapper>
  );
};

export default NutritionPage;
