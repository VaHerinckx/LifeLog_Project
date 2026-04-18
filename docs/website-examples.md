# Website Implementation Examples

Detailed code examples and workflows for building LifeLog website pages.

## Table of Contents
- [Component Usage Examples](#component-usage-examples)
- [Pagination Pattern](#pagination-pattern)
- [AnalysisTab Component](#analysistab-component)
- [Creating New Visualization Pages](#creating-new-visualization-pages)

## Component Usage Examples

### Standard Utilities
```javascript
import { formatDate, parseDate, isValidDate, isDateInRange } from '../../utils';
import { applyDateRangeFilter, applyMultiSelectFilter, applyFilters } from '../../utils';
import { calculateSum, calculateAverage, calculateStats, formatNumber } from '../../utils';
```

### Component Imports
```javascript
import { StarRating, FilteringPanel, CardsPanel } from '../../components/ui/';
import { readingFilterConfigs, filterBuilders } from '../../config/filterConfigs';
```

## Pagination Pattern

### How It Works

**Pagination is handled automatically by the `ContentCardsGroup` component. DO NOT implement manual pagination logic in page components.**

**Data Flow:**
- `ContentTab` receives full unfiltered data array from parent page
- `ContentTab` passes data to `ContentCardsGroup` via `renderGrid`/`renderList` functions
- `ContentCardsGroup` automatically enables pagination for datasets >200 items
- Default: 100 items per page (configurable: 25, 50, 100, 200)
- Pagination UI appears below cards when dataset exceeds items-per-page threshold

### Implementation Pattern

```jsx
<ContentTab
  items={filteredData}  // Pass FULL array - ContentCardsGroup handles pagination
  viewMode={viewMode}
  renderGrid={(items) => (
    <ContentCardsGroup
      items={items}  // ContentCardsGroup automatically paginates
      viewMode="grid"
      renderItem={(item) => <ItemCard item={item} onClick={handleClick} />}
    />
  )}
/>
```

### Key Rules

1. **Parent Page**: Pass full filtered array to `ContentTab.items`
2. **ContentTab**: Passes full array to render functions
3. **ContentCardsGroup**: Handles pagination internally (slicing, page controls)
4. **NO Manual State**: Don't create `currentPage`, `itemsPerPage` state in parent
5. **Performance**: Pagination activates automatically for large datasets (>200 items)

### Pagination Controls

- First/Previous/Next/Last buttons
- Items-per-page dropdown (25, 50, 100, 200)
- Auto-scroll to top on page change
- Shows "Showing X-Y of Z items"

**Reference:** See `ReadingPage.jsx` for gold standard implementation (passes full `filteredBooks` array to `ContentTab`)

## AnalysisTab Component

### Component API

```jsx
<AnalysisTab
  renderCharts={() => (                // Render function for charts (required, no parameters)
    <>
      <TimeSeriesBarChart data={filteredData} ... />
      <IntensityHeatmap data={filteredHourlyData} ... />
    </>
  )}
/>
```

### Key Rules

1. **Content**: Charts only - no titles, descriptions, or filter panels
2. **Data**: Each chart specifies its own data prop directly (not passed from AnalysisTab)
3. **Filtering**: All logic handled by parent page components (FilteringPanel)
4. **Auto-wrapping**: AnalysisTab automatically wraps each chart in `analysis-chart-section` divs
5. **No Loading State**: Data loaded at page level before tab switch
6. **Multi-Source Support**: Charts can use different filtered data sources via their `data` prop or via `metricOptions` with per-metric data overrides

## Creating New Visualization Pages

### Workflow Overview

**CRITICAL**: Follow this workflow when creating new data visualization pages.

### 1. Preparation Phase

- Use `new_page_template.txt` as starting point
- Analyze CSV data structure to understand columns and relationships
- Determine if data needs grouping (item-level → aggregated)
- Identify unique identifiers for grouping
- Remove any existing data handling for this type from DataContext before starting

### 2. Template Filling

- Define filters based on both item-level and aggregated-level fields if applicable
- Configure KPI cards with appropriate data sources (use dual sources if needed: aggregated for counts, item-level for sums)
- Design card preview showing essential information
- Plan detail modal with complete information display
- Specify analysis charts (can start with placeholder TimeSeriesBarChart)

### 3. Implementation Requirements

- **Base strictly on ReadingPage.jsx** - Follow exact component structure and patterns
- **Never create**: Page-specific CSS files or AnalysisTab components (use reusable components only)
- **Card Component**: Must support grid/list views, use only CSS variables from variables.css
- **Details Modal**: Apply gradient background to icon container, use proper surface colors from variables.css
- **Styling**: Never hardcode values - always use CSS variables (spacing, colors, typography, borders, shadows)

### 4. Data Handling Patterns

- **Grouping**: Use Map with unique identifier to aggregate item-level data
- **Dual Sources**: Pass both aggregated and item-level data to KPICardsPanel when needed
- **Sorting**: Always sort by date descending, then time descending (most recent first)
- **Text Combination**: Join multiple text fields with ` | ` separator
- **Date/Time Parsing**: Convert time strings to minutes for comparison

### 5. Component Structure

- **Card Component**: Separate functions for formatting, truncation, combination logic
- **CSS**: Import variables.css, define base styles, grid/list variants, common elements, responsive adjustments
- **Modal**: Follow BookDetails pattern with gradient icon background, light purple tints for sections
- **Display Logic**: Show all relevant fields (don't hide information like drinks when foods are present)

### 6. Integration Steps

1. Add environment variable for Drive file ID
2. Update config.js with new data type
3. Add case to DataContext fetchData switch
4. Add route to App.jsx
5. Add navigation item to NavigationBar.jsx
6. Add category card to Homepage.jsx

## Reference Files

- **Gold Standard**: `src/pages/Reading/ReadingPage.jsx`
- **Filter Configs**: `src/config/filterConfigs.jsx`
- **Design System**: `src/styles/variables.css`
- **Reusable Components**: `src/components/ui/`
