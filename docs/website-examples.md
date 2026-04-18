# Website Implementation Examples

## Standard Imports

```javascript
import { formatDate, parseDate, isDateInRange, applyFilters, calculateStats } from '../../utils';
import { StarRating, FilteringPanel, CardsPanel, ContentCardsGroup } from '../../components/ui/';
import { readingFilterConfigs } from '../../config/filterConfigs';
```

## Pagination

Handled automatically by `ContentCardsGroup` — triggers at >200 items, 100/page default.

**Rule:** Pass the full filtered array to `ContentTab.items`. Never create `currentPage` state in the parent.

```jsx
<ContentTab
  items={filteredData}
  renderGrid={(items) => (
    <ContentCardsGroup
      items={items}
      renderItem={(item) => <ItemCard item={item} onClick={handleClick} />}
    />
  )}
/>
```

## AnalysisTab

```jsx
<AnalysisTab
  renderCharts={() => (
    <>
      <TimeSeriesBarChart data={filteredData} ... />
      <IntensityHeatmap data={filteredHourlyData} ... />
    </>
  )}
/>
```

Rules:
- `renderCharts` returns chart elements only — no titles, filters, or descriptions
- Each chart takes its own `data` prop directly
- Auto-wraps each chart in `analysis-chart-section` divs

## Creating New Visualization Pages

### 1. Preparation
- Use `docs/new_page_template.txt` as starting point
- Analyze CSV columns and relationships
- Decide if data needs grouping (item-level → aggregated)

### 2. Implementation Rules
- Base strictly on `ReadingPage.jsx` — follow exact component structure
- Never create page-specific CSS files or AnalysisTab components
- All styling via CSS variables only — no hardcoded values
- Card component must support grid/list views
- Detail modal: gradient background on icon container, surface colors from `variables.css`

### 3. Data Handling
- Group with `Map` using a unique identifier
- Sort: date descending, then time descending
- Join multiple text fields with ` | ` separator

### 4. Integration Checklist
1. Add `VITE_[TYPE]_FILE_ID` to `.env`
2. Update `src/config/config.js`
3. Add case to `DataContext` fetchData switch
4. Add route to `App.jsx`
5. Add nav item to `NavigationBar.jsx`
6. Add category card to `Homepage.jsx`

## Reference Files

- **Gold Standard page:** `src/pages/Reading/ReadingPage.jsx`
- **Filter Configs:** `src/config/filterConfigs.jsx`
- **Design tokens:** `src/styles/variables.css` + `src/styles/tokens/`
- **Reusable components:** `src/components/ui/`
- **Page template:** `docs/new_page_template.txt`
