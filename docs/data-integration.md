# Data Integration Guide

Guide for integrating new data sources into the LifeLog system.

## Table of Contents
- [Adding New Data Sources](#adding-new-data-sources)
- [Data Sources to Pages Mapping](#data-sources-to-pages-mapping)
- [Impact Testing Requirements](#impact-testing-requirements)

## Adding New Data Sources

### Quick Setup (3 Steps)

#### Step 1: Environment Variable

Add to `.env` file:

```bash
VITE_[DATATYPE]_FILE_ID=your_google_drive_file_id
```

**Example:**
```bash
VITE_FINANCE_FILE_ID=1a2b3c4d5e6f7g8h9i0j
```

#### Step 2: Config File

Update `src/config/config.js`:

```javascript
export const DRIVE_FILES = {
  // Existing entries...
  DATATYPE: import.meta.env.VITE_DATATYPE_FILE_ID,
};
```

**Example:**
```javascript
export const DRIVE_FILES = {
  MUSIC: import.meta.env.VITE_MUSIC_FILE_ID,
  READING: import.meta.env.VITE_READING_FILE_ID,
  FINANCE: import.meta.env.VITE_FINANCE_FILE_ID,  // New entry
};
```

#### Step 3: DataContext

Update `src/context/DataContext.jsx`:

**Add to initialData:**
```javascript
const initialData = {
  music: null,
  reading: null,
  datatype: null,  // Add new data type
};
```

**Add to fetchData switch:**
```javascript
case 'datatype':
  fileId = DRIVE_FILES.DATATYPE;
  break;
```

**Complete Example:**
```javascript
const fetchData = async (dataType) => {
  let fileId;
  switch (dataType) {
    case 'music':
      fileId = DRIVE_FILES.MUSIC;
      break;
    case 'reading':
      fileId = DRIVE_FILES.READING;
      break;
    case 'finance':  // New case
      fileId = DRIVE_FILES.FINANCE;
      break;
    default:
      return;
  }
  // Fetch logic...
};
```

### Data Format Requirements

**File Format:**
- Delimiter: Pipe (`|`)
- Encoding: UTF-8
- Headers: First row
- Dates: ISO 8601 format (YYYY-MM-DD)

**Example CSV:**
```
date|category|amount|description
2024-01-15|Food|25.50|Grocery shopping
2024-01-16|Transport|15.00|Bus ticket
```

### Usage in Pages

```javascript
import { useData } from '../../context/DataContext';

function MyPage() {
  const { data, fetchData } = useData();

  useEffect(() => {
    fetchData('datatype');
  }, [fetchData]);

  const myData = data.datatype;

  // Use data...
}
```

## Data Sources to Pages Mapping

### Active Sources (Implemented)

| Category | CSV Files | Page Component | Location |
|----------|-----------|----------------|----------|
| Music | `lfm_processed.csv` | MusicPage.jsx | `src/pages/Music/` |
| Reading | `kindle_gr_processed.csv` | ReadingPage.jsx | `src/pages/Reading/` |
| Podcasts | `pocket_casts_processed.csv` | PodcastPage.jsx | `src/pages/Podcast/` |
| Movies/TV | `letterboxd_processed.csv`, `trakt_processed.csv` | MoviesPage.jsx | `src/pages/Movies/` |
| Nutrition | `nutrilio_processed.csv` | NutritionPage.jsx | `src/pages/Nutrition/` |
| Health | `apple_processed.csv`, `garmin_*_processed.csv` | HealthPage.jsx | `src/pages/Health/` |

### Processed Sources (No Page Yet)

| Category | CSV Files | Status |
|----------|-----------|--------|
| Finance | `moneymgr_processed.csv` | Ready for implementation |
| Screen Time | `offscreen_processed.csv` | Ready for implementation |
| Weather | `weather_processed.csv` | Ready for implementation |

### Source Processing Locations

All processors located in: `lifelog_python_processing/src/[category]/[source]_processing.py`

**Examples:**
- Music: `src/music/lfm_processing.py`
- Reading: `src/books/books_processing.py` (coordination file)
- Finance: `src/finance/moneymgr_processing.py`

## Impact Testing Requirements

### When Testing is Required

Test website when Python processing changes affect:

**High Impact Changes (Mandatory Testing):**
1. File encoding changes (UTF-8 ↔ UTF-16)
2. Column name changes (renames, additions, deletions)
3. Column type changes (string ↔ number, date formats)
4. Data format changes (date formats, boolean values)
5. File structure changes (delimiter, headers)

### Testing Workflow

#### Step 1: Identify Affected Pages

Use the [Data Sources to Pages Mapping](#data-sources-to-pages-mapping) table above to identify which pages use the modified CSV file.

**Example:**
- Modified: `lfm_processed.csv`
- Affected Page: `Music/MusicPage.jsx`

#### Step 2: Test CSV Parsing

1. Start development server: `npm run dev:all`
2. Navigate to affected page
3. Open browser DevTools → Console
4. Check for parsing errors

**Common Issues:**
- Encoding errors: `UnicodeDecodeError`, garbled text
- Column errors: `Cannot read property 'X' of undefined`
- Type errors: `NaN`, `Invalid Date`

#### Step 3: Test Page Functionality

Verify these components work correctly:

- [ ] **Data Loading**: No errors in console, data displays
- [ ] **Filtering**: All filters work with new column names/types
- [ ] **KPI Cards**: Stats calculate correctly
- [ ] **Content Display**: Grid/list views show data properly
- [ ] **Charts**: Analysis tab charts render without errors
- [ ] **Detail Modal**: Modal displays all fields correctly

#### Step 4: Check Specific Features

**Filtering:**
- Date range filters work with date column
- Multi-select filters work with categorical columns
- Search filters work with text columns

**Charts:**
- Time series charts parse dates correctly
- Bar/pie charts aggregate data correctly
- No `NaN` or `undefined` in chart labels/values

**Performance:**
- Page loads in <2 seconds
- No lag when switching filters
- Smooth scrolling in content area

#### Step 5: Verify Data Integrity

Use browser DevTools → Network tab:

1. Find CSV download request
2. Preview raw CSV data
3. Verify:
   - [ ] Correct encoding (no �� characters)
   - [ ] Column headers match component expectations
   - [ ] Data types are correct (dates, numbers, booleans)
   - [ ] No missing/corrupted values

### Testing Checklist Template

Use this checklist when testing changes:

```markdown
## Testing Checklist: [Data Source Name]

**Modified Files:**
- Python: `[source]_processing.py`
- CSV: `[source]_processed.csv`

**Changes Made:**
- [ ] Encoding change
- [ ] Column rename: `[old]` → `[new]`
- [ ] Column addition: `[new_column]`
- [ ] Data format change: [description]

**Affected Pages:**
- [ ] `[PageName]Page.jsx`

**Tests Performed:**
- [ ] CSV parsing (no console errors)
- [ ] Data loading (displays correctly)
- [ ] Filtering (all filters work)
- [ ] KPI cards (stats correct)
- [ ] Content display (grid/list views)
- [ ] Charts (no errors, correct data)
- [ ] Detail modal (all fields display)
- [ ] Performance (<2s load time)
- [ ] Data integrity (encoding, types, values)

**Issues Found:**
- [List any issues]

**Status:** ✅ Pass / ❌ Fail
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Garbled text (��) | UTF-16 encoding | Change to UTF-8 in Python processor |
| `Cannot read property` | Column name mismatch | Update component to use new column name |
| `Invalid Date` | Wrong date format | Use ISO 8601 (YYYY-MM-DD) in Python |
| Charts show NaN | Type mismatch | Ensure numeric columns are numbers, not strings |
| Slow loading | Large dataset | Implement chunked processing in Python |

### Testing Tools

**Browser DevTools:**
- **Console**: Check for JavaScript errors
- **Network**: Verify CSV download, inspect raw data
- **Performance**: Profile page load time
- **Elements**: Inspect DOM for rendering issues

**Development Server:**
```bash
npm run dev:all  # Start both frontend and backend
```

**Manual Testing:**
1. Visit page in browser
2. Test all filters and interactions
3. Check console for errors
4. Verify data displays correctly

## Reference Files

- **DataContext**: `lifelog_website/src/context/DataContext.jsx`
- **Config**: `lifelog_website/src/config/config.js`
- **Example Page**: `lifelog_website/src/pages/Reading/ReadingPage.jsx`
- **Page Template**: `new_page_template.txt`
