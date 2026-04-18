# Data Integration Guide

## Adding a New Data Source (3 Steps)

**Step 1 — `.env`:**
```bash
VITE_[DATATYPE]_FILE_ID=your_google_drive_file_id
```

**Step 2 — `src/config/config.js`:**
```javascript
export const DRIVE_FILES = {
  DATATYPE: import.meta.env.VITE_DATATYPE_FILE_ID,
};
```

**Step 3 — `src/context/DataContext.jsx`:**
```javascript
// initialData
const initialData = { datatype: null };

// fetchData switch
case 'datatype':
  fileId = DRIVE_FILES.DATATYPE;
  break;
```

**Usage in page:**
```javascript
const { data, fetchData } = useData();
useEffect(() => { fetchData('datatype'); }, [fetchData]);
const myData = data.datatype;
```

## Data Sources Map

### Active (implemented)

| Category | CSV Files | Page | Location |
|----------|-----------|------|----------|
| Music | `lfm_processed.csv` | MusicPage.jsx | `src/pages/Music/` |
| Reading | `kindle_gr_processed.csv` | ReadingPage.jsx | `src/pages/Reading/` |
| Podcasts | `pocket_casts_processed.csv` | PodcastPage.jsx | `src/pages/Podcast/` |
| Movies/TV | `letterboxd_processed.csv`, `trakt_processed.csv` | MoviesPage.jsx | `src/pages/Movies/` |
| Nutrition | `nutrilio_processed.csv` | NutritionPage.jsx | `src/pages/Nutrition/` |
| Health | `apple_processed.csv`, `garmin_*_processed.csv` | HealthPage.jsx | `src/pages/Health/` |

### Processed but no page yet

| Category | CSV | Status |
|----------|-----|--------|
| Finance | `moneymgr_processed.csv` | Ready |
| Screen Time | `offscreen_processed.csv` | Ready |
| Weather | `weather_processed.csv` | Ready |

Processors: `pipeline/src/sources_processing/[category]/[source]_processing.py`

## Impact Testing (After Python Changes)

Required when changing: encoding, column names/types, date formats, delimiters.

### Testing Checklist

```markdown
**Modified:** [source]_processing.py / [source]_processed.csv
**Changes:** [describe]
**Affected page:** [PageName]Page.jsx

- [ ] No console errors on page load
- [ ] Data displays correctly (grid + list views)
- [ ] All filters work
- [ ] KPI cards calculate correctly
- [ ] Charts render without NaN/undefined
- [ ] Detail modal shows all fields
- [ ] Page loads in <2s

Issues: [list]
Status: ✅ Pass / ❌ Fail
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Garbled text (��) | UTF-16 encoding | Switch to UTF-8 |
| `Cannot read property` | Column name mismatch | Update component |
| `Invalid Date` | Wrong date format | Use ISO 8601 (YYYY-MM-DD) |
| Charts show NaN | Type mismatch | Ensure numeric columns aren't strings |

**Dev server:** `cd website && npm run dev:all`

## Reference Files

- **DataContext:** `website/src/context/DataContext.jsx`
- **Config:** `website/src/config/config.js`
- **Gold standard page:** `website/src/pages/Reading/ReadingPage.jsx`
- **Page template:** `docs/new_page_template.txt`
