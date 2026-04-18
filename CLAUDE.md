# CLAUDE.md

## Project Overview

LifeLog Project - Personal data aggregation and visualization system processing data from multiple sources (fitness, reading, music, etc.) into a comprehensive dashboard.

**Architecture:** Raw exports → Python processing → Google Drive → React frontend

**Components:**
- **Python Pipeline** (`pipeline/`) - Processes raw service exports
- **React Dashboard** (`website/`) - Frontend visualization

> Single git repo — `pipeline/` and `website/` are plain subdirectories, not submodules. All commits and pushes go through the root repo only.

## Development Commands

```bash
# Python Pipeline
pyenv activate general_coding  # activate environment first
cd pipeline && python src/process_exports.py  # Interactive CLI

# React Website
cd website && npm install
npm run dev        # Frontend only
npm run dev:all    # With backend server
npm run build      # Production build
```

## Architecture

**Data Flow:** Downloads → `files/exports/` → Process → `files/processed_files/` → Google Drive → React frontend

**Python Pipeline:**
- Entry: `src/process_exports.py` (interactive CLI)
- Structure: Themed directories (`books/`, `music/`, `health/`)
- Utils: `drive_operations.py`, `file_operations.py`, `utils_functions.py`
- Auth: OAuth2 via `credentials/`
- File mappings: `dict_upload` in `process_exports.py`

**React Frontend:**
- Stack: React + Vite, Express proxy, Recharts
- Backend: `server.js` proxies Google Drive API calls
- Config: `src/config/config.js` (Drive file IDs)
- Output format: Pipe-delimited UTF-8 CSV (UTF-16 breaks website parsing)

## Data Sources Mapping

| Category | CSV Files | Page Component |
|----------|-----------|----------------|
| Music | `lfm_processed.csv` | `Music/MusicPage.jsx` |
| Reading | `kindle_gr_processed.csv` | `Reading/ReadingPage.jsx` |
| Podcasts | `pocket_casts_processed.csv` | `Podcast/PodcastPage.jsx` |
| Movies/TV | `letterboxd_processed.csv`, `trakt_processed.csv` | `Movies/MoviesPage.jsx` |
| Nutrition | `nutrilio_processed.csv` | `Nutrition/NutritionPage.jsx` |
| Health | `apple_processed.csv`, `garmin_*_processed.csv` | `Health/HealthPage.jsx` |

**When modifying Python processing:** Check mapping above to identify affected pages. Test CSV parsing, filtering, and charts after any encoding/column/format changes.

## Implementation References

**Website:**
- Page Structure: `ReadingPage.jsx` (gold standard — all pages must follow this)
- Filters: `filterConfigs.jsx`
- Design System: `src/styles/variables.css` + `src/styles/tokens/`
- `docs/website-examples.md` — Component patterns, pagination, AnalysisTab API, new page workflow (6-step integration checklist)

**Python:**
- Pipeline Pattern: `moneymgr_processing.py` (gold standard)
- Multi-Source: `books_processing.py`
- Utils: `utils_functions.py`
- `docs/python-examples.md` — Code templates for pipeline pattern, function naming, UTF-8 encoding, timezone correction, status messages, run tracking

**Integration:**
- `docs/data-integration.md` — 3-step new data source setup (.env → config.js → DataContext), data sources→pages mapping (incl. unimplemented sources), impact testing checklist
- `docs/python_processing_compliance_checklist.txt` — 200+ compliance items

## Deployment

**Website** is hosted on Render (static site + Express backend).
- Render "Root Directory" must be set to `website/` (not the repo root)
- Build command: `npm run build` | Publish directory: `dist`
- If the `website/` directory is ever renamed, update Render's Root Directory setting

## Component-Specific Standards

See subdirectory CLAUDE.md files for detailed standards:
- Python: [pipeline/CLAUDE.md](pipeline/CLAUDE.md)
- Website: [website/CLAUDE.md](website/CLAUDE.md)
