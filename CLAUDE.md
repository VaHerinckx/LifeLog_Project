# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

LifeLog Project - Personal data aggregation and visualization system processing data from multiple sources (fitness, reading, music, etc.) into a comprehensive dashboard.

**Architecture:** Raw exports → Python processing → Google Drive → React frontend

**Components:**
- **Python Pipeline** (`pipeline/`) - Processes raw service exports
- **React Dashboard** (`website/`) - Frontend visualization

## Development Commands

```bash
# Python Pipeline
cd pipeline && pip install -r requirements.txt
python src/process_exports.py  # Interactive CLI

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
- Output format: Pipe-delimited UTF-8 CSV

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
- Design System: `variables.css`
- Detailed guide: [docs/website-examples.md](docs/website-examples.md)

**Python:**
- Pipeline Pattern: `moneymgr_processing.py` (gold standard)
- Multi-Source: `books_processing.py`
- Utils: `utils_functions.py`
- Detailed guide: [docs/python-examples.md](docs/python-examples.md)

**Integration:**
- Adding data sources / testing checklist: [docs/data-integration.md](docs/data-integration.md)
- Compliance improvements (200+ items): [docs/python_processing_compliance_checklist.txt](docs/python_processing_compliance_checklist.txt)

## Component-Specific Standards

See subdirectory CLAUDE.md files for detailed standards:
- Python: [pipeline/CLAUDE.md](pipeline/CLAUDE.md)
- Website: [website/CLAUDE.md](website/CLAUDE.md)
