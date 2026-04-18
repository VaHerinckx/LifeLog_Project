# Python Processing — CLAUDE.md

Standards for the `pipeline/` directory.

## File Structure

- One file per data source, in its themed directory (`sources_processing/`, `topic_processing/`)
- Each file owns the full workflow: download → process → upload
- Multi-source coordination: separate file merges sources (e.g. `books_processing.py`)
- Utils: cross-source functions only (no source-specific logic in utils)

## Standard Pipeline (Every Processor Must Implement)

Three menu options:
1. Download new data, process, and upload to Drive
2. Process existing data and upload to Drive
3. Upload existing processed files to Drive

**Reference implementation:** `moneymgr_processing.py`

**Function naming convention:**
- `download_[source]_data()` — Download
- `create_[source]_file()` — Process
- `upload_[source]_results()` — Upload
- `full_[source]_pipeline()` — Complete workflow

See [../docs/python-examples.md](../docs/python-examples.md) for complete code templates.

## Data Output Non-Negotiables

| Rule | Value |
|------|-------|
| Delimiter | Pipe `\|` |
| Encoding | **UTF-8** (UTF-16 breaks website parsing) |
| Output path | `files/processed_files/[category]/[source]_processed.csv` |
| Column style | snake_case |
| Date column | Always named `date` (never `Period`, `timestamp`) |
| Boolean columns | `is_*` or `has_*` prefix |
| Column order | identifiers → descriptive → numerical → dates → booleans |

## Code Organization Rules

- **Imports:** Always at top of file — never inside functions
- **Timezone:** Use `time_difference_correction()` from `src.utils.utils_functions` for UTC/GMT timestamps
- **Tracking:** Call `record_successful_run('category_source', 'active')` on success
  - Tracking file: `files/tracking/last_successful_runs.csv`
- **Status messages:** 🚀 Starting, ✅ Completed, ❌ Failed, ⚠️ Warning

## References

- Code templates: [../docs/python-examples.md](../docs/python-examples.md)
- Integration testing: [../docs/data-integration.md](../docs/data-integration.md)
- Compliance checklist (200+ items): [../docs/python_processing_compliance_checklist.txt](../docs/python_processing_compliance_checklist.txt)
