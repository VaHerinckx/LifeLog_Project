# Python Processing Examples

## Standard Pipeline Pattern

All processors must implement three menu options:

```python
def main():
    print("\n=== [Source Name] Processing ===")
    print("1. Download new data, process, and upload to Drive")
    print("2. Process existing data and upload to Drive")
    print("3. Upload existing processed files to Drive")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == '1':
        full_[source]_pipeline()
    elif choice == '2':
        create_[source]_file()
        upload_[source]_results()
    elif choice == '3':
        upload_[source]_results()
    else:
        print("❌ Invalid choice")
```

**Reference:** `src/sources_processing/moneymgr/moneymgr_processing.py`

## Function Naming Convention

```python
def download_[source]_data(): ...   # Download raw data
def create_[source]_file(): ...     # Process → CSV
def upload_[source]_results(): ...  # Upload to Drive
def full_[source]_pipeline():       # Complete workflow
    download_[source]_data()
    create_[source]_file()
    upload_[source]_results()
```

## Import Management

Always at top of file — never inside functions.

```python
import os
import pandas as pd
from src.utils.drive_operations import upload_multiple_files
from src.utils.utils_functions import record_successful_run, time_difference_correction
```

## UTF-8 Encoding

UTF-16 breaks website CSV parsing. Always use UTF-8.

```python
df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
```

## Timezone Correction

For UTC/GMT timestamps:

```python
from src.utils.utils_functions import time_difference_correction
df = time_difference_correction(df, 'timestamp_column', source_timezone='UTC')
```

## Status Messages

```
🚀 Starting [operation]...
✅ [Operation] completed
❌ [Operation] failed: [reason]
⚠️  Warning: [issue]
```

## Run Tracking

Call on every successful pipeline completion:

```python
from src.utils.utils_functions import record_successful_run

if success:
    record_successful_run('category_source', 'active')
    # Types: 'active' | 'coordination' | 'legacy' | 'inactive'
```

Tracking file: `files/tracking/last_successful_runs.csv`

## Reference Files

- **Gold Standard:** `pipeline/src/sources_processing/moneymgr/moneymgr_processing.py`
- **Multi-Source:** `pipeline/src/sources_processing/books/books_processing.py`
- **Utils:** `pipeline/src/utils/utils_functions.py`
- **Compliance checklist:** `docs/python_processing_compliance_checklist.txt`
