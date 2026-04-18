# Python Processing Examples

Detailed code examples and patterns for LifeLog Python processing pipeline.

## Table of Contents
- [Standard Pipeline Pattern](#standard-pipeline-pattern)
- [Function Naming Convention](#function-naming-convention)
- [Import Management](#import-management)
- [UTF-8 Encoding](#utf-8-encoding)
- [Timezone Correction](#timezone-correction)
- [Status Messages](#status-messages)
- [Data Source Tracking](#data-source-tracking)

## Standard Pipeline Pattern

**Reference:** `moneymgr_processing.py`

### 3-Option Menu

All processors must implement three options:

```python
print("1. Download new data, process, and upload to Drive")
print("2. Process existing data and upload to Drive")
print("3. Upload existing processed files to Drive")
```

### Complete Implementation Example

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

if __name__ == "__main__":
    main()
```

## Function Naming Convention

### Standard Function Names

```python
def download_[source]_data():
    """Download raw data from source API/export"""
    # Download logic
    pass

def create_[source]_file():
    """Process raw data and create CSV output"""
    # Processing logic
    pass

def upload_[source]_results():
    """Upload processed CSV to Google Drive"""
    # Upload logic
    pass

def full_[source]_pipeline():
    """Complete workflow: download → process → upload"""
    download_[source]_data()
    create_[source]_file()
    upload_[source]_results()
```

### Example (Money Manager)

```python
def download_moneymgr_data():
    """Download Money Manager backup"""
    pass

def create_moneymgr_file():
    """Process Money Manager data"""
    pass

def upload_moneymgr_results():
    """Upload to Google Drive"""
    pass

def full_moneymgr_pipeline():
    """Complete Money Manager pipeline"""
    download_moneymgr_data()
    create_moneymgr_file()
    upload_moneymgr_results()
```

## Import Management

### Correct: Imports at Top of File

```python
# ✅ CORRECT - imports at top of file
import os
import pandas as pd
from datetime import datetime
from src.utils.drive_operations import upload_multiple_files
from src.utils.file_operations import ensure_directory
from src.utils.utils_functions import record_successful_run
```

### Incorrect: Imports Inside Functions

```python
# ❌ INCORRECT - never import inside functions
def process_data():
    import pandas as pd  # WRONG
    import os  # WRONG
    # function logic
```

## UTF-8 Encoding

### Critical: Always Use UTF-8

UTF-16 encoding breaks website CSV parsing. Always use UTF-8.

```python
# ✅ CORRECT
df.to_csv(output_file, sep='|', index=False, encoding='utf-8')

# ❌ INCORRECT - breaks website parsing
df.to_csv(output_file, sep='|', index=False, encoding='utf-16')
```

### Complete Example

```python
def create_processed_file(input_file, output_file):
    # Read data
    df = pd.read_csv(input_file)

    # Process data
    # ... processing logic ...

    # Write with UTF-8 encoding
    df.to_csv(
        output_file,
        sep='|',
        index=False,
        encoding='utf-8'  # CRITICAL
    )

    print(f"✅ File saved: {output_file}")
```

## Timezone Correction

### When to Use

Apply for UTC/GMT timestamps to convert to local time based on physical location.

```python
from src.utils.utils_functions import time_difference_correction

def process_timestamps(df):
    # Convert UTC timestamps to local time
    df = time_difference_correction(
        df,
        'timestamp_column',
        source_timezone='UTC'
    )
    return df
```

### Complete Example

```python
import pandas as pd
from src.utils.utils_functions import time_difference_correction

def create_music_file():
    # Read data
    df = pd.read_csv('files/exports/music/scrobbles.csv')

    # Apply timezone correction for UTC timestamps
    df = time_difference_correction(df, 'timestamp', source_timezone='UTC')

    # Continue processing
    # ...
```

## Status Messages

### Standard Emojis

Use consistent emojis for status messages:

```python
print("🚀 Starting [operation]...")
print("✅ [Operation] completed")
print("❌ [Operation] failed: [reason]")
print("⚠️  Warning: [issue]")
```

### Complete Example

```python
def download_data():
    print("🚀 Starting data download...")

    try:
        # Download logic
        response = requests.get(url)

        if response.status_code == 200:
            print("✅ Download completed")
            return True
        else:
            print(f"❌ Download failed: Status {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False

def process_data():
    print("🚀 Starting data processing...")

    if not os.path.exists(input_file):
        print("⚠️  Warning: Input file not found")
        return

    # Processing logic

    print("✅ Processing completed")
```

## Data Source Tracking

### Recording Successful Runs

All pipelines must track completion in `last_successful_runs.csv`.

```python
from src.utils.utils_functions import record_successful_run

def full_pipeline():
    success = False

    try:
        download_data()
        process_data()
        upload_results()
        success = True

    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        success = False

    # Record completion
    if success:
        record_successful_run('category_source', 'active')
```

### Source Types

- `active` - Currently active data source
- `coordination` - Multi-source coordination file
- `legacy` - Old/deprecated source
- `inactive` - Not currently processed

### Example

```python
# Music source (active)
if success:
    record_successful_run('music_lastfm', 'active')

# Books coordination (merges Kindle + Goodreads)
if success:
    record_successful_run('books_combined', 'coordination')
```

## Reference Files

- **Gold Standard**: `lifelog_python_processing/src/finance/moneymgr_processing.py`
- **Multi-Source**: `lifelog_python_processing/src/books/books_processing.py`
- **Utils**: `lifelog_python_processing/src/utils/utils_functions.py`
- **Compliance**: `python_processing_compliance_checklist.txt`
