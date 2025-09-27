# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the LifeLog Project - a personal data aggregation and visualization system that processes data from multiple sources (fitness trackers, reading apps, music services, etc.) and creates a comprehensive dashboard. The project consists of two main components:

1. **Python Processing Pipeline** (`lifelog_python_processing/`) - Processes raw exports from various services
2. **React Web Dashboard** (`lifelog_website/`) - Frontend visualization of the processed data

## Development Commands

### Python Processing
```bash
# Navigate to Python directory
cd lifelog_python_processing

# Install dependencies
pip install -r requirements.txt

# Run main processing pipeline
python src/process_exports.py
```

### React Website
```bash
# Navigate to website directory
cd lifelog_website

# Install dependencies
npm install

# Development (frontend only)
npm run dev

# Development (with backend server)
npm run dev:all

# Backend server only
npm run server

# Build for production
npm run build

# Lint code
npm run lint

# Preview production build
npm run preview
```

## Architecture Overview

### Python Processing Pipeline
- **Main Entry Point**: `src/process_exports.py` - Interactive CLI for processing different data sources
- **Data Sources**: Each source has its own processor in dedicated directories (`books/`, `music/`, `health/`, etc.)
- **Google Drive Integration**: Processed files are uploaded to Google Drive via `src/utils/drive_storage.py`
- **File Structure**: Raw exports go in `files/exports/`, processed files in `files/processed_files/`

### React Website
- **Frontend**: React with Vite, uses Recharts for visualizations
- **Backend**: Express server (`server.js`) acts as proxy for Google Drive API calls
- **Data Flow**: Frontend fetches CSV data from Google Drive via the Express proxy
- **Configuration**: Google Drive file IDs stored in environment variables (see `src/config/config.js`)

### Data Processing Workflow
1. User downloads exports from various services (Spotify, Garmin, Goodreads, etc.)
2. Files are placed in appropriate `files/exports/` subdirectories
3. `process_exports.py` processes each export type with dedicated processors
4. Processed CSV files are uploaded to Google Drive
5. React frontend fetches and visualizes the data

## Key Integration Points

- **Google Drive Authentication**: Python uses OAuth2 with credentials in `credentials/`, website uses public file sharing
- **Data Format**: All processed files use pipe-separated values (`|`) with UTF-16 encoding
- **Cross-Platform**: Python handles data processing, React handles visualization
- **File Mapping**: `dict_upload` in `process_exports.py` defines which files belong to each data category

## Environment Setup

### Python
- Requires Google Drive API credentials in `credentials/client_secrets.json`
- Settings configured in `credentials/settings.yaml`

### React
- Environment variables for Google Drive file IDs (prefix: `VITE_`)
- Backend server runs on port 3001, frontend on Vite default port

## Testing
- Sample files are generated in `files/sample_files/` for testing purposes
- Use option 3 in `process_exports.py` to create sample files

## Design System Guidelines

The LifeLog website uses a comprehensive design system with standardized variables and components for consistent styling across all pages and components.

### Design System Architecture

#### CSS Variables File: `src/styles/variables.css`
- **Central location** for all design tokens
- Contains 80+ CSS variables covering colors, typography, spacing, borders, shadows, and responsive breakpoints
- **Must be imported first** in any CSS file that needs design system variables

#### Component Classes File: `src/styles/components.css`
- Reusable component classes built using design system variables
- Provides consistent patterns for common UI elements
- Classes include: `.chart-container-base`, `.page-container`, `.page-title`, etc.

### Key Design Principles

#### 1. Container and Layout Standards
- **Max width**: 1400px (`--container-max-width`)
- **Container width**: 95% (`--container-width`)
- **Padding**: 2rem desktop / 1rem mobile (`--container-padding` / `--container-padding-mobile`)

#### 2. Color System
- **Primary color**: #3423A6 (`--color-primary`)
- **Primary dark**: #2A1C85 (`--color-primary-dark`)
- **Accent color**: #FB4B4E (`--color-accent`)
- **Accent dark**: #DC2626 (`--color-accent-dark`)
- **Background**: #171738 (`--color-background`) - purple/dark blue theme
- **Surface**: #D8DCFF (`--color-surface`) - light backgrounds for charts, cards, and components
- **Surface elevated**: #D8DCFF (`--color-surface-elevated`)
- **Text primary**: #2A1C85 (`--color-text-primary`) - purple text for page background
- **Text secondary**: rgba(255, 255, 255, 0.7) (`--color-text-secondary`) - muted white text
- **Text on surface**: #171738 (`--color-text-on-surface`) - dark text for light surfaces
- **Surface colors**: Light backgrounds for all charts, cards, and elevated components with dark text

#### 3. Typography Scale
- **Font family**: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif (`--font-primary`)
- **Size scale**: xs (0.75rem) → sm (0.875rem) → base (1rem) → lg (1.125rem) → xl-4xl (1.25-2.25rem)
- **Weight scale**: normal (400) → medium (500) → semibold (600) → bold (700)
- **Line heights**: tight (1.25) → normal (1.5) → relaxed (1.75)

#### 4. Spacing System
- **8px base unit system**: xs (0.5rem) → 4xl (5rem)
- **Consistent gaps**: Use spacing variables for margins, padding, and gaps
- **Responsive spacing**: Different values for mobile vs desktop

#### 5. Icon System
- **Icon sizes**: xs (1rem) → sm (1.25rem) → md (1.5rem) → lg (2rem) → xl (3rem)
- **Consistent sizing**: Use icon size variables for all icons and logos

#### 6. Border and Shadow System
- **Border widths**: thin (1px) → medium (2px) → thick (4px)
- **Border radius**: sm (4px) → md (8px) → lg (12px) → xl (16px)
- **Shadow scale**: sm → md → lg → xl for depth hierarchy

### Implementation Rules

#### 1. CSS Variable Usage
**✅ ALWAYS DO:**
```css
/* Use design system variables */
.my-component {
  padding: var(--spacing-lg);
  color: var(--color-text-primary);
  background-color: var(--color-surface);
  border-radius: var(--radius-md);
  font-size: var(--font-size-base);
}
```

**❌ NEVER DO:**
```css
/* Don't use hardcoded values */
.my-component {
  padding: 24px;
  color: #ffffff;
  background-color: #1a1a1a;
  border-radius: 8px;
  font-size: 16px;
}
```

#### 2. Chart Component Patterns
All chart components must follow this pattern:
```css
.chart-container {
  background-color: var(--chart-background);
  border-radius: var(--chart-border-radius);
  padding: var(--chart-padding);
  margin-bottom: var(--chart-margin-bottom);
  box-shadow: var(--chart-shadow);
}

.chart-title {
  color: var(--chart-title-color);
  font-size: var(--chart-title-size);
  font-weight: var(--chart-title-weight);
  margin: 0 0 var(--chart-title-margin) 0;
  text-align: var(--chart-title-align);
}
```

**IMPORTANT**: All chart titles must be:
- **Color**: Red (`--color-accent`)
- **Alignment**: Left aligned (`--chart-title-align: left`)
- This ensures consistency across all charts

#### 3. Page Structure Standards
All pages must follow this structure:
```css
.page-container {
  width: var(--container-width);
  max-width: var(--container-max-width);
  margin: 0 auto;
  padding: var(--container-padding);
  background-color: var(--color-background);
  color: var(--color-text-primary);
}

/* Mobile responsive */
@media (max-width: 768px) {
  .page-container {
    padding: var(--container-padding-mobile);
  }
}
```

#### 4. Responsive Design Requirements
- **Mobile-first approach**: Define mobile styles first, then add desktop enhancements
- **Breakpoints**: 768px for mobile/tablet split, 1024px and 1200px for larger screens
- **Touch-friendly**: Minimum 44px touch targets on mobile
- **Font size**: Minimum 16px on mobile to prevent zoom

### File Organization Rules

#### 1. Import Order
Always import design system files first:
```css
@import './styles/variables.css';
@import './styles/components.css';

/* Then component-specific styles */
.my-specific-styles {
  /* ... */
}
```

#### 2. CSS File Structure
Organize CSS files in this order:
1. Design system imports
2. Main component styles
3. Modifier classes
4. State-based styles (:hover, :focus, etc.)
5. Responsive media queries

#### 3. Variable Naming Convention
- Use semantic names: `--color-text-primary` not `--color-white`
- Follow hierarchy: `--spacing-xs` to `--spacing-4xl`
- Be specific: `--chart-title-color` not `--title-color`

### Testing and Validation

#### Before Creating New CSS:
1. Check if existing variables/components can be used
2. Ensure new styles follow spacing/color/typography scales
3. Test responsive behavior on mobile and desktop
4. Verify accessibility (contrast ratios, touch targets)

#### When Modifying Existing Styles:
1. Check impact across all components using the variable
2. Maintain consistent visual hierarchy
3. Preserve responsive breakpoints
4. Test in both light and dark color schemes (if applicable)

### Future Maintenance

#### Adding New Variables:
1. Add to appropriate section in `variables.css`
2. Follow existing naming conventions
3. Update this documentation if adding new categories
4. Test across multiple components before committing

#### Deprecating Variables:
1. Mark as deprecated in comments
2. Provide migration path in this documentation
3. Update all usages before removing
4. Remove only after full migration

### Analysis Tab Component Standards

All `*AnalysisTab` components (MusicAnalysisTab, PodcastAnalysisTab, ReadingAnalysisTab, MoviesAnalysisTab) must follow standardized patterns for consistent behavior and styling.

#### Required Structure
All analysis tab components must use this minimal structure (charts only, no titles/descriptions):

```jsx
<div className="analysis-tab-container">
  {data.length === 0 ? (
    <div className="analysis-empty-state">
      <p>No data available with current filters. Try adjusting your filter criteria.</p>
    </div>
  ) : (
    <div className="analysis-charts-grid">
      <div className="analysis-chart-section">
        {/* Chart components */}
      </div>
    </div>
  )}
</div>
```

#### Empty State Pattern
For empty states, use the standardized class:
```jsx
<div className="analysis-empty-state">
  <p>No data available message</p>
</div>
```

#### Class Definitions (from components.css)
- `.analysis-tab-container`: Full width container with consistent sizing
- `.analysis-charts-grid`: Responsive grid layout for charts (1 column mobile, 2 columns desktop at 1200px+)
- `.analysis-chart-section`: Individual chart wrapper with full width
- `.analysis-empty-state`: Standardized empty state styling with padding and surface background

#### Implementation Rules
1. **Content Scope**: Analysis tabs contain **ONLY charts** - no titles, descriptions, or filter panels
2. **Data Flow**: Analysis tabs receive pre-filtered data as props from parent page components
3. **Filter Management**: All filtering logic must be handled by parent page components (outside analysis tabs)
4. **Width Behavior**: All analysis tabs take full width of parent container
5. **Responsive Design**: Charts stack vertically on mobile, 2-column grid on desktop (1200px+)
6. **CSS Inheritance**: Individual CSS files should be minimal and inherit from design system
7. **Consistent Empty States**: Use standardized empty state pattern for no-data scenarios

#### Filter Panel Location
Filter panels (FilteringPanel, AnalysisFilterPane, etc.) must ALWAYS be placed in the parent page component, never inside analysis tab components. This ensures:
- Consistent UI patterns across all pages
- Clear separation between filtering (page-level) and visualization (component-level)
- Proper data flow from filters → page → analysis tab

This standardization ensures all analysis tabs have consistent width behavior, preventing the width constraint issue that was present in ReadingAnalysisTab, and provides a scalable pattern for future analysis components.

This design system ensures visual consistency, maintainability, and scalability across the entire LifeLog website. All new CSS files and modifications must follow these guidelines.

## Adding New Data Sources to the Website

This section documents the standardized process for integrating new data sources into the LifeLog website, connecting processed CSV files from Google Drive to the frontend.

### Phase 1: Configuration Setup (Required for all new data sources)

#### Step 1: Environment Variables Setup (`.env`)
Add the Google Drive file ID for your processed data file:

```bash
# Add this line to the .env file in lifelog_website directory
VITE_[DATATYPE]_FILE_ID=your_google_drive_file_id_here
```

**How to get Google Drive file ID:**
1. Upload your processed CSV file to Google Drive
2. Right-click the file → "Get link" → "Copy link"
3. Extract the file ID from the URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`
4. Use this ID in the environment variable

**Examples:**
```bash
VITE_HEALTH_FILE_ID=1abc123def456ghi789jkl
VITE_FITNESS_FILE_ID=2def456ghi789jkl123abc
```

#### Step 2: Configuration Layer (`src/config/config.js`)
Add your data source to the `DRIVE_FILES` object:

```javascript
export const DRIVE_FILES = {
  // Existing entries...
  MUSIC: import.meta.env.VITE_MUSIC_FILE_ID,
  MOVIES: import.meta.env.VITE_MOVIES_FILE_ID,
  
  // Add your new data source here
  HEALTH: import.meta.env.VITE_HEALTH_FILE_ID,
  FITNESS: import.meta.env.VITE_FITNESS_FILE_ID,
};
```

#### Step 3: Data Context Integration (`src/context/DataContext.jsx`)
Add your data source to the DataContext:

1. **Add to initial state:**
```javascript
const initialData = {
  // Existing data types...
  music: null,
  movies: null,
  
  // Add your new data type
  health: null,
  fitness: null,
};
```

2. **Add to fetchData switch statement:**
```javascript
const fetchData = async (dataType) => {
  try {
    setLoading(prev => ({ ...prev, [dataType]: true }));
    setError(prev => ({ ...prev, [dataType]: null }));

    let fileId;
    switch (dataType) {
      // Existing cases...
      case 'music':
        fileId = DRIVE_FILES.MUSIC;
        break;
      case 'movies':
        fileId = DRIVE_FILES.MOVIES;
        break;
        
      // Add your new case here
      case 'health':
        fileId = DRIVE_FILES.HEALTH;
        break;
      case 'fitness':
        fileId = DRIVE_FILES.FITNESS;
        break;
        
      default:
        throw new Error(`Unknown data type: ${dataType}`);
    }
    
    // Rest of fetchData logic...
  }
};
```

### Data Format Requirements

All processed CSV files must follow these standards:
- **Delimiter**: Pipe-separated values (`|`)
- **Encoding**: UTF-16 (automatically detected by Papa Parse)
- **Headers**: First row contains column names
- **Date format**: ISO 8601 format preferred (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`)

### Usage in Pages

Once configured, use the data in your page components:

```javascript
import { useData } from '../../context/DataContext';

function YourPage() {
  const { data, loading, error, fetchData } = useData();
  
  useEffect(() => {
    fetchData('health'); // or your data type
  }, [fetchData]);
  
  const healthData = data.health;
  
  // Your component logic...
}
```

### Error Handling and Troubleshooting

Common issues and solutions:
- **"File not found"**: Check Google Drive file ID and sharing permissions
- **"Parsing error"**: Verify CSV format (pipe-delimited, proper encoding)
- **"Large file timeout"**: Files over 100MB may timeout (increase timeout in server.js)
- **"Environment variable not found"**: Ensure `.env` file exists and variable name is correct

### Reference Examples

Examine existing implementations for patterns:
- **Movies**: Simple data structure with date filtering
- **Music**: Large dataset with special handling and chunked processing
- **Books**: Multiple related data files (books + audiobooks)

This process ensures consistent data integration patterns across all LifeLog data sources.

## Python Processing Pipeline Guidelines

This section provides comprehensive guidelines for developing and maintaining consistent Python data processing code within the LifeLog project. All Python processing code must follow these standards to ensure maintainability, consistency, and reliability.

### Core Architecture Principles

#### 1. Source Processor Structure
- **One File Per Source**: Each data source (Spotify, Garmin, Goodreads, etc.) has its own dedicated processing file
- **Themed Directories**: Group related sources in logical directories (`books/`, `music/`, `health/`, `movies/`, etc.)
- **Complete Processing**: Each source processor handles the entire workflow from download to upload within a single file
- **Self-Contained Logic**: Source-specific logic must remain in the source file, not in utils

#### 2. Multi-Source Topic Handling
**CRITICAL REQUIREMENT**: When a topic has multiple data sources, create a dedicated coordination file that processes and merges results.

**Pattern Example - Books Topic**:
- `books/goodreads_processing.py` - Processes Goodreads exports
- `books/kindle_processing.py` - Processes Kindle exports  
- `books/books_processing.py` - **Coordination file** that:
  - Imports and calls individual source processors
  - Checks for prerequisite processed files
  - Merges data from multiple sources into unified output
  - Handles combined upload operations

**Implementation Rule**: 
- Individual source processors (e.g., `goodreads_processing.py`) handle their specific data source
- Coordination file (e.g., `books_processing.py`) orchestrates multi-source workflows
- The coordination file becomes the primary entry point for the topic in `process_exports.py`

#### 3. Utils Layer Separation
Functions that can be used across multiple sources belong in utils modules:

- **`utils/drive_operations.py`**: Google Drive upload, connection testing, credential management
- **`utils/file_operations.py`**: File system operations, folder management, file validation
- **`utils/web_operations.py`**: Browser automation, download prompting, URL handling
- **`utils/utils_functions.py`**: Common data processing, timezone corrections, general utilities

**Never Put in Utils**:
- Source-specific data transformations
- API calls specific to one service
- Business logic tied to a particular data format

### Standardized Pipeline Patterns

#### 1. Pipeline Function Naming
All processors must implement these standard function names:

```python
# Core pipeline functions
def download_[source]_data()           # Download new data from service
def move_[source]_files()              # Move/organize downloaded files
def create_[source]_file()             # Process and create output CSV
def upload_[source]_results()          # Upload processed files to Drive

# Main pipeline orchestrator
def full_[source]_pipeline()           # Complete workflow with user options

# Legacy compatibility (if needed)
def process_[source]_export()          # Backward-compatible wrapper
```

#### 2. Standard Pipeline Options
Every `full_*_pipeline()` function must offer these 4 standard options:

```python
def full_[source]_pipeline():
    print(f"🎵 {Source} Processing Pipeline")
    print("=" * 50)
    print("1. Download new data, process, and upload to Drive")
    print("2. Process existing data and upload to Drive") 
    print("3. Upload existing processed files to Drive")
    print("4. Full pipeline (download + process + upload)")
    
    choice = input("Select option (1-4): ").strip()
    # Implementation logic...
```

### Data Output Standards

#### 1. File Format Requirements
- **Delimiter**: Pipe-separated values (`|`) - never comma or tab
- **Encoding**: UTF-16 with BOM for proper character handling
- **Headers**: First row must contain descriptive column names
- **Output Location**: `files/processed_files/[category]/[source]_processed.csv`

#### 2. Column Naming and Ordering Standards

**Column Naming Rules:**
- **snake_case**: All column names must use snake_case format: `release_date`, `track_name`, `artist_name`
- **Boolean Columns**: Use `is_*` or `has_*` prefixes: `is_favorite`, `has_lyrics`, `is_explicit`, `has_cover_art`
- **Descriptive Names**: Use full descriptive names, not abbreviations:
  - `release_date` not `date` or `dt`
  - `track_name` not `name` or `title`
  - `user_rating` not `rating` or `stars`
  - `duration_seconds` not `duration` or `time`

**Column Ordering Standards:**
Columns must be ordered logically in this sequence:
1. **Primary identifiers**: `id`, `name`, `title`, etc.
2. **Descriptive attributes**: `artist_name`, `album_name`, `genre`, etc.
3. **Numerical values**: `duration_seconds`, `user_rating`, `play_count`, etc.
4. **Dates and timestamps**: `release_date`, `last_played_date`, `created_at`, etc.
5. **Boolean columns**: `is_favorite`, `has_lyrics`, `is_explicit`, etc.

**Example Column Order:**
```
track_id | track_name | artist_name | album_name | genre | duration_seconds | user_rating | release_date | last_played_date | is_favorite | has_lyrics | is_explicit
```

#### 3. Date Handling Standards
- **Timezone Correction**: Use `time_difference_correction()` from utils for location-based timezone adjustments
- **Date Format**: Prefer ISO 8601 format (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`)
- **Consistency**: Ensure all dates within a file use the same format

### User Interface Standards

#### 1. Status Message Patterns
Use consistent emoji-based messages throughout all processors:

```python
# Status indicators
print("🚀 Starting [operation]...")
print("✅ [Operation] completed successfully")
print("❌ [Operation] failed: [reason]")
print("⚠️  Warning: [issue]")
print("📁 File found: [filename]")
print("📂 Creating directory: [path]")
print("🔗 Connecting to [service]...")
print("⬆️  Uploading to Google Drive...")
```

#### 2. User Interaction Patterns
- **Consistent Prompting**: Use standard input patterns with clear options
- **Confirmation Workflows**: Ask for confirmation on destructive operations
- **Progress Indication**: Show clear step-by-step progress through pipelines

#### 3. Error Handling Standards
- **Graceful Failures**: Allow pipeline to continue after non-critical errors
- **Clear User Feedback**: Use standardized emoji-based status messages
- **Detailed Logging**: Log errors with context for debugging

### Implementation Templates

#### 1. New Source Processor Template
When creating a new source processor, follow this structure:

```python
# Standard imports
import os
import pandas as pd
import time
from datetime import datetime

# Utils imports  
from src.utils.file_operations import [needed_functions]
from src.utils.web_operations import [needed_functions]
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection

# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

def download_[source]_data():
    """Download new data from [source] service"""
    print("🚀 Starting [source] data download...")
    # Implementation
    print("✅ Download completed")

def move_[source]_files():
    """Move and organize [source] files"""
    print("📁 Moving [source] files...")
    # Implementation  
    print("✅ Files moved successfully")

def create_[source]_file():
    """Process [source] data and create output CSV"""
    print("🔄 Processing [source] data...")
    # Data processing logic
    # Save as pipe-delimited UTF-16 CSV with proper column naming/ordering
    print("✅ Processing completed")

def upload_[source]_results():
    """Upload processed [source] files to Google Drive"""
    print("⬆️  Uploading [source] results...")
    files_to_upload = [
        'files/processed_files/[category]/[source]_processed.csv'
    ]
    upload_multiple_files(files_to_upload)
    print("✅ Upload completed")

def full_[source]_pipeline():
    """Complete [source] processing pipeline with user options"""
    print("🎯 [Source] Processing Pipeline")
    print("=" * 50)
    print("1. Download new data, process, and upload to Drive")
    print("2. Process existing data and upload to Drive")
    print("3. Upload existing processed files to Drive") 
    print("4. Full pipeline (download + process + upload)")
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        download_[source]_data()
        create_[source]_file()
        upload_[source]_results()
    elif choice == "2":
        create_[source]_file()
        upload_[source]_results()
    elif choice == "3":
        upload_[source]_results()
    elif choice == "4":
        download_[source]_data()
        move_[source]_files()
        create_[source]_file()
        upload_[source]_results()
    else:
        print("❌ Invalid choice")

# Legacy compatibility wrapper (if needed)
def process_[source]_export():
    """Legacy function - redirects to full pipeline"""
    full_[source]_pipeline()
```

#### 2. Multi-Source Coordination Template
For topics with multiple sources, create a coordination file:

```python
# Import individual source processors
from src.[category].[source1]_processing import full_[source1]_pipeline
from src.[category].[source2]_processing import full_[source2]_pipeline
from src.utils.drive_operations import upload_multiple_files, verify_drive_connection

def check_prerequisite_files():
    """Check if required processed files from individual sources exist"""
    required_files = {
        '[source1]': 'files/processed_files/[category]/[source1]_processed.csv',
        '[source2]': 'files/processed_files/[category]/[source2]_processed.csv'
    }
    
    file_status = {}
    for source, file_path in required_files.items():
        exists = os.path.exists(file_path)
        file_status[source] = {
            'exists': exists,
            'path': file_path
        }
        print(f"{'✅' if exists else '❌'} {source}: {file_path}")
    
    return file_status

def merge_[category]_data():
    """Merge data from multiple sources into unified output"""
    print("🔄 Merging [category] data from multiple sources...")
    
    # Load individual processed files
    df_source1 = pd.read_csv('files/processed_files/[category]/[source1]_processed.csv', sep='|')
    df_source2 = pd.read_csv('files/processed_files/[category]/[source2]_processed.csv', sep='|')
    
    # Merge logic specific to the category
    # Ensure final output follows column naming and ordering standards
    
    # Save merged result
    output_path = 'files/processed_files/[category]/[category]_combined.csv'
    merged_df.to_csv(output_path, sep='|', index=False, encoding='utf-16')
    print(f"✅ Merged data saved to {output_path}")

def full_[category]_pipeline():
    """Complete [category] processing pipeline coordinating multiple sources"""
    print("📚 [Category] Multi-Source Processing Pipeline")
    print("=" * 50)
    print("1. Process all sources and merge results")
    print("2. Process [source1] only")
    print("3. Process [source2] only")
    print("4. Merge existing processed files")
    print("5. Upload merged results to Drive")
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == "1":
        full_[source1]_pipeline()
        full_[source2]_pipeline()
        merge_[category]_data()
        upload_[category]_results()
    elif choice == "2":
        full_[source1]_pipeline()
    elif choice == "3":
        full_[source2]_pipeline()
    elif choice == "4":
        merge_[category]_data()
    elif choice == "5":
        upload_[category]_results()
    else:
        print("❌ Invalid choice")
```

### Environment and Configuration Standards

#### 1. API Key Management
- Store API keys in environment variables via `.env` file
- Use descriptive names: `TMDB_Key`, `LASTFM_API_KEY`, `SPOTIFY_CLIENT_ID`
- Always check for key existence before making API calls
- Provide clear error messages when keys are missing

#### 2. File Path Management
- Use relative paths from project root
- Create directory structure programmatically if it doesn't exist
- Validate file existence before processing
- Use Path objects for cross-platform compatibility

### Testing and Validation

#### 1. Sample File Generation
- Every processor should support sample file creation for testing
- Place sample files in `files/sample_files/[category]/`
- Sample files should represent realistic data variety
- Include edge cases in sample data

#### 2. Data Validation
- Validate file format before processing
- Check for required columns in input files
- Verify data types and ranges
- Handle missing or malformed data gracefully

### Integration with Main Pipeline

#### 1. Adding to process_exports.py
When adding a new processor to the main pipeline:

```python
# Import the main pipeline function
from src.[category].[processor]_processing import full_[processor]_pipeline

# Add to menu options in main()
print("X. Process [Source] Data")

# Add to choice handling
elif choice == "X":
    full_[processor]_pipeline()
```

#### 2. Google Drive Integration
- All processors must use `upload_multiple_files()` from `drive_operations.py`
- Test drive connection before upload operations
- Handle authentication failures gracefully
- Provide clear feedback on upload success/failure

### Performance and Optimization

#### 1. Large Dataset Handling
- Process data in chunks for large files (>100MB)
- Use efficient pandas operations (vectorization over loops)
- Cache API responses when possible
- Implement progress indicators for long operations

#### 2. API Rate Limiting
- Respect API rate limits with appropriate delays
- Implement exponential backoff for failed requests
- Cache results to minimize API calls
- Provide clear feedback on rate limit delays

These guidelines ensure all Python processing code follows consistent patterns, making the codebase maintainable, extensible, and reliable. All new processors and modifications to existing ones must follow these standards.
