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
