# Chart Responsiveness Guide

This guide documents the standard approach for creating responsive charts in the LifeLog Project using Recharts and CSS design tokens.

## Table of Contents
1. [Overview](#overview)
2. [The Gold Standard Pattern](#the-gold-standard-pattern)
3. [Key Principles](#key-principles)
4. [Implementation Guide](#implementation-guide)
5. [CSS Variables Reference](#css-variables-reference)
6. [Testing Checklist](#testing-checklist)
7. [Common Mistakes to Avoid](#common-mistakes-to-avoid)

---

## Overview

All charts in the LifeLog Project should be fully responsive and use CSS design tokens for consistent sizing, colors, and spacing. This ensures:
- Charts adapt automatically to any viewport size
- Styling can be changed globally by editing CSS variables
- Consistent behavior across all chart types
- Maintainable codebase with DRY principles

**Reference Implementation**: `TimeSeriesBarChart` is the gold standard that all charts should follow.

---

## The Gold Standard Pattern

### Component Structure

```jsx
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import './ChartComponent.css';

const ChartComponent = ({ data, title }) => {
  return (
    <div className="chart-container">
      {/* Header Section - 20% height */}
      <div className="chart-header">
        <h2 className="chart-title">{title}</h2>
        <div className="chart-controls">
          {/* Filters, dropdowns, etc. */}
        </div>
      </div>

      {/* Chart Content Section - 80% height */}
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
```

### CSS Structure

```css
/* Chart Container - Fills parent analysis-chart-section */
.chart-container {
  display: flex;
  flex-direction: column;
  height: 100%;  /* CRITICAL: Fill parent container */
  box-sizing: border-box;
  overflow: hidden;

  background-color: var(--chart-background);
  border-radius: var(--chart-border-radius);
  padding: var(--chart-padding);
  box-shadow: var(--chart-shadow);
}

/* Header Section - Fixed ratio of container */
.chart-header {
  height: var(--chart-header-height-ratio);  /* 20% */
  flex-shrink: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--chart-title-margin);
}

.chart-title {
  font-size: var(--chart-title-size);
  font-weight: var(--chart-title-weight);
  color: var(--chart-title-color);
  margin: 0;
}

/* Chart Content Area - Remaining space */
.recharts-responsive-container {
  height: var(--chart-content-height-ratio) !important;  /* 80% */
  flex-shrink: 0;
}

/* Mobile Responsive */
@media (max-width: 768px) {
  .chart-container {
    padding: var(--spacing-md);
  }

  .chart-header {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--spacing-sm);
  }

  .chart-title {
    font-size: var(--font-size-lg);
  }
}
```

---

## Key Principles

### 1. ResponsiveContainer Usage

**✅ CORRECT:**
```jsx
<ResponsiveContainer width="100%" height="100%">
```

**❌ INCORRECT:**
```jsx
<ResponsiveContainer width="100%" height={500}>  // Hardcoded!
<ResponsiveContainer width="100%" height={400}>  // Hardcoded!
```

**Why**: The parent container (`.analysis-chart-section`) provides explicit height via CSS variables. ResponsiveContainer should fill 100% of that space, allowing global control of chart sizing.

### 2. Flex Layout Pattern

**Required structure**:
```css
.chart-container {
  display: flex;
  flex-direction: column;
  height: 100%;  /* Fill parent */
}
```

**Why**: This allows the chart to:
- Fill the parent container completely
- Split space between header (20%) and chart content (80%)
- Respond to parent container size changes automatically

### 3. CSS Variables Only

**✅ CORRECT:**
```css
.chart-container {
  padding: var(--chart-padding);
  border-radius: var(--chart-border-radius);
  background-color: var(--chart-background);
}
```

**❌ INCORRECT:**
```css
.chart-container {
  padding: 24px;              // Hardcoded!
  border-radius: 12px;        // Hardcoded!
  background-color: #f5f5f8;  // Hardcoded!
}
```

**Why**: CSS variables provide a single source of truth. Changing dimensions globally becomes trivial.

### 4. Parent Container System

Charts are automatically wrapped in `.analysis-chart-section` by the `AnalysisTab` component:

```css
/* From components.css */
.analysis-chart-section {
  height: var(--chart-height-mobile);  /* 400px on mobile */
}

@media (min-width: 768px) {
  .analysis-chart-section {
    height: var(--chart-height-tablet);  /* 500px on tablet */
  }
}

@media (min-width: 1500px) {
  .analysis-chart-section {
    height: var(--chart-height-desktop);  /* 600px on desktop */
  }
}
```

**Your chart container fills this parent at 100% height**, automatically adapting to viewport breakpoints.

---

## Implementation Guide

### Step 1: Create Component Structure

```jsx
// src/components/charts/YourChart/YourChart.jsx
import { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import './YourChart.css';

const YourChart = ({ data, title }) => {
  return (
    <div className="your-chart-container">
      <div className="chart-header">
        <h2 className="chart-title">{title}</h2>
        {/* Add controls here */}
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="var(--chart-primary-color)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default YourChart;
```

### Step 2: Create CSS File

```css
/* src/components/charts/YourChart/YourChart.css */

/* Import design tokens */
@import '../../../styles/variables.css';

/* Container */
.your-chart-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  box-sizing: border-box;
  overflow: hidden;

  background-color: var(--chart-background);
  border-radius: var(--chart-border-radius);
  padding: var(--chart-padding);
  box-shadow: var(--chart-shadow);
}

/* Header */
.your-chart-container .chart-header {
  height: var(--chart-header-height-ratio);
  flex-shrink: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--chart-title-margin);
}

.your-chart-container .chart-title {
  font-size: var(--chart-title-size);
  font-weight: var(--chart-title-weight);
  color: var(--chart-title-color);
  margin: 0;
}

/* Chart Content */
.your-chart-container .recharts-responsive-container {
  height: var(--chart-content-height-ratio) !important;
  flex-shrink: 0;
}

/* Mobile Responsive */
@media (max-width: 768px) {
  .your-chart-container {
    padding: var(--spacing-md);
  }

  .your-chart-container .chart-header {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--spacing-sm);
  }

  .your-chart-container .chart-title {
    font-size: var(--font-size-lg);
  }
}
```

### Step 3: Use in Analysis Tab

```jsx
// In your page component
<AnalysisTab
  data={filteredData}
  renderCharts={(data) => (
    <>
      <YourChart
        data={data}
        title="Your Chart Title"
      />
      {/* More charts... */}
    </>
  )}
/>
```

---

## CSS Variables Reference

### Chart Container Styling
```css
--chart-border-radius      /* Border radius (12px) */
--chart-padding            /* Internal padding (24px) */
--chart-margin-bottom      /* Bottom margin (32px) */
--chart-shadow             /* Box shadow */
```

### Chart Dimensions
```css
--chart-height-default     /* Default height (500px) */
--chart-height-mobile      /* Mobile height (400px) */
--chart-height-tablet      /* Tablet height (500px) */
--chart-height-desktop     /* Desktop height (600px) */
```

### Chart Layout Proportions
```css
--chart-header-height-ratio    /* Header height (20%) */
--chart-content-height-ratio   /* Content height (80%) */
```

### Chart Typography
```css
--chart-title-size          /* Title font size */
--chart-title-weight        /* Title font weight */
--chart-title-color         /* Title text color */
--chart-title-margin        /* Title margin */

/* Fluid typography (responsive scaling) */
--chart-fluid-text-base     /* clamp(0.7rem, 2vw, 1rem) */
--chart-fluid-text-small    /* clamp(0.6rem, 1.5vw, 0.875rem) */
--chart-fluid-text-large    /* clamp(0.9rem, 2.5vw, 1.25rem) */
```

### Chart Colors
```css
--chart-primary-color       /* Primary bars/lines */
--chart-secondary-color     /* Secondary series */
--chart-tertiary-color      /* Third series */
--chart-background          /* Container background */
--chart-grid-line           /* Grid line color */
```

### Chart Controls
```css
--chart-control-width-sm    /* Small dropdown (100px) */
--chart-control-width-md    /* Medium dropdown (120px) */
--chart-control-width-lg    /* Large dropdown (160px) */
--chart-control-gap         /* Space between controls */
```

### Chart Margins
```css
--chart-margin-top          /* Top margin for Recharts */
--chart-margin-right        /* Right margin (for labels) */
--chart-margin-bottom       /* Bottom margin */
--chart-margin-left         /* Left margin (for Y-axis labels) */
```

**Note**: All these variables are defined in `/src/styles/tokens/charts.css` and can be changed globally.

---

## Testing Checklist

Before considering a chart complete, verify the following:

### Visual Testing
- [ ] Chart fills parent container at 100% height
- [ ] No overflow or scrollbars within chart
- [ ] Header and controls are clearly visible
- [ ] Title text is readable and properly sized
- [ ] Chart elements (axes, labels, tooltips) are clearly visible

### Responsive Behavior (Test at these widths)
- [ ] **320px** (Mobile small): Chart renders without horizontal scroll
- [ ] **768px** (Tablet): Controls stack properly, layout adjusts
- [ ] **1024px** (Tablet landscape): Chart uses appropriate sizing
- [ ] **1500px+** (Desktop): Chart fills larger container properly

### Breakpoint Transitions
- [ ] At 768px: Controls switch from horizontal to vertical layout
- [ ] At 1500px: Chart height increases from 500px to 600px
- [ ] Text remains legible at all sizes (no smaller than 10px)

### Grid Layout Testing (Multiple charts)
- [ ] Charts align properly in 2-column layout (1200px+)
- [ ] Charts stack to 1 column on mobile
- [ ] All charts in grid have equal height
- [ ] Gap spacing is consistent between charts

### Functional Testing
- [ ] Tooltips display correctly on hover
- [ ] Legend items are clickable (if applicable)
- [ ] Chart controls (dropdowns, buttons) function properly
- [ ] Data updates reflect immediately in chart
- [ ] Empty state displays when no data available

### CSS Variables Testing
- [ ] No hardcoded pixel values in component code
- [ ] All colors reference design tokens
- [ ] All spacing uses spacing scale variables
- [ ] All typography uses font scale variables

### Browser DevTools Checks
- [ ] No console errors
- [ ] No layout warnings
- [ ] ResponsiveContainer has proper dimensions in inspector
- [ ] Chart container shows `height: 100%` in computed styles

---

## Common Mistakes to Avoid

### 1. Hardcoding ResponsiveContainer Height
```jsx
// ❌ BAD
<ResponsiveContainer width="100%" height={500}>

// ✅ GOOD
<ResponsiveContainer width="100%" height="100%">
```

### 2. Missing Flex Layout
```css
/* ❌ BAD */
.chart-container {
  height: var(--chart-height-default);  /* Fixed height */
}

/* ✅ GOOD */
.chart-container {
  display: flex;
  flex-direction: column;
  height: 100%;  /* Fill parent */
}
```

### 3. Not Using CSS Variables
```css
/* ❌ BAD */
.chart-title {
  font-size: 20px;
  color: #333;
}

/* ✅ GOOD */
.chart-title {
  font-size: var(--chart-title-size);
  color: var(--chart-title-color);
}
```

### 4. Forgetting Mobile Breakpoints
```css
/* ❌ BAD - No mobile adjustments */
.chart-header {
  display: flex;
  justify-content: space-between;
}

/* ✅ GOOD - Mobile-responsive */
.chart-header {
  display: flex;
  justify-content: space-between;
}

@media (max-width: 768px) {
  .chart-header {
    flex-direction: column;
    gap: var(--spacing-sm);
  }
}
```

### 5. Not Testing with Real Data
- Always test with production-like data volumes
- Verify performance with large datasets
- Check edge cases (empty data, single data point, extreme values)

### 6. Ignoring Parent Container Context
```jsx
// ❌ BAD - Chart sets its own height
<div style={{ height: '500px' }}>
  <ResponsiveContainer width="100%" height="100%">

// ✅ GOOD - Parent AnalysisTab sets height via CSS
<AnalysisTab renderCharts={() => (
  <YourChart data={data} />  /* No height styling needed */
)} />
```

---

## Quick Reference

### Minimal Chart Component Template

```jsx
// Component
const ChartName = ({ data, title }) => (
  <div className="chart-name-container">
    <div className="chart-header">
      <h2 className="chart-title">{title}</h2>
    </div>
    <ResponsiveContainer width="100%" height="100%">
      {/* Your Recharts component */}
    </ResponsiveContainer>
  </div>
);
```

```css
/* CSS */
.chart-name-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: var(--chart-background);
  border-radius: var(--chart-border-radius);
  padding: var(--chart-padding);
}

.chart-name-container .chart-header {
  height: var(--chart-header-height-ratio);
  flex-shrink: 0;
}

.chart-name-container .recharts-responsive-container {
  height: var(--chart-content-height-ratio) !important;
  flex-shrink: 0;
}

@media (max-width: 768px) {
  .chart-name-container {
    padding: var(--spacing-md);
  }
}
```

---

## Additional Resources

- **Reference Implementation**: [TimeSeriesBarChart.jsx](../lifelog_website/src/components/charts/TimeSeriesBarChart/TimeSeriesBarChart.jsx)
- **Design Tokens**: [charts.css](../lifelog_website/src/styles/tokens/charts.css)
- **Analysis Tab Container**: [AnalysisTab.jsx](../lifelog_website/src/components/ui/page-sections/AnalysisTab/AnalysisTab.jsx)
- **Recharts Documentation**: https://recharts.org/

---

## Summary

**The Three Golden Rules:**

1. **ResponsiveContainer**: Always use `width="100%" height="100%"`
2. **Flex Layout**: Container must be `display: flex; flex-direction: column; height: 100%`
3. **CSS Variables**: Never hardcode dimensions, colors, or spacing

Follow these principles and your charts will be responsive, maintainable, and consistent across the entire application.
