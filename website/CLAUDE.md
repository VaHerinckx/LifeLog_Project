# Website — CLAUDE.md

Standards for the `website/` React dashboard.

## Gold Standard: ReadingPage.jsx

**All pages must strictly follow this component structure:**

1. `PageHeader` — Title and description
2. `TabNavigation` — Content/analysis tab switching
3. `FilteringPanel` — Centralized filtering (uses `filterConfigs.jsx`)
4. `CardsPanel` — KPI stats display
5. `ContentTab` — Grid/list/timeline views (auto-pagination via `ContentCardsGroup`)
6. `AnalysisTab` — Charts (auto-wrapping, data passed per-chart)
7. Detail Modal — Item details view

Never create page-specific CSS files or page-specific AnalysisTab components.

## DRY — Check Before Writing

| Need | Where to look |
|------|--------------|
| Reusable components | `/src/components/ui/` |
| Date operations, filtering, statistics | `/src/utils/` |
| Filter configs | `/src/config/filterConfigs.jsx` |
| Charts | `/src/components/charts/` |
| Centralized configs | `/src/config/` |

## Design System Rules

- **Always use CSS variables** — `var(--spacing-lg)` not `16px`
- **Import order** — `variables.css` first, then `components.css`
- **Mobile-first** — base styles for mobile, enhance for desktop
- **Breakpoints** — 768px (tablet), 1024px, 1200px (desktop)
- Design tokens: `src/styles/variables.css` + `src/styles/tokens/` (colors, spacing, typography, shadows, borders, charts)
- Reusable classes: `src/styles/components.css`

## Known Gotchas

- `fetchData('health')` does **not** exist — use `'healthDaily'` or `'healthHourly'`
- Music data uses **string timestamps** (`YYYY-MM-DD HH:MM:SS`), not Date objects — filter with string operations, not Date parsing
- Music loads in chunks (10K rows) — avoid Date parsing on the full dataset
- `KpiCard` renders as its own styled card — never nest it inside another card component
- `KPICardsPanel` injects data via `dataSource` prop using `React.cloneElement`

## References

- Component patterns, pagination, AnalysisTab API, new page workflow: [../docs/website-examples.md](../docs/website-examples.md)
- New data source setup + impact testing checklist: [../docs/data-integration.md](../docs/data-integration.md)
