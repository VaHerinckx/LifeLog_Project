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
- Design tokens: `src/styles/variables.css` (80+ tokens)
- Reusable classes: `src/styles/components.css`

## Component Patterns

**Pagination:** Auto-handled by `ContentCardsGroup` (triggers at >200 items, 100/page default)
- Pass the full filtered array to `ContentTab.items` — no manual `currentPage` state in parent

**AnalysisTab:** Use the reusable component from `/src/components/ui/AnalysisTab/`
- Render function returns chart elements only
- Each chart specifies its own `data` prop
- Filtering handled by parent `FilteringPanel`, not inside AnalysisTab

**Adding a data source:** 3-step setup: `.env` → `config.js` → `DataContext`

## References

- Detailed workflow: [../docs/website-examples.md](../docs/website-examples.md)
- Adding data sources + testing checklist: [../docs/data-integration.md](../docs/data-integration.md)
