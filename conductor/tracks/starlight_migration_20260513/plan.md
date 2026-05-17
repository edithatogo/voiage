# Plan: Starlight Documentation Migration

## Phase 1: Scaffold Starlight Site
- [x] Create `docs/astro-site/package.json` with all dependencies
- [x] Create `docs/astro-site/astro.config.mjs` with Starlight config and plugins
- [x] Create `docs/astro-site/tsconfig.json`
- [x] Create `docs/astro-site/public/.nojekyll`
- [x] Create `docs/astro-site/src/content/docs/index.mdx` (homepage)

## Phase 2: Convert Core Documentation
- [x] Create `introduction.mdx` from Sphinx `introduction.rst`
- [x] Create `getting-started.mdx` from Sphinx `getting_started.rst`
- [x] Create `cross-domain-usage.mdx` from Sphinx `cross_domain_usage.rst`
- [x] Create `user-guide/index.mdx` with feature page links
- [x] Create `user-guide/metamodeling.mdx`
- [x] Create `user-guide/features/basic-voi.mdx`
- [x] Create `user-guide/features/advanced-voi.mdx`
- [x] Create `user-guide/features/cli.mdx`
- [x] Create `user-guide/features/network-nma.mdx`
- [x] Create `user-guide/features/portfolio-optimization.mdx`
- [x] Create `user-guide/migration-guide.mdx`
- [x] Create `user-guide/performance-guide.mdx`

## Phase 3: Convert VOI Methods Pages
- [x] Create `methods/index.mdx` with method family listing
- [x] Create `methods/evpi.mdx`, `evppi.mdx`, `evsi.mdx`, `enbs.mdx`
- [x] Create `methods/structural-voi.mdx`, `network-nma.mdx`
- [x] Create `methods/adaptive-trials.mdx`, `portfolio-optimization.mdx`
- [x] Create `methods/sequential-voi.mdx`, `observational-voi.mdx`
- [x] Create `methods/calibration-voi.mdx`, `ceaf.mdx`
- [x] Create `methods/dominance.mdx`, `heterogeneity.mdx`

## Phase 4: Convert Developer Guide
- [x] Create `developer-guide/index.mdx`
- [x] Create architecture, profiling, Rust-core pages
- [x] Create versioning, community, contributing pages
- [x] Create polyglot-tooling, HPC contract pages
- [x] Create starlight-docs-platform page

## Phase 5: Create Supporting Pages
- [x] Create `contributing.mdx`, `changelog.mdx`
- [x] Create `cli-reference.mdx`, `data-structures.mdx`, `backends.mdx`
- [x] Create `examples/index.mdx`
- [x] Create `api-reference/index.mdx` with polyglot component

## Phase 6: Configure Polyglot Plugin
- [x] Configure `starlight-polyglot` for Python entry point `voiage`
- [x] Configure `starlight-polyglot` for TypeScript entry point `voiage`
- [x] Verify polyglot API reference generation

## Phase 7: Update CI/CD
- [x] Update `.github/workflows/docs.yml` for Node.js + pnpm build
- [x] Add starlight-links-validator check
- [x] Verify GitHub Pages deployment path (`dist/`)

## Phase 8: Update Documentation Records
- [x] Update `conductor/tech-stack.md` — replace Sphinx with Starlight + polyglot
- [x] Create track spec and plan
- [x] Archive track on completion

## Phase 9: Verification
- [x] Run `pnpm install` and `pnpm run build` to verify build
- [x] Verify all sidebar links resolve correctly
- [x] Verify static assets (`.nojekyll`) in output
- [x] Confirm Sphinx source files remain intact
