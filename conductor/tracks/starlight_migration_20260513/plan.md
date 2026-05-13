# Plan: Starlight Documentation Migration

## Phase 1: Scaffold Starlight Site
- [ ] Create `docs/astro-site/package.json` with all dependencies
- [ ] Create `docs/astro-site/astro.config.mjs` with Starlight config and plugins
- [ ] Create `docs/astro-site/tsconfig.json`
- [ ] Create `docs/astro-site/public/.nojekyll`
- [ ] Create `docs/astro-site/src/content/docs/index.mdx` (homepage)

## Phase 2: Convert Core Documentation
- [ ] Create `introduction.mdx` from Sphinx `introduction.rst`
- [ ] Create `getting-started.mdx` from Sphinx `getting_started.rst`
- [ ] Create `cross-domain-usage.mdx` from Sphinx `cross_domain_usage.rst`
- [ ] Create `user-guide/index.mdx` with feature page links
- [ ] Create `user-guide/metamodeling.mdx`
- [ ] Create `user-guide/features/basic-voi.mdx`
- [ ] Create `user-guide/features/advanced-voi.mdx`
- [ ] Create `user-guide/features/cli.mdx`
- [ ] Create `user-guide/features/network-nma.mdx`
- [ ] Create `user-guide/features/portfolio-optimization.mdx`
- [ ] Create `user-guide/migration-guide.mdx`
- [ ] Create `user-guide/performance-guide.mdx`

## Phase 3: Convert VOI Methods Pages
- [ ] Create `methods/index.mdx` with method family listing
- [ ] Create `methods/evpi.mdx`, `evppi.mdx`, `evsi.mdx`, `enbs.mdx`
- [ ] Create `methods/structural-voi.mdx`, `network-nma.mdx`
- [ ] Create `methods/adaptive-trials.mdx`, `portfolio-optimization.mdx`
- [ ] Create `methods/sequential-voi.mdx`, `observational-voi.mdx`
- [ ] Create `methods/calibration-voi.mdx`, `ceaf.mdx`
- [ ] Create `methods/dominance.mdx`, `heterogeneity.mdx`

## Phase 4: Convert Developer Guide
- [ ] Create `developer-guide/index.mdx`
- [ ] Create architecture, profiling, Rust-core pages
- [ ] Create versioning, community, contributing pages
- [ ] Create polyglot-tooling, HPC contract pages
- [ ] Create starlight-docs-platform page

## Phase 5: Create Supporting Pages
- [ ] Create `contributing.mdx`, `changelog.mdx`
- [ ] Create `cli-reference.mdx`, `data-structures.mdx`, `backends.mdx`
- [ ] Create `examples/index.mdx`
- [ ] Create `api-reference/index.mdx` with polyglot component

## Phase 6: Configure Polyglot Plugin
- [ ] Configure `starlight-polyglot` for Python entry point `voiage`
- [ ] Configure `starlight-polyglot` for TypeScript entry point `voiage`
- [ ] Verify polyglot API reference generation

## Phase 7: Update CI/CD
- [ ] Update `.github/workflows/docs.yml` for Node.js + pnpm build
- [ ] Add starlight-links-validator check
- [ ] Verify GitHub Pages deployment path (`dist/`)

## Phase 8: Update Documentation Records
- [ ] Update `conductor/tech-stack.md` — replace Sphinx with Starlight + polyglot
- [ ] Create track spec and plan
- [ ] Archive track on completion

## Phase 9: Verification
- [ ] Run `pnpm install` and `pnpm run build` to verify build
- [ ] Verify all sidebar links resolve correctly
- [ ] Verify static assets (`.nojekyll`) in output
- [ ] Confirm Sphinx source files remain intact
