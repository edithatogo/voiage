# Track: Starlight Documentation Migration

## ID
`starlight_migration_20260513`

## Fulfills
- **REQ-MIG-002**: Replace Sphinx documentation system with Starlight + polyglot

## Status
Active

## Objective
Migrate the voiage documentation system from Sphinx (RST-based) to Starlight
(Astro-based, MDX) with starlight-polyglot integration for auto-generated
multi-language API references.

## Motivation
The previous tech-stack.md listed Starlight as a "candidate" platform. This
track makes Starlight the official documentation platform, replacing Sphinx,
while preserving the existing Sphinx source files as a backup.

## Scope

### In Scope
1. Create `docs/astro-site/` with full Starlight scaffold
2. Convert existing Sphinx RST documentation to MDX equivalents
3. Integrate starlight-polyglot for Python (and TypeScript if available) API
   reference generation
4. Add starlight-versions for versioned documentation navigation
5. Add starlight-links-validator for CI link validation
6. Add starlight-llms-txt for LLM-friendly text export
7. Update `.github/workflows/docs.yml` to build and deploy Starlight site
8. Update `conductor/tech-stack.md` to reflect Sphinx → Starlight replacement

### Out of Scope
- Removal or modification of existing Sphinx files (kept as backup)
- Content rewrite of existing canonical markdown guides
- Jupyter notebook conversion
- Advanced UX plugins (image-zoom, heading-badges, sidebar-topics, utils)

## Deliverables
- Fully scaffolded Starlight site at `docs/astro-site/`
- MDX content files covering all existing documentation sections
- Updated CI/CD pipeline for Starlight build + deploy
- Updated tech-stack documentation
- Track closure archive

## Dependencies
- `starlight-polyglot` at `file:/Users/doughnut/GitHub/starlight-polyglot/packages/starlight-polyglot`
- Node.js 22 and pnpm 10 for CI build

## Risk and Mitigation
| Risk | Mitigation |
|------|------------|
| Polyglot plugin may require path adjustments | Use file: link for local development; update before npm publish |
| Sphinx content may have RST-specific formatting | Manual MDX conversion preserving content hierarchy |
| Broken links in migrated content | starlight-links-validator plugin catches issues in CI |

## Acceptance Criteria
- [ ] Starlight site builds successfully with `pnpm run build`
- [ ] All existing documentation sections have MDX equivalents
- [ ] Polyglot API reference renders in the site
- [ ] CI pipeline deploys Starlight site to GitHub Pages
- [ ] Tech-stack.md accurately reflects the new documentation toolchain
- [ ] Sphinx source files remain intact as backup
