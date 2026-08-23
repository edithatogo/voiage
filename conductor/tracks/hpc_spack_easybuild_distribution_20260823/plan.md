# Track Plan: HPC Spack & EasyBuild Distribution

## Phases

- [ ] **Phase 1: Spack & EasyBuild Recipe Authoring**
  - [ ] Write `packaging/spack/package.py`.
  - [ ] Write `packaging/easybuild/voiage-2.1.0-foss-2023a.eb`.
- [ ] **Phase 2: HPC Documentation & Upstream Handoff**
  - [ ] Document cluster usage in `docs/astro-site/src/content/docs/developer-guide/hpc-deployment.mdx`.
  - [ ] Package upstream PR documentation in `docs/release/hpc-distribution-handoff.md`.
- [ ] **Phase 3: Automated Verification & Staging**
  - [ ] Run full test suite and quality harness.
  - [ ] Commit, open PR, verify green CI, and merge.
