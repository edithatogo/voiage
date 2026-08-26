# Track: HPC Spack & EasyBuild Distribution

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Evidence](./evidence.jsonl)
- [GitHub issue #1025](https://github.com/edithatogo/voiage/issues/1025)
- [Project 28](https://github.com/users/edithatogo/projects/28)
- [Registration PR #1027](https://github.com/edithatogo/voiage/pull/1027)

Status: in progress. Repository implementation remains pending, and upstream
Spack or EasyBuild pull requests remain maintainer-controlled.

---

## Objectives
1. Author and validate `py-voiage` Python package recipe in `packaging/spack/package.py`.
2. Author and validate EasyBuild easyconfig in `packaging/easybuild/voiage-2.1.0-foss-2023a.eb`.
3. Document HPC cluster module usage (`module load voiage`) and execution verification.
4. Prepare upstream PR staging packets for `spack/spack` and `easybuilders/easybuild-easyconfigs`.
