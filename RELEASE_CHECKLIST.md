# Release Checklist for voiage

## Pre-Release Preparation

### 1. Code Quality and Testing
- [ ] Run all tests: `tox -e py311` (or current Python version)
- [ ] Run linting checks: `tox -e lint`
- [ ] Run type checking: `tox -e typecheck`
- [ ] Run security scanning: `tox -e security`
- [ ] Run safety dependency checks: `tox -e safety`
- [ ] Check code coverage: `tox -e coverage_report`
- [ ] Verify documentation builds: `tox -e docs`

### 2. Version Updates
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `voiage/__init__.py`
- [ ] Update version in `docs/conf.py`
- [ ] Update changelog in `CHANGELOG.md`
- [ ] Update release date in documentation

### 3. Documentation
- [ ] Review and update README.md
- [ ] Update installation instructions if needed
- [ ] Verify all links are working
- [ ] Check that examples are up-to-date
- [ ] Review API documentation for accuracy

### 4. Final Validation
- [ ] Run integration tests
- [ ] Test CLI functionality: `python -m voiage.cli --help`
- [ ] Test web API functionality
- [ ] Test widget functionality in Jupyter
- [ ] Verify package builds correctly: `python -m build`

## Release Process

### 1. Git Operations
- [ ] Commit all changes with descriptive message
- [ ] Create and push tag: `git tag -a v0.X.X -m "Release version 0.X.X"`
- [ ] Push tags: `git push origin --tags`

### 2. GitHub Release
- [ ] Create GitHub release from tag
- [ ] Add release notes with key changes
- [ ] Include any migration notes if breaking changes
- [ ] Add contributors list

### 3. PyPI Publishing
- [ ] GitHub Actions will automatically publish to PyPI
- [ ] Verify package is available on PyPI
- [ ] Test installation: `pip install voiage==0.X.X`

### 4. Conda-Forge Update
- [ ] Update conda recipe in `conda-recipe/meta.yaml`
- [ ] Submit PR to conda-forge feedstock
- [ ] Verify package is available on conda-forge
- [ ] Test installation: `conda install -c conda-forge voiage`

### 5. Docker Images
- [ ] Build and push Docker images
- [ ] Update Docker Hub with new tags
- [ ] Test Docker images: `docker run voiage:0.X.X --help`

## Post-Release Verification

### 1. Package Availability
- [ ] Verify PyPI package: https://pypi.org/project/voiage/
- [ ] Verify conda-forge package: https://anaconda.org/conda-forge/voiage
- [ ] Verify GitHub release assets
- [ ] Verify documentation deployment

### 2. Installation Testing
- [ ] Test PyPI installation in clean environment
- [ ] Test conda-forge installation in clean environment
- [ ] Test Docker image functionality
- [ ] Test CLI commands work correctly

### 3. Communication
- [ ] Announce release on relevant channels
- [ ] Update project website if applicable
- [ ] Notify collaborators and contributors
- [ ] Update any dependent projects

## Versioning Guidelines

Following Semantic Versioning (SemVer):
- **Major**: Breaking changes to API
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, documentation updates, non-breaking improvements

## Emergency Procedures

### Hotfix Process
1. Create hotfix branch from latest release tag
2. Make minimal changes to fix the issue
3. Follow abbreviated release checklist
4. Merge to main and create new patch release

### Rollback Procedure
1. If PyPI release has critical issues, yank the version
2. Create issue explaining the problem
3. Work on fix following hotfix process
4. Communicate with users about workaround

This checklist ensures consistent, reliable releases for the voiage library.