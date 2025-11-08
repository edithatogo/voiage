# Branch Architecture for voiage Repository

## Overview

This document describes the separated branch architecture implemented for the `voiage` repository, where the software library and the academic paper are maintained in separate branches for better organization and focused development.

## Branch Structure

### Main Branch (`main`)
The `main` branch contains only the core software library and related code:

- `voiage/` - Core library source code
- `tests/` - Software tests
- `examples/` - Software usage examples
- `docs/` - Software documentation (excluding paper)
- `pyproject.toml`, `setup.py` - Packaging configuration
- `README.md` - Software README
- `CHANGELOG.md` - Software change log
- Other standard software project files

### Paper Branch (`paper`)
The `paper` branch contains only the academic paper and related documentation:

- `paper/` - Complete academic paper files
  - `paper.qmd` - Main Quarto paper source file
  - `references_corrected.bib` - Paper bibliography
  - `SUPPLEMENTARY_METHODS_AND_FORMULAE.md` - Supplementary methods documentation
  - `output/` - Compiled paper outputs (PDF, LaTeX)
  - JSS/Quarto configuration files
  - Paper-specific examples and validation code
  - Additional paper documentation and review materials

## Benefits of Separation

### For Software Development
- Cleaner, more focused repository for code contributions
- No paper-related files in software package
- Easier dependency management for software users
- Simpler CI/CD workflows for software changes
- Uncluttered git history for software development

### For Paper Development
- Dedicated branch for academic paper work
- Independent versioning and review process
- Separate review and approval workflows
- Ability to track paper-specific changes separately
- No interference with software development workflows

### For Collaboration
- Software developers can focus on code without paper files cluttering the repository
- Academic collaborators can work on paper without accessing full software history
- Different timelines for software releases vs. paper publication
- Separate issue tracking and project management

## Workflow

### For Software Updates
1. Work on features, bug fixes, and improvements in feature branches
2. Merge to `main` branch after review
3. Software releases are tagged from the `main` branch
4. Paper branch remains unaffected

### For Paper Updates
1. Work on paper content in the dedicated `paper` branch
2. Paper revisions, reviews, and submission happen in the `paper` branch
3. Paper-related artifacts are isolated in the dedicated branch
4. Software branch remains unaffected

## Integration Points

Though the branches are separate, they maintain appropriate linking:

- Paper references the software repository and specific tagged versions
- Software documentation may reference the paper for methodological details
- DOI and citation information is maintained in both locations
- GitHub Actions can be configured to build paper artifacts independently

## Migration Notes

This architecture was implemented after the initial paper development to improve project organization. The paper content was moved to the dedicated `paper` branch, and the main branch has been cleaned of paper-related files. The paper is now maintained entirely in its dedicated branch, while the main branch focuses solely on the software library.