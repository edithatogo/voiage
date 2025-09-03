# Repository Structure for Qoder Quest

## Overview

This document explains the repository structure and how to properly use the documentation files with Qoder Quest.

## Repository Structure

The voiage repository has the following structure:

```
voiage/                    # Repository root
└── voiage/                # Main Python package (current directory)
    ├── .qoder/            # Qoder-specific files
    │   └── quests/
    │       └── map-repo.md  # Original design document
    ├── REPO_MAP.md        # Repository map (detailed documentation)
    ├── ROADMAP_STATUS.md  # Roadmap status
    ├── REPO_MAP_CORRECTED.md  # Corrected repository map
    ├── ROADMAP_STATUS_CORRECTED.md  # Corrected roadmap status
    ├── README_FOR_QODER_QUEST.md  # This file
    ├── __init__.py
    ├── analysis.py
    ├── backends.py
    ├── cli.py
    ├── config.py
    ├── core/
    ├── exceptions.py
    ├── metamodels.py
    ├── methods/
    ├── plot/
    ├── schema.py
    └── stats.py
```

## Using Documentation with Qoder Quest

To properly use the documentation files with Qoder Quest:

1. **Move Documentation Files**: For optimal use with Qoder Quest, the documentation files should be moved to the repository root directory:
   - Move `REPO_MAP_CORRECTED.md` to `../REPO_MAP.md`
   - Move `ROADMAP_STATUS_CORRECTED.md` to `../ROADMAP_STATUS.md`
   - Move `README_FOR_QODER_QUEST.md` to `../README.md`

2. **Understanding the Context**: The documentation was originally created while inside the inner [voiage](file:///Users/doughnut/GitHub/voiage/voiage/__init__.py) directory, but Qoder Quest should be run from the repository root to properly understand the entire project structure.

3. **Key Documentation Files**:
   - `REPO_MAP_CORRECTED.md`: Comprehensive repository map with detailed component descriptions
   - `ROADMAP_STATUS_CORRECTED.md`: Comparison of design documentation with actual implementation status
   - `.qoder/quests/map-repo.md`: Original design document

## Repository Context

The voiage repository is a Python library for Value of Information Analysis (VOIA) in health economics and decision modeling. It provides tools for calculating various VOI metrics including EVPI, EVPPI, and EVSI, as well as portfolio optimization for research prioritization.

## Current Status

The repository is currently in an early development stage (v0.1), with core infrastructure implemented but many advanced features still as placeholders. Most files explicitly mention "v0.1" and indicate they are placeholders for future implementation.

## For Developers

When working with this repository:
1. Refer to `REPO_MAP_CORRECTED.md` for detailed component documentation
2. Check `ROADMAP_STATUS_CORRECTED.md` for implementation status
3. Note that many advanced methods are currently placeholders with "NotImplementedError"
4. Follow the backward compatibility practices documented in the codebase