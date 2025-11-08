# Repository Structure Summary

## Branch Architecture

The `voiage` repository implements a separated branch architecture:

### Paper Branch
Contains:
- Complete JSS paper in Quarto format
- All paper-related files (bibliography, references, figures, examples)
- Supplementary materials and mathematical documentation
- Paper-specific validation and testing files

### Main Branch
Contains:
- Core software library code
- Tests for software functionality (excluding paper-specific tests)
- Examples for software usage (excluding paper-specific examples)
- Documentation for software usage
- Configuration files for software development
- All files necessary for installing and using the `voiage` library

## Benefits of This Architecture

1. **Clean Software Development**: The main branch focuses solely on the software library without paper-related files
2. **Independent Release Cycles**: Software and paper can follow different development and review schedules
3. **Focused Collaboration**: Different teams can work on software and paper independently
4. **Simplified Dependency Management**: Software users only get the library files without paper dependencies
5. **Clear Separation of Concerns**: Academic paper development is separated from software development

## Accessing the Paper

The paper describing the `voiage` library is available in the `paper` branch of this repository. To access it:

```bash
git checkout paper
git pull origin paper
cd paper
# Paper files are in this directory
```

## Paper Organization

In the paper branch:
- `paper.qmd` - Main paper in Quarto format
- `references_corrected.bib` - Bibliography file
- `output/` - Compiled PDF and other outputs
- `SUPPLEMENTARY_METHODS_AND_FORMULAE.md` - Supplementary methods documentation
- Additional example files and validation materials