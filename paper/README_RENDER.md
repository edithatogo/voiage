# Rendering Instructions for voiage Paper

This repository contains the source files for the "voiage: A Python Library for Value of Information Analysis" paper, with configurations for both arXiv and Journal of Statistical Software (JSS) formats.

## File Structure

- `paper.qmd` - Main paper source file in Quarto markdown format
- `references_corrected.bib` - Bibliography in BibLaTeX format
- `_quarto_arxiv.yml` - Configuration for arXiv format
- `_quarto_jss.yml` - Configuration for JSS format
- `arxiv_output/` - Directory where arXiv PDF will be generated
- `jss_output/` - Directory where JSS PDF will be generated

## Requirements

To render the paper, you need:
- [Quarto](https://quarto.org/) installed on your system
- A LaTeX distribution (e.g., TeX Live, MiKTeX) for PDF generation
- Python (for any computational elements in the document)

## Rendering Instructions

### For arXiv Format

To render the paper in arXiv format, run one of these commands:

```bash
# Using the arXiv profile (if you've set up profiles)
quarto render paper.qmd --profile arxiv

# Or directly specify the configuration
quarto render paper.qmd --config _quarto_arxiv.yml

# Or render with specific output
quarto render paper.qmd --output paper_arxiv.pdf --output-dir arxiv_output
```

### For JSS Format

To render the paper in JSS format, run one of these commands:

```bash
# Using the JSS profile (if you've set up profiles)
quarto render paper.qmd --profile jss

# Or directly specify the configuration
quarto render paper.qmd --config _quarto_jss.yml

# Or render with specific output
quarto render paper.qmd --output paper_jss.pdf --output-dir jss_output
```

## Format-Specific Notes

### arXiv Format
- Uses standard LaTeX article class
- Uses the mathpazo font for a more professional appearance
- Includes all necessary packages for mathematical notation
- Optimized for arXiv's submission requirements

### JSS Format
- Uses the JSS LaTeX document class
- Follows the Journal of Statistical Software formatting requirements
- Includes JSS-specific header information (running titles, etc.)
- Optimized for JSS publication standards

## Expected Output

When rendered successfully, both formats will generate:
- A PDF file with proper formatting
- A LaTeX .tex file (if keep-tex is enabled in the configuration)
- Properly formatted citations and bibliography
- All figures and tables positioned as specified

## Troubleshooting

If you encounter issues:
1. Ensure your LaTeX installation is complete and includes all required packages
2. Make sure Quarto is properly installed and in your PATH
3. Check that all referenced files (bibliography, figures, etc.) exist in the expected locations
4. Verify that any computational cells have their required Python dependencies installed

## Verification

To verify the output matches expectations:
- Check that citations appear correctly in both numerical and author-year formats
- Verify that mathematical notation renders properly
- Confirm that the bibliography includes all references with complete information
- Ensure the document structure matches the target format requirements