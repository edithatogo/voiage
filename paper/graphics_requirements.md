# Graphics for voiage Repository

## Purpose
To create attractive scientific graphics that illustrate the unique or best parts of the voiage algorithm, for use in the GitHub repository README and paper/website.

## Graphics to Create

### 1. Conceptual VOI Workflow Diagram
- Title: "Value of Information Analysis Workflow"
- Description: A flowchart showing the progression from uncertainty in health economic models to quantified value of potential research investments
- Elements:
  - Initial model uncertainty
  - Parameter sensitivity analysis
  - VOI calculation (EVPI/EVPPI)
  - Research prioritization
  - Optimal study design
- Style: Clean, scientific, using color-coding for different VOI methods

### 2. Uncertainty Decomposition Visualization
- Title: "Decomposing Decision Uncertainty"
- Description: A tornado diagram or similar visualization showing which parameters contribute most to decision uncertainty
- Elements:
  - Parameter names on y-axis
  - Uncertainty contribution on x-axis
  - Color coding by parameter category
- Style: Information-rich, clearly labeled for non-experts

### 3. Value of Information Curve
- Title: "Value of Information vs. Research Investment"
- Description: A curve showing how the value of information changes with sample size or precision
- Elements:
  - Research investment (sample size, precision) on x-axis
  - Expected value of information on y-axis
  - Optimal point marking cost-effectiveness
- Style: Clear, intuitive for stakeholders

### 4. Comparison Visualization
- Title: "voiage vs. Alternative Approaches"
- Description: Visual comparison showing advantages of voiage over existing tools
- Elements:
  - Comparison table with features
  - Performance benchmarks
  - Integration capabilities
- Style: Clean, highlighting key differentiators

### 5. Application Example Diagram
- Title: "Practical Applications in Health Economics"
- Description: Illustration of how voiage can be applied to real-world problems
- Elements:
  - Healthcare decision context
  - Model structure
  - VOI results
  - Decision implications
- Style: Practical, relatable to health economists

### 6. Technical Architecture Diagram
- Title: "voiage Architecture"
- Description: Schematic showing the library's modular design and integration capabilities
- Elements:
  - Core VOI methods
  - Computational backends
  - Healthcare utilities
  - Integration points
- Style: Technical but accessible

## Technical Specifications

### Format
- SVG for scalability and editability
- PNG for direct use in README
- High resolution (300 DPI) for print applications

### Color Scheme
- Primary: Scientific blue (#1f77b4) and teal (#2ca02c)
- Secondary: Orange (#ff7f0e) for highlights
- Neutral: Grays (#7f7f7f) for backgrounds
- Accessibility: Colorblind-friendly palette

### Style Guidelines
- Consistent iconography
- Sans-serif font (e.g., Helvetica, Arial)
- Clear labeling
- Minimal but informative

## Implementation Notes

These graphics should be created using tools such as:
- Python with matplotlib/seaborn for data-driven visualizations
- Adobe Illustrator or Inkscape for detailed diagrams
- Jupyter notebooks to generate reproducible graphics

The graphics should be stored in a `/graphics` directory in the repository with clear licensing information.

## Integration Points

- README.md: Use graphics to visually explain the library's purpose and capabilities
- Documentation: Include graphics to illustrate concepts
- Paper: Use graphics to enhance understanding of methods
- Website: Feature graphics prominently to attract users