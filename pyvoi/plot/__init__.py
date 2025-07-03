# pyvoi/plot/__init__.py

"""
Plotting utilities for pyVOI.

This package provides functions to generate common visualizations used in
Value of Information analyses.

Modules:
- voi_curves: Plotting EVPI, EVPPI, EVSI over sample size, time, or other parameters.
- ceac: Plotting Cost-Effectiveness Acceptability Curves (CEACs),
        Cost-Effectiveness Planes, EVPPI surfaces.

These modules will typically use Matplotlib and Seaborn (or ArviZ for Bayesian plots)
as backend plotting libraries.
"""

# Import key plotting functions to make them available at `pyvoi.plot.<plot_function>`
# from .voi_curves import plot_evsi_vs_sample_size, plot_dynamic_voi
# from .ceac import plot_ceac, plot_ce_plane, plot_evppi_surface

# To be populated as plotting functions are implemented.

# Example:
# def example_plot():
#     """A placeholder for a future plotting function."""
#     try:
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.title("Example pyVOI Plot")
#         plt.xlabel("X-axis")
#         plt.ylabel("Y-axis")
#         # plt.show() # Usually not called directly in library code
#         print("Example plot function called (matplotlib would show a plot).")
#         return plt.gca() # Return axis for further customization
#     except ImportError:
#         print("Matplotlib not installed. Plotting functions will not work.")
#         return None
