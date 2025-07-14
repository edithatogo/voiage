# voiage/methods/__init__.py

"""
Value of Information (VOI) calculation methods.

This package contains modules for different types of VOI analyses.
Each module typically implements one or more related VOI methods.

Modules:
- basic: EVPI (Expected Value of Perfect Information) and
         EVPPI (Expected Value of Partial Perfect Information).
- sample_information: EVSI (Expected Value of Sample Information) and
                      ENBS (Expected Net Benefit of Sampling).
- structural: Structural EVPI/EVPPI for model uncertainty.
- network_nma: EVSI for Network Meta-Analysis.
- adaptive: EVSI for adaptive trial designs.
- portfolio: Portfolio VOI for research prioritization.
- sequential: Dynamic and sequential VOI.
- observational: VOI for observational data.
- calibration: VOI for model calibration data.

Each method should be designed to be as modular and extensible as possible,
potentially allowing for different computational backends (NumPy, JAX) or
algorithmic variations where appropriate.
"""

# Import key functions from submodules to make them available at `voiage.methods.<method_name>`
# or even `voiage.<method_name>` if re-exported in `voiage/__init__.py`.

# from .basic import evpi, evppi
# from .sample_information import evsi, enbs
# from .structural import structural_evpi, structural_evppi
# from .network_nma import evsi_nma
# from .adaptive import adaptive_evsi
# from .portfolio import portfolio_voi
# from .sequential import sequential_voi
# from .observational import voi_observational
# from .calibration import voi_calibration

# To be populated as methods are implemented.

# Example of a registry for methods (optional, for advanced plugin architecture)
# _method_registry = {}

# def register_method(name: str, func: Callable):
#     if name in _method_registry:
#         print(f"Warning: Method '{name}' is already registered. Overwriting.")
#     _method_registry[name] = func

# def get_method(name: str) -> Callable:
#     if name not in _method_registry:
#         raise ValueError(f"Method '{name}' not found in registry.")
#     return _method_registry[name]
