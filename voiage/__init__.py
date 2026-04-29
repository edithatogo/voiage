"""Top-level public API for `voiage`.

The package exposes the curated core analysis surface together with the main
subpackage namespaces for advanced workflows.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from . import (
    analysis,
    backends,
    cli,
    config,
    core,
    ecosystem_integration,
    exceptions,
    factory,
    fluent,
    health_economics,
    hta_integration,
    methods,
    multi_domain,
    plot,
    schema,
)
from .analysis import DecisionAnalysis
from .ecosystem_integration import HeomlRunBundle, load_heoml_run_bundle
from .methods.basic import evpi, evppi
from .methods.distributional import (
    DistributionalEquityResult,
    value_of_distributional_equity,
)
from .methods.implementation import (
    ImplementationAdjustedResult,
    value_of_implementation,
)
from .methods.perspective import (
    Perspective,
    PerspectiveSet,
    ValueOfPerspectiveResult,
    value_of_perspective,
)
from .methods.sample_information import enbs, evsi
from .schema import (
    DecisionOption,
    ParameterSet,
    PortfolioSpec,
    PortfolioStudy,
    TrialDesign,
    ValueArray,
)

try:
    __version__ = _package_version("voiage")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.0.0"

__all__ = [
    "DecisionAnalysis",
    "DecisionOption",
    "DistributionalEquityResult",
    "HeomlRunBundle",
    "ImplementationAdjustedResult",
    "ParameterSet",
    "Perspective",
    "PerspectiveSet",
    "PortfolioSpec",
    "PortfolioStudy",
    "TrialDesign",
    "ValueArray",
    "ValueOfPerspectiveResult",
    "analysis",
    "backends",
    "cli",
    "config",
    "core",
    "ecosystem_integration",
    "enbs",
    "evpi",
    "evppi",
    "evsi",
    "exceptions",
    "factory",
    "fluent",
    "health_economics",
    "hta_integration",
    "load_heoml_run_bundle",
    "methods",
    "multi_domain",
    "plot",
    "schema",
    "value_of_distributional_equity",
    "value_of_implementation",
    "value_of_perspective",
]
