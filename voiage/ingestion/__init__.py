"""Optional source-format adapters for the normalized VOI input contract."""

from typing import TYPE_CHECKING

from .base import (
    INGESTION_PROVIDER_SDK_VERSION,
    IngestionError,
    IngestionProvider,
    ProviderCapabilities,
    SourceAccessPolicy,
)
from .dataframe import from_dataframe
from .live_probe import AuthoritativeProbeGateError, run_authoritative_probe
from .registry import ProviderRegistry, default_registry, discover_entry_point_providers

if TYPE_CHECKING:
    from .croissant import CroissantProvider
    from .frictionless import FrictionlessProvider

__all__ = [
    "INGESTION_PROVIDER_SDK_VERSION",
    "AuthoritativeProbeGateError",
    "CroissantProvider",
    "FrictionlessProvider",
    "IngestionError",
    "IngestionProvider",
    "ProviderCapabilities",
    "ProviderRegistry",
    "SourceAccessPolicy",
    "default_registry",
    "discover_entry_point_providers",
    "from_dataframe",
    "run_authoritative_probe",
]


def __getattr__(name: str) -> object:
    """Load built-in source adapters only when their public names are requested."""
    if name == "CroissantProvider":
        from .croissant import CroissantProvider

        return CroissantProvider
    if name == "FrictionlessProvider":
        from .frictionless import FrictionlessProvider

        return FrictionlessProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
