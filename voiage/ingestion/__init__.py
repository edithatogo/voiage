"""Optional source-format adapters for the normalized VOI input contract."""

from .base import (
    IngestionError,
    IngestionProvider,
    ProviderCapabilities,
    SourceAccessPolicy,
)
from .croissant import CroissantProvider
from .dataframe import from_dataframe
from .frictionless import FrictionlessProvider
from .live_probe import AuthoritativeProbeGateError, run_authoritative_probe
from .registry import ProviderRegistry, default_registry, discover_entry_point_providers

__all__ = [
    "CroissantProvider",
    "AuthoritativeProbeGateError",
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
