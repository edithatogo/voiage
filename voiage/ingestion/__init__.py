"""Optional source-format adapters for the normalized VOI input contract."""

from .base import IngestionError, IngestionProvider, SourceAccessPolicy
from .croissant import CroissantProvider
from .dataframe import from_dataframe
from .frictionless import FrictionlessProvider
from .registry import ProviderRegistry, default_registry

__all__ = [
    "CroissantProvider",
    "FrictionlessProvider",
    "IngestionError",
    "IngestionProvider",
    "ProviderRegistry",
    "SourceAccessPolicy",
    "default_registry",
    "from_dataframe",
]
