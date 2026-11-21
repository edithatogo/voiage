"""Environment for isolated installed-wheel consumer tests."""

import os
from pathlib import Path


def wheel_consumer_environment(tmp_path: Path) -> dict[str, str]:
    """Keep isolation by default; reuse only an explicitly selected cache."""
    environment = os.environ.copy()
    selected_cache = environment.pop("VOIAGE_TEST_UV_CACHE_DIR", None)
    cache_dir = tmp_path / "uv-cache"
    if selected_cache:
        cache_dir = Path(selected_cache).expanduser().resolve()
        if not cache_dir.is_dir() or not os.access(cache_dir, os.W_OK):
            raise ValueError("VOIAGE_TEST_UV_CACHE_DIR must be a writable directory")
    environment["UV_CACHE_DIR"] = str(cache_dir)
    environment["UV_LINK_MODE"] = "copy"
    return environment
