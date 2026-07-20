"""Small fail-closed C14 policy decisions protected by strict mutation CI."""

from __future__ import annotations

_SENSITIVE_LOG_FRAGMENTS = (
    "access_key",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "passphrase",
    "password",
    "private_key",
    "secret",
    "session",
    "signature",
    "token",
)


def is_sensitive_log_key(key: str) -> bool:
    """Classify credential-bearing log keys conservatively and case-insensitively."""
    folded = key.casefold()
    return any(fragment in folded for fragment in _SENSITIVE_LOG_FRAGMENTS)


def require_schema_fingerprint(declared: object, computed: str, *, label: str) -> None:
    """Fail closed unless a declared schema fingerprint matches computed bytes."""
    if not isinstance(declared, str) or declared != computed:
        raise ValueError(f"{label} schema fingerprint does not match fields")


__all__ = ["is_sensitive_log_key", "require_schema_fingerprint"]
