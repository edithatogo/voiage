"""Mutation-complete tests for C14 assurance policy decisions."""

from __future__ import annotations

import pytest

from voiage.assurance_policy import is_sensitive_log_key, require_schema_fingerprint


@pytest.mark.parametrize(
    "key",
    [
        "ACCESS_KEY_ID",
        "api_key",
        "authorization",
        "cookie_header",
        "credential",
        "passphrase",
        "password_hash",
        "private_key",
        "client_secret",
        "session_id",
        "signature",
        "refresh_token",
    ],
)
def test_sensitive_log_key_covers_every_credential_family(key: str) -> None:
    assert is_sensitive_log_key(key) is True


def test_sensitive_log_key_does_not_hide_safe_scientific_fields() -> None:
    assert is_sensitive_log_key("numerical_policy_id") is False
    assert is_sensitive_log_key("incremental_cost") is False


def test_schema_fingerprint_requires_exact_computed_identity() -> None:
    require_schema_fingerprint("a" * 64, "a" * 64, label="current")
    with pytest.raises(ValueError, match="current schema fingerprint"):
        require_schema_fingerprint("b" * 64, "a" * 64, label="current")
    with pytest.raises(ValueError, match="previous schema fingerprint"):
        require_schema_fingerprint(None, "a" * 64, label="previous")
