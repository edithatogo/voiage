"""Minimal OTLP/HTTP JSON privacy and correlation contract for VOIAGE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import ipaddress
import json
import math
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_16 = re.compile(r"^[0-9a-f]{16}$")
_SENSITIVE_KEY = re.compile(
    r"(?:authorization|api.?key|access.?token|client.?secret|cookie|credential|jwt|passphrase|password|private.?key|refresh.?token|secret|session|signature|token)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:bearer\s+\S+|(?:password|secret|token|authorization|api.?key)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)
REDACTED = "[REDACTED]"


class TelemetryContractError(ValueError):
    """Raised for unsafe or malformed telemetry exports."""


def _contains_secret(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _SENSITIVE_KEY.search(str(key)) or _contains_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_secret(item) for item in value)
    return isinstance(value, str) and bool(_SENSITIVE_VALUE.search(value))


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Identifiers that must survive export and collector receipt unchanged."""

    run_id: str
    trace_id: str
    span_id: str
    backend: str
    numerical_policy_id: str

    def __post_init__(self) -> None:
        """Reject empty, secret-bearing, or malformed identifiers."""
        fields = (self.run_id, self.backend, self.numerical_policy_id)
        if any(not value.strip() for value in fields):
            raise TelemetryContractError("correlation fields must not be empty")
        if any(_contains_secret(value) for value in fields):
            raise TelemetryContractError(
                "correlation fields contain secret-bearing text"
            )
        if _HEX_32.fullmatch(self.trace_id) is None or int(self.trace_id, 16) == 0:
            raise TelemetryContractError(
                "trace_id must be non-zero 32-character lowercase hex"
            )
        if _HEX_16.fullmatch(self.span_id) is None or int(self.span_id, 16) == 0:
            raise TelemetryContractError(
                "span_id must be non-zero 16-character lowercase hex"
            )


def _safe_value(key: str, value: object) -> object:
    if _SENSITIVE_KEY.search(key):
        return REDACTED
    if isinstance(value, Mapping):
        return {str(name): _safe_value(str(name), item) for name, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_value(key, item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise TelemetryContractError("telemetry numbers must be finite")
    if isinstance(value, str) and _contains_secret(value):
        return REDACTED
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    coerced = str(value)
    return REDACTED if _contains_secret(coerced) else coerced


def _any_value(value: object) -> dict[str, object]:  # noqa: PLR0911
    if value is None:
        return {"stringValue": "null"}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    if isinstance(value, Mapping):
        return {
            "kvlistValue": {
                "values": [
                    {"key": str(key), "value": _any_value(item)}
                    for key, item in sorted(value.items())
                ]
            }
        }
    raise TelemetryContractError("unsupported telemetry value")


def build_otlp_log_request(
    message: str,
    correlation: CorrelationContext,
    *,
    attributes: Mapping[str, object] | None = None,
    observed_time_unix_nano: int | None = None,
) -> dict[str, object]:
    """Build a privacy-screened OTLP/HTTP JSON log request."""
    if not isinstance(message, str) or not message.strip():
        raise TelemetryContractError("telemetry message must not be empty")
    if _contains_secret(message):
        raise TelemetryContractError("telemetry message contains secret-bearing text")
    safe = {
        str(key): _safe_value(str(key), value)
        for key, value in (attributes or {}).items()
    }
    safe.update(
        {
            "run.id": correlation.run_id,
            "analysis.backend": correlation.backend,
            "analysis.numerical_policy_id": correlation.numerical_policy_id,
        }
    )
    timestamp = (
        time.time_ns() if observed_time_unix_nano is None else observed_time_unix_nano
    )
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int)
        or not 0 < timestamp < 2**64
    ):
        raise TelemetryContractError("observed timestamp must be a positive uint64")
    record = {
        "timeUnixNano": str(timestamp),
        "observedTimeUnixNano": str(timestamp),
        "severityText": "INFO",
        "body": {"stringValue": message},
        "traceId": correlation.trace_id,
        "spanId": correlation.span_id,
        "attributes": [
            {"key": key, "value": _any_value(value)}
            for key, value in sorted(safe.items())
        ],
    }
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "voiage"}}
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "voiage.c15", "version": "1.0.0"},
                        "logRecords": [record],
                    }
                ],
            }
        ]
    }


def _safe_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.path != "/v1/logs" or parsed.query or parsed.fragment:
        raise TelemetryContractError("OTLP endpoint must use the exact /v1/logs path")
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and parsed.hostname:
        try:
            if ipaddress.ip_address(parsed.hostname).is_loopback:
                return
        except ValueError:
            if parsed.hostname.casefold() == "localhost":
                return
    raise TelemetryContractError("OTLP export requires HTTPS or loopback HTTP")


def export_otlp_http(
    endpoint: str, payload: Mapping[str, object], *, timeout: float = 5
) -> None:
    """POST one finite OTLP JSON request to an allowed endpoint."""
    _safe_endpoint(endpoint)
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise TelemetryContractError("OTLP payload must be finite JSON") from exc
    request = Request(  # noqa: S310
        endpoint,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # nosec B310  # noqa: S310
        if not 200 <= response.status < 300:
            raise TelemetryContractError(
                f"collector returned HTTP status {response.status}"
            )


__all__ = [
    "CorrelationContext",
    "TelemetryContractError",
    "build_otlp_log_request",
    "export_otlp_http",
]
