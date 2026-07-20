from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Self

import pytest

from scripts.c15_otel_probe import probe, received_contract
from voiage import c15_otel
from voiage.c15_otel import (
    CorrelationContext,
    TelemetryContractError,
    build_otlp_log_request,
    export_otlp_http,
)


def _context() -> CorrelationContext:
    return CorrelationContext(
        run_id="run-c15",
        trace_id="0123456789abcdef0123456789abcdef",
        span_id="0123456789abcdef",
        backend="numpy",
        numerical_policy_id="policy",
    )


def test_probe_uses_independently_received_payload() -> None:
    report = probe()
    assert report["correlation_source"] == "collector_received_payload"
    assert report["privacy_screened"] is True


def test_nested_secrets_are_redacted_and_safe_values_retained() -> None:
    payload = build_otlp_log_request(
        "analysis.completed",
        _context(),
        attributes={
            "authorization": "Bearer private",
            "nested": {"token": "private"},
            "message": "token=private",
            "safe": "ok",
        },
    )
    encoded = json.dumps(payload)
    assert "private" not in encoded
    assert "[REDACTED]" in encoded
    contains_secret = c15_otel._contains_secret
    assert contains_secret({"nested": ["safe", "token=private"]})
    assert contains_secret(["safe", "password=hunter2"])


@pytest.mark.parametrize(
    "message", ["Bearer secret", "password=hunter2", "token=private"]
)
def test_secret_bearing_messages_are_rejected(message: str) -> None:
    with pytest.raises(TelemetryContractError, match="secret-bearing"):
        build_otlp_log_request(message, _context())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_attributes_and_exports_are_rejected(value: float) -> None:
    with pytest.raises(TelemetryContractError, match="finite"):
        build_otlp_log_request("safe", _context(), attributes={"value": value})
    with pytest.raises(TelemetryContractError, match="finite JSON"):
        export_otlp_http("http://127.0.0.1:1/v1/logs", {"value": value})


@pytest.mark.parametrize(
    ("trace_id", "span_id"),
    [
        ("0" * 32, "1" * 16),
        ("1" * 32, "0" * 16),
        ("INVALID", "1" * 16),
        ("1" * 32, "INVALID"),
    ],
)
def test_invalid_or_zero_otel_ids_are_rejected(trace_id: str, span_id: str) -> None:
    with pytest.raises(TelemetryContractError):
        CorrelationContext("run", trace_id, span_id, "numpy", "policy")


def test_received_payload_parser_rejects_correlation_drift() -> None:
    context = _context()
    payload = build_otlp_log_request(
        "analysis.completed",
        context,
        attributes={"authorization": "secret=x", "safe": "retained"},
    )
    drifted = deepcopy(payload)
    drifted["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["traceId"] = "f" * 32
    with pytest.raises(RuntimeError, match="correlation"):
        received_contract(drifted, context)


def test_all_supported_otlp_value_shapes_are_encoded() -> None:
    payload = build_otlp_log_request(
        "safe",
        _context(),
        attributes={
            "none": None,
            "boolean": True,
            "integer": 7,
            "floating": 1.5,
            "sequence": [1, "two"],
            "mapping": {"nested": False},
            "object": object(),
        },
        observed_time_unix_nano=1,
    )
    assert json.dumps(payload, allow_nan=False)
    with pytest.raises(TelemetryContractError, match="unsupported"):
        c15_otel._any_value(object())


def test_safe_endpoint_accepts_https_and_loopback_only() -> None:
    safe_endpoint = c15_otel._safe_endpoint
    safe_endpoint("https://collector.example/v1/logs")
    safe_endpoint("http://localhost:4318/v1/logs")
    with pytest.raises(TelemetryContractError, match="HTTPS"):
        safe_endpoint("http://192.0.2.1/v1/logs")


@pytest.mark.parametrize("message", ["", " ", None])
def test_empty_messages_are_rejected(message: str | None) -> None:
    with pytest.raises(TelemetryContractError, match="empty"):
        build_otlp_log_request(message, _context())  # type: ignore[arg-type]


def test_export_rejects_non_success_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 503

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(c15_otel, "urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(TelemetryContractError, match="503"):
        export_otlp_http("https://collector.example/v1/logs", {})


@pytest.mark.parametrize(
    "endpoint",
    [
        "file:///v1/logs",
        "http://example.com/v1/logs",
        "https://example.com/wrong",
        "https://example.com/v1/logs?query=1",
    ],
)
def test_export_rejects_unsafe_or_malformed_endpoints(endpoint: str) -> None:
    with pytest.raises(TelemetryContractError, match="endpoint|HTTPS"):
        export_otlp_http(endpoint, {})


@pytest.mark.parametrize("timestamp", [0, -1, True, 2**64, 1.5])
def test_timestamp_must_be_positive_uint64(timestamp: object) -> None:
    with pytest.raises(TelemetryContractError, match="timestamp"):
        build_otlp_log_request("safe", _context(), observed_time_unix_nano=timestamp)


def test_correlation_rejects_empty_and_secret_bearing_fields() -> None:
    with pytest.raises(TelemetryContractError, match="empty"):
        CorrelationContext("", "1" * 32, "1" * 16, "numpy", "policy")
    with pytest.raises(TelemetryContractError, match="secret"):
        CorrelationContext("run", "1" * 32, "1" * 16, "numpy", "token=must-not-export")


def test_received_payload_parser_rejects_malformed_and_leaked_payloads() -> None:
    context = _context()
    with pytest.raises(RuntimeError, match="malformed"):
        received_contract({}, context)
    payload = build_otlp_log_request(
        "analysis.completed",
        context,
        attributes={"authorization": "secret=x", "safe": "retained"},
    )
    payload["leak"] = "must-not-export"
    with pytest.raises(RuntimeError, match="privacy"):
        received_contract(payload, context)
