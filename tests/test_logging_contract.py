from __future__ import annotations

import json
import logging as stdlib_logging

import numpy as np
from pydantic import ValidationError
import pytest

from voiage.contracts.analysis import AnalysisSpec, DiagnosticRecord, NumericalPolicy
from voiage.contracts.kernel import run_evpi
import voiage.logging as logging_module


def test_logging_settings_are_pydantic_v2_and_strict() -> None:
    assert logging_module.LoggingSettings.model_fields
    with pytest.raises(ValidationError):
        logging_module.LoggingSettings(level="verbose")


def test_json_logging_carries_run_and_bound_context(tmp_path) -> None:
    destination = tmp_path / "run.jsonl"
    logger = logging_module.configure_logging(
        logging_module.LoggingSettings(console=False, log_file=destination, run_id="test-run")
    )
    with logging_module.log_context(
        track="C08",
        command="evpi",
        access_token="never-log-this",  # noqa: S106 - verifies redaction
    ):
        logger.info("analysis_started")
    for handler in logger.handlers:
        handler.flush()

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert records[-1]["message"] == "analysis_started"
    assert records[-1]["run_id"] == "test-run"
    assert records[-1]["track"] == "C08"
    assert records[-1]["access_token"] == "[REDACTED]"  # noqa: S105


def test_configuration_preserves_root_handlers() -> None:
    root = stdlib_logging.getLogger()
    sentinel = stdlib_logging.NullHandler()
    root.addHandler(sentinel)
    try:
        logging_module.configure_logging(logging_module.LoggingSettings(console=False))
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)


def test_shared_trace_context_is_w3c_compatible_and_strict() -> None:
    trace = logging_module.TraceContext(trace_id="1" * 32, span_id="2" * 16, trace_flags="01")
    assert trace.traceparent == f"00-{'1' * 32}-{'2' * 16}-01"

    for field, value in (
        ("trace_id", "0" * 32),
        ("trace_id", "A" * 32),
        ("span_id", "0" * 16),
        ("span_id", "2" * 15),
        ("trace_flags", "zz"),
    ):
        with pytest.raises(ValidationError):
            logging_module.TraceContext.model_validate(
                {"trace_id": "1" * 32, "span_id": "2" * 16, field: value}
            )


def test_analysis_logging_correlates_shared_fields_and_redacts_recursively(
    tmp_path,
) -> None:
    destination = tmp_path / "correlated.jsonl"
    logger = logging_module.configure_logging(
        logging_module.LoggingSettings(console=False, log_file=destination, run_id="settings-run")
    )
    policy_id = logging_module.numerical_policy_digest({"dtype": "float64", "rtol": 1e-9})
    context = logging_module.AnalysisLogContext(
        run_id="analysis-run",
        trace=logging_module.TraceContext(trace_id="1" * 32, span_id="2" * 16),
        analysis_id="analysis-1",
        backend_requested="jax",
        backend_selected="numpy",
        fallback_code="backend_fallback",
        numerical_policy_id=policy_id,
    )

    with (
        logging_module.analysis_log_context(context),
        logging_module.log_context(
            request={
                "headers": {"Authorization": "Bearer top-secret"},
                "items": [
                    {"api_token": "nested-secret"},
                    {"private_key": "private-key-value"},
                    {"cookie": "session-cookie-value"},
                    {"session_id": "session-value"},
                ],
            },
            note=(
                "password=hunter2 "
                "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signature "
                "url=https://example.test/path?signature=query-secret&safe=value"
            ),
        ),
    ):
        logger.error("failed with Bearer message-secret")
    for handler in logger.handlers:
        handler.flush()

    rendered = destination.read_text(encoding="utf-8")
    for secret in (
        "top-secret",
        "nested-secret",
        "private-key-value",
        "session-cookie-value",
        "session-value",
        "hunter2",
        "eyJhbGciOiJIUzI1NiJ9",
        "query-secret",
        "message-secret",
    ):
        assert secret not in rendered
    record = json.loads(rendered.splitlines()[-1])
    assert record["run_id"] == "analysis-run"
    assert record["trace_id"] == "1" * 32
    assert record["span_id"] == "2" * 16
    assert record["traceparent"] == f"00-{'1' * 32}-{'2' * 16}-00"
    assert record["backend_requested"] == "jax"
    assert record["backend_selected"] == "numpy"
    assert record["fallback_code"] == "backend_fallback"
    assert record["numerical_policy_id"] == policy_id
    assert record["request"] == {
        "headers": {"Authorization": "[REDACTED]"},
        "items": [
            {"api_token": "[REDACTED]"},
            {"private_key": "[REDACTED]"},
            {"cookie": "[REDACTED]"},
            {"session_id": "[REDACTED]"},
        ],
    }


@pytest.mark.parametrize(
    "reserved",
    [
        "analysis_id",
        "backend_requested",
        "backend_selected",
        "exception",
        "fallback_code",
        "level",
        "logger",
        "message",
        "numerical_policy_id",
        "run_id",
        "service",
        "span_id",
        "timestamp",
        "trace_flags",
        "trace_id",
        "traceparent",
    ],
)
def test_untrusted_context_cannot_override_reserved_fields(reserved: str) -> None:
    with pytest.raises(ValueError, match="reserved logging context field"):
        with logging_module.log_context(**{reserved: "forged"}):
            pass


def test_analysis_result_adapts_to_shared_logging_contract() -> None:
    policy = NumericalPolicy(backend_preference=("numpy",))
    spec = AnalysisSpec(
        analysis_id="reference-analysis",
        decision_problem_id="reference-decision",
        method_family="evpi",
        method_contract_version="1.0.0",
        strategy_names=("A", "B"),
        numerical_policy=policy,
    )
    result = run_evpi(np.array([[0.0, 10.0], [8.0, 0.0]]), spec=spec)

    context = logging_module.analysis_log_context_from_result(
        result, trace=logging_module.TraceContext(trace_id="1" * 32, span_id="2" * 16)
    )

    assert context.run_id == result.run_context.run_id
    assert context.analysis_id == "reference-analysis"
    assert context.backend_requested == "numpy"
    assert context.backend_selected == "numpy"
    assert context.fallback_code == "none"
    assert context.numerical_policy_id == logging_module.numerical_policy_digest(policy)

    unrelated_warning = DiagnosticRecord(
        severity="warning",
        code="rounding_notice",
        message="not a backend fallback",
        backend="numpy",
    )
    warned = result.model_copy(
        update={
            "diagnostics": result.diagnostics.model_copy(
                update={"warnings": (unrelated_warning,)}
            )
        }
    )
    assert logging_module.analysis_log_context_from_result(warned).fallback_code == "none"

    fallback_warning = DiagnosticRecord(
        severity="warning",
        code="backend_fallback",
        message="fell back",
        backend="numpy",
    )
    fallback_result = result.model_copy(
        update={
            "run_context": result.run_context.model_copy(
                update={"requested_backend": "jax"}
            ),
            "diagnostics": result.diagnostics.model_copy(
                update={"warnings": (fallback_warning,)}
            ),
        }
    )
    assert logging_module.analysis_log_context_from_result(fallback_result).fallback_code == (
        "backend_fallback"
    )


def test_run_evpi_emits_correlated_analysis_boundary_event(tmp_path) -> None:
    destination = tmp_path / "analysis-boundary.jsonl"
    logger = logging_module.configure_logging(
        logging_module.LoggingSettings(console=False, log_file=destination, run_id="settings-run")
    )
    policy = NumericalPolicy(backend_preference=("numpy",))
    spec = AnalysisSpec(
        analysis_id="logged-analysis",
        decision_problem_id="logged-decision",
        method_family="evpi",
        method_contract_version="1.0.0",
        strategy_names=("A", "B"),
        numerical_policy=policy,
    )

    result = run_evpi(np.array([[0.0, 10.0], [8.0, 0.0]]), spec=spec)
    for handler in logger.handlers:
        handler.flush()
    records = [json.loads(line) for line in destination.read_text().splitlines()]
    event = next(item for item in records if item["message"] == "analysis_completed")

    assert event["run_id"] == result.run_context.run_id
    assert event["analysis_id"] == "logged-analysis"
    assert event["backend_requested"] == "numpy"
    assert event["backend_selected"] == "numpy"
    assert event["fallback_code"] == "none"
    assert event["numerical_policy_id"] == logging_module.numerical_policy_digest(policy)


def test_redaction_handles_scalars_objects_and_exceptions() -> None:
    class SecretObject:
        def __str__(self) -> str:
            return "token=object-secret"

    assert logging_module._redact_value(None) is None
    assert logging_module._redact_value(True) is True
    assert logging_module._redact_value(3.5) == 3.5
    assert "object-secret" not in str(logging_module._redact_value(SecretObject()))

    formatter = logging_module.JsonFormatter()
    error = RuntimeError("token=exception-secret")
    record = stdlib_logging.LogRecord(
        "voiage",
        stdlib_logging.ERROR,
        __file__,
        1,
        "failed",
        (),
        (RuntimeError, error, None),
    )
    rendered = formatter.format(record)
    assert "exception-secret" not in rendered
