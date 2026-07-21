#!/usr/bin/env python3
"""Export through an ephemeral collector and independently inspect received bytes."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import ClassVar, cast

from voiage.c15_otel import (
    CorrelationContext,
    TelemetryContractError,
    build_otlp_log_request,
    export_otlp_http,
)


class _Collector(BaseHTTPRequestHandler):
    payloads: ClassVar[list[dict[str, object]]] = []

    def do_POST(self) -> None:
        if (
            self.path != "/v1/logs"
            or self.headers.get("Content-Type") != "application/json"
        ):
            self.send_error(400)
            return
        length = int(self.headers.get("Content-Length", "0"))
        value: object = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            self.send_error(400)
            return
        self.payloads.append(cast("dict[str, object]", value))
        self.send_response(200)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def received_contract(
    payload: dict[str, object], expected: CorrelationContext
) -> dict[str, object]:
    """Parse only collector-received payload and validate privacy plus correlation."""
    try:
        resource_logs = cast("list[dict[str, object]]", payload["resourceLogs"])
        scope_logs = cast("list[dict[str, object]]", resource_logs[0]["scopeLogs"])
        records = cast("list[dict[str, object]]", scope_logs[0]["logRecords"])
        record = records[0]
        raw_attributes = cast("list[dict[str, object]]", record["attributes"])
        attributes: dict[str, object] = {}
        for item in raw_attributes:
            value = cast("dict[str, object]", item["value"])
            attributes[cast("str", item["key"])] = next(iter(value.values()))
    except (IndexError, KeyError, StopIteration, TypeError) as exc:
        raise RuntimeError("collector received malformed OTLP payload") from exc
    if len(resource_logs) != 1 or len(records) != 1:
        raise RuntimeError("collector must receive exactly one log record")
    observed = {
        "run_id": attributes.get("run.id"),
        "trace_id": record.get("traceId"),
        "span_id": record.get("spanId"),
        "backend": attributes.get("analysis.backend"),
        "numerical_policy_id": attributes.get("analysis.numerical_policy_id"),
    }
    required = {
        "run_id": expected.run_id,
        "trace_id": expected.trace_id,
        "span_id": expected.span_id,
        "backend": expected.backend,
        "numerical_policy_id": expected.numerical_policy_id,
    }
    if observed != required:
        raise RuntimeError("collector-observed correlation does not match export")
    encoded = json.dumps(payload, sort_keys=True).casefold()
    if "must-not-export" in encoded:
        raise RuntimeError("collector received privacy-sensitive telemetry")
    if (
        attributes.get("authorization") != "[REDACTED]"
        or attributes.get("safe") != "retained"
    ):
        raise RuntimeError("collector redaction/safe-attribute contract failed")
    return observed


def probe() -> dict[str, object]:
    """Export once to an ephemeral collector and validate received bytes."""
    _Collector.payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = CorrelationContext(
        run_id="voiage-c15-probe",
        trace_id="0123456789abcdef0123456789abcdef",
        span_id="0123456789abcdef",
        backend="numpy",
        numerical_policy_id="voiage-c15-policy",
    )
    try:
        payload = build_otlp_log_request(
            "analysis.completed",
            context,
            attributes={"authorization": "Bearer must-not-export", "safe": "retained"},
        )
        export_otlp_http(f"http://127.0.0.1:{server.server_port}/v1/logs", payload)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    if len(_Collector.payloads) != 1:
        raise RuntimeError("collector did not receive exactly one export")
    return {
        "schema_version": "1.0.0",
        "collector": "ephemeral-loopback-otlp-http-json",
        "exports_received": 1,
        "privacy_screened": True,
        "correlation_source": "collector_received_payload",
        "correlation": received_contract(_Collector.payloads[0], context),
    }


def main() -> int:
    """Run the collector probe and persist its assurance report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = probe()
    except (OSError, RuntimeError, TelemetryContractError, ValueError) as exc:
        print(f"C15 telemetry assurance failed: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
