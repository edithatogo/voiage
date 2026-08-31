#!/usr/bin/env python3
"""Compare one public metadata request with temporary replay and mock transport.

This explicit opt-in experiment never records application traffic. Its only
network target is public, version-specific PyPI metadata. The cassette is
removed on exit; reports retain only response hashes and timing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any
import urllib.request

URL = "https://pypi.org/pypi/pytest-testmon/2.2.0/json"


def permitted_request(request: Any) -> Any:
    """Reject anything outside the fixed public, credential-free request."""
    if request.method != "GET" or request.uri != URL or request.body:
        raise ValueError("only the fixed public PyPI metadata GET may be recorded")
    if any(
        name.lower() in ("authorization", "cookie", "proxy-authorization")
        for name in request.headers
    ):
        raise ValueError("credential-bearing requests must not be recorded")
    request.headers = {}
    return request


def public_response(response: dict[str, Any]) -> dict[str, Any]:
    """Retain only public JSON and remove all transport headers."""
    response["headers"] = {}
    return response


def main() -> None:
    """Perform a bounded experiment only when explicitly invoked."""
    import httpx
    import vcr

    rows = []
    # No proxy or credential handlers; redirects fail the cassette URI allowlist.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryDirectory(prefix="voiage-public-replay-") as directory:
        cassette = str(Path(directory) / "public.json")
        recorder = vcr.VCR(
            serializer="json",
            before_record_request=permitted_request,
            before_record_response=public_response,
            filter_headers=["authorization", "cookie", "proxy-authorization"],
        )
        expected = None
        for mode in ("once", "none"):
            started = time.perf_counter()
            with recorder.use_cassette(cassette, record_mode=mode) as tape:
                with opener.open(URL, timeout=20) as response:
                    payload = response.read()
                assert json.loads(payload)["info"]["version"] == "2.2.0"
                assert len(tape.requests) == 1
                digest = hashlib.sha256(payload).hexdigest()
                if expected is not None:
                    assert digest == expected
                expected = digest
            rows.append(
                {
                    "profile": "public_record"
                    if mode == "once"
                    else "network_disabled_replay",
                    "wall_seconds": round(time.perf_counter() - started, 6),
                    "checks": 3,
                    "response_sha256": digest,
                }
            )
        started = time.perf_counter()
        # Unit tests should keep small generated response contracts in-process.
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"info": {"version": "2.2.0"}})
        )
        with httpx.Client(transport=transport) as client:
            response = client.get(URL)
            assert response.status_code == 200
            assert response.json()["info"]["version"] == "2.2.0"
        rows.append(
            {
                "profile": "in_process_mock",
                "wall_seconds": round(time.perf_counter() - started, 6),
                "checks": 2,
            }
        )
    result = {
        "url": URL,
        "measurements": rows,
        "cassette_retained": False,
        "credentials_or_restricted_payloads": False,
        "decision": "Retain in-process transports for deterministic unit tests; temporary VCR is useful only for explicitly scoped public integration investigations.",
    }
    output = (
        Path(__file__).resolve().parents[1]
        / ".conductor/local/http-replay-evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
