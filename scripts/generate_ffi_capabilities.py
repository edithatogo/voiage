#!/usr/bin/env python3
"""Generate the C ABI capability document from stable-core status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "specs/v1/stable-core-status.json"
OUTPUT = ROOT / "rust/crates/voiage-ffi/src/generated_capabilities.rs"


def _document() -> dict[str, Any]:
    status = json.loads(SOURCE.read_text(encoding="utf-8"))
    return {
        "aggregate": status["aggregate"],
        "contract_version": status["contract_version"],
        "methods": status["methods"],
        "schema_version": "1.0",
        "source": str(SOURCE.relative_to(ROOT)),
        "status": status["status"],
    }


def _render() -> str:
    document = json.dumps(_document(), separators=(",", ":"), sort_keys=True)
    return (
        "//! Generated from specs/v1/stable-core-status.json; do not edit.\n\n"
        "pub(crate) const CAPABILITY_DOCUMENT_JSON_NUL: &[u8] =\n"
        f'    concat!(r#"{document}"#, "\\0").as_bytes();\n'
    )


def main() -> int:
    """Write the Rust artifact or fail when its checked-in copy has drifted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated Rust artifact is absent or stale",
    )
    args = parser.parse_args()
    rendered = _render()

    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print(
                f"{OUTPUT.relative_to(ROOT)} is missing or stale; "
                "run scripts/generate_ffi_capabilities.py",
                file=sys.stderr,
            )
            return 2
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
