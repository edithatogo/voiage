#!/usr/bin/env python3
"""Compare a candidate C ABI with an immutable released baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

_DECLARATION = re.compile(
    r"VOIAGE_V1_API\s+(?P<declaration>.*?\b(?P<symbol>voiage_v1_[a-z0-9_]+)\s*\([^;]*\));",
    re.DOTALL,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _layouts(path: Path) -> dict[str, str]:
    return {line.split(maxsplit=1)[0]: line for line in _records(path)}


def _declarations(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group("symbol"): " ".join(match.group("declaration").split())
        for match in _DECLARATION.finditer(text)
    }


def compare(
    baseline: Path,
    candidate_header: Path,
    candidate_symbols: Path,
    candidate_layouts: Path,
) -> dict[str, Any]:
    """Return a machine-readable additive-compatibility result."""
    metadata = json.loads((baseline / "metadata.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for name, expected in metadata["artifacts_sha256"].items():
        actual = _sha256(baseline / name)
        if actual != expected:
            errors.append(f"released baseline digest mismatch: {name}")

    baseline_symbols = set(_records(baseline / "symbols.txt"))
    candidate_symbol_set = set(_records(candidate_symbols))
    removed = sorted(baseline_symbols - candidate_symbol_set)
    if removed:
        errors.append(f"removed released symbols: {', '.join(removed)}")

    baseline_declarations = _declarations(baseline / "voiage_v1.h")
    candidate_declarations = _declarations(candidate_header)
    errors.extend(
        f"changed released declaration: {symbol}"
        for symbol in sorted(baseline_symbols)
        if baseline_declarations.get(symbol) != candidate_declarations.get(symbol)
    )
    undocumented = sorted(candidate_symbol_set - candidate_declarations.keys())
    if undocumented:
        errors.append(
            f"candidate symbols absent from header: {', '.join(undocumented)}"
        )

    baseline_layouts = _layouts(baseline / "layouts.txt")
    candidate_layout_map = _layouts(candidate_layouts)
    for name, released_record in baseline_layouts.items():
        if candidate_layout_map.get(name) != released_record:
            errors.append(f"changed released layout: {name}")

    return {
        "schema_version": "1.0.0",
        "baseline_release": metadata["release"],
        "baseline_source_commit": metadata["source_commit"],
        "compatible": not errors,
        "errors": errors,
        "retained_symbols": sorted(baseline_symbols & candidate_symbol_set),
        "additive_symbols": sorted(candidate_symbol_set - baseline_symbols),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-header", type=Path, required=True)
    parser.add_argument("--candidate-symbols", type=Path, required=True)
    parser.add_argument("--candidate-layouts", type=Path, required=True)
    return parser


def main() -> int:
    """Run the compatibility comparison and print its JSON receipt."""
    args = _parser().parse_args()
    try:
        result = compare(
            args.baseline,
            args.candidate_header,
            args.candidate_symbols,
            args.candidate_layouts,
        )
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"compatible": False, "errors": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["compatible"] else 1


if __name__ == "__main__":
    sys.exit(main())
