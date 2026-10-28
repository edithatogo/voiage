"""Capture and verify the source-derived translation reference, without runtime edits."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from html import escape
import importlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

TRACK = Path("conductor/tracks/translation_reference_20261026")
PAGE = Path(
    "docs/astro-site/src/content/docs/user-guide/tabular-translation-reference.mdx"
)
BASE = "f7fcbe4d0a27174d5ae69c33da46575f149f5c1e"
CONSUMER_FILES = (
    "scripts/run_empirical_longitudinal.py",
    "scripts/run_release_calibration.py",
    "scripts/verify_empirical_longitudinal.py",
)
NATIVE_FILES = (
    "voiage/schema.py",
    "voiage/ecosystem_integration.py",
    "voiage/core/io.py",
    "voiage/methods/basic.py",
)
TOPICS = {
    "index": r"\.iloc\[|strategy_names\[",
    "columns": r"\.(?:year|interpretation)\b|(?:table|frame)\[",
    "rows": r"\.itertuples\(|\.iterrows\(|\.iter_rows\(",
    "grouping": r"\.groupby\(|\.group_by\(",
    "concat": r"\bpd\.concat\(|\bpl\.concat\(",
    "reduction": r"\bnp\.mean\(|\.mean\(",
    "nulls": r"\bnp\.nan\b|\.null_count\(|\.is_nan\(",
    "float-format": r"float_format=|csv\.writer\(",
}
IDS = (
    "index",
    "columns",
    "rows",
    "grouping",
    "concat",
    "alignment",
    "reduction",
    "rolling",
    "resample",
    "interpolation",
    "nulls",
    "float-format",
    "attrs",
    "nd-interpolation",
)


def sha(data: bytes) -> str:
    """Digest source bytes or explicitly encoded excerpts."""
    return hashlib.sha256(data).hexdigest()


def git_blob(repo: Path, revision: str, name: str) -> bytes:
    """Read an exact committed blob without executing consumer code."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required for capture")
    return subprocess.check_output([git, "-C", str(repo), "show", f"{revision}:{name}"])  # noqa: S603


def snapshot(repo: Path, revision: str, files: tuple[str, ...]) -> dict[str, Any]:
    """Capture bounded one-line citations with whole-blob and excerpt digests."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Capture requires a full immutable commit ID")
    records = []
    for name in files:
        raw = git_blob(repo, revision, name)
        lines = raw.decode().splitlines(keepends=True)
        snippets = []
        for topic, pattern in TOPICS.items():
            matches = [i for i, line in enumerate(lines, 1) if re.search(pattern, line)]
            for number in matches[:3]:
                text = lines[number - 1]
                snippets.append(
                    {
                        "id": f"{name}:{number}:{topic}",
                        "topic": topic,
                        "start": number,
                        "end": number,
                        "text": text,
                        "sha256": sha(text.encode()),
                    }
                )
        records.append(
            {
                "path": name,
                "sha256": sha(raw),
                "line_count": len(lines),
                "snippets": snippets,
            }
        )
    return {"schema_version": "1.0", "source_commit": revision, "files": records}


def load_cases(root: Path) -> list[dict[str, Any]]:
    """Load the committed semantic cases in publication order."""
    return json.loads((root / TRACK / "semantic-cases.json").read_text())["rows"]


def run_case(row: dict[str, Any], mode: str) -> Any:
    """Run trusted, committed example code and assert its frozen semantic result."""
    scope: dict[str, Any] = {
        "pl": importlib.import_module("polars"),
        "np": importlib.import_module("numpy"),
        "datetime": datetime,
        "io": io,
    }
    if mode == "legacy":
        module = row["legacy_module"]
        scope["pd" if module == "pandas" else "xr"] = importlib.import_module(module)
    code = row[f"{mode}_code"]
    exec(compile(code, f"reference:{row['id']}:{mode}", "exec"), scope)  # noqa: S102 - owned executable documentation
    result = scope["result"]
    if result != row["expected"]:
        raise AssertionError(f"{row['id']}:{mode}: {result!r} != {row['expected']!r}")
    return result


def render(rows: list[dict[str, Any]]) -> str:
    """Generate the single reference page from exactly the tested code forms."""
    lines = [
        "---",
        "title: pandas and xarray to Polars reference",
        "description: Source-derived, executable semantic differences for the v3 migration",
        "---",
        "",
        "This reference targets Python 3.15.1 and the qualified Polars 2.1 floor.",
        "The v3 containers use wide frames. Public array constructors, numerical",
        "methods, wire contracts and bindings retain their existing behavior.",
        "",
        "Legacy blocks describe caller-owned pandas/xarray code; these packages",
        "are not installed or imported by the base Voiage runtime. They are external",
        "test oracles only. Polars blocks run in the clean no-extras environment.",
        "Removed `dataset`/`from_dataset`/`to_dataset` APIs are not replacements.",
        "",
        "The committed capture records source commits, file and excerpt digests,",
        "and one-based line ranges. CI needs no external consumer checkout.",
        "The source manifests are `tobac-consumer-idioms.json` and",
        "`voiage-idioms.json` under `conductor/tracks/translation_reference_20261026/`.",
        "See the [migration guide](/voiage/user-guide/migration-guide).",
        "",
        "Fixtures below establish the stated relationships only: value/order",
        "equality does not promise dtype, index, time-zone or metadata equivalence.",
        "No approximation is promoted into a numerical method.",
        "",
        "Examples assume `import polars as pl`, `import numpy as np`,",
        "`import io` and `from datetime import datetime`.",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['id']}: {row['title']}",
            "",
            row["explanation"],
            "",
            f"Evidence: **{row['evidence']}**. {row['provenance_note']}",
            "",
        ]
        for citation in row["citations"]:
            lines += [f"- `{citation['source']}`: `{citation['id']}`"]
        lines += [
            "",
            f"Relationship: **{row['relation']}**. {escape(row['qualification'])}",
            "",
            f"Legacy `{row['legacy_module']}` form:",
            "",
            "```python",
            row["legacy_code"].rstrip(),
            "```",
            "",
            "Polars/caller replacement:",
            "",
            "```python",
            row["target_code"].rstrip(),
            "```",
            "",
            f"Both return `{json.dumps(row['expected'])}` on the fixture.",
            "",
        ]
    lines += [
        "## Reproduce and interpret the evidence",
        "",
        "Run `python scripts/check_translation_reference.py --mode target` in the",
        "clean installed base environment. `--mode legacy --require-all` requires",
        "both external oracles; missing capabilities fail that qualification run.",
        "Pytest oracle cases skip only when their declared external package is absent.",
        "The track receipt distinguishes base, producer and isolated full-oracle runs.",
        "",
        "The caller formatter is a local recipe, not a new Voiage API or persistence",
        "format. The library's existing CSV writer remains unchanged.",
        "",
        "API references: [as-of join](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html),",
        "[CSV writer](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.write_csv.html),",
        "[dynamic windows](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by_dynamic.html).",
        "",
    ]
    return "\n".join(lines)


def validate(root: Path) -> list[str]:
    """Validate hermetic citations, observed/anticipatory scope and rendered code."""
    errors = []
    citations: dict[str, set[str]] = {}
    for source, filename in [
        ("tobac", "tobac-consumer-idioms.json"),
        ("voiage", "voiage-idioms.json"),
    ]:
        path = root / TRACK / filename
        if not path.is_file():
            errors.append(f"Missing capture: {filename}")
            continue
        data = json.loads(path.read_text())
        if not re.fullmatch(r"[0-9a-f]{40}", data["source_commit"]):
            errors.append(f"Unpinned capture: {filename}")
        if source == "tobac":
            for notice in ("LICENSE", "NOTICE"):
                retained = root / TRACK / "consumer-notices" / f"tobac-{notice}"
                if not retained.is_file() or sha(
                    retained.read_bytes().replace(b"\r\n", b"\n")
                ) != data.get("notices", {}).get(notice):
                    errors.append(f"Retained notice drift: {notice}")
        citations[source] = set()
        expected_files = CONSUMER_FILES if source == "tobac" else NATIVE_FILES
        if [item["path"] for item in data["files"]] != list(expected_files):
            errors.append(f"Capture scope drift: {source}")
        for item in data["files"]:
            if item["path"] not in expected_files:
                errors.append(f"Unexpected capture path: {item['path']}")
                continue
            if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                errors.append(f"Invalid source digest: {item['path']}")
            if source == "voiage":
                actual = root / item["path"]
                if (
                    actual.is_symlink()
                    or not actual.resolve().is_relative_to(root.resolve())
                    or sha(actual.read_bytes().replace(b"\r\n", b"\n"))
                    != item["sha256"]
                ):
                    errors.append(f"Current source drift: {item['path']}")
                    continue
            for snippet in item["snippets"]:
                citations[source].add(snippet["id"])
                if (
                    not (1 <= snippet["start"] <= snippet["end"] <= item["line_count"])
                    or len(snippet["text"].splitlines())
                    != snippet["end"] - snippet["start"] + 1
                    or sha(snippet["text"].encode()) != snippet["sha256"]
                ):
                    errors.append(f"Invalid excerpt: {snippet['id']}")
                if source == "voiage" and actual.read_text().splitlines(keepends=True)[
                    snippet["start"] - 1 : snippet["end"]
                ] != snippet["text"].splitlines(keepends=True):
                    errors.append(f"Current citation drift: {snippet['id']}")
    rows = load_cases(root)
    if [row["id"] for row in rows] != list(IDS):
        errors.append("Missing, duplicated or reordered semantic rows")
    for row in rows:
        if row.get("evidence") not in {"derived", "anticipatory"} or not row.get(
            "provenance_note"
        ):
            errors.append(f"Unclassified evidence: {row['id']}")
        if row.get("evidence") == "derived" and not row.get("citations"):
            errors.append(f"Uncited derived row: {row['id']}")
        errors.extend(
            f"Missing citation: {citation['id']}"
            for citation in row.get("citations", [])
            if citation["id"] not in citations.get(citation["source"], set())
        )
        for mode in ("target", "legacy"):
            compile(row[f"{mode}_code"], f"reference:{row['id']}:{mode}", "exec")
    if not (root / PAGE).is_file() or (root / PAGE).read_text() != render(rows):
        errors.append("Published reference differs from tested code/wording")
    return errors


def main() -> int:
    """Capture source once, render explicitly, or verify committed forms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-consumer", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--write-doc", action="store_true")
    parser.add_argument("--mode", choices=["target", "legacy"])
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 15, 1):
        parser.error("Reference qualification requires Python 3.15.1")
    root = Path(__file__).resolve().parents[1]
    if args.capture_consumer:
        if not args.revision:
            parser.error("Capture requires --revision with a full commit ID")
        consumer = snapshot(args.capture_consumer, args.revision, CONSUMER_FILES)
        consumer["notices"] = {
            name: sha(git_blob(args.capture_consumer, args.revision, name))
            for name in ("LICENSE", "NOTICE")
        }
        consumer["attribution"] = (
            "tobac project-owned source excerpts, Apache-2.0; retained LICENSE/NOTICE in consumer-notices. No datasets captured or executed."
        )
        for filename, data in [
            ("tobac-consumer-idioms.json", consumer),
            ("voiage-idioms.json", snapshot(root, BASE, NATIVE_FILES)),
        ]:
            (root / TRACK / filename).write_text(json.dumps(data, indent=2) + "\n")
        notices = root / TRACK / "consumer-notices"
        notices.mkdir(exist_ok=True)
        for name in ("LICENSE", "NOTICE"):
            (notices / f"tobac-{name}").write_bytes(
                git_blob(args.capture_consumer, args.revision, name)
            )
        return 0
    rows = load_cases(root)
    if args.write_doc:
        (root / PAGE).write_text(render(rows))
        return 0
    errors = validate(root)
    outcomes = []
    if args.mode and not errors:
        for row in rows:
            if (
                args.mode == "legacy"
                and importlib.util.find_spec(row["legacy_module"]) is None
            ):
                outcomes.append(
                    {
                        "id": row["id"],
                        "status": "unavailable",
                        "capability": row["legacy_module"],
                    }
                )
                continue
            run_case(row, args.mode)
            outcomes.append({"id": row["id"], "status": "passed"})
    if args.mode == "target":
        loaded = {name.split(".", 1)[0] for name in sys.modules}
        if loaded & {"pandas", "xarray"}:
            errors.append("Target recipes imported an external legacy engine")
    missing = any(item["status"] == "unavailable" for item in outcomes)
    passed = not errors and not (args.require_all and missing)
    print(
        json.dumps(
            {
                "passed": passed,
                "python": sys.version.split()[0],
                "errors": errors,
                "mode": args.mode,
                "outcomes": outcomes,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
