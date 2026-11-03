"""Bind staged concrete Spack archives to the frozen authenticated mirror."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

from scripts.easybuild_source_hooks import resolved_file


def staged_source(path: Path, work: Path) -> dict:
    """Accept Spack's stage aliases only when they target this frozen cache."""
    if not path.absolute().is_relative_to(work.absolute()):
        raise ValueError("Staged archive path escapes private work root")
    target = path.resolve(strict=True)
    if path.is_symlink() and not target.is_relative_to((work / "cache").resolve()):
        raise ValueError("Staged archive alias escapes frozen cache")
    record = resolved_file(target, [work])
    record["archive_path"] = str(path)
    record["link"] = str(path.readlink()) if path.is_symlink() else None
    return record


def main() -> None:
    """Execute inside the qualified Spack Python runtime and active environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(sys.argv[1:])
    work, output = args.work, args.output
    environment = importlib.import_module("spack.environment").active_environment()
    traverse = importlib.import_module("spack.traverse").traverse_nodes
    authenticated = {
        entry["sha256"]
        for name in ("cache", "mirror")
        for entry in json.loads((output / f"{name}-manifest.json").read_text())
    }
    records = []
    for spec in traverse(environment.concrete_roots()):
        if spec.external or not spec.package.has_code:
            continue
        for stage in spec.package.stage:
            archive = stage.archive_file
            if archive is None:
                raise ValueError(f"Missing concrete staged source archive: {spec}")
            record = staged_source(Path(archive), work)
            if record["sha256"] not in authenticated:
                raise ValueError(f"Staged source is absent from frozen mirror: {spec}")
            records.append(
                {"spec": str(spec), "dag_hash": spec.dag_hash(), "source": record}
            )
    if not records:
        raise ValueError("No concrete staged source witnesses")
    (output / "staged-sources.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
