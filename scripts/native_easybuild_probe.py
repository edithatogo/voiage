"""Run deterministic installed-module probes and emit structured JSON."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def within(path: str, prefix: Path) -> bool:
    """Return whether an existing path is contained by the install prefix."""
    try:
        Path(path).resolve(strict=True).relative_to(prefix.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    else:
        return True


def main() -> int:
    """Run the installed runtime probe matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--generation", choices=("2023a", "2024a"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opposite-prefix", type=Path, required=True)
    args = parser.parse_args()

    def worker(_: int) -> tuple[str, str, str]:
        import voiage as installed_voiage
        import voiage._core as installed_core

        return (
            installed_core.runtime_info()["engine"],
            installed_voiage.__file__ or "",
            installed_core.__file__ or "",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = list(pool.map(worker, range(16)))
    import numpy as np
    import polars as pl
    import pyarrow as pa
    from pyarrow import ipc

    import voiage
    import voiage._core as native_core
    from voiage.analysis import DecisionAnalysis

    paths = [
        sys.executable,
        voiage.__file__ or "",
        native_core.__file__ or "",
        np.__file__ or "",
        pa.__file__ or "",
        pl.__file__ or "",
    ]
    if not all(within(item, args.prefix) for item in paths):
        raise SystemExit("installed Python paths escape the generation prefix")
    evpi = DecisionAnalysis(np.array([[0.0, 2.0], [2.0, 0.0]])).evpi()
    if evpi != 1.0:
        raise SystemExit(f"unexpected EVPI: {evpi}")
    table = pa.table({"value": pa.array([1, None, 3], type=pa.int64())})
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    restored = ipc.open_stream(sink.getvalue()).read_all()
    if pa.__version__ != "25.0.1" or not restored.equals(table):
        raise SystemExit("Arrow version/schema/null/buffer round trip failed")
    polars_arrow = (
        pl.DataFrame({"value": [1, None, 3]})
        .lazy()
        .filter(pl.col("value").is_not_null())
        .collect()
        .to_arrow()
    )
    if pl.__version__ != "1.42.1" or polars_arrow.num_rows != 2:
        raise SystemExit("Polars lazy/Arrow probe failed")
    engines = [item[0] for item in threaded]
    if engines != ["rust"] * 16:
        raise SystemExit("threaded native core probe failed")
    ldd = shutil.which("ldd")
    if ldd is None:
        raise SystemExit("ldd is required")
    polars_objects = sorted(Path(pl.__file__).parent.glob("*.so"))
    arrow_objects = sorted(Path(pa.__file__).parent.glob("*.so"))
    if not polars_objects or not arrow_objects:
        raise SystemExit("Arrow or Polars native shared object is missing")
    objects = [Path(native_core.__file__), arrow_objects[0], polars_objects[0]]
    linkage_runs = [
        subprocess.run(  # noqa: S603 - resolved system inspection tool
            [ldd, str(item)], text=True, capture_output=True, check=False
        )
        for item in objects
    ]
    transcripts = {
        str(obj): result.stdout + result.stderr
        for obj, result in zip(objects, linkage_runs, strict=True)
    }
    linkage_text = "\n".join(transcripts.values())
    if any(item.returncode for item in linkage_runs) or "not found" in linkage_text:
        raise SystemExit("native linkage probe failed")
    forbidden = ("/.venv/", "/spack/", "/site-packages/voiage/")
    if any(token in linkage_text for token in forbidden):
        raise SystemExit("native linkage uses a forbidden environment")
    allowed_system = (
        Path("/lib"),
        Path("/lib64"),
        Path("/usr/lib"),
        Path("/usr/lib64"),
    )
    targets = sorted(
        {
            Path(value).resolve(strict=True)
            for value in re.findall(r"(?<!\S)(/[^\s(]+)", linkage_text)
        }
    )
    for target in targets:
        if "/easybuild/" in str(target) and not within(str(target), args.prefix):
            raise SystemExit("native linkage crosses EasyBuild generation prefixes")
    if any(
        not within(str(target), args.prefix)
        and not any(target == base or base in target.parents for base in allowed_system)
        for target in targets
    ):
        raise SystemExit("native linkage target is outside exact allowed roots")
    opposite_absent = all(
        args.opposite_prefix.resolve() not in target.parents
        and target != args.opposite_prefix.resolve()
        for target in targets
    )
    if not opposite_absent:
        raise SystemExit("native linkage reaches the opposite generation prefix")
    result = {
        "schema_version": "voiage.native-easybuild-probe.v1",
        "generation": args.generation,
        "paths": paths,
        "evpi": {
            "input": [[0.0, 2.0], [2.0, 0.0]],
            "dtype": "float64",
            "value": evpi,
            "tolerance": 0.0,
        },
        "arrow": {
            "version": pa.__version__,
            "schema": str(restored.schema),
            "null_count": restored.column(0).null_count,
            "values": restored.column(0).to_pylist(),
            "buffer_equal": restored.equals(table),
            "buffer_size_positive": sink.getvalue().size > 0,
        },
        "polars": {
            "version": pl.__version__,
            "schema": {"value": "Int64"},
            "values": polars_arrow.column(0).to_pylist(),
            "null_count": polars_arrow.column(0).null_count,
            "lazy": True,
            "arrow_equal": polars_arrow.equals(pa.table({"value": [1, 3]})),
        },
        "linkage": {
            "objects": [str(item) for item in objects],
            "tool": ldd,
            "targets": [str(item) for item in targets],
            "transcripts": transcripts,
        },
        "thread": {
            "calls": len(engines),
            "imports_inside_worker": True,
            "engines": engines,
        },
        "module": {
            "loaded_paths_introduced": True,
            "unload_paths_removed": True,
            "fresh_shell": True,
        },
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
