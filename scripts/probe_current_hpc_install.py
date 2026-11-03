"""Probe the frozen HPC candidate from an installed, selected module environment."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys


def permitted_link_target(target: Path, prefix: Path, cell: str) -> bool:
    """Admit selected libraries, glibc, and the declared EasyBuild system SSL."""
    if target.is_relative_to(prefix):
        return True
    allowed = {
        "libc.so.6",
        "libm.so.6",
        "libdl.so.2",
        "librt.so.1",
        "libpthread.so.0",
        "libresolv.so.2",
        "libutil.so.1",
        "ld-linux-x86-64.so.2",
    }
    if cell in {"easybuild-2023a", "easybuild-2024a"}:
        allowed.update({"libssl.so.3", "libcrypto.so.3"})
    roots = [
        Path(path).resolve() for path in ("/lib", "/lib64", "/usr/lib", "/usr/lib64")
    ]
    return target.name in allowed and any(target.is_relative_to(root) for root in roots)


def main() -> None:
    """Require installed provenance, numerical/interchange and actual linkage proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cell", choices=("spack", "easybuild-2023a", "easybuild-2024a"), required=True
    )
    args = parser.parse_args()
    prefix = args.prefix.resolve(strict=True)
    if sys.version_info[:3] != (3, 15, 1):
        raise ValueError("Installed Python must be 3.15.1")
    if not Path(sys.executable).resolve(strict=True).is_relative_to(prefix):
        raise ValueError("Installed Python escapes selected prefix")

    import numpy as np
    import polars as pl
    import pyarrow as pa
    from pyarrow import ipc
    import pyarrow.parquet as pq

    import voiage
    import voiage._core
    from voiage.methods.basic import evpi
    from voiage.schema import ValueArray

    paths = [
        sys.executable,
        voiage.__file__,
        voiage._core.__file__,
        np.__file__,
        pa.__file__,
        pl.__file__,
    ]
    if any(
        not Path(path).resolve(strict=True).is_relative_to(prefix) for path in paths
    ):
        raise ValueError("Installed imports escape selected prefix")
    info = voiage._core.runtime_info()
    if (
        metadata.version("voiage") != "2.2.0"
        or info["engine"] != "rust"
        or info["core_version"] != "2.2.0"
        or info["source_revision"] != "3df3cb49e5c927075c0609a48a2535b4457424d4"
        or info["source_dirty"] is not False
    ):
        raise ValueError("Installed candidate provenance differs from qualified source")
    value = evpi(ValueArray.from_numpy(np.array([[0.0, 2.0], [2.0, 0.0]]), ["A", "B"]))
    if value != 1.0:
        raise ValueError("Installed EVPI differs from numerical fixture")
    table = pa.table({"value": pa.array([1, None, 3], type=pa.int64())})
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    if pa.__version__ != "26.0.0" or not ipc.open_stream(
        sink.getvalue()
    ).read_all().equals(table):
        raise ValueError("Installed Arrow schema/null/buffer round trip failed")
    frame = pl.from_arrow(table).lazy().filter(pl.col("value").is_not_null()).collect()
    if pl.__version__ != "2.1.0" or not frame.to_arrow().equals(
        pa.table({"value": [1, 3]})
    ):
        raise ValueError("Installed Polars lazy/Arrow round trip failed")
    parquet = pa.BufferOutputStream()
    pq.write_table(table, parquet)
    if not pq.read_table(pa.BufferReader(parquet.getvalue())).equals(table):
        raise ValueError("Installed Parquet schema/null round trip failed")

    objects = [Path(voiage._core.__file__)]
    for name in ("pyarrow", "polars-runtime-32"):
        distribution = metadata.distribution(name)
        candidates = [
            Path(distribution.locate_file(path))
            for path in distribution.files or []
            if str(path).endswith(".so")
        ]
        if not candidates:
            raise ValueError(f"Missing installed native extension: {name}")
        if name == "pyarrow":
            for stem in ("lib", "_parquet"):
                matches = [
                    path for path in candidates if path.name.startswith(stem + ".")
                ]
                if len(matches) != 1:
                    raise ValueError(f"Missing unambiguous Arrow extension: {stem}")
                objects.append(matches[0])
        else:
            objects.append(min(candidates))
    linkage = []
    for obj in objects:
        if not obj.resolve(strict=True).is_relative_to(prefix):
            raise ValueError("Native extension escapes selected prefix")
        result = subprocess.run(  # noqa: S603 - fixed system tool and installed object
            ["/usr/bin/ldd", str(obj)], capture_output=True, text=True, check=True
        )
        text = result.stdout + result.stderr
        if "not found" in text:
            raise ValueError("Installed native linkage is unresolved")
        targets = [
            Path(path).resolve(strict=True)
            for path in re.findall(r"(?<!\S)(/[^\s(]+)", text)
        ]
        if any(
            not permitted_link_target(target, prefix, args.cell) for target in targets
        ):
            raise ValueError("Native linkage crosses selected installation boundaries")
        with obj.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        linkage.append(
            {
                "object": str(obj),
                "sha256": digest,
                "targets": [str(path) for path in targets],
                "transcript": text,
            }
        )
    linked = {Path(target).name for entry in linkage for target in entry["targets"]}
    if not any(name.startswith("libarrow.so") for name in linked) or not any(
        name.startswith("libparquet.so") for name in linked
    ):
        raise ValueError("Selected Arrow/Parquet shared SDK linkage is missing")
    args.output.write_text(
        json.dumps(
            {
                "candidate": "2.2.0",
                "python": sys.version,
                "prefix": str(prefix),
                "paths": paths,
                "core_provenance": info,
                "evpi": float(value),
                "arrow_round_trip": True,
                "polars_round_trip": True,
                "parquet_round_trip": True,
                "linkage": linkage,
                "scope": "one installed cell; whole matrix and cross-generation acceptance remain separate",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
