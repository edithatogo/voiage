"""Minimal Python 3.14t observation independent of the scientific stack."""

from __future__ import annotations

import json
import sys

import polars as pl
import pyarrow as pa


def main() -> None:
    source = pa.table({"id": [1, 2], "value": [1.5, 2.5]})
    restored = pl.from_arrow(source).to_arrow()
    assert restored.to_pydict() == source.to_pydict()
    assert restored.__arrow_c_stream__() is not None
    is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: None)()
    print(
        json.dumps(
            {
                "gil_enabled": is_gil_enabled,
                "pyarrow": pa.__version__,
                "polars": pl.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
