---
title: Arrow interchange and compatibility
description: Versioned Arrow, Parquet, and evidence-manifest policy for VOIAGE.
---

VOIAGE uses Apache Arrow as its in-memory interchange contract and compressed
Parquet or Arrow IPC files at durable boundaries. Perspective-result tables
carry the method contract version, estimand, and interchange name in schema
metadata.

The versioned golden artifacts are under
`specs/frontier/perspective/v1/interchange/`. Regenerate and verify them with:

```console
uv run python scripts/perspective_interchange_manifest.py --write
uv run python scripts/perspective_interchange_manifest.py --check
```

The manifest fingerprints the serialized Arrow schema. Text-source hashes are
computed after UTF-8 decoding and LF normalization; binary Parquet and IPC
files are hashed as raw bytes. This keeps evidence hashes stable across Windows
and Linux while retaining byte-exact binary verification.

CI verifies PyArrow-to-Polars round trips, fresh-process reads, the current and
previous fixture version, and relative serialization performance against JSON
Lines. Python 3.12–3.14 is the supported runtime range. Python 3.14t is observed in a
non-blocking job while its ecosystem support remains beta.
