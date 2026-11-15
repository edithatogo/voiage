# Recorded reproduction environment

`uv.lock` preserves the exact bytes whose SHA-256 is recorded in the original
paper receipt: `e5bfefe59aa2920d5b28e9beaa5a6e05cc8b08e4a91e74ed8589d7fc3354f7c5`.
The matching `pyproject.toml` and lock were copied without modification from
commit `27488e817238d6fe63016f2f5a5b15f91b1acda7`.

`original-reproduction-manifest.json` preserves the original receipt byte for
byte. Its `source_reference: v2.0.0` label does not establish that the lock came
from that release: the actual tag's lock has a different digest. Its recorded
root-relative command describes the old receipt; it is not the current replay
instruction. No historical source identity or human use is certified here.

The historical `paper/reproduction-manifest.json` resolves these archived
paths and their digests. The earlier Python 3.15 replay is retained in
`paper/reproduction-manifest-python315.json`. Both frozen locks contain
dependencies with open security advisories; retain them as historical evidence.

The original lock and the earlier Python 3.15 lock are evidence, not supported
installation or replay inputs. Their SHA-256 values are, respectively,
`e5bfefe59aa2920d5b28e9beaa5a6e05cc8b08e4a91e74ed8589d7fc3354f7c5` and
`ee0d8fbaab209be5f858e32e8d2e8d3c43b710cf10b696dc6b24ecc77baa2d57`.
The verifier rejects their historical replay descriptors before starting a
subprocess; the hosted paper workflow selects the current patched descriptor.
The root and current replay locks are checked against the advisory floors for
Pillow, soupsieve, Tornado and OpenTelemetry API.

GitHub may report the frozen locks as runtime dependencies because it discovers
lockfiles independently of the replay selection. Assess each alert individually:
the 11 alerts numbered 335–338 and 340–346 were traced to these two files, and
may be classified as not used only while these exact vulnerable archive entries
remain outside supported execution. This does not fix the archived packages.
Reopen the affected alerts before enabling an archived environment or if an
affected version enters an active environment; assess new alerts separately.
Do not rewrite the archives or disable dependency scanning to resolve alerts.

The previous patched Python 3.15.1 replay remains in
`paper/reproduction-manifest-python315-patched.json`. The earlier 3.15.1
profiles remain verifiable as historical records, but the current verifier
does not execute them because the active repository runtime is Python 3.15.2.

The current patched Python 3.15.2 replay uses
`paper/reproduction-manifest-python3152-patched.json`. Its source commit
identifies a new replay selection, not an attribution of the historical
execution. Run from the repository root:

```sh
.venv/bin/python scripts/verify_paper_reproduction.py --manifest paper/reproduction-manifest-python3152-patched.json
```

The verifier requires the exact source commit in local Git history, Git, uv,
and the Python/Rust build toolchains. It clones into a temporary directory,
checks out that exact commit, and confirms that its project and lock bytes
match the declared archives. It runs the original generator with `uv run
--frozen` inside that isolated project, verifies tracked output hashes, and
prints a digest-bound result only after success. Dependency installation may
access package registries. The temporary checkout is removed afterward.

Current dependency updates change the root lock without changing this replay
environment. The frozen-lock CI jobs still validate the current root project.
A successful replay is new verification evidence; it does not authenticate the
original receipt's release label.
