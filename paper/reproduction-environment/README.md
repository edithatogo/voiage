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

The current patched replay uses
`paper/reproduction-manifest-python315-patched.json`. Its source commit
identifies a new replay selection, not an attribution of the historical
execution. Run from the repository root:

```sh
python scripts/verify_paper_reproduction.py --manifest paper/reproduction-manifest-python315-patched.json
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
