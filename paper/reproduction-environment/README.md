# Recorded reproduction environment

`uv.lock` preserves the exact bytes whose SHA-256 is recorded in
`paper/reproduction-manifest.json`:
`e5bfefe59aa2920d5b28e9beaa5a6e05cc8b08e4a91e74ed8589d7fc3354f7c5`.
They were copied without modification from commit `27488e81` on 31 August 2026.

The historical manifest and generated outputs remain unchanged. Its
`source_reference: v2.0.0` label does not establish that this lock came from that
release tag: the actual tag's lock has a different digest. This snapshot records
only the environment bytes named by the existing receipt. It does not certify a
new execution, revise the historical source identity, or claim a new human use.

Current dependency updates change the root lock without rewriting these bytes.
The reproduction test checks this snapshot against the existing manifest digest;
the frozen-lock CI jobs continue to validate the current root environment.
