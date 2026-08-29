# G5 conformance and pathological review protocol

This protocol is the reproducible review artifact for AC-03. It defines the
negative, reference, property and pathological cases that must be exercised
before a release-automation or Decision Studio claim is promoted. Cases are
deterministic, use repository fixtures only, and must run without credentials,
network access, uploads, registry publication or release approval. A runner
records the case ID, input fixture hash, observed disposition and artifact
hash; a mismatch is a failure, not a warning.

## Execution contract

1. Run from a clean checkout with a fixed Python/Rust toolchain and
   `SOURCE_DATE_EPOCH` set.
2. Disable network and credential discovery; use temporary directories outside
   the repository for generated artifacts.
3. For every case below, assert the expected disposition and stable diagnostic
   code. Redact secret-like values before writing logs.
4. Store the manifest, command line, tool versions, fixture hashes and output
   hashes in the Conductor evidence ledger. Do not interpret a local pass as
   hosted, scientific, registry or publication approval.

## Required cases

| ID | Class | Deterministic input / mutation | Expected disposition and evidence |
|---|---|---|---|
| G5-REG-01 | Conformance | Registry card omits its owning issue or native child link | Reject with a stable cross-reference diagnostic; no status transition. |
| G5-REG-02 | Pathological | Two cards claim the same track/issue with conflicting lifecycle states | Reject as duplicate/conflicting ownership; emit both card hashes. |
| G5-REG-03 | Property | For every valid lifecycle transition, replaying the transition is idempotent | State and serialized registry hash remain unchanged on replay. |
| G5-REG-04 | Reference | Compare registry, `metadata.json`, `plan.md` and `spec.md` against the canonical Conductor schema | Any missing, stale or mismatched field fails the review with a file/field path. |
| G5-XREF-01 | Pathological | Child issue, Project item or PR reference points to a deleted/non-numeric target | Fail closed; preserve the unresolved reference and remediation note. |
| G5-UI-01 | Conformance | Decision Studio request contains estimator execution or credential fields | Reject at the adapter boundary; estimator and secrets never reach UI output. |
| G5-UI-02 | Pathological | Adapter receives an unsupported mode, malformed JSON, or unknown schema version | Return a typed unsupported-input diagnostic; perform no I/O or partial write. |
| G5-UI-03 | Property | Replaying a valid local request with the same seed produces byte-identical output | Compare canonical JSON and artifact SHA-256; any drift fails. |
| G5-REL-01 | Conformance | Version differs across Python, Rust, R/Julia manifests and release metadata | Block release and identify every divergent manifest. |
| G5-REL-02 | Pathological | Dirty checkout, detached/unexpected head, or untracked generated file | Block packaging; record git status and exact head. |
| G5-REL-03 | Reference | Build the same source twice with fixed epoch and isolated temp dirs | Wheel/sdist/bindings hashes and manifest contents must match. |
| G5-REL-04 | Pathological | Artifact is missing SBOM, provenance, signature, checksum or required license | Block publication; identify missing evidence without attempting upload. |
| G5-REL-05 | Conformance | Prerelease or registry publication requested without explicit approval evidence | Remain gated; produce an approval-required diagnostic. |
| G5-SEC-01 | Pathological | Fixture path uses `../`, absolute paths, symlinks escaping the root, or archive traversal | Reject before extraction/read; no file outside the sandbox is touched. |
| G5-SEC-02 | Pathological | Malformed JSON/schema, duplicate keys, oversized payload or zip-bomb ratio | Reject with bounded-resource diagnostic; process stays within configured limits. |
| G5-SEC-03 | Pathological | Inputs and logs contain token/password/API-key-shaped strings | Redacted diagnostics contain no secret value or reversible encoding. |
| G5-SEC-04 | Reference | Redirect/cache metadata changes source URL, content hash or policy decision | Treat as a new artifact and revalidate; never reuse a stale cache decision. |
| G5-ASSURE-01 | Reference | Counterfactual, policy, regret, constraint and calibration fixtures | Published invariants and tolerances match the versioned reference outputs. |
| G5-ASSURE-02 | Property | Permuting independent rows, then canonicalizing, leaves the result invariant | Canonical result and evidence hash are unchanged. |
| G5-ASSURE-03 | Conformance | Provenance is absent, incomplete or inconsistent with fixture hashes | Block positive claim; emit provenance-required diagnostic. |

## Review record

The reviewer records the exact command, case manifest hash, fixture hashes,
toolchain versions, pass/fail counts and artifact hashes. Failures are linked
to the relevant Conductor task and GitHub issue/PR; they are not silently
converted into exclusions. This protocol is evidence preparation only: an
independent evidence review, hosted checks, release authority, registry
acceptance and scientific/practitioner review remain separate gates.

## Independent repository references

The review compares observed behavior with these independently maintained
contracts rather than with this protocol's expected results alone:

- `conductor/workflow.md` — task lifecycle, validation and evidence rules.
- `conductor/github-cross-references.json` — issue, Project and PR linkage.
- `specs/v1/stable-api.json` — stable API and version compatibility contract.
- `.github/workflows/` — hosted quality, security and release automation.
- `conductor/tracks/quality_release_automation_20260723/spec.md` — this
  track's maturity, privacy and external-gate boundaries.
