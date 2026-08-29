# Blocker-Resolution Plan

## Options

1. **Repository transparency baseline (recommended):** maintain the AI-use
   disclosure, accountable governance language, schema/ledger design, and
   release-scope validation locally; require a human maintainer to attest
   authorship, CRediT roles, and retained-output review before publication.
2. **Publication-bound attestation:** collect a signed, release-digest-bound
   CRediT and AI-assistance record and run the full manuscript/release audit.
   This is the correct next step for publication but requires the accountable
   human and exact release artifact.
3. **Defer detailed provenance:** retain only the current general disclosure.
   This avoids new records but leaves the track's machine-readable provenance
   requirement unresolved.

## Recommendation

Use option 1 now, then option 2 when a human maintainer supplies the release
identity and attestation. AI output must remain non-authorial, and raw prompts,
secrets, confidential data, and chain-of-thought must not enter the ledger.

## Contingencies and exit criteria

- If model/tool identity is unavailable, record it as unknown rather than infer
  it from commits or environment observations.
- If the maintainer cannot attest retained-output review, keep publication and
  CRediT promotion blocked.
- Archive requires canonical CRediT data, release-linked provenance, human
  attestation, review evidence, and hosted checks; repository prose alone is
  insufficient.
