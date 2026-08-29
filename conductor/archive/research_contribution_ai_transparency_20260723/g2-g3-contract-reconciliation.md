# G2–G3 artifact reconciliation and frozen contract

## Existing-artifact disposition

`codemeta.json` and `paper/metadata.json` remain authoritative for package and
manuscript metadata respectively. This track does not overwrite either file or
infer contributor roles from commit history. No canonical CRediT record or
release-linked AI ledger was present in the inspected source tree; those are
new, versioned artifacts to be added only with accountable human confirmation.

The manuscript disclosure is a human-owned publication input. A generated
disclosure, panel output, or local validation result cannot certify authorship,
consent, or submission.

## Frozen AI-assistance record contract

Each ledger event must contain exactly these fields:

```json
{
  "schema_version": "1.0",
  "event_id": "stable-event-id",
  "recorded_at": "UTC ISO-8601 timestamp",
  "release_ref": "immutable tag or commit",
  "provider": "known provider or null",
  "tool": "known tool or null",
  "model": "known model or null",
  "purpose": "bounded engineering/research purpose",
  "scope": ["path or artifact"],
  "disposition": "accepted|rejected|pending-human-review",
  "verification": {"reviewer": "human identifier or null", "method": "..."},
  "limitations": ["known limitation"],
  "redactions": ["prompt|secret|confidential-data"],
  "previous_event_sha256": "hash or null",
  "event_sha256": "sha256(canonical event without event_sha256)"
}
```

Events are append-only and hash-chained. Raw prompts, chain-of-thought,
credentials, personal data and confidential inputs are prohibited. Unknown
provider/tool/model values are represented as `null`, never guessed. AI may
assist work but is not an author or CRediT contributor; human role assignment,
materiality, verification and manuscript wording remain explicit fields and
human decisions.

## Maturity boundary

This contract is repository-ready evidence, not a completed disclosure. It
does not claim a release, authorship, human approval, scientific review, JOSS
or arXiv submission, or external acceptance. A future release ledger must bind
events to immutable artifacts and a human-confirmed disclosure before any
positive public claim.
