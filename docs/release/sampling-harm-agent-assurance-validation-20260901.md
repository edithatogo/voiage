# Agent-only sampling-harm assurance validation

The existing August 4 assurance record is non-human preparation for issues
#876, #853 and #850. Its schema alone did not detect a wrong report digest,
duplicate report role or changed candidate identifier. The semantic validator
now checks those bindings and composes the existing challenge and source
readiness validators:

```sh
python scripts/validate_sampling_harm_agent_assurance.py --repository-root .
```

The historical record, reports, source observations and finding register remain
unchanged. Report references use the repository's canonical JSON SHA-256 with
the self-digest field excluded. The assurance record's synthesis, source and
register references use file-byte SHA-256. These conventions are deliberately
different; neither is replaced with a newly computed historical receipt.

The existing register schema already pins the synthesis digest and rejects a
fully rebound report graph. The additional independent canonical digest pins
the complete historical assurance record,
including its narrative and linked hashes. It was verified against the unchanged
record at commit `25fc585c0bf9fea69d9bcc4220a2d667975ef3eb`. It also rejects
schema-valid changes to the assurance record's own narrative, which the earlier
validator accepted. No schema is changed to demonstrate this narrower gap.
Schema and semantic checks run first to retain specific failure diagnostics;
JSON formatting alone does not change the canonical record identity.

The validator rejects altered references, redirected paths, inconsistent report
roles, candidate or finding bindings, and unavailable authority. It enforces
the recorded November 30 review deadline and rejects known supersession passed
to its Python interface. Callers must supply known source drift, candidate
changes, substantive remediation or new jurisdictions; this local check does
not monitor external sources or determine applicability.

Passing validates historical agent-only evidence. It does not refresh the
expired August 9 H8-C governance snapshot, freeze a qualified replacement
packet, assign humans, resolve findings, or authorize a kernel or study.
The response explicitly retains historical status, nineteen pending findings
and human review not performed.

The owner-selected path remains independent review of a proposed generic-kernel
exclusion. Agent-only preparation and three repository fixes are already
recorded as complete preparation; their independent finding dispositions remain
pending. Source rights and applicability, eligible independent reviewers,
candidate-bound confirmations and accountable disposition still require their
separate evidence. No issue closure or external action follows from this check.

The new module is explicitly classified as assurance in both current v2
inventories. A separate September 1 structure delta retains the original
August 29 audit bytes and its count of 176 modules. It records full membership
from a later verified checkout with that same count, then the exact additional
assurance module. The current 177-module check rejects removals, unregistered
additions and substitutions; it does not relabel that later checkout as the
original audit's provenance or claim a new scientific or ABI audit.
