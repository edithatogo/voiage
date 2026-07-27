# SciCrunch/RRID registration handoff

Checked: 27 July 2026

Status: **submitted; SciCrunch curation pending; no RRID assigned**

SciCrunch accepts software tools in its Registry. Its current curation guidance
requires a resource name, URL, and description. Other fields may help curation
but are not mandatory. A submitted suggestion does not receive an RRID until a
SciCrunch curator approves it.

The authenticated portal duplicate check for `voiage` returned “There was no
resource similar in our system.” The full Resource Submission form was
submitted at 2026-07-27T06:07:06Z after the account holder confirmed the
rendered answers and current terms. SciCrunch displayed “Thank you for your
submission!” at `https://scicrunch.org/scicrunch/about/thanks`. The portal did
not provide a submission number or RRID on that confirmation page.

## Required answers

**Resource name**

`voiage`

**URL**

`https://github.com/edithatogo/voiage`

**Description**

> voiage is software for Value of Information analysis in research,
> implementation, and decision making. It calculates the expected value of
> reducing uncertainty about decisions, parameters, models, studies,
> populations, implementation, and timing. A Rust core supplies stable
> numerical kernels; Python provides the broad analysis interface, with R and
> Julia bindings to the shared native library.

## Recommended optional answers

- Resource type: software resource; data analysis software; software library;
  source code.
- Availability: free and openly available.
- Licence: Apache-2.0.
- Current version: 2.0.0, released 26 July 2026.
- Creator: Dylan A. Mordaunt,
  ORCID `https://orcid.org/0000-0002-9775-0603`.
- Alternate URLs: documentation, the exact PyPI release, and the root
  `voiage-core` crates.io package.
- Keywords: copy the ordered list from the machine-readable packet.
- Defining citation: leave empty until the arXiv preprint or JOSS article has a
  permanent identifier. Do not use an incomplete submission identifier.
- Funding: leave empty unless the account holder supplies an attributable
  funding statement.
- Abbreviation: leave empty; `voiage` is the resource name, not a declared
  expansion.

The complete copyable answer set and evidence ledger are in
`scicrunch-rrid-registration.json`.

## Submission evidence

- Duplicate check: completed; no similar resource found.
- Account-bound accuracy and terms confirmation: completed.
- Submission: completed at 2026-07-27T06:07:06Z.
- Confirmation message: “Thank you for your submission!”
- Confirmation URL: `https://scicrunch.org/scicrunch/about/thanks`.
- Confirmation identifier: none displayed.
- RRID: not assigned; SciCrunch curation remains pending.

## Curator-response protocol

- Record questions and answers verbatim or link to an attributable copy.
- Answer from versioned repository evidence; do not infer funding, institutional
  ownership, a defining publication, or registry acceptance.
- If accepted, record the assigned identifier in the
  `RRID:SCR_######` form, the assignment timestamp, and the resolver URL.
- Allow the documented indexing delay before treating the resolver record as
  publicly discoverable.
- If rejected or identified as a duplicate, preserve the decision and use the
  curator-provided canonical record.

## Validation

Run:

```bash
python scripts/validate_scicrunch_rrid.py
pytest tests/test_scicrunch_rrid_readiness.py --no-cov -q
```

The validator distinguishes prepared and submitted states, requires
account-bound declarations and confirmation evidence for a submitted record,
and fails if curation or an RRID is invented.

## Authoritative guidance

- Registration form: `https://scicrunch.org/create/resource`
- Registry overview and curation guidance:
  `https://scicrunch.org/scicrunch/about/Curation%20Guidelines`
- RRID system: `https://www.rrids.org/rrid-system`
