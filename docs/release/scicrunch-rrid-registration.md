# SciCrunch/RRID registration handoff

Checked: 27 July 2026

Status: **ready for account-bound submission; not submitted; no RRID assigned**

SciCrunch accepts software tools in its Registry. Its current curation guidance
requires a resource name, URL, and description. Other fields may help curation
but are not mandatory. A submitted suggestion does not receive an RRID until a
SciCrunch curator approves it.

Before entering a new record, use the form's first step to check for an existing
`voiage` record. The public searches checked during preparation returned no
matching record, but the form performs the authoritative duplicate check.

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

## Account-holder checkpoint

The authenticated account holder should:

1. Open `https://scicrunch.org/create/resource`.
2. Run the portal's duplicate check for `voiage`.
3. Enter the prepared answers and inspect the rendered record.
4. Confirm that the information is accurate and accept the current portal
   terms.
5. Submit and retain the confirmation email, identifier, URL, and timestamp.

After submission, update the machine-readable state to `submitted`, set
`submission.performed` to `true`, and record authoritative confirmation
evidence. Keep `curation.rrid` null until SciCrunch assigns an identifier.

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

The validator fails if local preparation is represented as submission or
curation, if an RRID is invented, if required answers are absent, or if the
release and Software Heritage evidence is malformed.

## Authoritative guidance

- Registration form: `https://scicrunch.org/create/resource`
- Registry overview and curation guidance:
  `https://scicrunch.org/scicrunch/about/Curation%20Guidelines`
- RRID system: `https://www.rrids.org/rrid-system`
