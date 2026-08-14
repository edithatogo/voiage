# Reproducibility and numerical audit

Score: **808/1000**
Disposition: **not ready for submission**
Fail-closed cap: **applies**

## Checks reproduced

- Frozen clean regeneration matched every tracked worked-example output.
- All six hashes in `paper/reproduction.sha256` matched.
- The recorded `uv.lock` digest matched.
- Focused JOSS, worked-example, and numerical-reference tests passed.
- The signed `v2.0.0` tag resolved to
  `e849e89152c306e79c96d0a8a9815ee5faca0529`.
- EVPI, EVPPI, EVSI, ENBS, bootstrap, and theoretical benchmark values
  reconciled.

## Traceability findings

- “Reviewed commit” was ambiguous because release evidence binds the v2.0.0
  software commit, not an arbitrary current manuscript checkout.
- Official PDF evidence must be refreshed after any manuscript revision and
  bound to the exact committed source.
- A dirty checkout is not submission evidence.
- The platform statement must identify whether it describes the released or
  current workflow revision.

## Gates

All source checks were queued, the AI attestation remained pending, actual
research use was absent, and no independent human engagement was documented.
