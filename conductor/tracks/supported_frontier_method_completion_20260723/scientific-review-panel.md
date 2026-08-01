# Scientific review panel protocol

Experimental frontier promotion uses a panel of relevant subagents instead of
a single independent scientific reviewer. The panel gathers challenge evidence
but does not authorize stable promotion, publication, registry submission, or
release.

## Required composition

Each method family requires at least three role-specific subagents:

1. **Estimand reviewer** — definitions, assumptions, units, utility, cost,
   policy, and limiting cases.
2. **Numerical reviewer** — fixtures, identities, convergence, pathologies,
   ties, feasibility, and reproducibility.
3. **Boundary reviewer** — API maturity, language dispositions, provenance,
   unsupported claims, security, and release wording.

Add a domain reviewer where the method requires specialist context. A panel
requires all core roles to report, no unresolved High/Critical finding, and an
explicit disposition for every Medium finding.

## Evidence boundary

Record each panel report, reviewed revision, commands, artifacts, and finding
dispositions in the Conductor ledger. The maintainer retains the final
scientific, stable-promotion, release, and publication decision. If the panel
disagrees, retain the method as experimental and record the disagreement and
required follow-up; do not promote by majority vote alone.
