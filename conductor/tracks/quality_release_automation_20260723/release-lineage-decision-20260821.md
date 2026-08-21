# Release-lineage decision — 2026-08-21

PyPI and GitHub already contain stable release `2.0.1`, while the repository
manifests had subsequently returned to `2.0.1-rc.4`. Publishing that candidate
would regress the public version line and is therefore rejected.

The owner-approved Option A release is assigned version `2.1.0`. A minor
version accurately represents the backward-compatible public capabilities
added since 2.0.1. The release includes the stable core while retaining all
frontier APIs, including M22–M31, at their documented experimental maturity.
No scientific promotion follows from the package version.

Repository merge, signed tag, staged private draft, digest review, publication,
and external registry acceptance remain distinct receipts. This decision
authorizes the repository-owned release path but does not claim conda-forge,
Julia General, CRAN, SciCrunch, arXiv, or journal acceptance.
