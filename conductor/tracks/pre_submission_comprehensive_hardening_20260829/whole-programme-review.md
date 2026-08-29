# Whole-programme automated review

## Outcome

The complete consolidated programme was reviewed across requirements, feature
gaps, analytical and data contracts, API/ABI ownership, installed polyglot
packages, dependencies, security, CI/test performance, governance, manuscript,
release alignment, and venue-readiness evidence. No unresolved Critical, High,
or Medium repository finding remains.

The review does not reinterpret external gates as repository defects. The
published v2.1.0 artifacts precede the hardened source, so the frozen result is
explicitly **not eligible for submission** until a separately authorized future
release binds the final source, SBOM, provenance, and registry artifacts.

## Final gates

- Full tox at closeout revision `0f193d65`: all 15 environments passed in
  697.71 seconds.
- Full coverage suite: 4,503 passed, 16 intentionally skipped, 95.16% coverage.
- Repository harness, Ruff, formatting, ty, Vale, JOSS validator, dependency
  frontier, version synchronization, Python 3.12–3.14, minimum/maximum
  dependencies, ingestion conformance, and 1,030-page docs build: passed.
- Exact-head hosted preview run 33262673063: workflow passed; both deterministic
  shards and the repaired sccache/nextest lane passed. Python 3.15 and Ruff 0.16
  remain intentional non-promotion findings.
- Exact-main retained-binding workflow 33260609561 and pinned `pkgcheck`:
  passed.
- Full Conductor validation, GitHub cross-reference validation, submission
  readiness validation, and append-only evidence-ledger integrity: passed.

## External and human gates

No pyOpenSci, rOpenSci, JOSS, arXiv, registry, archive, badge, funding,
publication, release, or external-attestation action was performed. Authorship,
source confirmation, community engagement, venue communication, editorial
review, acceptance, and the future release decision remain with their
accountable human or external authorities.
