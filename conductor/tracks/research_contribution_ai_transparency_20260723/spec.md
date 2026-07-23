# Track Specification: Research Contribution And AI Transparency

## Overview

Create canonical human-contribution and auditable AI-assistance statements for
the repository, releases, and manuscript.

## Requirements

1. Add `CONTRIBUTORS.md`, confirmed CRediT roles in canonical LaTeX, and
   synchronized `CITATION.cff` and `codemeta.json`.
2. Add `AI_CONTRIBUTION.md`, a versioned schema, append-only ledger, and
   synchronized manuscript disclosure.
3. Record tool/provider/model when known, date, purpose, affected components,
   accepted/revised/rejected disposition, verification, limitations, privacy,
   issue/track/PR/commit, release, and accountable human.
4. Add PR/issue disclosure fields and optional `Assisted-by:` trailers.
5. Run Authentext as claim analysis, never scientific approval.

## Privacy, authorship, and failure policy

Do not store chain-of-thought, secrets, personal/confidential data, or raw
prompts by default. AI is not an author or CRediT contributor. Do not infer
human roles from commit counts. Missing confirmation or human review blocks
publication/release claims.

## Acceptance criteria

Human roles are confirmed; material AI assistance is reviewed and release-
linked; repository, metadata, generated statement, and manuscript agree.

## External gates

Human contributor confirmation, authorship, journal-specific disclosure, and
manuscript submission.

