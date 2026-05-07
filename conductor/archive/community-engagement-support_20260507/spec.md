# Community Engagement And Support

## Overview

`voiage` already has contributor docs, issue templates, and release automation,
but the community-facing support surface is still incomplete. This track makes
the support and governance path explicit by adding repository-level guidance
for getting help, reporting security issues, and understanding community
expectations. The goal is to make the project easier to adopt, safer to use,
and simpler to contribute to without changing the library's runtime behavior.

## Goals

1. Provide a clear support and escalation path for users and contributors.
2. Add the standard community-health documents that are still missing from the
   repository.
3. Make the issue and pull-request surfaces better aligned with the current
   contribution workflow.
4. Keep the community-facing docs consistent with the existing pragmatic,
   contract-first voice of the repository.

## Functional Requirements

1. The repository must have a support-oriented community page that explains:
   - where to ask questions
   - how to report bugs or request features
   - what information to include in a good report
   - how the maintainers expect issues to be triaged
2. The repository must have a code-of-conduct document and a security policy.
3. The issue and pull-request templates must align with the current support and
   contribution workflow.
4. The top-level README and contributor guidance must link to the new
   community-health documents.
5. The roadmap and backlog must reflect the community engagement surface as
   implemented work, not as an unresolved placeholder.

## Non-Functional Requirements

1. The new documents should be concise and unambiguous.
2. The support and security guidance should be repository-local and
   non-interactive.
3. No runtime behavior or release channels should change as a result of this
   track.

## Acceptance Criteria

1. Community-health files exist and are linked from the main documentation.
2. The issue/PR templates reflect the support and contribution workflow.
3. The roadmap and backlog state the community engagement work as implemented.
4. The repository still validates cleanly after the docs and template updates.

## Out of Scope

1. Running a community program, forum, or external discussion platform.
2. Introducing new runtime features or changing the VOI analysis surface.
3. Changing release channels, registry targets, or versioning policy.
