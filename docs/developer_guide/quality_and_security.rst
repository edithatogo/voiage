Quality and security harness
============================

``voiage`` treats repository automation as a release surface. The local
harness and the GitHub workflows therefore validate both Python behavior and
the controls that protect changes, dependencies, and published artifacts.

Local gates
-----------

Run the complete gate before opening or updating a pull request::

   uv run python scripts/repo_harness.py
   tox

The ``harness`` tox environment checks that every workflow declares explicit
permissions, every external action is pinned to an immutable commit, required
security workflows exist, and the repository governance files are present.
The full tox environment additionally runs linting, type checking, docs,
contract validation, the supported Python matrix, and the coverage threshold.
For dependency-only changes, also run ``uvx --from pip-audit pip-audit
--local --progress-spinner off`` and ``pnpm audit --prod`` from
``docs/astro-site``.

GitHub gates
------------

The repository runs the following controls on the hosted surface:

* CodeQL with the security-and-quality query suite and a blocking alert gate.
* OpenSSF Scorecard with SARIF upload and signed result publication.
* Dependency Review for pull-request dependency changes.
* Zizmor auditing of GitHub Actions workflows.
* SHA-pinned actions, least-privilege permissions, and disabled credential
  persistence on security-sensitive checkouts.
* Release artifact provenance attestations alongside trusted package publishing.

The protected ``main`` ruleset requires pull requests, resolved review threads,
successful required checks, signed commits, and linear history. The required
approval count is zero for the repository's single-maintainer operating model;
CODEOWNERS still routes review responsibility without creating an impossible
independent-approval gate. Repository settings are verified with the GitHub API
because branch protection and security-analysis state are hosted controls
rather than files in the checkout.

Owner and organization gates
----------------------------

Some controls cannot be truthfully configured from this repository alone:

* organization-wide Actions policies, runner groups, and required workflows;
* billing- or plan-dependent secret-scanning validity checks;
* enterprise security policy and outside-collaborator restrictions;
* environment reviewers and production deployment approvals.

These remain explicit administrator-owned follow-up gates. They must not be
represented as locally complete merely because the repository workflows pass.
