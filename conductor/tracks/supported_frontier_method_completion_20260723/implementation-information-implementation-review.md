# Independent implementation review — implementation-information decomposition

Issue: [#593](https://github.com/edithatogo/voiage/issues/593), with delivery
subissues #766–#768 under frontier programme #318.

Review date: 2026-08-01. Reviewed revision:
`b6d8bedb77bbff6e3da3161e3569024eb7a64fc8`. Reviewed maturity:
experimental Python implementation planned for v1.3.0 under canonical C18/M25.

This is an independent repository implementation and contract review. It is
not named scientific approval, hosted exact-head or installed-wheel evidence,
Rust/R/Julia parity, stable promotion, release evidence, or issue-closure
authorization.

## Scope reviewed

- issue #593, M25-U1–M25-U3, the Mermaid design and F593-1–F593-4 plan;
- the Johannesen et al. EVPIM/EVSIM definitions and Heath
  implementation-adjusted EVSI definition cited by the contract;
- the exact finite Python evaluator, four-cell matrix, costs, scaling,
  conditional implementation scenarios, public API and CLI;
- strict input/result schemas, the normative fixture, deterministic fixture
  generator, capability/registry records and shared evidence pins;
- state/action/signal dependence, complete ties, switches, interaction and
  identity residuals, uptake deltas, failure boundaries and language/maturity
  dispositions; and
- documentation, migration guidance, roadmap, todo and changelog projections.

The review compared the implementation with the frozen equations and primary
definitions rather than treating schemas, generated fixtures, method names or
issue status as numerical evidence.

## Equations and semantics checked

For uncertain state `s`, intended action `a`, realised action `r`, state law
`p(s)`, net benefit `NB(s,r)` and implementation scenario
`q(r | s,a)`, the implementation-weighted state/action value is

\[
u_q(s,a)=\sum_r q(r\mid s,a)NB(s,r).
\]

The four reviewed per-person-time cells are

\[
C_{00}=\max_a\sum_s p(s)u_{current}(s,a),\qquad
C_{10}=\sum_s p(s)\max_a u_{current}(s,a),
\]

\[
C_{01}=\max_a\sum_s p(s)NB(s,a),\qquad
C_{11}=\sum_s p(s)\max_a NB(s,a).
\]

The implementation computes realizable EVPI as `C10-C00`, EVPIM as
`C01-C00`, EVP as `C11-C00`, and the signed interaction as
`C11-C10-C01+C00`. Therefore the reported residual for

\[
EVP=\operatorname{realizable\ EVPI}+EVPIM+interaction
\]

is exactly zero up to floating-point arithmetic. EVSIM replaces the current
scenario with the declared specific implementation scenario under current
information.

For signal `x` with likelihood `p(x | s)` and signal-dependent implementation
`q_x`, IA-EVSI is evaluated jointly as

\[
\sum_x\max_a\sum_s p(s)p(x\mid s)u_{q_x}(s,a)-C_{00}.
\]

This is the finite joint-law version of the sample-dependent implementation
value in Heath; it does not multiply ordinary EVSI by fixed uptake. Population
and discounted-time scaling is applied once to gross values. Declared costs
are aggregate, nonnegative action costs and are subtracted once from their
matched gross component. Net values remain signed.

An independent enumeration of the normative fixture reproduced cells
`[12.0, 12.4, 12.0, 14.0]`, realizable EVPI `400`, EVPIM `0`, EVP `2000`,
interaction `1600`, EVSIM `400`, and IA-EVSI `1000` after the declared scale.
State, action and signal reordering preserved all estimands and audits.

## Findings and remediation verification

### Critical

None.

### High

None open. The re-review verified remediation of every original High finding:

1. `voiage.methods` now resolves both advertised lazy exports, and the curated
   package-export contract includes them.
2. Perfect-information switching compares each state policy with the baseline
   policy set; a case retaining the same action in every state reports no
   switch.
3. Installed Draft 2020-12 schemas constrain all scientific envelopes, while
   runtime semantic checks reject unknown/missing keys, malformed identifiers,
   incoherent state/action/signal maps, unknown costs, empty units, malformed
   chronology and non-finite intermediate or aggregate values.
4. Results now expose implementation scenarios, uptake changes,
   state/action cell values and signal/action joint values required to audit
   current, specific, perfect and post-sample implementation.

### Medium

None open. The re-review verified that:

- complete ties and the false-switch regression are directly tested;
- state/action permutations are regression-tested and an independent
  signal-order probe preserves gross/net components, policies and uptake
  audits;
- unit, chronology, unknown-cost, non-finite-value and scaling-overflow
  pathologies fail closed;
- Ruff, formatting and BasedPyright report zero findings;
- the deterministic fixture generator reproduces the checked-in result byte
  for byte with SHA-256
  `c770273d92cc2361689abfb9465f897f70064f1bfab7a930b0f2846da6d02c4d`;
- shared DSA, Value of Flexibility and Distribution-Family Information
  evidence pins remain valid after the new CLI/export surfaces; and
- the legacy scalar multiplier, migration boundary, roadmap, todo and
  changelog are synchronized without relabelling the helper as EVPIM, EVSIM or
  IA-EVSI.

### Low

None open in the reviewed experimental repository scope.

## Validation evidence

- Focused implementation, package-export, frontier-fixture, CLI E2E, legacy
  implementation and frontier-CLI suite: **91 passed**.
- Shared evidence-pin and CLI registry slice: **4 passed**.
- Ruff check and format check over changed Python surfaces: **passed**.
- BasedPyright over implementation, tests and generator: **0 errors, 0
  warnings**.
- Independent adversarial and numerical probes: **passed**.
- Fixture generator rerun: **byte-identical**, no worktree diff.
- Frontier contract validator: **passed**, including
  `implementation_information_decomposition`.
- Full Conductor validator: **146 tracks, zero errors, zero warnings**.
- Astro documentation check: **zero errors, warnings or hints**.
- Evidence-ledger append-only validation and `git diff --check`: **passed**.

Coverage collection in the local Python 3.14 environment remains unavailable:
both pytest-cov and direct coverage execution fail before test collection when
NumPy is loaded twice during plugin initialization. This is not accepted as
coverage evidence and does not replace the pending installed-wheel and hosted
exact-head gates.

## Remaining gates

F593-4 remains in progress. Before it, #593, stable promotion or release can be
completed, the repository still requires:

1. installed-wheel and hosted exact-head required-check evidence for the final
   branch revision;
2. named scientific review of terminology, assumptions, implementation
   dynamics and empirical applicability;
3. Rust, R and Julia shared-fixture execution before any parity claim, with
   Mojo retained as an external boundary;
4. separate stable-promotion, merge, signing, release and issue-closure
   decisions; and
5. final Project 28, issue and parent-programme reconciliation from their own
   governed evidence.

Subject to those external and later gates, the reviewed experimental Python
implementation satisfies the repository-owned #593/M25 contract.

## Rebase ratification

The reviewed six-commit range was rebased without patch changes onto
`origin/main` at `76e7f397239ccdbcb683ca23db87e675ec115388`. Stable range-diff
matched each historical reviewed commit to its signed replacement:

- implementation: `4f7f1dec` → `9cc93c63`;
- governance: `e3797ee7` → `4d7b7151`;
- shared assurance: `a01aa606` → `db1baf89`;
- hardening: `4b8bb869` → `b6d8bedb`;
- review artifact: `8570b4e3` → `f945f87b`; and
- review boundary: `8959c720` → `f52feb28`.

All six replacement commits have valid repository-configured GPG signatures.
[PR #787](https://github.com/edithatogo/voiage/pull/787) is open against
`main` at exact head `f52feb2877d4802e9ef9da8a918eb5df46afd81b`.
The hosted-check snapshot contained both successful and in-progress checks, so
this ratification is content-equivalence and review provenance evidence, not a
hosted-gate completion claim. Historical hashes remain unchanged in the
append-only evidence ledger; a new correction entry binds this ratification to
the rebased exact head and PR.

## Reviewer attestation

Reviewer role: independent Codex review agent, separate from the implementing
root agent. Attested result: **PASS**, with no open Critical, High, Medium or
Low implementation findings at reviewed revision `b6d8bedb77bbff6e3da3161e3569024eb7a64fc8`.
This artifact is committed in the repository's configured GPG-signed review
commit; its exact commit is recorded in F593-4 and the append-only evidence
ledger.
