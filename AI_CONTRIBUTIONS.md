# AI-Assisted Development & Transparency Statement

`voiage` adheres to transparent, responsible disclosure of AI-assisted engineering and research tooling.

## Core Governance Principles

1. **Non-Authorship:** AI systems, LLMs, and automated agents are not authors, contributors, or copyright holders.
2. **Human Accountability:** The human maintainer (@edithatogo) retains scientific and engineering accountability. Authorization for autonomous commits and merges is not evidence that the maintainer personally reviewed every output. Current submission materials require an explicit human-review confirmation.
3. **No Chain-of-Thought / Secret Storage:** Raw internal reasoning traces, API keys, credentials, and confidential personal data are never stored in repository files.
4. **Verification Requirement:** AI-generated proposals or code suggestions are never accepted without automated test coverage, type checking, linter validation, and local/hosted CI execution.

---

## AI Systems & Tools Utilized

| System / Tool | Provider | Purpose | Accountable Human Reviewer |
| :--- | :--- | :--- | :--- |
| **Antigravity CLI / Gemini 2.5 Pro** | Google DeepMind | Pair programming, test generation, schema validation, refactoring, and documentation drafting | Dylan A Mordaunt (`@edithatogo`) |
| **GitHub Copilot** | GitHub / OpenAI | Contextual IDE code completion and snippet expansion | Dylan A Mordaunt (`@edithatogo`) |

---

## Transparency Ledger (Append-Only)

The rows below are historical ledger entries, not a current certification of
coverage or personal sign-off. Read the dated correction after the table before
using them as submission evidence.

| Release / Target | Focus Component | AI Tooling Used | Verification Method | Human Sign-Off |
| :--- | :--- | :--- | :--- | :--- |
| `v1.0.0` | Initial VOI Core | IDE completion | Pytest unit suite, tox matrix | Dylan A Mordaunt |
| `v2.0.0` | Rust Polyglot ABI | Antigravity CLI / Copilot | Cargo test suite, PyO3 bindings, cross-platform CI | Dylan A Mordaunt |
| `v2.1.0` | Stable Core & Frontier | Antigravity CLI (Gemini) | 60+ GitHub Actions checks, property tests, branch coverage | Dylan A Mordaunt |
| `v2.2.0` (In Progress) | Enterprise Decision Suite & Assurance | Antigravity CLI (Gemini) | Exhaustive decision correctness suite, 100% branch coverage | Dylan A Mordaunt |

### 2026-08-30 correction and continuation

The earlier v2.2.0 row is superseded as a current-status statement: v2.2.0 is
published, and the recorded full Python gate reports approximately 95 percent
combined coverage, not 100 percent branch coverage. The release, hardening and
venue-packet work also used OpenAI Codex for substantial code/test generation,
repairs, workflow changes, documentation and manuscript editing. The tools
table above is therefore a historical, incomplete inventory, not an exhaustive
list. JOSS's dated tool record additionally includes Google Jules. Exact model
identifiers and a repository-wide generated-code percentage were not retained
for every session; neither is inferred from commit authorship.

| Release / Target | Focus Component | AI Tooling Used | Verification Method | Human Sign-Off |
| :--- | :--- | :--- | :--- | :--- |
| Published v2.2.0 and subsequent unsubmitted packet repairs | Release evidence, supported-environment research replay, factual manuscript and disclosure repairs | OpenAI Codex; historical tool records remain separate | Full tox, numerical-reference tests, exact-head CI, citation/prose audits and PDF review | Autonomous delivery authorized; final personal review of current submission outputs pending |

The human author confirmed review of the then-retained AI-assisted outputs on
27 July 2026. That statement is preserved in
`paper/joss-editorial-assurance.json`; it does not automatically cover later
commits. Before submission, the maintainer must review the current material and
confirm the extent and accuracy of this disclosure, including any primarily
generated components. No AI system is an author or an independent reviewer.
