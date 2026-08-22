# AI-Assisted Development & Transparency Statement

`voiage` adheres to transparent, responsible disclosure of AI-assisted engineering and research tooling.

## Core Governance Principles

1. **Non-Authorship:** AI systems, LLMs, and automated agents are not authors, contributors, or copyright holders.
2. **Human Accountability:** The human maintainer (@edithatogo) reviews, tests, approves, and assumes sole scientific and engineering accountability for all merged changes.
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

| Release / Target | Focus Component | AI Tooling Used | Verification Method | Human Sign-Off |
| :--- | :--- | :--- | :--- | :--- |
| `v1.0.0` | Initial VOI Core | IDE completion | Pytest unit suite, tox matrix | Dylan A Mordaunt |
| `v2.0.0` | Rust Polyglot ABI | Antigravity CLI / Copilot | Cargo test suite, PyO3 bindings, cross-platform CI | Dylan A Mordaunt |
| `v2.1.0` | Stable Core & Frontier | Antigravity CLI (Gemini) | 60+ GitHub Actions checks, property tests, branch coverage | Dylan A Mordaunt |
| `v2.2.0` (In Progress) | Enterprise Decision Suite & Assurance | Antigravity CLI (Gemini) | Exhaustive decision correctness suite, 100% branch coverage | Dylan A Mordaunt |
