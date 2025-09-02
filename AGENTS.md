# Agent Protocol for `voiage`

This document outlines the standard operating procedure for AI agents contributing to the `voiage` repository. Adherence to this protocol is mandatory for all agent-driven development.

## High-Level Goal

The primary objective is to advance the `voiage` library by implementing features, fixing bugs, and improving the codebase, guided by the `roadmap.md` and `todo.md` files. The ultimate vision is to establish `voiage` as the premier, cross-domain, high-performance library for Value of Information analysis.

## Agent Operating Procedure

Agents must follow this sequence for every contribution:

1.  **Synchronize:** Read `AGENTS.md` (this file), `roadmap.md`, and `todo.md` to get the latest project status, priorities, and protocols.
2.  **Select Task:** Choose a single, discrete task from the top of the `todo.md` list. Do not assign yourself the task. If the `todo.md` is empty, consult the `roadmap.md` for the next logical feature to implement and add it to `todo.md`.
3.  **Understand Context:** Before writing any code, thoroughly read all files related to the task. Use file search and read tools to understand the existing patterns, data structures, and APIs.
4.  **Implement:**
    *   Write clean, readable, and well-documented code that adheres to the style defined in `CONTRIBUTING.md`.
    *   Add or update unit tests for any new or modified code. The goal is to maintain or increase test coverage.
    *   Ensure all new code is fully type-hinted.
5.  **Verify:**
    *   Run the full test suite locally using `tox`. This command will also run the linter, formatter, and type checker.
    *   Fix any and all errors reported by the verification step. Do not proceed until `tox` passes cleanly.
6.  **Document:**
    *   Update the `changelog.md` with a concise, user-friendly description of the change.
    *   Update the `todo.md` by moving the completed task to the "Done" section.
    *   If the changes have a significant impact on the project's status, update the `roadmap.md` accordingly.
7.  **Commit:**
    *   Use `git status` and `git diff HEAD` to review your changes.
    *   Propose a commit with a message that follows the conventions outlined in `CONTRIBUTING.md`.

## Key Files & Artifacts

*   **`AGENTS.md`**: The primary protocol document.
*   **`roadmap.md`**: The high-level vision and phased development plan.
*   **`todo.md`**: The actionable backlog of tasks for agents.
*   **`changelog.md`**: A log of all user-facing changes.
*   **`CONTRIBUTING.md`**: Technical guidelines for development (code style, testing, etc.).
*   **`pyproject.toml`**: Defines project dependencies and tool configurations (Ruff, MyPy, etc.).
*   **`tox.ini`**: Defines the test environments for verification.

## Development Philosophy

*   **Robustness over Speed:** Prefer solutions that are well-tested and stable.
*   **Clarity is Key:** Write code and documentation that is easy for both humans and other agents to understand.
*   **Follow Conventions:** Adhere strictly to the existing code style and project structure.
