# Project Workflow

## Source of Truth

- **Repository protocol**: `AGENTS.md`.
- **Action backlog**: `todo.md`.
- **Strategic status**: `roadmap.md`.
- **Conductor state**: `conductor/tracks.md`, `conductor/tracks/`, and
  `conductor/archive/`.
- **Technical conventions**: `CONTRIBUTING.md`, `pyproject.toml`, and
  `tox.ini`.

The previous generic `plan.md` workflow is not the root workflow for this
repository. Use Conductor `plan.md` files only within their specific track
folders.

## Standard Cleanup and Development Loop

1. Read `AGENTS.md`, `roadmap.md`, and `todo.md`.
2. Select one discrete task from `todo.md`; if no task is open, choose the next
   logical roadmap cleanup and record the result in `todo.md`.
3. Inspect the existing code, docs, fixtures, and tests before editing.
4. Implement the smallest coherent slice.
5. Add or update tests so the cleanup or feature cannot regress silently.
6. Update `changelog.md` and `todo.md`; update `roadmap.md` when status changes.
7. Run focused tests first, then the full tox gate.
8. Stage the completed slice and review `git status` plus `git diff HEAD`.

## Verification

Use the repo-defined full gate before treating a slice as complete:

```bash
uv run --with tox tox -e lint,typecheck,docs,py314,coverage_report
pixi run verify
```

Focused checks should be run before the full gate when they are relevant, for
example:

```bash
uv run pytest --no-cov tests/test_repo_cleanup.py
uv run python examples/cli_example.py
```

## Cleanup Rules

- Remove generated artifacts, stale duplicate status reports, local machine
  paths, and root scratch scripts only when they are not part of maintained
  package, docs, tests, CI, or release surfaces.
- Prefer durable guard tests and `.gitignore` coverage so removed artifacts do
  not return.
- Keep external gates explicit. Registry approval, hardware access, fabricated
  silicon evidence, and external curation are not repo-local completion
  criteria.
- Do not rewrite user work or unstaged changes that are unrelated to the active
  cleanup slice.

## Commit Messages

Follow Conventional Commits as documented in `CONTRIBUTING.md`, for example:

```bash
chore(cleanup): remove stale generated artifacts
docs(workflow): align root workflow with agent protocol
```
