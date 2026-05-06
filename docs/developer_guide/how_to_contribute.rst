How To Contribute
=================

This guide is the short route for contributors who want to add or change a
VOI method, a plotting helper, a contract, or a docs page.

Start here
----------

1. Read ``AGENTS.md``, ``roadmap.md``, and ``todo.md``.
2. Pick a single task from the active backlog.
3. Read the surrounding implementation, tests, and contract files before
   editing.
4. Keep the change small enough to review in one pass.

Working style
-------------

- Use ``tox`` as the main verification gate.
- Use ``uv run nox`` when you want the uv-backed local session runner.
- Keep new code type-hinted.
- Add regression tests for behavior changes.
- Keep prose changes aligned with Vale and the docs build.

Conductor workflow
------------------

The repository uses Conductor to keep spec, implementation, and review in
sync. For a track-based change:

1. Read the track spec and implementation plan.
2. Implement one task at a time.
3. Run the relevant tests before widening scope.
4. Update the track plan and project docs when the task is complete.
5. Let the review/checkpoint step decide whether the phase is ready to move on.

Branching and PRs
-----------------

- Create a feature branch for larger changes.
- Keep commits conventional, focused, and easy to revert.
- Include the docs and tests that explain the behavior change.
- Do not bundle unrelated cleanup with contract changes.

Testing and coverage
--------------------

- Run the full suite with ``tox`` before finishing a substantive change.
- Keep coverage at or above the project threshold.
- Run targeted tests while iterating on a narrow slice.
- Use property-based tests or fuzzing where invariants matter.

If you are adding a new VOI method
----------------------------------

Use the method implementation guide in this section as the canonical shape
for the stable core methods. Add the method, its schema surface, its tests,
and any CLI or docs wiring in the same change set when the feature is public.
