# Python Release Submission Working Notes

Submission path:

1. `v*` tag triggers the release workflow.
2. PyPI/TestPyPI publish through trusted publishing.
3. In-repo conda-forge recipe update PR is produced.
4. External conda-forge feedstock merge remains manual.

Release contract reminders:

- The Python façade remains the stable release surface.
- The release story should stay aligned with the binding submission checklist.
- Live registry state is not verified in-repo; only the documented submission
  path is.
