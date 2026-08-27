# Support

`voiage` is maintained as a contract-first library, so the best support path is
to start with the shipped documentation and the issue templates.

## Where to start

1. Read the relevant docs in `README.md`, `docs/`, and the binding-specific
   walkthroughs.
2. Check the existing issues and pull requests for the same problem.
3. Use the matching issue template:
   - bug reports for reproducible failures
   - feature requests for new VOI methods, outputs, or workflows
   - support questions for usage help that does not fit a bug report

## What to include

Good reports are specific and reproducible. Include:

- the version of `voiage`
- the language and runtime you are using
- the smallest possible input that reproduces the issue
- the exact command or code you ran
- the observed output and the output you expected

## Triage expectations

Support requests are triaged against the current roadmap and the implemented
contract surface. Issues that are missing a reproducible example or that ask
for broad roadmap planning will usually be redirected to the feature template
or the existing documentation.

For a complete, reproducible report affecting a supported release, the
maintainer aims to acknowledge and triage the issue within 14 calendar days.
Critical defects receive priority investigation. Fix timing depends on
severity, reproducibility, maintainer capacity, compatibility risk, and the
release process; no resolution deadline or continuous-availability guarantee
is offered. The maintenance scope and succession policy are documented in
`docs/release/pyopensci-readiness.md`.

## Security issues

Do not report security vulnerabilities in public issues. Use `SECURITY.md` for
private reporting instructions.
