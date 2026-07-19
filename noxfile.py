"""Nox sessions for local development.

The repository still uses tox as the CI source of truth, but Nox provides a
uv-backed local runner with the same core sessions.
"""

import nox

PYPROJECT = nox.project.load_toml("pyproject.toml")
PYTHON_VERSIONS = nox.project.python_versions(PYPROJECT)

nox.options.default_venv_backend = "uv"


def _sync_project(session: nox.Session) -> None:
    session.run(
        "uv",
        "sync",
        "--locked",
        "--all-extras",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the main pytest suite across the supported Python versions."""
    _sync_project(session)
    session.run(
        "pytest",
        "tests/",
        "-m",
        "not integration and not e2e and not benchmark",
        "--cov=voiage",
        "--cov-fail-under=0",
        "--cov-report=xml",
        "--cov-report=term-missing",
        "--numprocesses=auto",
        "-v",
        *session.posargs,
    )


@nox.session
def lint(session: nox.Session) -> None:
    """Run Ruff and Bandit checks."""
    _sync_project(session)
    session.run("ruff", "check", "voiage", "tests", "--fix", "--exit-non-zero-on-fix")
    session.run("ruff", "format", "voiage", "tests", "--check")
    session.run(
        "bandit", "-r", "voiage", "-s", "B101,B110,B405,B314", "-c", "pyproject.toml"
    )


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run the static type checker."""
    _sync_project(session)
    session.run(
        "ty",
        "check",
        "voiage",
        "--python-version=3.14",
        "--ignore",
        "invalid-argument-type",
        "--ignore",
        "invalid-assignment",
        "--ignore",
        "unresolved-attribute",
        "--ignore",
        "unresolved-import",
        "--ignore",
        "no-matching-overload",
        "--ignore",
        "unsupported-operator",
        "--ignore",
        "call-non-callable",
        "--ignore",
        "not-iterable",
        "--ignore",
        "not-subscriptable",
        "--ignore",
        "invalid-return-type",
        "--ignore",
        "invalid-method-override",
        "--ignore",
        "unused-type-ignore-comment",
        "--ignore",
        "redundant-cast",
        *session.posargs,
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Check and build the Astro/Starlight documentation."""
    site = "docs/astro-site"
    session.run("pnpm", "install", "--frozen-lockfile", external=True, cwd=site)
    session.run("pnpm", "run", "check", external=True, cwd=site)
    session.run("pnpm", "run", "build", *session.posargs, external=True, cwd=site)


@nox.session
def coverage_report(session: nox.Session) -> None:
    """Generate the HTML and XML coverage outputs."""
    _sync_project(session)
    session.run(
        "pytest",
        "tests/",
        "--cov=voiage",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-report=html",
        "--cov-fail-under=90",
        *session.posargs,
    )


@nox.session
def frontier_contract(session: nox.Session) -> None:
    """Validate the frontier contract registry and fixtures."""
    _sync_project(session)
    session.run("python", "scripts/validate_frontier_contract.py", *session.posargs)
