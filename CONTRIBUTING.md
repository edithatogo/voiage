# Contributing to `voiage`

We welcome contributions from the community, whether from humans or AI agents. To ensure a smooth and effective development process, please adhere to the following guidelines.

## Getting Started

### Prerequisites

*   Python 3.8+
*   [Poetry](https://python-poetry.org/docs/#installation) for dependency management (recommended)
*   [pre-commit](https://pre-commit.com/#installation) for automated checks

### Setting Up the Development Environment

1.  **Fork and Clone:**
    *   Fork the repository on GitHub.
    *   Clone your fork locally:
        ```bash
        git clone https://github.com/YOUR_USERNAME/voiage.git
        cd voiage
        ```

2.  **Install Dependencies:**
    *   If using Poetry:
        ```bash
        poetry install
        ```
    *   If using pip:
        ```bash
        pip install -e .[dev]
        ```

3.  **Install Pre-commit Hooks:**
    *   This will install the hooks defined in `.pre-commit-config.yaml`, which automatically run checks before each commit.
        ```bash
        pre-commit install
        ```

## Development Workflow

1.  **Create a Branch:**
    *   Create a new branch for your feature or bugfix:
        ```bash
        git checkout -b feature/my-new-feature
        ```

2.  **Write Code:**
    *   Make your changes, following the code style guidelines below.
    *   Ensure new code is well-tested and fully type-hinted.

3.  **Verify Changes:**
    *   Run the full suite of tests, linting, and type checks using `tox`. This is the same check that runs in our CI pipeline.
        ```bash
        tox
        ```
    *   Fix any errors reported by `tox` before proceeding.

4.  **Commit Changes:**
    *   Stage your changes (`git add .`).
    *   Commit them with a descriptive message. The pre-commit hooks will run automatically.
        ```bash
        git commit -m "feat: Add my new feature"
        ```

5.  **Push and Open a Pull Request:**
    *   Push your branch to your fork and open a pull request against the `main` branch of the original repository.

## Code Style and Conventions

### Formatting and Linting

*   We use **Ruff** for all formatting and linting to ensure a consistent code style.
*   The configuration is defined in `pyproject.toml`.
*   The pre-commit hooks will automatically format your code. You can also run it manually:
    ```bash
    ruff format .
    ruff check --fix .
    ```

### Type Hinting

*   All new functions and methods must include type hints.
*   We use **MyPy** for static type analysis. The pre-commit hook will run `mypy` on your changes.

### Testing

*   We use **pytest** for testing.
*   All new features must be accompanied by tests.
*   All bug fixes should include a regression test.
*   Run tests using `tox` or directly with `pytest`:
    ```bash
    pytest
    ```

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps in automating changelog generation and makes the commit history more readable.

Each commit message should be in the format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

*   **Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `build`, `ci`.
*   **Example:**
    ```
    feat(analysis): Add EVSI regression method
    ```
    ```
    fix: Correct population scaling in EVPI calculation
    ```

## Documentation

*   Public functions and classes should have docstrings following the **NumPy docstring convention**.
*   The project's documentation is in the `docs/` directory and is built with Sphinx.
