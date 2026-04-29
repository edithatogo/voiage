"""Allow running voiage CLI as a module: python -m voiage."""

from voiage.cli import app

if __name__ == "__main__":  # pragma: no cover
    app()
