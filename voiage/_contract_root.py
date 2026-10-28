"""Locate source contracts in a checkout or the private installed package."""

from pathlib import Path


def contract_root() -> Path:
    """Return the directory containing the canonical ``specs`` tree."""
    package = Path(__file__).resolve().parent
    private = package / "_contracts"
    if (private / "specs").is_dir():
        return private
    checkout = package.parent
    if (checkout / "pyproject.toml").is_file() and (checkout / "specs").is_dir():
        return checkout
    return private
