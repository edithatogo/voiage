#!/usr/bin/env python3
"""Validate canonical version synchronization across package manifests."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from voiage.versioning import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--repo-root", str(REPO_ROOT)]))
