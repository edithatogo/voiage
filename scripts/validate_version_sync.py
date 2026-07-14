#!/usr/bin/env python3
"""Validate canonical version synchronization across package manifests."""

from pathlib import Path

from voiage.versioning import main

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    raise SystemExit(main(["--repo-root", str(repo_root)]))
