#!/usr/bin/env python3
"""Validate H8-D/H8-E automated challenge evidence without granting authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voiage.sampling_harm_automated_challenge import (
    SamplingHarmAutomatedChallengeError,
    load_and_validate_sampling_harm_automated_challenge,
)


def main() -> int:
    """Validate command-line arguments and print a deterministic receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("synthesis", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        receipt = load_and_validate_sampling_harm_automated_challenge(
            args.synthesis, repository_root=args.repository_root
        )
    except (OSError, SamplingHarmAutomatedChallengeError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "valid", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
