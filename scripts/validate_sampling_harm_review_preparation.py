#!/usr/bin/env python3
"""Validate the sampling-harm H8-C frozen review preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voiage.sampling_harm_review_preparation import (
    SamplingHarmReviewPreparationError,
    load_and_validate_sampling_harm_review_preparation,
)


def main() -> int:
    """Validate command-line arguments and return a process status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("envelope", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-candidate-commit", required=True)
    parser.add_argument("--expected-package-commit", required=True)
    args = parser.parse_args()
    receipt: dict[str, str] = {}
    try:
        receipt = load_and_validate_sampling_harm_review_preparation(
            args.envelope,
            repository_root=args.repository_root,
            expected_candidate_commit=args.expected_candidate_commit,
            expected_package_commit=args.expected_package_commit,
        )
    except (OSError, SamplingHarmReviewPreparationError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "valid", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
