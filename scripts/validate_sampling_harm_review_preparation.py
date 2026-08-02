#!/usr/bin/env python3
"""Validate the sampling-harm H8-C frozen review preparation."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--expected-candidate-commit")
    args = parser.parse_args()
    try:
        load_and_validate_sampling_harm_review_preparation(
            args.envelope,
            repository_root=args.repository_root,
            expected_candidate_commit=args.expected_candidate_commit,
        )
    except (OSError, SamplingHarmReviewPreparationError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
