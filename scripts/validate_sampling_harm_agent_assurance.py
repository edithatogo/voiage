#!/usr/bin/env python3
"""Check the agent-only assurance manifest without claiming independent review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voiage.sampling_harm_agent_assurance import (
    SamplingHarmAgentAssuranceError,
    load_and_validate_sampling_harm_agent_assurance,
)


def main() -> int:
    """Print a non-authorizing receipt using the current observation time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        receipt = load_and_validate_sampling_harm_agent_assurance(
            repository_root=args.repository_root
        )
    except (OSError, SamplingHarmAgentAssuranceError) as error:
        return parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
