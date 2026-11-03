"""Supervise a native driver and privileged stage monitors from one root owner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.observe_native_stage import StageObserver


def main() -> int:
    """Keep supervision privileged while all source/build commands run as runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.geteuid() != 0 or args.uid <= 0 or args.gid <= 0:
        parser.error("Root supervisor and non-root driver UID/GID are required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("An owned native driver command is required")
    unprivileged = [
        "/usr/bin/setpriv",
        f"--reuid={args.uid}",
        f"--regid={args.gid}",
        "--init-groups",
        "--",
        *command,
    ]
    observer = StageObserver(args.evidence, args.workspace, timeout=args.timeout)
    return observer.run(unprivileged, cwd=args.cwd)


if __name__ == "__main__":
    raise SystemExit(main())
