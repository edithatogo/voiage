"""Enable loopback only in a fresh build namespace, then drop all root identities."""

from __future__ import annotations

import argparse
import os
import socket
import struct


def enter_namespace(
    parent_namespace: str, uid: int, gid: int, command: list[str]
) -> None:
    """Reject the host network before configuring the isolated local interface."""
    if os.geteuid() != 0 or uid <= 0 or gid <= 0 or not command:
        raise ValueError("Root setup, non-root identities and a command are required")
    if os.readlink("/proc/self/ns/net") == parent_namespace:
        raise ValueError("Refusing to configure the parent network namespace")
    if sorted(name for _index, name in socket.if_nameindex()) != ["lo"]:
        raise ValueError("Build namespace must contain only loopback")

    import fcntl

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
        request = struct.pack("16sH22x", b"lo", 0)
        response = fcntl.ioctl(control.fileno(), 0x8913, request)  # SIOCGIFFLAGS
        flags = struct.unpack_from("H", response, 16)[0]
        fcntl.ioctl(
            control.fileno(), 0x8914, struct.pack("16sH22x", b"lo", flags | 1)
        )  # SIOCSIFFLAGS, IFF_UP
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    os.execvpe(command[0], command, os.environ)  # noqa: S606 - trusted stage argv


def main() -> None:
    """Enter the already isolated namespace without a shell or external routes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-namespace", required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    enter_namespace(args.parent_namespace, args.uid, args.gid, command)


if __name__ == "__main__":
    main()
