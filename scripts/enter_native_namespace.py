"""Enable loopback only in a fresh build namespace, then drop all root identities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import struct
lazy import pwd

# These are build inputs, paths and compiler controls. Authentication is never
# forwarded from the privileged monitor. Module names carry only tool locations.
BUILD_ENVIRONMENT = frozenset(
    [
        "RUNNER_TEMP",
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SOURCE_DATE_EPOCH",
        "CARGO",
        "CARGO_HOME",
        "CARGO_NET_OFFLINE",
        "CARGO_NET_RETRY",
        "CARGO_BUILD_JOBS",
        "CARGO_TARGET_DIR",
        "CARGO_HTTP_CAINFO",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "RUSTC",
        "RUSTDOC",
        "RUSTFLAGS",
        "RUSTDOCFLAGS",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "CC",
        "CXX",
        "FC",
        "F77",
        "F90",
        "AR",
        "AS",
        "LD",
        "NM",
        "OBJCOPY",
        "OBJDUMP",
        "RANLIB",
        "STRIP",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "FCFLAGS",
        "FFLAGS",
        "LDFLAGS",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "SPACK_ROOT",
        "SPACK_PYTHON",
        "SPACK_ENV",
        "SPACK_USER_CONFIG_PATH",
        "SPACK_USER_CACHE_PATH",
        "SPACK_DISABLE_LOCAL_CONFIG",
        "VOIAGE_COMPILER_RECEIPTS",
        "VOIAGE_SOURCE_ROOTS",
        "VOIAGE_SOURCE_RECEIPTS",
        "VOIAGE_QUALIFIED_SOURCE_ARCHIVE",
        "VOIAGE_QUALIFIED_SOURCE_SHA256",
        "VOIAGE_SOURCE_REVISION",
        "VOIAGE_SOURCE_TREE_GIT_OID",
        "VOIAGE_SOURCE_CLEAN",
        "VOIAGE_NATIVE_USER_NAMESPACE",
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "UV_OFFLINE",
        "UV_NO_PROGRESS",
        "MAKEFLAGS",
        "FPATH",
        "MODULEPATH",
        "MODULESHOME",
        "LOADEDMODULES",
        "_LMFILES_",
        "LMOD_CMD",
        "LMOD_DIR",
        "LMOD_PKG",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ]
)


def preserved_environment_names() -> str:
    """Name declared sudo inputs, including settings created later in a driver."""
    names = (
        BUILD_ENVIRONMENT
        | {"HOME", "USER", "LOGNAME"}
        | {
            key
            for key in os.environ
            if key.startswith(("EBROOT", "EBVERSION", "EBDEVEL"))
        }
    )
    return ",".join(sorted(names))


def build_environment(uid: int) -> dict[str, str]:
    """Select the real runner profile and declared inputs without root credentials."""
    identity = pwd.getpwuid(uid)
    selected = {
        key: value
        for key, value in os.environ.items()
        if key in BUILD_ENVIRONMENT
        or key.startswith(("EBROOT", "EBVERSION", "EBDEVEL"))
    }
    selected.update(
        HOME=identity.pw_dir, USER=identity.pw_name, LOGNAME=identity.pw_name
    )
    return selected


def enter_namespace(
    parent_namespace: str,
    uid: int,
    gid: int,
    command: list[str],
    *,
    require_identity_maps: bool = False,
) -> None:
    """Reject the host network before configuring the isolated local interface."""
    if os.geteuid() != 0 or uid <= 0 or gid <= 0 or not command:
        raise ValueError("Root setup, non-root identities and a command are required")
    if os.readlink("/proc/self/ns/net") == parent_namespace:
        raise ValueError("Refusing to configure the parent network namespace")
    if sorted(name for _index, name in socket.if_nameindex()) != ["lo"]:
        raise ValueError("Build namespace must contain only loopback")

    if require_identity_maps:
        for name, identity in (("uid_map", uid), ("gid_map", gid)):
            observed = {
                tuple(map(int, line.split()))
                for line in Path(f"/proc/self/{name}").read_text().splitlines()
            }
            if observed != {(0, 0, 1), (identity, identity, 1)}:
                raise ValueError(
                    "Namespace identity mappings differ from exact runner IDs"
                )
        if Path("/proc/self/setgroups").read_text().strip() != "allow":
            raise ValueError(
                "Namespace must permit explicit supplementary-group clearing"
            )

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
    os.execvpe(command[0], command, build_environment(uid))  # noqa: S606 - trusted stage argv


def main() -> None:
    """Enter the already isolated namespace without a shell or external routes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-namespace", required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("--require-identity-maps", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    enter_namespace(
        args.parent_namespace,
        args.uid,
        args.gid,
        command,
        require_identity_maps=args.require_identity_maps,
    )


if __name__ == "__main__":
    main()
