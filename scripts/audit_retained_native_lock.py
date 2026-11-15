"""Prepare and verify an exact-source mirror for the frozen native lock audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKET = Path(
    "conductor/tracks/scientific_ecosystem_alignment_20261029/installed-adapters"
)
REVISION = "83d556ba4995fd5f5238f4953e9a8af34976f464"
OLD_DEPENDENCY = b'innovate-rs = { path = "../inputs/innovate-src/bindings/rust" }'
NEW_DEPENDENCY = (
    b'innovate-rs = { git = "https://github.com/edithatogo/innovate.git", '
    b'rev = "83d556ba4995fd5f5238f4953e9a8af34976f464" }'
)
GIT_SOURCE = f"git+https://github.com/edithatogo/innovate.git?rev={REVISION}#{REVISION}"
FROZEN = {
    "Cargo.toml": "e74a1d4a73c0148fe7e08baac91405cf27c7342f93d2f7dbf9355639c63a2eff",
    "Cargo.lock": "767684c75c3eac7f32b52460ee116926141035af9d7318ce2b28701168c33868",
    "src/main.rs": "7c9f046bcf67da15a0e1c73e18db4eca06edbae7a58c7a733dd282e7a7a42c33",
}


def frozen_files(root: Path) -> dict[str, bytes]:
    """Authenticate the original packet before using its dependency graph."""
    packet = root / PACKET
    files = {}
    for name, expected in FROZEN.items():
        path = packet / "native" / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe frozen file: {name}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Frozen file changed: {name}")
        files[name] = data
    identity = json.loads((packet / "package-identities.json").read_text())
    if identity["innovate"]["source"] != REVISION:
        raise ValueError("The retained Innovate source identity changed")
    if files["Cargo.toml"].count(OLD_DEPENDENCY) != 1:
        raise ValueError("Unexpected retained Innovate dependency")
    return files


def prepare(root: Path, destination: Path) -> None:
    """Create a new private replica with the same immutable upstream source."""
    files = frozen_files(root)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "src").mkdir()
    (destination / "Cargo.toml").write_bytes(
        files["Cargo.toml"].replace(OLD_DEPENDENCY, NEW_DEPENDENCY)
    )
    (destination / "Cargo.lock").write_bytes(files["Cargo.lock"])
    (destination / "src/main.rs").write_bytes(files["src/main.rs"])


def verify(root: Path, destination: Path) -> dict[str, object]:
    """Reject any lock change other than the exact pinned Git source field."""
    files = frozen_files(root)
    for name in FROZEN:
        if (destination / name).is_symlink() or not (destination / name).is_file():
            raise ValueError(f"Missing or unsafe replica file: {name}")
    expected_manifest = files["Cargo.toml"].replace(OLD_DEPENDENCY, NEW_DEPENDENCY)
    if (destination / "Cargo.toml").read_bytes() != expected_manifest:
        raise ValueError("Replica manifest changed")
    if (destination / "src/main.rs").read_bytes() != files["src/main.rs"]:
        raise ValueError("Replica native source changed")
    original = tomllib.loads(files["Cargo.lock"].decode())
    candidate = tomllib.loads((destination / "Cargo.lock").read_text())
    old_entries = [p for p in original["package"] if p["name"] == "innovate-rs"]
    new_entries = [p for p in candidate["package"] if p["name"] == "innovate-rs"]
    if len(old_entries) != 1 or len(new_entries) != 1:
        raise ValueError("Expected exactly one Innovate package")
    if "source" in old_entries[0] or new_entries[0].pop("source", None) != GIT_SOURCE:
        raise ValueError("Replica Innovate source identity differs")
    if candidate != original:
        raise ValueError("Replica dependency graph differs from the frozen lock")
    return {
        "passed": True,
        "frozen_lock_sha256": FROZEN["Cargo.lock"],
        "innovate_revision": REVISION,
        "replica_lock_sha256": hashlib.sha256(
            (destination / "Cargo.lock").read_bytes()
        ).hexdigest(),
        "scope": "dependency graph equivalence, not native execution",
    }


def main() -> int:
    """Prepare or verify a private replica; advisory scanning is a separate step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "prepare":
            prepare(ROOT, args.destination)
            result: dict[str, object] = {"prepared": True}
        else:
            result = verify(ROOT, args.destination)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
