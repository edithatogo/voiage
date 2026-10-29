"""Exercise Cargo's MSRV rejection and optional-backend controls offline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

EXPECTED_VERSION = "1.85.0"


def tool_version(executable: str, name: str) -> str:
    """Require the exact qualified MSRV tool, not a newer PATH fallback."""
    result = subprocess.run(  # noqa: S603 - explicitly selected tool
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout.strip()
    if not re.match(rf"^{name} {re.escape(EXPECTED_VERSION)}(?:\s|$)", output):
        raise ValueError(f"{name} must be {EXPECTED_VERSION}; observed {output!r}")
    return output


def verify(cargo: str, rustc: str) -> None:
    """Run isolated locked builds with incompatible and compatible backends."""
    print(tool_version(cargo, "cargo"), flush=True)
    print(tool_version(rustc, "rustc"), flush=True)
    with tempfile.TemporaryDirectory(prefix="voiage-msrv-witness-") as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "backend/src").mkdir(parents=True)
        (root / "Cargo.toml").write_text(
            '[package]\nname = "msrv-root-witness"\nversion = "0.1.0"\n'
            'edition = "2021"\nrust-version = "1.85"\n'
            '[features]\nbackend = ["dep:backend"]\n'
            '[dependencies]\nbackend = { path = "backend", optional = true }\n',
            encoding="utf-8",
        )
        backend = root / "backend/Cargo.toml"
        backend.write_text(
            '[package]\nname = "backend"\nversion = "0.1.0"\n'
            'edition = "2021"\nrust-version = "1.86"\n',
            encoding="utf-8",
        )
        (root / "src/lib.rs").write_text(
            '#[cfg(feature = "backend")] pub fn witness() -> u32 { backend::witness() }\n',
            encoding="utf-8",
        )
        (root / "backend/src/lib.rs").write_text(
            "pub fn witness() -> u32 { 1 }\n", encoding="utf-8"
        )
        env = os.environ.copy()
        env.update(
            RUSTC=rustc,
            RUSTUP_TOOLCHAIN=EXPECTED_VERSION,
            CARGO_TARGET_DIR=str(root / "target"),
        )

        def run(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(  # noqa: S603 - explicitly selected cargo
                [cargo, *args],
                cwd=root,
                check=False,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

        def require_success(*args: str) -> None:
            result = run(*args)
            if result.returncode:
                raise RuntimeError(f"cargo {args} failed: {result.stderr}")

        require_success("generate-lockfile", "--offline")
        lock = (root / "Cargo.lock").read_bytes()
        require_success("check", "--locked", "--offline", "--no-default-features")
        rejected = run("check", "--locked", "--offline", "--features", "backend")
        if rejected.returncode == 0 or "requires rustc 1.86" not in rejected.stderr:
            raise RuntimeError(
                f"incompatible backend was not rejected for MSRV: {rejected.stderr}"
            )
        backend.write_text(
            backend.read_text(encoding="utf-8").replace('"1.86"', '"1.85"'),
            encoding="utf-8",
        )
        require_success("check", "--locked", "--offline", "--features", "backend")
        if (root / "Cargo.lock").read_bytes() != lock:
            raise RuntimeError("locked fixture unexpectedly changed")
    print(
        "MSRV witness passed: feature-off and compatible controls pass; incompatible backend rejected"
    )


def main() -> int:
    """Select qualified binaries and execute the witness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo", default=shutil.which("cargo"))
    parser.add_argument("--rustc", default=shutil.which("rustc"))
    args = parser.parse_args()
    if not args.cargo or not args.rustc:
        parser.error("cargo and rustc are required")
    verify(args.cargo, args.rustc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
