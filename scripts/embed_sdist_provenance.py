"""Embed immutable Git identity for downstream Git-less sdist builds."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "rust/crates/voiage-python/source-provenance.txt"
GIT = shutil.which("git")
if GIT is None:
    raise RuntimeError("git is required to embed sdist provenance")


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - arguments are fixed by this module
        [GIT, *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    """Write a clean, immutable Git identity into the sdist-only manifest."""
    tracked_dirty = (
        subprocess.run(  # noqa: S603 - fixed Git invocation
            [GIT, "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False
        ).returncode
        != 0
        or subprocess.run(  # noqa: S603 - fixed Git invocation
            [GIT, "diff", "--cached", "--quiet", "HEAD", "--"],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    )
    untracked = [
        path
        for path in _git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
        if path
        and "/.astro/" not in f"/{path}"
        and not path.startswith(("rust/target/", "target/"))
    ]
    if tracked_dirty or untracked:
        raise SystemExit("refusing to embed provenance from a dirty source tree")
    revision = _git("rev-parse", "--verify", "HEAD^{commit}")
    tree = _git("rev-parse", "--verify", "HEAD^{tree}")
    OUTPUT.write_text(
        f"revision={revision}\ntree={tree}\nclean=true\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
