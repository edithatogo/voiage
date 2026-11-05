"""Record actual checked EasyBuild sources through its supported post-fetch hook."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tomllib
from typing import Any


def resolved_file(
    path: Path, roots: list[Path], expected: str | None = None
) -> dict[str, Any]:
    """Bind regular source bytes inside explicit roots, rejecting shadow paths."""
    path = path.absolute()
    if any(parent.is_symlink() for parent in [path, *path.parents]):
        raise ValueError(f"Symlinked source path: {path}")
    real = path.resolve(strict=True)
    if not real.is_file() or not any(
        real.is_relative_to(root.resolve(strict=True)) for root in roots
    ):
        raise ValueError(f"Resolved source escapes allowed roots: {path}")
    with real.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Source digest mismatch: {path}")
    return {"path": str(real), "size": real.stat().st_size, "sha256": digest}


def post_fetch_hook(app: Any) -> None:
    """Run after EasyBuild's fetch/checksum methods, including extensions."""
    roots = [Path(item) for item in os.environ["VOIAGE_SOURCE_ROOTS"].split(os.pathsep)]
    entries = list(app.src) + list(app.patches)
    if hasattr(app, "config_guess"):
        if not app.config_guess:
            raise ValueError("Unresolved auxiliary config.guess source")
        # ConfigureMake.obtain_config_guess already checks its pinned SHA-256.
        entries.append({"path": app.config_guess})
    for extension in app.exts:
        if extension.get("src"):
            entries.append({"path": extension["src"]})
        entries.extend(extension.get("patches", []))
    repository = Path(__file__).resolve().parents[1]
    inputs = json.loads(
        (repository / "packaging/current/source-vendor-inputs.json").read_text()
    )
    patches = {
        repository / entry["lock_patch"]["path"]: entry["lock_patch"]["sha256"]
        for entry in inputs["entries"].values()
        if "lock_patch" in entry
    }
    records = []
    for entry in entries:
        path = Path(entry["path"])
        if path in patches:
            records.append(resolved_file(path, [repository], patches[path]))
        else:
            records.append(resolved_file(path, roots))
    output = Path(os.environ["VOIAGE_SOURCE_RECEIPTS"])
    output.mkdir(parents=True, exist_ok=True)
    name = f"{app.name}-{app.version}-{app.toolchain.name}-{app.toolchain.version}"
    if Path(name).name != name:
        raise ValueError("Unsafe source receipt identity")
    (output / f"{name}.json").write_text(
        json.dumps({"module": name, "sources": records}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def post_configure_hook(app: Any) -> None:
    """Freeze effective source-compiler settings before native compilation."""
    if app.name not in {"Rust", "Rust-nightly"}:
        return
    output = Path(os.environ["VOIAGE_COMPILER_RECEIPTS"])
    path = Path(app.cfg["start_dir"]) / "bootstrap.toml"
    with path.open("rb") as stream:
        effective = tomllib.load(stream)
    if (
        effective["llvm"]["download-ci-llvm"] is not False
        or effective["llvm"]["link-jobs"] != 1
        or effective["rust"]["download-rustc"] is not False
        or effective["build"]["vendor"] is not True
        or effective["install"]["prefix"] != app.installdir
    ):
        raise ValueError("Unqualified effective source compiler configuration")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        path, output / f"{app.name}-{app.version}-{app.toolchain.version}.toml"
    )
