"""Record actual checked EasyBuild sources through its supported post-fetch hook."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
    records = [resolved_file(Path(entry["path"]), roots) for entry in entries]
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
