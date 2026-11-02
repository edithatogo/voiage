"""Qualify EasyBuild's reproducible Git source archives against the pinned sdist."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import tomllib
from typing import Any
from urllib.parse import urlsplit

from qualify_source_vendor import digest, fetch, unpack


def recipe_values(path: Path) -> dict[str, Any]:
    """Read only literal source declarations without executing a recipe."""
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"crates", "checksums"}
    }


def qualify(work: Path, output: Path, eb: Path) -> None:
    """Fetch fixed revisions twice using the exact supported EasyBuild CLI."""
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir()
    inputs = json.loads(Path("packaging/current/source-vendor-inputs.json").read_text())
    entry = inputs["entries"]["polars-runtime-32"]
    archive = fetch(entry["source"], work / "primary")
    source = unpack(archive, work / "unpacked")
    lock = tomllib.loads((source / "Cargo.lock").read_text())
    expected_git = set()
    for package in lock["package"]:
        if package.get("source", "").startswith("git+"):
            url = urlsplit(package["source"][4:])
            expected_git.add(
                (
                    package["name"],
                    package["version"],
                    f"{url.scheme}://{url.netloc}{url.path}",
                    url.fragment,
                )
            )
    recipes = [recipe_values(Path(path)) for path in entry["recipes"]]
    crates = [tuple(crate) for crate in recipes[0]["crates"] if len(crate) == 4]
    if set(crates) != expected_git or crates != [
        tuple(c) for c in recipes[1]["crates"] if len(c) == 4
    ]:
        raise ValueError("Recipe Git identities differ from authenticated Cargo.lock")
    receipts = []
    for index in range(2):
        root = work / f"fetch-{index}"
        root.mkdir()
        probe = root / "Voiage-git-source-probe-1.0.eb"
        probe.write_text(
            'easyblock = "Cargo"\nname = "Voiage-git-source-probe"\nversion = "1.0"\n'
            'homepage = "https://github.com/edithatogo/voiage"\n'
            'description = "Fixed revision source fetch qualification"\n'
            "toolchain = SYSTEM\nsources = []\ncrates = " + repr(crates) + "\n"
        )
        command = [
            str(eb),
            str(probe),
            "--fetch",
            "--inject-checksums-to-json",
            f"--sourcepath={root / 'sources'}",
            f"--buildpath={root / 'build'}",
            f"--prefix={root / 'prefix'}",
            "--modules-tool=EnvironmentModules",
            "--allow-use-as-root-and-accept-consequences",
        ]
        with (output / f"fetch-{index}.log").open("wb") as stream:
            subprocess.run(  # noqa: S603 - fixed supported CLI and qualified tool
                command,
                env=dict(os.environ, CARGO_NET_RETRY="0"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        checksums = json.loads((root / "checksums.json").read_text())
        actual = {}
        for name, version, _repo, revision in crates:
            filename = f"{name}-{version}-{revision[:8]}.tar.xz"
            matches = list((root / "sources").rglob(filename))
            if len(matches) != 1 or matches[0].is_symlink():
                raise ValueError("Missing or shadowed resolved Git archive")
            actual[filename] = digest(matches[0])
            if checksums[filename] != actual[filename]:
                raise ValueError(
                    "EasyBuild checksum differs from resolved archive bytes"
                )
        receipts.append(actual)
        (output / f"checksums-{index}.json").write_text(
            json.dumps(actual, indent=2) + "\n"
        )
    if receipts[0] != receipts[1]:
        raise ValueError("Fresh EasyBuild Git archives are not reproducible")
    result = {
        "source_sha256": digest(archive),
        "git_sources": crates,
        "archives": receipts[0],
        "fresh_fetches": 2,
        "reproducible": True,
    }
    (output / "git-archives.json").write_text(json.dumps(result, indent=2) + "\n")
    for recipe in recipes:
        pinned = {
            name: sha for item in recipe["checksums"] for name, sha in item.items()
        }
        if any(pinned.get(name) != sha for name, sha in receipts[0].items()):
            raise ValueError(
                "Qualified Git archive digest is missing from a provider recipe"
            )


def main() -> None:
    """Run the bounded source-only probe, never a native build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eb", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.work.resolve(), args.output.resolve(), args.eb.resolve(strict=True))


if __name__ == "__main__":
    main()
