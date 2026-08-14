"""PEP 517 wrapper that makes Git-less source archives provenance-complete."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
from typing import TYPE_CHECKING

import maturin

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import Any

ROOT = Path(__file__).resolve().parent
_PROVENANCE_RELATIVE = PurePosixPath("rust/crates/voiage-python/source-provenance.txt")
PROVENANCE = ROOT.joinpath(*_PROVENANCE_RELATIVE.parts)
_OID = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_VARIABLES = (
    "VOIAGE_SOURCE_REVISION",
    "VOIAGE_SOURCE_TREE_GIT_OID",
    "VOIAGE_SOURCE_CLEAN",
)
_MAX_PROVENANCE_BYTES = 256
GIT = shutil.which("git")


def _git(*arguments: str) -> str:
    """Run one fixed Git query in the source root."""
    if GIT is None:
        raise FileNotFoundError("git is unavailable")
    return subprocess.run(  # noqa: S603 - executable and arguments are fixed
        [GIT, *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _environment_identity() -> tuple[str, str] | None:
    """Return a complete clean identity from the environment, if supplied."""
    supplied = tuple(os.environ.get(name) for name in _SOURCE_VARIABLES)
    if not any(value is not None for value in supplied):
        return None

    revision, tree, clean = supplied
    if (
        revision is None
        or tree is None
        or clean != "true"
        or _OID.fullmatch(revision) is None
        or _OID.fullmatch(tree) is None
    ):
        raise RuntimeError(
            "source provenance requires all of VOIAGE_SOURCE_REVISION, "
            "VOIAGE_SOURCE_TREE_GIT_OID, and VOIAGE_SOURCE_CLEAN; the OIDs "
            "must be 40 lowercase hexadecimal characters and clean must be true"
        )
    return revision, tree


def _git_identity() -> tuple[str, str, bool] | None:
    """Return the checkout identity and whether relevant source is dirty."""
    try:
        top_level = Path(_git("rev-parse", "--show-toplevel")).resolve()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    if top_level != ROOT.resolve():
        return None
    try:
        revision = _git("rev-parse", "--verify", "HEAD^{commit}")
        tree = _git("rev-parse", "--verify", "HEAD^{tree}")
        status = _git("status", "--porcelain=v1", "--untracked-files=all")
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("failed to inspect the source Git checkout") from error

    if _OID.fullmatch(revision) is None or _OID.fullmatch(tree) is None:
        raise RuntimeError("Git returned an invalid source identity")
    relevant = [
        line
        for line in status.splitlines()
        if line[3:] != _PROVENANCE_RELATIVE.as_posix()
    ]
    return revision, tree, bool(relevant)


def _source_identity() -> tuple[str, str]:
    """Resolve and verify an immutable clean identity for an sdist."""
    supplied = _environment_identity()
    git = _git_identity()
    if git is not None:
        revision, tree, dirty = git
        if dirty:
            raise RuntimeError(
                "refusing to create a provenance-complete source distribution "
                "from a dirty source tree"
            )
        if supplied is not None and supplied != (revision, tree):
            raise RuntimeError(
                "supplied source identity does not match the clean Git checkout"
            )
        return revision, tree
    if supplied is not None:
        return supplied
    raise RuntimeError(
        "source-distribution provenance requires either a clean Git checkout "
        "or a complete clean VOIAGE source identity"
    )


def _provenance_bytes(identity: tuple[str, str]) -> bytes:
    """Serialize one canonical immutable provenance record."""
    revision, tree = identity
    return f"revision={revision}\ntree={tree}\nclean=true\n".encode()


def _parse_provenance(contents: bytes) -> tuple[str, str]:
    """Parse an exact canonical immutable provenance record."""
    if len(contents) > _MAX_PROVENANCE_BYTES:
        raise RuntimeError("embedded source provenance is unexpectedly large")
    try:
        text = contents.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("embedded source provenance must be ASCII") from error
    match = re.fullmatch(
        r"revision=([0-9a-f]{40})\ntree=([0-9a-f]{40})\nclean=true\n",
        text,
    )
    if match is None:
        raise RuntimeError(
            "embedded source provenance must be the canonical three-line "
            "record with lowercase Git OIDs and clean=true"
        )
    return match.group(1), match.group(2)


def _embedded_identity() -> tuple[str, str]:
    """Read and validate the identity carried by a Git-less source archive."""
    if PROVENANCE.is_symlink() or not PROVENANCE.is_file():
        raise RuntimeError(
            "Git-less wheel builds require the regular embedded source "
            f"provenance file {_PROVENANCE_RELATIVE}"
        )
    return _parse_provenance(PROVENANCE.read_bytes())


@contextmanager
def _gitless_wheel_identity() -> Iterator[None]:
    """Bind a Git-less wheel build to the exact identity in its sdist."""
    supplied = _environment_identity()
    git = _git_identity()
    if git is not None:
        revision, tree, dirty = git
        if supplied is not None and (dirty or supplied != (revision, tree)):
            raise RuntimeError(
                "supplied clean source identity does not match the Git checkout"
            )
        yield
        return

    embedded = _embedded_identity()
    if supplied is not None and supplied != embedded:
        raise RuntimeError(
            "supplied source identity does not match embedded sdist provenance"
        )

    previous = {name: os.environ.get(name) for name in _SOURCE_VARIABLES}
    revision, tree = embedded
    os.environ.update(
        {
            "VOIAGE_SOURCE_REVISION": revision,
            "VOIAGE_SOURCE_TREE_GIT_OID": tree,
            "VOIAGE_SOURCE_CLEAN": "true",
        }
    )
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _verify_sdist(
    sdist_directory: str,
    filename: str,
    expected: bytes,
) -> None:
    """Verify the returned sdist contains one regular, exact identity record."""
    directory = Path(sdist_directory).resolve()
    archive = (directory / filename).resolve()
    if archive.parent != directory or not archive.is_file():
        raise RuntimeError("Maturin returned an unsafe or missing sdist path")

    with tarfile.open(archive, mode="r:*") as package:
        matches = [
            member
            for member in package.getmembers()
            if len(PurePosixPath(member.name).parts)
            == len(_PROVENANCE_RELATIVE.parts) + 1
            and PurePosixPath(member.name).parts[0] not in {"", ".", "..", "/"}
            and PurePosixPath(member.name).parts[1:] == _PROVENANCE_RELATIVE.parts
        ]
        if len(matches) != 1 or not matches[0].isfile():
            raise RuntimeError(
                "source distribution must contain exactly one regular embedded "
                "source provenance file"
            )
        member = matches[0]
        if member.size > _MAX_PROVENANCE_BYTES:
            raise RuntimeError("sdist source provenance is unexpectedly large")
        extracted = package.extractfile(member)
        if extracted is None or extracted.read(_MAX_PROVENANCE_BYTES + 1) != expected:
            raise RuntimeError(
                "source distribution did not preserve its immutable source identity"
            )


def build_sdist(
    sdist_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    """Build an sdist containing the immutable identity used downstream."""
    identity = _source_identity()
    expected = _provenance_bytes(identity)
    if PROVENANCE.is_symlink():
        raise RuntimeError("refusing to replace a symlinked source provenance file")
    previous = PROVENANCE.read_bytes() if PROVENANCE.exists() else None
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_bytes(expected)
    filename: str | None = None
    try:
        filename = maturin.build_sdist(sdist_directory, config_settings)
        _verify_sdist(sdist_directory, filename, expected)
    except Exception:
        if filename is not None:
            candidate = (Path(sdist_directory).resolve() / filename).resolve()
            if candidate.parent == Path(sdist_directory).resolve():
                candidate.unlink(missing_ok=True)
        raise
    else:
        return filename
    finally:
        if previous is None:
            PROVENANCE.unlink(missing_ok=True)
        else:
            PROVENANCE.write_bytes(previous)


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build a wheel, requiring exact embedded identity when Git is absent."""
    with _gitless_wheel_identity():
        return maturin.build_wheel(
            wheel_directory,
            config_settings,
            metadata_directory,
        )


def build_editable(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Delegate editable builds to Maturin."""
    return maturin.build_editable(
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def get_requires_for_build_sdist(
    config_settings: Mapping[str, Any] | None = None,
) -> Sequence[str]:
    """Return Maturin's source-build requirements."""
    return maturin.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_wheel(
    config_settings: Mapping[str, Any] | None = None,
) -> Sequence[str]:
    """Return Maturin's wheel-build requirements."""
    return maturin.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(
    config_settings: Mapping[str, Any] | None = None,
) -> Sequence[str]:
    """Return Maturin's editable-build requirements."""
    return maturin.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    """Delegate wheel metadata generation to Maturin."""
    return maturin.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings,
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    """Delegate editable metadata generation to Maturin."""
    return maturin.prepare_metadata_for_build_editable(
        metadata_directory,
        config_settings,
    )
