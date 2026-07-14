#!/usr/bin/env python3
"""Refresh the per-language binding registry-audit snapshot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING
from urllib import error as url_error
from urllib import request

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "docs" / "release" / "registry_audit_snapshot.json"
DEFAULT_TIMEOUT_SECONDS = 8


@dataclass(frozen=True)
class Channel:
    """Descriptor for one language binding registry channel."""

    key: str
    package: str
    registry: str
    registry_url: str
    check_url: str
    notes: str
    evaluator: Callable[[int | None, bytes | None, str | None], str]
    confidence: str


def _fetch_json_bytes(
    url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> tuple[int | None, bytes | None, str | None]:
    """Fetch raw bytes from ``url`` and return status code, payload, and optional error."""
    try:
        request_obj = request.Request(url, method="GET")  # noqa: S310
        with request.urlopen(  # noqa: S310
            request_obj, timeout=timeout
        ) as response:
            body = response.read()
            status = getattr(response, "status", 200)
            return status, body, None
    except url_error.HTTPError as exc:
        return exc.code, exc.read(), str(exc)
    except OSError as exc:
        return None, None, str(exc)


def _evaluator_http_hit_if_200(
    _status: int | None, body: bytes | None, _error: str | None
) -> str:
    """Return a live-or-not status for generic package-index endpoints."""
    if _status == 200 and body:
        return "confirmed"
    if _status == 404:
        return "not_found"
    if _status is None:
        return "not_checked"
    return "unconfirmed"


def _evaluator_go_module_proxy(
    status: int | None, body: bytes | None, error: str | None
) -> str:
    if error is not None and status is None:
        return "not_checked"
    if status == 404:
        result = "not_found"
    elif status != 200 or body is None:
        result = "unconfirmed"
    else:
        text = body.decode("utf-8", errors="replace").strip()
        if not text or "not found:" in text.lower():
            result = "no_released_versions"
        elif text.splitlines():
            result = "confirmed"
        else:
            result = "no_released_versions"
    return result


def _evaluator_julia_general(
    status: int | None, body: bytes | None, error: str | None
) -> str:
    if error is not None and status is None:
        return "not_checked"
    if status in {404, 403}:
        return "not_present"
    if status == 200 and body:
        return "confirmed"
    if status == 200 and not body:
        return "not_present"
    return "unconfirmed"


def _evaluator_external_manual(
    _status: int | None, _body: bytes | None, _error: str | None
) -> str:
    """Return the conservative status for curation targets without package APIs."""
    return "external_manual"


CHANNELS: tuple[Channel, ...] = (
    Channel(
        key="python",
        package="voiage",
        registry="PyPI",
        registry_url="https://pypi.org/project/voiage/",
        check_url="https://pypi.org/pypi/voiage/json",
        notes=(
            "Automated PyPI/TestPyPI publish exists on tags; external "
            "conda-forge feedstock merge is manual."
        ),
        evaluator=_evaluator_http_hit_if_200,
        confidence="high",
    ),
    Channel(
        key="r",
        package="voiageR",
        registry="CRAN",
        registry_url="https://cran.r-project.org/web/packages/voiageR/index.html",
        check_url="https://crandb.r-pkg.org/voiageR",
        notes=(
            "GitHub Release source archives are produced from r-v* tags; CRAN and "
            "r-universe are external."
        ),
        evaluator=_evaluator_http_hit_if_200,
        confidence="medium",
    ),
    Channel(
        key="julia",
        package="Voiage",
        registry="Julia General",
        registry_url="https://github.com/JuliaRegistries/General",
        check_url=(
            "https://raw.githubusercontent.com/JuliaRegistries/General/"
            "master/V/Voiage/Package.toml"
        ),
        notes=(
            "TagBot sync and GitHub release artifacts are in-repo. Registry approval "
            "still depends on the external Julia ecosystem."
        ),
        evaluator=_evaluator_julia_general,
        confidence="medium",
    ),
    Channel(
        key="typescript",
        package="@voiage/core",
        registry="npm",
        registry_url="https://www.npmjs.com/package/%40voiage%2Fcore",
        check_url="https://registry.npmjs.org/%40voiage/core",
        notes="Automated npm publish exists on typescript-v* tags.",
        evaluator=_evaluator_http_hit_if_200,
        confidence="high",
    ),
    Channel(
        key="go",
        package="github.com/edithatogo/voiage/bindings/go",
        registry="Go module proxy",
        registry_url=(
            "https://proxy.golang.org/github.com/edithatogo/voiage/bindings/go/@v/list"
        ),
        check_url=(
            "https://proxy.golang.org/github.com/edithatogo/voiage/bindings/go/@v/list"
        ),
        notes="Semver tags under bindings/go/v* are in-repo publication boundary.",
        evaluator=_evaluator_go_module_proxy,
        confidence="medium",
    ),
    Channel(
        key="rust",
        package="voiage-core",
        registry="crates.io",
        registry_url="https://crates.io/crates/voiage-core",
        check_url="https://crates.io/api/v1/crates/voiage-core",
        notes="Automated cargo publish exists on rust-v* tags.",
        evaluator=_evaluator_http_hit_if_200,
        confidence="high",
    ),
    Channel(
        key="dotnet",
        package="Voiage.Core",
        registry="NuGet",
        registry_url="https://www.nuget.org/packages/Voiage.Core",
        check_url=(
            "https://api.nuget.org/v3/registration5-semver1/voiage.core/index.json"
        ),
        notes="Automated NuGet publish exists on dotnet-v* tags.",
        evaluator=_evaluator_http_hit_if_200,
        confidence="high",
    ),
    Channel(
        key="conda_forge",
        package="voiage",
        registry="conda-forge",
        registry_url="https://anaconda.org/conda-forge/voiage",
        check_url="https://api.anaconda.org/package/conda-forge/voiage",
        notes=(
            "The in-repo workflow can create a feedstock update PR; feedstock "
            "merge and channel indexing remain external conda-forge gates."
        ),
        evaluator=_evaluator_http_hit_if_200,
        confidence="medium",
    ),
    Channel(
        key="r_universe",
        package="voiageR",
        registry="r-universe",
        registry_url="https://edithatogo.r-universe.dev/voiageR",
        check_url="https://edithatogo.r-universe.dev/api/packages/voiageR",
        notes=(
            "r-universe indexing is external to this repository and must be "
            "verified independently from GitHub Release source archives."
        ),
        evaluator=_evaluator_http_hit_if_200,
        confidence="medium",
    ),
    Channel(
        key="spack",
        package="py-voiage",
        registry="Spack",
        registry_url="https://packages.spack.io/package.html?name=py-voiage",
        check_url=(
            "https://raw.githubusercontent.com/spack/spack/develop/"
            "var/spack/repos/builtin/packages/py-voiage/package.py"
        ),
        notes=(
            "Spack publication requires an upstream package PR, maintainer "
            "review, merge, and package visibility."
        ),
        evaluator=_evaluator_http_hit_if_200,
        confidence="medium",
    ),
    Channel(
        key="easybuild",
        package="voiage",
        registry="EasyBuild",
        registry_url="https://github.com/easybuilders/easybuild-easyconfigs",
        check_url=(
            "https://api.github.com/repos/easybuilders/"
            "easybuild-easyconfigs/contents/easybuild/easyconfigs/v/voiage"
            "?ref=develop"
        ),
        notes=(
            "EasyBuild publication requires an upstream easyconfig PR, "
            "maintainer review, merge, and easyconfig visibility."
        ),
        evaluator=_evaluator_http_hit_if_200,
        confidence="medium",
    ),
    Channel(
        key="hpsf",
        package="voiage",
        registry="HPSF",
        registry_url="https://hpsf.io/",
        check_url="https://hpsf.io/",
        notes=(
            "HPSF curation has no repo-owned package API check here; portal or "
            "curation review evidence must be attached by the follow-through track."
        ),
        evaluator=_evaluator_external_manual,
        confidence="low",
    ),
    Channel(
        key="e4s",
        package="voiage",
        registry="E4S",
        registry_url="https://e4s.io/",
        check_url="https://e4s.io/",
        notes=(
            "E4S inclusion depends on external curation and usually on upstream "
            "HPC packaging evidence such as Spack or EasyBuild."
        ),
        evaluator=_evaluator_external_manual,
        confidence="low",
    ),
)


def _status_details(status: str) -> str:
    if status == "confirmed":
        return "Live package payload or metadata endpoint was reachable."
    if status == "not_found":
        return "Manual check returned HTTP 404."
    if status == "not_present":
        return "General registry contents lookup returned no active package entry."
    if status == "no_released_versions":
        return "Endpoint was reachable but reported no released versions."
    if status == "not_checked":
        return (
            "Status was not refreshed because the registry endpoint "
            "could not be reached."
        )
    return "Manual check did not produce a confirmed publication state."


def _snapshot_entry(channel: Channel, status: str) -> dict[str, object]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "package": channel.package,
        "registry": channel.registry,
        "registry_url": channel.registry_url,
        "status": status,
        "details": _status_details(status),
        "evidence": "docs/release/binding-submission-checklist.md",
        "notes": channel.notes,
        "checked_at": now,
        "evidence_confidence": channel.confidence,
    }


def refresh_snapshot() -> dict[str, dict[str, object]]:
    """Return a live channel-status snapshot from current registry endpoints."""
    snapshot: dict[str, dict[str, object]] = {}
    for channel in CHANNELS:
        status_code, body, error = _fetch_json_bytes(channel.check_url)
        status = channel.evaluator(status_code, body, error)
        snapshot[channel.key] = _snapshot_entry(channel, status)
    return snapshot


def write_snapshot(snapshot_path: Path, snapshot: dict[str, dict[str, object]]) -> None:
    """Write an audit snapshot to disk in a stable format."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot": snapshot,
    }
    with snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main() -> int:
    """Entrypoint for command-line registry audit refresh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        default=str(DEFAULT_SNAPSHOT_PATH),
        help="Path to the JSON registry-audit snapshot file.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Write the current snapshot file in-place with a synthetic offline status.",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if args.offline:
        snapshot = {
            channel.key: {
                **_snapshot_entry(channel, "not_checked"),
                "details": "Offline mode: existing live status was not rechecked.",
            }
            for channel in CHANNELS
        }
        write_snapshot(snapshot_path, snapshot)
        print(f"Registry audit written in offline mode to {snapshot_path}")
        return 0

    if not snapshot_path.exists():
        print(
            "Warning: expected snapshot file does not exist yet; "
            f"creating {snapshot_path}"
        )
    snapshot = refresh_snapshot()
    write_snapshot(snapshot_path, snapshot)
    print(f"Registry audit refreshed: {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
