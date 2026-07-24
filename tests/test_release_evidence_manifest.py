from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.release_evidence_manifest import (
    create_manifest,
    validate_manifest,
)

COMMIT = "05cc373d78ae74143194e889ff1317de4dfea52e"
SWHID = "swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    assets = tmp_path / "assets"
    assets.mkdir()
    wheel = assets / "voiage-1.0.0-py3-none-any.whl"
    sdist = assets / "voiage-1.0.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    checksums = assets / "SHA256SUMS"
    checksums.write_text(
        f"{_sha256(sdist)}  {sdist.name}\n{_sha256(wheel)}  {wheel.name}\n",
        encoding="utf-8",
    )

    sbom = tmp_path / "sbom.cdx.json"
    sbom.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "metadata": {
                    "component": {
                        "type": "library",
                        "name": "voiage",
                        "version": "1.0.0",
                    },
                    "properties": [
                        {"name": "voiage:source:commit", "value": COMMIT},
                        {"name": "voiage:source:tag", "value": "v1.0.0"},
                    ],
                },
                "components": [],
                "dependencies": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            [
                {
                    "verificationResult": {
                        "statement": {
                            "predicateType": "https://slsa.dev/provenance/v1",
                            "subject": [
                                {
                                    "name": wheel.name,
                                    "digest": {"sha256": _sha256(wheel)},
                                },
                                {
                                    "name": sdist.name,
                                    "digest": {"sha256": _sha256(sdist)},
                                },
                            ],
                            "predicate": {
                                "buildDefinition": {
                                    "resolvedDependencies": [
                                        {
                                            "uri": (
                                                "git+https://github.com/"
                                                "edithatogo/voiage@refs/tags/v1.0.0"
                                            ),
                                            "digest": {"gitCommit": COMMIT},
                                        }
                                    ]
                                },
                                "runDetails": {
                                    "metadata": {
                                        "invocationId": (
                                            "https://github.com/edithatogo/voiage/"
                                            "actions/runs/1/attempts/1"
                                        )
                                    }
                                },
                            },
                        }
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    return assets, sbom, attestation


def test_complete_release_manifest_binds_every_evidence_layer(
    tmp_path: Path,
) -> None:
    assets, sbom, attestation = _write_fixture(tmp_path)
    manifest = create_manifest(
        repository="edithatogo/voiage",
        version="1.0.0",
        tag="v1.0.0",
        commit=COMMIT,
        release_url="https://github.com/edithatogo/voiage/releases/tag/v1.0.0",
        asset_directory=assets,
        sbom_path=sbom,
        sbom_recorded_path="sbom.cdx.json",
        software_heritage_id=SWHID,
        attestation_report=attestation,
        sbom_attached_to_release=False,
    )

    assert (
        validate_manifest(
            manifest,
            artifact_root=tmp_path,
            repository_root=None,
        )
        == []
    )
    assert manifest["release"]["commit"] == COMMIT
    assert manifest["sbom"]["sha256"] == _sha256(sbom)
    assert manifest["sbom"]["attached_to_github_release"] is False
    assert manifest["attestation"]["status"] == "verified"
    assert {item["name"] for item in manifest["attestation"]["subjects"]} == {
        "voiage-1.0.0-py3-none-any.whl",
        "voiage-1.0.0.tar.gz",
    }


def test_validation_fails_closed_when_asset_or_sbom_changes(tmp_path: Path) -> None:
    assets, sbom, attestation = _write_fixture(tmp_path)
    manifest = create_manifest(
        repository="edithatogo/voiage",
        version="1.0.0",
        tag="v1.0.0",
        commit=COMMIT,
        release_url="https://github.com/edithatogo/voiage/releases/tag/v1.0.0",
        asset_directory=assets,
        sbom_path=sbom,
        sbom_recorded_path="sbom.cdx.json",
        software_heritage_id=SWHID,
        attestation_report=attestation,
        sbom_attached_to_release=False,
    )

    (assets / "voiage-1.0.0.tar.gz").write_bytes(b"changed")
    sbom.write_text("{}\n", encoding="utf-8")

    errors = validate_manifest(
        manifest,
        artifact_root=tmp_path,
        repository_root=None,
    )
    assert any("release asset digest mismatch" in error for error in errors)
    assert any("SBOM digest mismatch" in error for error in errors)


def test_creation_rejects_attestation_for_a_stale_artifact(tmp_path: Path) -> None:
    assets, sbom, attestation = _write_fixture(tmp_path)
    report = json.loads(attestation.read_text(encoding="utf-8"))
    report[0]["verificationResult"]["statement"]["subject"][0]["digest"]["sha256"] = (
        "0" * 64
    )
    attestation.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="matching provenance attestation",
    ):
        create_manifest(
            repository="edithatogo/voiage",
            version="1.0.0",
            tag="v1.0.0",
            commit=COMMIT,
            release_url="https://github.com/edithatogo/voiage/releases/tag/v1.0.0",
            asset_directory=assets,
            sbom_path=sbom,
            sbom_recorded_path="sbom.cdx.json",
            software_heritage_id=SWHID,
            attestation_report=attestation,
            sbom_attached_to_release=False,
        )


def test_creation_cannot_claim_an_unattached_sbom(tmp_path: Path) -> None:
    assets, sbom, attestation = _write_fixture(tmp_path)

    with pytest.raises(
        ValueError,
        match="SBOM cannot be marked attached",
    ):
        create_manifest(
            repository="edithatogo/voiage",
            version="1.0.0",
            tag="v1.0.0",
            commit=COMMIT,
            release_url="https://github.com/edithatogo/voiage/releases/tag/v1.0.0",
            asset_directory=assets,
            sbom_path=sbom,
            sbom_recorded_path="sbom.cdx.json",
            software_heritage_id=SWHID,
            attestation_report=attestation,
            sbom_attached_to_release=True,
        )


def test_checked_in_release_manifest_matches_tag_and_local_sbom() -> None:
    manifest_path = Path("docs/release/v1.0.0-release-evidence.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert (
        validate_manifest(
            manifest,
            artifact_root=Path.cwd(),
            repository_root=Path.cwd(),
            require_asset_files=False,
        )
        == []
    )
    assert manifest["release"]["tag"] == "v1.0.0"
    assert manifest["release"]["commit"] == COMMIT
    assert manifest["software_heritage"]["snapshot_swhid"] == SWHID
    assert manifest["sbom"]["attached_to_github_release"] is False


def test_sbom_workflow_is_tag_aware_and_retains_digest_bound_evidence() -> None:
    workflow = Path(".github/workflows/sbom.yml").read_text(encoding="utf-8")

    for required in (
        "workflow_dispatch:",
        "release_tag:",
        'tags: ["v*"]',
        "github.event.release.tag_name",
        "SOURCE_COMMIT",
        "SOURCE_TAG",
        "release-evidence.json",
        "release-evidence-checksums.txt",
        "scripts/release_evidence_manifest.py create",
        "scripts/release_evidence_manifest.py validate",
        "gh attestation verify",
        "sbom.cdx.json",
        "sha256sum",
        "retention-days: 90",
    ):
        assert required in workflow

    assert "gh release upload" not in workflow
    assert "attached_to_github_release=true" not in workflow
