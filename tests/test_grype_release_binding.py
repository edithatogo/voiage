"""Negative qualification of retained SBOM binding and declared coverage."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROJECTION = load_script("prepare_grype_sbom")
BINDING = load_script("verify_grype_release_binding")
SOURCE_SHA = "a" * 40


def full_inventory():
    return {
        "$schema": "http://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "bom-ref": "root",
                "name": "voiage",
                "type": "application",
                "version": "3.1.0",
            },
            "properties": [
                {"name": "voiage:source:commit", "value": SOURCE_SHA},
                {"name": "voiage:source:tag", "value": "v3.1.0"},
            ],
        },
        "components": [
            {
                "bom-ref": "python",
                "type": "library",
                "name": "polars",
                "version": "2.1.0",
                "purl": "pkg:pypi/polars@2.1.0",
            },
            {
                "bom-ref": "rust",
                "type": "library",
                "name": "serde",
                "version": "1.0.0",
                "purl": "pkg:cargo/serde@1.0.0",
            },
            {
                "bom-ref": "r",
                "type": "library",
                "name": "jsonlite",
                "purl": "pkg:cran/jsonlite",
            },
            {
                "bom-ref": "julia",
                "type": "library",
                "name": "JSON3",
                "purl": "pkg:julia/JSON3?uuid=fixture",
            },
        ],
        "dependencies": [
            {"ref": "root", "dependsOn": ["python", "rust", "r", "julia"]}
        ],
    }


class ProjectionTests(unittest.TestCase):
    def test_projection_retains_supported_versions_and_declares_exclusions(self):
        raw = json.dumps(full_inventory()).encode()
        inventory, receipt = PROJECTION.project_inventory(raw)
        assert [p["name"] for p in inventory["components"]] == ["polars", "serde"]
        assert inventory["dependencies"][0]["dependsOn"] == ["python", "rust"]
        assert len(receipt["excluded_components"]) == 2
        assert receipt["parent_sbom_sha256"] == hashlib.sha256(raw).hexdigest()

    def test_supported_unresolved_or_new_ecosystem_is_an_error(self):
        for change in ({"version": ">=2.1"}, {"purl": "pkg:npm/example@1.0.0"}):
            fixture = full_inventory()
            fixture["components"][0].update(change)
            with self.subTest(change=change), pytest.raises(ValueError):
                PROJECTION.project_inventory(json.dumps(fixture).encode())

    def test_empty_supported_graph_cannot_be_called_clean(self):
        fixture = full_inventory()
        fixture["components"] = fixture["components"][2:]
        with pytest.raises(ValueError):
            PROJECTION.project_inventory(json.dumps(fixture).encode())


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        self.sbom = self.root / "polyglot.sbom.cdx.json"
        self.inventory = full_inventory()
        self.sbom.write_text(json.dumps(self.inventory))
        projection, _ = PROJECTION.project_inventory(self.sbom.read_bytes())
        self.resolved = self.root / "resolved-grype.sbom.cdx.json"
        self.resolved.write_text(json.dumps(projection))
        self.subjects = [
            self.dist / "voiage-3.1.0.whl",
            self.dist / "voiage-3.1.0.tar.gz",
        ]
        for path in self.subjects:
            path.write_bytes(b"fixture distribution " + path.name.encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = [
                {
                    "verificationResult": {
                        "statement": {
                            "predicateType": "https://cyclonedx.org/bom",
                            "predicate": self.inventory,
                            "subject": [
                                {"name": path.name, "digest": {"sha256": digest}}
                            ],
                        }
                    }
                }
            ]
            (self.root / f"{path.name}.attestation.json").write_text(json.dumps(result))
        self.manifest = self.root / "release-manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "source_commit": SOURCE_SHA,
                    "artifacts": [
                        {
                            "name": path.name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                        for path in [*self.subjects, self.sbom, self.resolved]
                    ],
                }
            )
        )

    def verify(self):
        return BINDING.verify_binding(
            self.manifest,
            self.sbom,
            self.dist,
            self.root,
            SOURCE_SHA,
            "v3.1.0",
            self.resolved,
        )

    def test_accepts_bound_reviewed_payload(self):
        assert self.verify()["distribution_count"] == 2

    def test_rejects_different_distribution_bytes(self):
        self.subjects[0].write_bytes(b"other build")
        with pytest.raises(ValueError, match="digest mismatch"):
            self.verify()

    def test_rejects_missing_attestation(self):
        (self.root / f"{self.subjects[0].name}.attestation.json").unlink()
        with pytest.raises(ValueError, match="missing regular evidence"):
            self.verify()

    def test_rejects_a_different_attested_inventory(self):
        path = self.root / f"{self.subjects[0].name}.attestation.json"
        result = json.loads(path.read_text())
        result[0]["verificationResult"]["statement"]["predicate"] = copy.deepcopy(
            self.inventory
        )
        result[0]["verificationResult"]["statement"]["predicate"]["components"] = []
        path.write_text(json.dumps(result))
        with pytest.raises(ValueError, match="verified predicate differs"):
            self.verify()

    def test_rejects_extra_manifest_subject(self):
        document = json.loads(self.manifest.read_text())
        document["artifacts"].append({"name": "unknown.whl", "sha256": "0" * 64})
        self.manifest.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="exact candidate set"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
