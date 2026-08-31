"""Source identity and dependency-boundary checks for local Spack overlays."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "packaging/spack-overlay"


def test_overlay_recipe_manifest_binds_every_package() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    recipes = sorted((ROOT / "packages").glob("*/package.py"))
    assert manifest["recipe_sha256"] == {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in recipes
    }
    assert manifest["patch_sha256"] == {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "packages").rglob("*.patch"))
    }
    assert "packages/py-pyarrow/for_aarch64.patch" in manifest["patch_sha256"]
    for path in recipes:
        ast.parse(path.read_text())
    assert not manifest["native_builds_executed"]
    assert not manifest["upstream_submitted"]


def test_solver_receipt_binds_inspectable_logs_and_package_manifest() -> None:
    receipt = json.loads((ROOT / "solver-receipt.json").read_text())
    assert (
        receipt["manifest_sha256"]
        == hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest()
    )
    for result in receipt["results"]:
        path = (ROOT / result["log_path"]).resolve()
        assert path.is_relative_to((ROOT / "solver-logs").resolve())
        assert result["log_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        text = path.read_text()
        for private_prefix in ["/Users/", "/Volumes/", "/private/", "/tmp/"]:  # noqa: S108 - forbidden log text, not a filesystem operation
            assert private_prefix not in text
        if result["exit_code"] == 0:
            assert result["status"] == "concretized"
            assert result["spec"] in text
            assert "Error: failed to concretize" not in text
            if result["spec"] == "py-pyarrow@25.0.0":
                arrow = next(
                    line for line in text.splitlines() if "^arrow@25.0.0" in line
                )
                for feature in ["python", "csv", "dataset", "filesystem", "parquet"]:
                    assert f"+{feature}" in arrow
        else:
            assert result["status"] == "failed"
            assert "Error: failed to concretize" in text
    assert not receipt["builds_executed"]


def test_overlay_versions_match_verified_source_digests() -> None:
    audit = json.loads((ROOT / "source-audit.json").read_text())
    audit += json.loads((ROOT / "transitive-source-audit.json").read_text())
    for entry in audit:
        source = entry.get("sources", [entry])[0]
        assert source.get("download_hash_verified", source.get("hash_verified"))
        recipe = ROOT / "packages" / f"py-{entry['name']}" / "package.py"
        assert (
            f'"{entry["version"]}", sha256="{source["sha256"]}"' in recipe.read_text()
        )
    arrow = json.loads((ROOT / "arrow-source-audit.json").read_text())
    assert arrow["sha256"] in (ROOT / "packages/arrow/package.py").read_text()
    assert arrow["sha512_verified_against"].endswith(".sha512")
    assert arrow["signature_verified"] is False


def test_native_dependency_requirements_are_not_downgraded() -> None:
    core = (ROOT / "packages/py-pydantic-core/package.py").read_text()
    assert 'depends_on("rust@1.88:"' in core
    assert 'depends_on("py-maturin@1.10:1"' in core
    runtime = (ROOT / "packages/py-polars-runtime-32/package.py").read_text()
    assert 'depends_on("rust@nightly-2026-04-01"' in runtime
    polars = (ROOT / "packages/py-polars/package.py").read_text()
    assert 'depends_on("py-polars-runtime-32@1.42.1"' in polars
    assert '@when("@:1.29 ~nightly")' in polars
    arrow = (ROOT / "packages/py-pyarrow/package.py").read_text()
    assert 'depends_on("py-libcst@1.8.6:"' in arrow
    assert 'depends_on("py-scikit-build-core"' in arrow
    assert '"arrow@25.0.0+python+csv+dataset+filesystem+parquet"' in arrow


def test_spack_directives_are_explicit_including_license() -> None:
    for recipe in (ROOT / "packages").glob("*/package.py"):
        tree = ast.parse(recipe.read_text())
        names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "spack.package"
            for alias in node.names
        }
        assert "*" not in names
        # Python also defines a builtin named license; it is not Spack's directive.
        assert "license" in names
        assert "depends_on" in names


def test_upstream_notices_remain_with_derived_recipes() -> None:
    for name in ["COPYRIGHT", "NOTICE", "LICENSE-APACHE", "LICENSE-MIT"]:
        assert (ROOT / name).read_text().strip()
    for name in ["py-pydantic", "py-polars", "py-pyarrow", "arrow"]:
        assert (
            "Copyright Spack Project Developers"
            in (ROOT / "packages" / name / "package.py").read_text()
        )
