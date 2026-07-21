"""Keep the published Astro binding reference aligned with the v1 matrix."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_astro_binding_reference_covers_the_normative_matrix() -> None:
    matrix = json.loads(
        (ROOT / "specs/v1/binding-matrix.json").read_text(encoding="utf-8")
    )
    reference = (
        ROOT / "docs/astro-site/src/content/docs/reference/bindings.mdx"
    ).read_text(encoding="utf-8")

    assert "The matrix is normative" in reference
    display_names = {"r": "R"}
    adapter_names = {
        "native_rust_workspace": "Native Rust workspace",
        "c_abi": "C ABI",
        "pyo3": "PyO3",
    }
    for binding in matrix["bindings"]:
        if binding["status"] == "external_boundary":
            continue
        name = display_names.get(binding["id"], binding["id"].capitalize())
        assert f"| {name} |" in reference
        assert adapter_names[binding["adapter"]] in reference
        registry = binding["registry"]
        assert registry in reference or all(
            registry_part in reference
            for registry_part in registry.replace(";", " ").split()
            if registry_part not in {"no", "package"}
        )
        assert binding["tag_pattern"] in reference
        for version in binding["supported_runtime_versions"]:
            assert version in reference
