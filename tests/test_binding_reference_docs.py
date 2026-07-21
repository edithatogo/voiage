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
    display_names = {"dotnet": ".NET", "r": "R", "typescript": "TypeScript"}
    adapter_names = {
        "wasm_or_n_api": "WebAssembly",
        "native_rust_crate": "Native Rust crate",
        "c_abi": "C ABI",
        "pyo3": "PyO3",
    }
    for binding in matrix["bindings"]:
        name = display_names.get(binding["id"], binding["id"].capitalize())
        assert f"| {name} |" in reference
        assert adapter_names[binding["adapter"]] in reference
        for registry_part in binding["registry"].replace(";", " ").split():
            assert registry_part in reference
        assert binding["tag_pattern"] in reference
        for version in binding["supported_runtime_versions"]:
            assert version in reference
