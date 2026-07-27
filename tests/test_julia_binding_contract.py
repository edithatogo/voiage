"""Keep Julia on the Rust-normalized JSON and canonical Arrow contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_julia_contracts_use_direct_rust_json_and_arrow_adapters() -> None:
    source = (ROOT / "bindings/julia/src/Voiage.jl").read_text(encoding="utf-8")
    project = (ROOT / "bindings/julia/Project.toml").read_text(encoding="utf-8")
    assert ":voiage_v1_decision_problem_json" in source
    assert ":voiage_v1_statistical_assurance_json" in source
    assert "Arrow.Table(path)" in source
    assert "does not match a pinned voiage v1 schema" in source
    assert 'Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"' in project
    assert 'Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"' in project
