"""Keep the R .C adapter aligned with the pointer-safe Rust ABI entrypoint."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r_evpi_uses_pointer_safe_dimension_adapter() -> None:
    source = (ROOT / "r-package/voiageR/R/voiageR.R").read_text(encoding="utf-8")
    assert '"voiage_v1_evpi_i32_r"' in source
    assert '"voiage_v1_evpi_i32",' not in source
    assert 'PACKAGE = "voiageR"' not in source


def test_r_contracts_use_direct_rust_json_and_arrow_adapters() -> None:
    source = (ROOT / "r-package/voiageR/R/voiageR.R").read_text(encoding="utf-8")
    namespace = (ROOT / "r-package/voiageR/NAMESPACE").read_text(encoding="utf-8")
    assert '"voiage_v1_decision_problem_json_i32_r"' in source
    assert '"voiage_v1_statistical_assurance_json_i32_r"' in source
    assert "arrow::read_ipc_file" in source
    assert "does not match a pinned voiage v1 schema" in source
    assert "export(normalize_decision_problem)" in namespace
    assert "export(normalize_statistical_assurance)" in namespace
    assert "export(read_voiage_arrow)" in namespace
