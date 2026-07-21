"""Keep the R .C adapter aligned with the pointer-safe Rust ABI entrypoint."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r_evpi_uses_pointer_safe_dimension_adapter() -> None:
    source = (ROOT / "r-package/voiageR/R/voiageR.R").read_text(encoding="utf-8")
    assert '"voiage_v1_evpi_i32_r"' in source
    assert '"voiage_v1_evpi_i32",' not in source
