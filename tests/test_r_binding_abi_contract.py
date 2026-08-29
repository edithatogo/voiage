"""Keep the R .C adapter aligned with its package-owned native routines."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r_evpi_uses_pointer_safe_dimension_adapter() -> None:
    source = (ROOT / "r-package/voiageR/R/voiageR.R").read_text(encoding="utf-8")
    registration = (ROOT / "r-package/voiageR/src/init.c").read_text(encoding="utf-8")
    assert "voiageR_evpi" in source
    assert '{"voiageR_evpi", (DL_FUNC)&voiageR_evpi, 5}' in registration
    assert '"voiage_v1_evpi_i32_r"' not in source


def test_r_enbs_uses_pointer_safe_adapter() -> None:
    source = (ROOT / "r-package/voiageR/R/voiageR.R").read_text(encoding="utf-8")
    registration = (ROOT / "r-package/voiageR/src/init.c").read_text(encoding="utf-8")
    assert "voiageR_enbs" in source
    assert '{"voiageR_enbs", (DL_FUNC)&voiageR_enbs, 4}' in registration
    assert '"voiage_v1_enbs_r"' not in source
