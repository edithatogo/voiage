import pytest

from voiage.methods.qualitative import assess_qualitative_voi


def test_qualitative_voi_is_auditable_without_numeric_fabrication() -> None:
    result = assess_qualitative_voi("churn labels", "retention offer", "high", "changes targeting", ["study-1"], "analyst")
    assert result.value_rating == "high"
    assert result.evidence_ids == ("study-1",)
    assert not hasattr(result, "value")


def test_qualitative_voi_requires_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        assess_qualitative_voi("source", "decision", "low", "reason", [], "analyst")
