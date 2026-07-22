from pathlib import Path


def test_registry_readiness_contract():
    text = (Path(__file__).parents[1] / "docs" / "registry-readiness.md").read_text()
    assert "repository_ready_external_gates_pending" in text
    assert "authoritative provider evidence" in text
