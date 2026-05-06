from __future__ import annotations

import json
from pathlib import Path


def _load_notebook(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_validation_notebooks_exist_and_stay_compact() -> None:
    nma_path = Path("examples/nma_validation.ipynb")
    structural_path = Path("examples/structural_voi_validation.ipynb")

    for path in (nma_path, structural_path):
        assert path.exists(), f"expected {path} to exist"
        notebook = _load_notebook(path)
        cells = notebook.get("cells", [])
        assert isinstance(cells, list)
        assert len(cells) <= 5


def test_nma_validation_notebook_uses_public_nma_api() -> None:
    notebook = _load_notebook(Path("examples/nma_validation.ipynb"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for needle in (
        "NetworkMetaAnalysisData",
        "calculate_nma_evpi",
        "calculate_nma_evppi",
        "assert evppi <= evpi + 1e-9",
    ):
        assert needle in sources


def test_structural_validation_notebook_uses_public_structural_api() -> None:
    notebook = _load_notebook(Path("examples/structural_voi_validation.ipynb"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for needle in (
        "structural_evpi",
        "structural_evppi",
        "ValueArray.from_numpy",
        "assert sevppi <= sevpi + 1e-9",
    ):
        assert needle in sources
