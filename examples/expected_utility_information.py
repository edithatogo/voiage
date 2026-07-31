"""Run the experimental expected-utility information and VoC presentations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from voiage.methods.utility_information import (
    expected_utility_information_value,
    value_of_clairvoyance,
)


def main() -> None:
    """Evaluate the deterministic nonlinear clairvoyance fixture."""
    repository_root = Path(__file__).resolve().parents[1]
    fixture_path = (
        repository_root
        / "specs"
        / "frontier"
        / "expected-utility-information-pricing"
        / "v1"
        / "fixtures"
        / "normative"
        / "log-buy-sell-asymmetry.json"
    )
    fixture = cast(
        "dict[str, object]", json.loads(fixture_path.read_text(encoding="utf-8"))
    )
    request = cast("dict[str, object]", fixture["request"])

    canonical_request = dict(request)
    canonical_request["presentation_label"] = "canonical"
    result = expected_utility_information_value(canonical_request)
    voc = value_of_clairvoyance(request, selected_measure="bpi")

    measures = {
        name: cast("dict[str, object]", result[name])["value"]
        for name in ("eui", "cei", "bpi", "spi", "ppi")
    }
    output = {
        "method_maturity": result["method_maturity"],
        "utility": result["utility"],
        "measures": measures,
        "bpi_root_status": cast("dict[str, object]", result["bpi_root"])["status"],
        "spi_root_status": cast("dict[str, object]", result["spi_root"])["status"],
        "voc_presentation": voc["presentation"],
        "voc_method": voc["method"],
        "affine_reduction": result["affine_reduction"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
