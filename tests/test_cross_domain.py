"""Cross-domain smoke tests for the main factory entrypoints."""

import numpy as np

from voiage.factory import (
    create_environmental_analysis,
    create_financial_analysis,
    create_healthcare_analysis,
)


def _net_benefits() -> np.ndarray:
    return np.array(
        [
            [100.0, 101.5],
            [99.0, 102.0],
            [101.0, 100.5],
            [100.5, 103.0],
        ],
        dtype=np.float64,
    )


def test_healthcare_factory_smoke() -> None:
    analysis = create_healthcare_analysis(nb_array=_net_benefits(), use_jit=False)
    assert analysis.evpi() >= 0.0


def test_financial_factory_smoke() -> None:
    analysis = create_financial_analysis(nb_array=_net_benefits())
    assert analysis.evpi() >= 0.0


def test_environmental_factory_smoke() -> None:
    analysis = create_environmental_analysis(
        nb_array=_net_benefits(),
        carbon_intensity=0.5,
        energy_consumption=1000.0,
        water_intensity=0.2,
    )
    assert analysis.evpi() >= 0.0
