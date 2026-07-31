"""Expected-utility information pricing and VoC presentation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json

from voiage import _runtime
from voiage.exceptions import raise_input_error

_MEASURES = frozenset({"eui", "cei", "bpi", "spi", "ppi", "evpi"})


def expected_utility_information_value(
    request: Mapping[str, object],
) -> dict[str, object]:
    """Return the single Rust-owned expected-utility information result."""
    payload = deepcopy(dict(request))
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _runtime.compute_expected_utility_information(encoded)


def value_of_clairvoyance(
    request: Mapping[str, object], *, selected_measure: str = "eui"
) -> dict[str, object]:
    """Present a clairvoyant canonical result as VoC without recomputation."""
    if selected_measure not in _MEASURES:
        raise_input_error(f"Unsupported VoC selected measure: {selected_measure}.")
    payload = deepcopy(dict(request))
    raw_information = payload.get("information")
    if not isinstance(raw_information, Mapping):
        raise_input_error("VoC presentation requires an information mapping.")
    information = dict(raw_information)
    if information.get("kind") != "clairvoyant":
        raise_input_error("VoC presentation requires information.kind='clairvoyant'.")
    payload["presentation_label"] = "voc"
    result = expected_utility_information_value(payload)
    if selected_measure == "evpi":
        raw_reduction = result.get("affine_reduction")
        if not isinstance(raw_reduction, Mapping):
            raise_input_error("Canonical result omitted affine reduction metadata.")
        reduction = dict(raw_reduction)
        if (
            reduction.get("status") != "available"
            or reduction.get("monetary_measure") != "evpi"
        ):
            raise_input_error(
                "Monetary EVPI presentation requires an affine utility reduction.",
                diagnostic_code="affine_reduction_required",
            )
    raw_presentation = result.get("presentation")
    if not isinstance(raw_presentation, Mapping):
        raise_input_error("Canonical result omitted presentation metadata.")
    presentation = dict(raw_presentation)
    presentation["selected_measure"] = selected_measure
    result["presentation"] = presentation
    return result
