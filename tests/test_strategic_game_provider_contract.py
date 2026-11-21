"""Consumer checks for the qualified native Paignia static policy profile."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest

from voiage.exceptions import InputError
from voiage.experimental.paignia_static_policy import (
    PaigniaStaticPolicyBinding,
    value_paignia_static_policy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_FIXTURES = Path(__file__).parent / "fixtures" / "strategic_game_provider"

type BoundCase = tuple[bytes, bytes, PaigniaStaticPolicyBinding]


@pytest.fixture
def scoped_case() -> BoundCase:
    specification = (_FIXTURES / "static-information-input.json").read_bytes()
    projection = (_FIXTURES / "static-information-policy.json").read_bytes()
    binding = PaigniaStaticPolicyBinding(
        **json.loads((_FIXTURES / "static-information-binding.json").read_bytes())
    )
    return specification, projection, binding


def test_prequalified_static_policy_reaches_native_information_value(
    scoped_case: BoundCase,
) -> None:
    game_source = (_FIXTURES / "static_policy_reference.rs").read_bytes()
    assert sha256(game_source).hexdigest() == scoped_case[2].game_sha256
    result = value_paignia_static_policy(*scoped_case).to_contract_dict()
    design = next(
        item for item in result["designs"] if item["design_id"] == "qualified_response"
    )
    assert design["selected_policy_id"] == "native_policy"
    assert design["selection_mode"] == "declared_response"
    assert design["equilibrium_receipt"] is None
    assert design["signed_values"]["by_agent"]["institution"] == 1.0
    assert design["signed_values"]["by_agent"]["public"] == 2.0
    assert design["signed_values"]["social"] == 3.0


def test_stale_source_and_comparison_bytes_fail_before_evaluation(
    scoped_case: BoundCase,
) -> None:
    specification, projection, binding = scoped_case
    with pytest.raises(InputError, match="comparison.*digest"):
        value_paignia_static_policy(specification + b" ", projection, binding)
    with pytest.raises(InputError, match="projection.*digest"):
        value_paignia_static_policy(specification, projection + b" ", binding)


def _reseal(
    scoped_case: BoundCase,
    change: Callable[[dict[str, Any]], None],
    *,
    field: Literal["comparison", "projection"] = "projection",
) -> BoundCase:
    specification, projection, binding = scoped_case
    modified = json.loads(specification if field == "comparison" else projection)
    change(modified)
    changed = json.dumps(modified).encode()
    digest = sha256(changed).hexdigest()
    if field == "comparison":
        return changed, projection, replace(binding, comparison_sha256=digest)
    return specification, changed, replace(binding, projection_sha256=digest)


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("status", "not_converged", "not_converged"),
        ("solution_concept", "finite_nash", "solution concept"),
        ("catalogue", "enumerated", "catalogue"),
        ("selection_rule", "warm_start_previous", "selection rule"),
    ],
)
def test_provider_execution_and_selection_are_gated(
    scoped_case: BoundCase,
    field: str,
    value: str,
    failure: str,
) -> None:
    modified = _reseal(
        scoped_case, lambda packet: packet["native"].update({field: value})
    )
    with pytest.raises(InputError, match=failure):
        value_paignia_static_policy(*modified)


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("projection_version", "2.0.0", "unsupported projection version"),
        ("provider_revision", "0" * 40, "provider revision"),
        ("package_receipt_sha256", "0" * 64, "package receipt"),
        ("game_sha256", "0" * 64, "game digest"),
        ("native_player_id", "board", "native player"),
        ("analysis_id", "foreign", "different comparison"),
        ("design_id", "foreign", "declared-response design"),
        ("policy_id", "status_quo", "declared-response design"),
    ],
)
def test_projection_identity_and_scope(
    scoped_case: BoundCase, field: str, value: str, failure: str
) -> None:
    modified = _reseal(scoped_case, lambda packet: packet.update({field: value}))
    with pytest.raises(InputError, match=failure):
        value_paignia_static_policy(*modified)


@pytest.mark.parametrize(
    "field", ["provider_revision", "package_receipt_sha256", "game_sha256"]
)
def test_mutable_provider_identity_is_rejected(
    scoped_case: BoundCase, field: str
) -> None:
    specification, projection, binding = scoped_case
    with pytest.raises(InputError, match="immutable digests"):
        value_paignia_static_policy(
            specification, projection, replace(binding, **{field: "main"})
        )


def test_blank_native_actor_is_rejected_even_with_matching_binding(
    scoped_case: BoundCase,
) -> None:
    specification, projection, binding = _reseal(
        scoped_case, lambda packet: packet.update(native_player_id="")
    )
    with pytest.raises(InputError, match="native player identity is invalid"):
        value_paignia_static_policy(
            specification, projection, replace(binding, native_player_id="")
        )


@pytest.mark.parametrize(
    "field", ["specification_id", "game_instance_id", "solver_run_id"]
)
def test_wrong_producer_run_identity(scoped_case: BoundCase, field: str) -> None:
    modified = _reseal(
        scoped_case,
        lambda packet: packet["native"]["identities"].update({field: "foreign"}),
    )
    with pytest.raises(InputError, match=field):
        value_paignia_static_policy(*modified)


def test_producer_identity_must_be_nonempty(scoped_case: BoundCase) -> None:
    specification, projection, binding = scoped_case
    with pytest.raises(InputError, match="specification_id"):
        value_paignia_static_policy(
            specification, projection, replace(binding, specification_id="")
        )


def test_optional_native_identity_preserves_static_scope(
    scoped_case: BoundCase,
) -> None:
    modified = _reseal(
        scoped_case,
        lambda packet: packet["native"]["identities"].update(session_id="batch-1"),
    )
    assert value_paignia_static_policy(*modified).to_contract_dict()["analysis_id"] == (
        "si03-static-observation"
    )
    invalid = _reseal(
        scoped_case,
        lambda packet: packet["native"]["identities"].update(session_id=" batch-1"),
    )
    with pytest.raises(InputError, match="session_id identity is invalid"):
        value_paignia_static_policy(*invalid)


@pytest.mark.parametrize("tolerance", [True, -0.1, float("inf"), float("nan")])
def test_invalid_native_tolerance(scoped_case: BoundCase, tolerance: float) -> None:
    modified = _reseal(
        scoped_case, lambda packet: packet["native"].update(tolerance=tolerance)
    )
    with pytest.raises(InputError, match="finite and nonnegative"):
        value_paignia_static_policy(*modified)


@pytest.mark.parametrize(
    ("payload", "failure"),
    [
        (b"[]", "expected a JSON object"),
        (
            b'{"native":{"status":"completed","status":"not_converged"}}',
            "duplicate members",
        ),
        (b"{}", "projection_version"),
    ],
)
def test_invalid_projection_documents(
    scoped_case: BoundCase, payload: bytes, failure: str
) -> None:
    specification, _, binding = scoped_case
    with pytest.raises(InputError, match=failure):
        value_paignia_static_policy(
            specification,
            payload,
            replace(binding, projection_sha256=sha256(payload).hexdigest()),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_mode", "fixed"),
        ("selected_policy_id", "status_quo"),
        ("policy_ids", ["status_quo"]),
        ("equilibrium_receipt", {}),
    ],
)
def test_static_selection_is_bound_to_the_declared_response(
    scoped_case: BoundCase, field: str, value: object
) -> None:
    modified = _reseal(
        scoped_case,
        lambda input_case: input_case["designs"][1].update({field: value}),
        field="comparison",
    )
    with pytest.raises(InputError, match="declared-response design"):
        value_paignia_static_policy(*modified)


@pytest.mark.parametrize(
    ("field", "value"), [("policy_id", "absent"), ("decision_agent_id", "board")]
)
def test_selected_policy_belongs_to_the_bound_native_player(
    scoped_case: BoundCase, field: str, value: str
) -> None:
    modified = _reseal(
        scoped_case,
        lambda input_case: input_case["policies"][1].update({field: value}),
        field="comparison",
    )
    with pytest.raises(InputError, match="decision actor"):
        value_paignia_static_policy(*modified)


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("world_id", "foreign", "duplicate or foreign"),
        ("observation", "favourable", "permitted observation"),
        ("action_id", "maintain", "differs from the declared policy"),
        ("native_node_id", 0, "native decision-node identity"),
        ("native_node_id", "adverse", "conflicting choices"),
    ],
)
def test_projected_decision_is_bound_to_world_and_permitted_observation(
    scoped_case: BoundCase, field: str, value: object, failure: str
) -> None:
    modified = _reseal(
        scoped_case, lambda packet: packet["choices"][0].update({field: value})
    )
    with pytest.raises(InputError, match=failure):
        value_paignia_static_policy(*modified)


def test_projection_rejects_a_repeated_world(scoped_case: BoundCase) -> None:
    modified = _reseal(
        scoped_case, lambda packet: packet["choices"].append(packet["choices"][0])
    )
    with pytest.raises(InputError, match="duplicate or foreign world"):
        value_paignia_static_policy(*modified)


def test_hidden_world_cannot_change_an_action_at_the_same_observation(
    scoped_case: BoundCase,
) -> None:
    def share_observation(input_case: dict[str, Any]) -> None:
        input_case["worlds"][1]["signal"] = input_case["worlds"][0]["signal"]
        del input_case["policies"][1]["decisions"][1]

    modified = _reseal(scoped_case, share_observation, field="comparison")
    modified = _reseal(
        modified,
        lambda packet: packet["choices"][1].update(
            observation=packet["choices"][0]["observation"]
        ),
    )
    with pytest.raises(InputError, match="differs from the declared policy"):
        value_paignia_static_policy(*modified)


def test_projection_requires_whole_world_law(scoped_case: BoundCase) -> None:
    modified = _reseal(
        scoped_case, lambda packet: packet.update(choices=packet["choices"][:1])
    )
    with pytest.raises(InputError, match="complete world law"):
        value_paignia_static_policy(*modified)


def test_unobserved_policy_uses_one_action(scoped_case: BoundCase) -> None:
    unobserved = {"observation": "unobserved", "action_id": "maintain"}
    modified = _reseal(
        scoped_case,
        lambda input_case: input_case["policies"][1].update(decisions=[unobserved]),
        field="comparison",
    )

    def clear_observations(packet: dict[str, Any]) -> None:
        for choice in packet["choices"]:
            choice.update(unobserved)

    modified = _reseal(modified, clear_observations)
    assert (
        value_paignia_static_policy(*modified).to_contract_dict()["designs"][1][
            "signed_values"
        ]["social"]
        == 0.0
    )
    ambiguous = _reseal(
        scoped_case,
        lambda input_case: input_case["policies"][1]["decisions"].append(unobserved),
        field="comparison",
    )
    with pytest.raises(
        InputError, match="unobserved policy must have only one decision"
    ):
        value_paignia_static_policy(*ambiguous)


def test_comparison_still_requires_authorized_observation(
    scoped_case: BoundCase,
) -> None:
    modified = _reseal(
        scoped_case,
        lambda input_case: input_case["designs"][1].update(recipients=[]),
        field="comparison",
    )
    with pytest.raises(
        InputError, match="unshared designs may only use unobserved policies"
    ):
        value_paignia_static_policy(*modified)
