"""Admit a Paignia static-policy projection for finite information evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import TYPE_CHECKING, Any, cast

from voiage.exceptions import raise_input_error
from voiage.methods.signed_social_information import signed_social_information_value

if TYPE_CHECKING:
    from voiage.methods.signed_social_information import SignedSocialInformationResult


_NATIVE_FIELDS = (
    ("schema_version", "1.0.0", "schema version"),
    (
        "semantic_profile",
        "paignia/static-finite-perfect-information-sparse-zero/1.0.0",
        "semantic profile",
    ),
    ("game_class", "finite_acyclic_extensive", "game class"),
    ("information_structure", "perfect_information", "information structure"),
    (
        "solution_concept",
        "tolerance_aware_backward_induction_policy",
        "solution concept",
    ),
    ("algorithm", "paignia/backward-induction", "algorithm"),
    ("algorithm_version", "1.0.0", "algorithm version"),
    (
        "selection_rule",
        "input_order_pairwise_tolerance_then_lexicographic",
        "selection rule",
    ),
    ("catalogue", "one_found", "catalogue"),
)


@dataclass(frozen=True, slots=True)
class PaigniaStaticPolicyBinding:
    """Expected package, producer run and file digests from the caller's evidence."""

    provider_revision: str
    package_receipt_sha256: str
    game_sha256: str
    specification_id: str
    game_instance_id: str
    solver_run_id: str
    native_player_id: str
    comparison_sha256: str
    projection_sha256: str


def value_paignia_static_policy(
    comparison: bytes,
    projection: bytes,
    binding: PaigniaStaticPolicyBinding,
) -> SignedSocialInformationResult:
    """Evaluate one bound declared response in a signed-information v1 comparison.

    The binding comes from the caller's accepted package and producer evidence.
    The producer validates its native result before projecting its decisions.
    The two inputs are the exact file bytes recorded in that binding. This
    admission uses the native static result's one-policy selection scope.
    """
    try:
        specification = _read_object(
            comparison, binding.comparison_sha256, "comparison"
        )
        receipt = _read_object(projection, binding.projection_sha256, "projection")
        _validate_native_policy(receipt, binding)
        _validate_comparison_policy(specification, receipt)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise_input_error(f"Paignia static policy: {error}")
    return signed_social_information_value(specification)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("JSON object has duplicate members")
    return result


def _read_object(payload: bytes, digest: str, label: str) -> dict[str, Any]:
    if sha256(payload).hexdigest() != digest:
        raise ValueError(f"the {label} digest does not match the binding")
    value = json.loads(payload, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return cast("dict[str, Any]", value)


def _is_label(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _validate_native_policy(
    receipt: dict[str, Any],
    binding: PaigniaStaticPolicyBinding,
) -> None:
    if receipt["projection_version"] != "1.0.0":
        raise ValueError("unsupported projection version")
    if not re.fullmatch(r"[0-9a-f]{40}", binding.provider_revision) or not all(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in (binding.package_receipt_sha256, binding.game_sha256)
    ):
        raise ValueError(
            "provider, package and game bindings require immutable digests"
        )
    for field, expected, label in (
        ("provider_revision", binding.provider_revision, "provider revision"),
        ("package_receipt_sha256", binding.package_receipt_sha256, "package receipt"),
        ("game_sha256", binding.game_sha256, "game digest"),
        ("native_player_id", binding.native_player_id, "native player"),
    ):
        if receipt[field] != expected:
            raise ValueError(f"the {label} does not match the binding")
    if not _is_label(binding.native_player_id):
        raise ValueError("the native player identity is invalid")

    native = receipt["native"]
    if native["status"] != "completed":
        raise ValueError(
            f"provider result status {native['status']!r} is not completed"
        )
    for field, required, label in _NATIVE_FIELDS:
        if native[field] != required:
            raise ValueError(f"unsupported native {label}")
    identities = native["identities"]
    for field, expected in (
        ("specification_id", binding.specification_id),
        ("game_instance_id", binding.game_instance_id),
        ("solver_run_id", binding.solver_run_id),
    ):
        if not _is_label(expected) or identities[field] != expected:
            raise ValueError(f"the native {field} does not match the binding")
    for field in ("session_id", "policy_artifact_id", "solver_trace_id"):
        value = identities[field]
        if value is not None and not _is_label(value):
            raise ValueError(f"the native {field} identity is invalid")
    tolerance = native["tolerance"]
    if (
        type(tolerance) not in (int, float)
        or not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("native tolerance must be finite and nonnegative")


def _validate_comparison_policy(
    specification: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if receipt["analysis_id"] != specification["analysis_id"]:
        raise ValueError("the projection refers to a different comparison")
    design_id = receipt["design_id"]
    policy_id = receipt["policy_id"]
    design = next(
        (item for item in specification["designs"] if item["design_id"] == design_id),
        None,
    )
    if (
        design is None
        or design["selection_mode"] != "declared_response"
        or design["selected_policy_id"] != policy_id
        or policy_id not in design["policy_ids"]
        or design["equilibrium_receipt"] is not None
    ):
        raise ValueError(
            "the static policy requires a matching declared-response design"
        )
    policy = next(
        (item for item in specification["policies"] if item["policy_id"] == policy_id),
        None,
    )
    if policy is None or policy["decision_agent_id"] != receipt["native_player_id"]:
        raise ValueError("the policy decision actor does not match the projection")
    decisions = {row["observation"]: row["action_id"] for row in policy["decisions"]}
    uses_signal = "unobserved" not in decisions
    if not uses_signal and len(decisions) != 1:
        raise ValueError("an unobserved policy must have only one decision")
    expected_observations = {
        world["world_id"]: world["signal"] if uses_signal else "unobserved"
        for world in specification["worlds"]
    }
    observed: set[str] = set()
    node_choices: dict[str, tuple[str, str]] = {}
    for row in receipt["choices"]:
        world_id = row["world_id"]
        if world_id in observed or world_id not in expected_observations:
            raise ValueError("the projection has a duplicate or foreign world identity")
        observed.add(world_id)
        expected = expected_observations[world_id]
        if row["observation"] != expected:
            raise ValueError("the projection changes the world's permitted observation")
        if row["action_id"] != decisions[expected]:
            raise ValueError("the native action differs from the declared policy")
        node_id = row["native_node_id"]
        if not _is_label(node_id):
            raise ValueError("the native decision-node identity is invalid")
        choice = (row["observation"], row["action_id"])
        if node_choices.setdefault(node_id, choice) != choice:
            raise ValueError("one native decision node has conflicting choices")
    if observed != expected_observations.keys():
        raise ValueError("the projection does not cover the complete world law")
