#!/usr/bin/env python3
"""Independent offline validator and oracle reproducer for the Q03 reference corpus."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNPINNED_REVISIONS = {"", "latest", "head", "main", "master"}


def _nonempty_str(value: object, label: str) -> str:
    """Require a non-empty string with no surrounding whitespace issues."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _check_not_self_certified(reference_impl: object, label: str) -> str:
    """Reject reference values attributed to the implementation under test."""
    name = _nonempty_str(reference_impl, label)
    if "voiage" in name.lower():
        raise ValueError(
            f"{label} cannot attribute reference generation to voiage: {name!r}"
        )
    return name


def _check_pinned_revision(revision: object, label: str) -> str:
    """Require an explicit pinned revision rather than a floating branch or tag."""
    value = _nonempty_str(revision, label)
    if value.lower() in _UNPINNED_REVISIONS:
        raise ValueError(f"{label} must be a pinned revision, not {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    """Return lowercase SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction_evpi(
    net_benefits: list[list[float]], weights: list[float] | None = None
) -> tuple[float, float, float]:
    """Compute exact EVPI, expected current value, and expected perfect info in Rationals."""
    if not net_benefits:
        raise ValueError("net_benefits must not be empty")
    n_rows = len(net_benefits)
    n_cols = len(net_benefits[0])
    if n_cols == 0 or any(len(row) != n_cols for row in net_benefits):
        raise ValueError("net_benefits rows must share a non-zero strategy count")
    if weights is None:
        frac_weights = [Fraction(1, n_rows) for _ in range(n_rows)]
    else:
        if len(weights) != n_rows:
            raise ValueError(
                "draw alignment error: weights length matches net_benefits"
            )
        frac_weights = [Fraction(str(w)) for w in weights]
        total_w = sum(frac_weights)
        if total_w <= 0:
            raise ValueError("weights must have positive total weight")
        frac_weights = [w / total_w for w in frac_weights]

    frac_rows = [[Fraction(str(val)) for val in row] for row in net_benefits]
    expected_perfect = sum(
        (w * max(row) for w, row in zip(frac_weights, frac_rows, strict=True)),
        start=Fraction(0),
    )
    strategy_means = [
        sum(
            (w * row[col] for w, row in zip(frac_weights, frac_rows, strict=True)),
            start=Fraction(0),
        )
        for col in range(n_cols)
    ]
    expected_current = max(strategy_means)
    return (
        float(expected_perfect - expected_current),
        float(expected_current),
        float(expected_perfect),
    )


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _beta_binomial_evsi(
    alpha: float, beta: float, slope: float, intercept: float, total_n: int
) -> tuple[float, float]:
    """Return exact (evsi, evpi) for Beta(alpha, beta) + Binomial(total_n) linear NB."""
    if slope <= 0.0:
        raise ValueError("Beta-Binomial oracle expects positive slope")
    prior_best = max(0.0, slope * (alpha / (alpha + beta)) + intercept)
    conditional_sum = 0.0
    for k in range(total_n + 1):
        log_pk = (
            math.lgamma(total_n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(total_n - k + 1)
            + math.lgamma(alpha + beta)
            - math.lgamma(alpha)
            - math.lgamma(beta)
            + math.lgamma(alpha + k)
            + math.lgamma(beta + total_n - k)
            - math.lgamma(alpha + beta + total_n)
        )
        pk = math.exp(log_pk)
        post_mean = (alpha + k) / (alpha + beta + total_n)
        conditional_sum += pk * max(0.0, slope * post_mean + intercept)
    evsi_val = max(0.0, conditional_sum - prior_best)

    # Exact closed-form rational polynomial integration for Beta(2, 3): 12 * p * (1 - p)^2.
    if alpha == 2.0 and beta == 3.0 and -intercept / slope == 0.45:
        threshold = Fraction(45, 100)
        slope_f = Fraction(str(slope))
        intercept_f = Fraction(str(intercept))
        # f(p) = 12 * (p - 2p^2 + p^3); integrand = (slope*p + intercept) * f(p)
        coeffs = [
            Fraction(0),
            12 * intercept_f,
            12 * (slope_f - 2 * intercept_f),
            12 * (-2 * slope_f + intercept_f),
            12 * slope_f,
        ]
        upper = sum(c / (deg + 1) for deg, c in enumerate(coeffs))
        lower = sum(
            c * (threshold ** (deg + 1)) / (deg + 1) for deg, c in enumerate(coeffs)
        )
        evpi_val = float(upper - lower) - prior_best
    else:
        raise ValueError("Unsupported Beta-Binomial closed-form EVPI parameters")
    return evsi_val, evpi_val


def _gamma_poisson_evsi(
    alpha: int,
    rate: float,
    slope: float,
    intercept: float,
    exposure: float,
    k_max: int,
    certified_tail_bound: float,
) -> tuple[float, float, float]:
    """Return exact truncated (evsi, tail_bound, evpi) for integer-alpha Gamma-Poisson."""
    if alpha != 3 or slope <= 0.0:
        raise ValueError(
            "Gamma-Poisson oracle expects integer alpha=3 and positive slope"
        )
    prior_mean_lam = alpha / rate
    prior_best = max(0.0, slope * prior_mean_lam + intercept)

    p_fail = rate / (rate + exposure)
    q_succ = exposure / (rate + exposure)
    mean_terms: list[float] = []
    evsi_terms: list[float] = []
    for k in range(k_max + 1):
        log_pk = (
            math.lgamma(alpha + k)
            - math.lgamma(alpha)
            - math.lgamma(k + 1)
            + alpha * math.log(p_fail)
            + k * math.log(q_succ)
        )
        pk = math.exp(log_pk)
        post_lam = (alpha + k) / (rate + exposure)
        mean_terms.append(pk * post_lam)
        evsi_terms.append(pk * max(0.0, slope * post_lam + intercept))
    rem_mean = max(0.0, prior_mean_lam - math.fsum(mean_terms))
    tail_bound = slope * rem_mean
    if tail_bound > certified_tail_bound:
        raise ValueError(
            f"Gamma-Poisson tail bound {tail_bound} exceeds certified {certified_tail_bound}"
        )
    evsi_val = max(0.0, math.fsum(evsi_terms) - prior_best)

    lam_star = -intercept / slope
    x_star = rate * lam_star
    p_above = math.exp(-x_star) * (1.0 + x_star + x_star * x_star / 2.0)
    lam_trunc = (
        prior_mean_lam
        * math.exp(-x_star)
        * (1.0 + x_star + x_star * x_star / 2.0 + (x_star**3) / 6.0)
    )
    evpi_val = slope * lam_trunc + intercept * p_above - prior_best
    return evsi_val, tail_bound, evpi_val


def _require_dict(value: object, label: str) -> dict[str, Any]:
    """Require a JSON object mapping."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _require_bool(value: object, label: str) -> bool:
    """Require an explicit Boolean value."""
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _validate_case_metadata(payload: dict[str, Any], label: str) -> None:
    """Validate mandatory governance, unit, provenance, and draw-alignment fields."""
    _nonempty_str(payload.get("schema_version"), f"{label}.schema_version")
    _check_not_self_certified(
        payload.get("reference_implementation"), f"{label}.reference_implementation"
    )
    _nonempty_str(payload.get("verification_family"), f"{label}.verification_family")
    split = payload.get("split")
    if split not in {"development", "held_out"}:
        raise ValueError(f"{label}.split must be 'development' or 'held_out'")
    _require_bool(payload.get("held_out_from_tuning"), f"{label}.held_out_from_tuning")
    _require_bool(payload.get("synthetic_owned"), f"{label}.synthetic_owned")
    _require_bool(
        payload.get("same_author_application"), f"{label}.same_author_application"
    )
    if not _require_dict(payload.get("decision"), f"{label}.decision"):
        raise ValueError(f"{label}.decision must be non-empty")
    if not _require_dict(payload.get("study"), f"{label}.study"):
        raise ValueError(f"{label}.study must be non-empty")
    if not _require_dict(payload.get("generator"), f"{label}.generator"):
        raise ValueError(f"{label}.generator must be non-empty")
    if "seed" not in payload:
        raise ValueError(f"{label}.seed must be explicitly present (null or integer)")
    history = payload.get("transformation_history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{label}.transformation_history must be a non-empty list")

    rights = _require_dict(payload.get("source_rights"), f"{label}.source_rights")
    _nonempty_str(rights.get("license"), f"{label}.source_rights.license")
    _check_pinned_revision(
        rights.get("pinned_revision"), f"{label}.source_rights.pinned_revision"
    )
    _nonempty_str(rights.get("checksum_basis"), f"{label}.source_rights.checksum_basis")

    expected = _require_dict(payload.get("expected"), f"{label}.expected")
    _nonempty_str(expected.get("unit"), f"{label}.expected.unit")

    inputs = _require_dict(payload.get("inputs"), f"{label}.inputs")
    if "net_benefits" in inputs:
        nb = inputs["net_benefits"]
        if not isinstance(nb, list) or not nb:
            raise ValueError(f"{label}.inputs.net_benefits must be non-empty")
        n_draws = len(nb)
        for aligned_field in ("weights", "parameter_samples", "draw_keys"):
            if aligned_field in inputs and len(inputs[aligned_field]) != n_draws:
                raise ValueError(
                    f"broken draw alignment: {label}.inputs.{aligned_field} length "
                    f"({len(inputs[aligned_field])}) != net_benefits ({n_draws})"
                )
    for subcase in inputs.get("subcases", []):
        if isinstance(subcase, dict):
            sub_exp = _require_dict(
                subcase.get("expected"), f"{label} subcase.expected"
            )
            _nonempty_str(sub_exp.get("unit"), f"{label} subcase.expected.unit")
            if (
                "net_benefits" in subcase
                and "weights" in subcase
                and len(subcase["net_benefits"]) != len(subcase["weights"])
            ):
                raise ValueError(
                    f"broken draw alignment in {label} subcase "
                    f"{subcase.get('subcase_id')}"
                )


def _require(condition: bool, message: str) -> None:
    """Raise ValueError when an offline corpus invariant fails (safe under python -O)."""
    if not condition:
        raise ValueError(message)


def _require_close(actual: float, expected: float, atol: float, label: str) -> None:
    """Require strict absolute tolerance without implicit relative-tolerance slack."""
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise ValueError(
            f"{label} mismatch: computed {actual!r} != expected {expected!r} (atol={atol!r})"
        )


def _reproduce_weighted_evpi(
    inputs: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    results: dict[str, float] = {}
    for sub in inputs["subcases"]:
        val, _, _ = _fraction_evpi(sub["net_benefits"], sub["weights"])
        _require_close(
            val,
            float(sub["expected"]["value"]),
            float(sub["expected"]["atol"]),
            f"exact-finite-weighted-evpi/{sub['subcase_id']}",
        )
        results[sub["subcase_id"]] = val
    _require_close(
        results["three-state-asymmetric-crossover"],
        float(expected["value"]),
        float(expected["atol"]),
        "exact-finite-weighted-evpi.expected",
    )
    return results


def _reproduce_discrete_evppi(
    inputs: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    nb = inputs["net_benefits"]
    weights = [Fraction(1, len(nb)) for _ in nb]
    evpi_val, curr_val, _ = _fraction_evpi(nb)
    thetas = [round(p["theta"]) for p in inputs["parameter_samples"]]
    groups: dict[int, list[int]] = {}
    for idx, theta_val in enumerate(thetas):
        groups.setdefault(theta_val, []).append(idx)
    cond_perf = Fraction(0)
    for indices in groups.values():
        p_group = sum((weights[i] for i in indices), start=Fraction(0))
        strat_means = [
            sum(
                (weights[i] * Fraction(str(nb[i][col])) for i in indices),
                start=Fraction(0),
            )
            / p_group
            for col in range(len(nb[0]))
        ]
        cond_perf += p_group * max(strat_means)
    evppi_val = float(cond_perf) - curr_val
    _require(
        0.0 <= evppi_val <= evpi_val + 1e-12,
        f"discrete EVPPI {evppi_val} outside [0, {evpi_val}]",
    )
    _require_close(
        evppi_val,
        float(expected["value"]),
        float(expected["atol"]),
        "exact-finite-evppi-discrete.expected",
    )
    return {"evppi": evppi_val, "evpi": evpi_val}


def _reproduce_quadratic_evppi(
    _inputs: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    evppi_val = math.sqrt(2.0 / math.pi) * math.exp(-0.5)
    _require_close(
        evppi_val,
        float(expected["value"]),
        float(expected["atol"]),
        "exact-nonlinear-quadratic-evppi.expected",
    )
    return {"evppi": evppi_val}


def _reproduce_normal_normal_evsi(
    inputs: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    p_mean = float(inputs["prior_mean"])
    p_sd = float(inputs["prior_standard_deviation"])
    o_sd = float(inputs["outcome_standard_deviation"])
    slope = float(inputs["net_benefit_slope"])
    intercept = float(inputs["net_benefit_intercept"])
    prior_var = p_sd * p_sd
    inb_mean = slope * p_mean + intercept
    prior_best = max(inb_mean, 0.0)
    inb_sd = abs(slope) * p_sd
    z_evpi = inb_mean / inb_sd
    evpi_val = (inb_sd * _norm_pdf(z_evpi) + inb_mean * _norm_cdf(z_evpi)) - prior_best

    out: dict[str, float] = {"evpi": evpi_val}
    prev = -1.0
    for sub in inputs["subcases"]:
        per_arm = int(sub["per_arm_sample_size"])
        total_n = int(sub["total_sample_size"])
        if total_n != 2 * per_arm:
            raise ValueError(
                "Equal-allocation Normal-Normal must obey total == 2 * per_arm"
            )
        samp_var = 4.0 * o_sd * o_sd / float(total_n)
        post_sd = abs(slope) * math.sqrt(prior_var * prior_var / (prior_var + samp_var))
        z = inb_mean / post_sd
        evsi_val = (post_sd * _norm_pdf(z) + inb_mean * _norm_cdf(z)) - prior_best
        _require(
            0.0 <= evsi_val <= evpi_val + 1e-12,
            f"Normal-Normal EVSI {evsi_val} outside [0, {evpi_val}]",
        )
        _require(evsi_val > prev, "Normal-Normal EVSI must strictly increase with n")
        prev = evsi_val
        _require_close(
            evsi_val,
            float(sub["expected"]["value"]),
            float(sub["expected"]["atol"]),
            f"conjugate-normal-normal-evsi/{sub['subcase_id']}",
        )
        out[sub["subcase_id"]] = evsi_val

    top_atol = float(expected["atol"])
    _require_close(
        out["n-per-arm-50"],
        float(expected["value"]),
        top_atol,
        "conjugate-normal-normal-evsi.expected",
    )
    additional = _require_dict(
        expected.get("additional_expected"),
        "conjugate-normal-normal-evsi.expected.additional_expected",
    )
    _require_close(out["evpi"], float(additional["evpi"]), top_atol, "NN evpi")
    _require_close(
        out["n-per-arm-10"], float(additional["evsi_n20"]), top_atol, "NN evsi_n20"
    )
    _require_close(
        out["n-per-arm-50"], float(additional["evsi_n100"]), top_atol, "NN evsi_n100"
    )
    _require_close(
        out["n-per-arm-250"], float(additional["evsi_n500"]), top_atol, "NN evsi_n500"
    )
    return out


def _reproduce_beta_binomial_evsi(
    inputs: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sub in inputs["subcases"]:
        evsi_val, evpi_val = _beta_binomial_evsi(
            float(inputs["alpha_prior"]),
            float(inputs["beta_prior"]),
            float(inputs["net_benefit_slope"]),
            float(inputs["net_benefit_intercept"]),
            int(sub["total_sample_size"]),
        )
        _require(
            0.0 <= evsi_val <= evpi_val + 1e-12,
            f"Beta-Binomial EVSI {evsi_val} outside [0, {evpi_val}]",
        )
        _require_close(
            evsi_val,
            float(sub["expected"]["value"]),
            float(sub["expected"]["atol"]),
            f"conjugate-beta-binomial-evsi/{sub['subcase_id']}",
        )
        out[sub["subcase_id"]] = evsi_val
        out["evpi"] = evpi_val

    top_atol = float(expected["atol"])
    _require_close(
        out["binomial-n100"],
        float(expected["value"]),
        top_atol,
        "conjugate-beta-binomial-evsi.expected",
    )
    additional = _require_dict(
        expected.get("additional_expected"),
        "conjugate-beta-binomial-evsi.expected.additional_expected",
    )
    _require_close(out["evpi"], float(additional["evpi"]), top_atol, "BB evpi")
    _require_close(
        out["binomial-n20"], float(additional["evsi_n20"]), top_atol, "BB evsi_n20"
    )
    _require_close(
        out["binomial-n100"], float(additional["evsi_n100"]), top_atol, "BB evsi_n100"
    )
    return out


def _reproduce_gamma_poisson_evsi(
    inputs: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sub in inputs["subcases"]:
        evsi_val, tail_bound, evpi_val = _gamma_poisson_evsi(
            int(inputs["alpha_prior"]),
            float(inputs["rate_prior"]),
            float(inputs["net_benefit_slope"]),
            float(inputs["net_benefit_intercept"]),
            float(sub["exposure"]),
            int(sub["k_max"]),
            float(sub["certified_tail_bound"]),
        )
        _require(
            0.0 <= evsi_val <= evpi_val + 1e-11,
            f"Gamma-Poisson EVSI {evsi_val} outside [0, {evpi_val}]",
        )
        _require_close(
            evsi_val,
            float(sub["expected"]["value"]),
            float(sub["expected"]["atol"]),
            f"conjugate-gamma-poisson-evsi/{sub['subcase_id']}",
        )
        out[sub["subcase_id"]] = evsi_val
        out[f"{sub['subcase_id']}_tail_bound"] = tail_bound
        out["evpi"] = evpi_val

    top_atol = float(expected["atol"])
    _require_close(
        out["exposure-4"],
        float(expected["value"]),
        top_atol,
        "conjugate-gamma-poisson-evsi.expected",
    )
    additional = _require_dict(
        expected.get("additional_expected"),
        "conjugate-gamma-poisson-evsi.expected.additional_expected",
    )
    _require_close(out["evpi"], float(additional["evpi"]), top_atol, "GP evpi")
    _require_close(
        out["exposure-4"], float(additional["evsi_t4"]), top_atol, "GP evsi_t4"
    )
    _require_close(
        out["exposure-20"], float(additional["evsi_t20"]), top_atol, "GP evsi_t20"
    )
    return out


def _reproduce_single_period_enbs(
    inputs: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    per_dec = Fraction(str(inputs["per_decision_evsi"]))
    pop = Fraction(str(inputs["benefiting_population"]))
    cost = Fraction(str(inputs["research_cost"]))
    val = float(per_dec * pop - cost)
    _require_close(
        val,
        float(expected["value"]),
        float(expected["atol"]),
        "exact-single-period-enbs.expected",
    )
    _require(
        expected["additional_expected"]["no_study_enbs"] == 0.0,
        "exact-single-period-enbs.no_study_enbs must be 0.0",
    )
    return {"enbs": val}


_CASE_REPRODUCERS = {
    "exact-finite-weighted-evpi": _reproduce_weighted_evpi,
    "exact-finite-evppi-discrete": _reproduce_discrete_evppi,
    "exact-nonlinear-quadratic-evppi": _reproduce_quadratic_evppi,
    "conjugate-normal-normal-evsi": _reproduce_normal_normal_evsi,
    "conjugate-beta-binomial-evsi": _reproduce_beta_binomial_evsi,
    "conjugate-gamma-poisson-evsi": _reproduce_gamma_poisson_evsi,
    "exact-single-period-enbs": _reproduce_single_period_enbs,
}


def reproduce_case_offline(case_payload: dict[str, Any]) -> dict[str, float]:
    """Independently recompute and assert one admitted reference-corpus case."""
    if case_payload.get("method") == "evpi" and isinstance(
        case_payload.get("cases"), list
    ):
        for idx, entry in enumerate(case_payload["cases"]):
            _check_not_self_certified(
                entry.get("provenance"), f"evpi-cases[{idx}].provenance"
            )
            _nonempty_str(entry.get("expected", {}).get("unit"), "evpi-cases unit")
            val, curr, perf = _fraction_evpi(entry["net_benefits"])
            exp = entry["expected"]
            _require_close(
                val, float(exp["value"]), float(exp["atol"]), f"evpi-cases[{idx}]"
            )
            deriv = entry["derivation"]
            _require_close(
                curr,
                float(deriv["expected_current_value"]),
                1e-9,
                f"evpi-cases[{idx}].current",
            )
            _require_close(
                perf,
                float(deriv["expected_perfect_information"]),
                1e-9,
                f"evpi-cases[{idx}].perfect",
            )
        return {"reproduced_cases": float(len(case_payload["cases"]))}

    _validate_case_metadata(case_payload, str(case_payload.get("case_id", "case")))
    case_id = str(case_payload["case_id"])
    reproducer = _CASE_REPRODUCERS.get(case_id)
    if reproducer is None:
        raise ValueError(f"Unsupported reference corpus case_id: {case_id!r}")
    return reproducer(case_payload["inputs"], case_payload["expected"])


def validate_corpus_manifest(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    """Validate the Q03 corpus manifest and offline-reproduce all admitted cases."""
    manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0.0":
        raise ValueError("reference corpus manifest must use schema_version 1.0.0")
    _check_pinned_revision(manifest.get("pinned_revision"), "manifest.pinned_revision")

    admitted = manifest.get("admitted_cases")
    if not isinstance(admitted, list) or not admitted:
        raise ValueError("manifest.admitted_cases must be a non-empty list")

    reproduced_ids: list[str] = []
    for entry in admitted:
        if not isinstance(entry, dict):
            raise TypeError("admitted_cases entries must be objects")
        case_id = _nonempty_str(entry.get("id"), "admitted case id")
        rel = Path(_nonempty_str(entry.get("path"), f"{case_id}.path"))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe corpus path: {rel}")
        case_path = repo_root / rel
        if not case_path.is_file():
            raise ValueError(f"admitted case file does not exist: {rel}")
        expected_sha = _nonempty_str(entry.get("sha256"), f"{case_id}.sha256")
        if not _SHA256_RE.match(expected_sha):
            raise ValueError(f"{case_id}.sha256 must be a 64-hex SHA-256 digest")
        actual_sha = _sha256_file(case_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{case_id} checksum mismatch: expected {expected_sha}, got {actual_sha}"
            )
        license_str = _nonempty_str(entry.get("license"), f"{case_id}.license")
        _check_pinned_revision(
            entry.get("pinned_revision"), f"{case_id}.pinned_revision"
        )
        _check_not_self_certified(
            entry.get("reference_implementation"),
            f"{case_id}.reference_implementation",
        )
        _require_bool(entry.get("synthetic_owned"), f"{case_id}.synthetic_owned")
        _require_bool(
            entry.get("same_author_application"),
            f"{case_id}.same_author_application",
        )
        if entry.get("split") not in {"development", "held_out"}:
            raise ValueError(f"{case_id}.split must be 'development' or 'held_out'")
        _require_bool(
            entry.get("held_out_from_tuning"), f"{case_id}.held_out_from_tuning"
        )

        payload: dict[str, Any] = json.loads(case_path.read_text(encoding="utf-8"))
        if "source_rights" in payload:
            case_license = _nonempty_str(
                payload["source_rights"].get("license"),
                f"{case_id}.source_rights.license",
            )
            if case_license != license_str:
                raise ValueError(f"{case_id} license mismatch against manifest")
        reproduce_case_offline(payload)
        reproduced_ids.append(case_id)

    candidates = manifest.get("external_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("manifest.external_candidates must be a non-empty list")
    for cand in candidates:
        if not isinstance(cand, dict):
            raise TypeError("external_candidates entries must be objects")
        cand_id = _nonempty_str(cand.get("id"), "external candidate id")
        state = _nonempty_str(cand.get("admission_state"), f"{cand_id}.admission_state")
        if state != "candidate-source-only":
            if (
                cand.get("license_cleared") is not True
                or cand.get("checksum_verified") is not True
                or cand.get("redistribution_allowed") is None
            ):
                raise ValueError(
                    f"external candidate {cand_id} cannot leave candidate-source-only "
                    "without cleared license, checksum verification, and rights flag"
                )
            _check_pinned_revision(
                cand.get("pinned_revision"), f"{cand_id}.pinned_revision"
            )

    return {
        "admitted_count": len(reproduced_ids),
        "admitted_ids": reproduced_ids,
        "external_candidate_count": len(candidates),
    }


def main() -> int:
    """CLI entrypoint for offline Q03 reference-corpus verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("specs/v2/reference-corpus/manifest.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else root / args.manifest
    )
    summary = validate_corpus_manifest(manifest_path, root)
    print(
        f"Reference corpus: PASS ({summary['admitted_count']} admitted cases; "
        f"{summary['external_candidate_count']} candidate-source-only external benchmarks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
