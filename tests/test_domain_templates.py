"""Tests for Curated Domain Template and Adapter Registry (#577)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiage.domain_templates import (
    DomainTemplate,
    get_domain_template,
    list_domain_templates,
    load_domain_template_registry,
    validate_domain_template_registry,
)
from voiage.exceptions import InputError


def test_domain_template_registry_json_schema_validation() -> None:
    assert validate_domain_template_registry() is True


def test_load_all_domain_templates_coverage() -> None:
    templates = load_domain_template_registry()
    assert len(templates) == 9

    template_ids = {t.template_id for t in templates}
    expected_ids = {
        "churn_retention",
        "dynamic_pricing",
        "marketing_measurement",
        "inventory_replenishment",
        "predictive_maintenance",
        "fraud_credit",
        "finance_data_vendor",
        "market_entry",
        "hr_attrition",
    }
    assert template_ids == expected_ids


def test_get_domain_template_lookup_and_not_found() -> None:
    churn = get_domain_template("churn_retention")
    assert churn.template_id == "churn_retention"
    assert churn.domain == "customer_success"
    assert "evpi" in churn.capabilities
    assert "base_churn" in churn.required_fields
    assert len(churn.decisions) == 3

    with pytest.raises(ValueError, match="not found in registry"):
        get_domain_template("non_existent_template")


def test_list_domain_templates_filtering() -> None:
    # Filter by domain
    cs_templates = list_domain_templates(domain="customer_success")
    assert len(cs_templates) == 1
    assert cs_templates[0].template_id == "churn_retention"

    # Filter by capability
    risk_templates = list_domain_templates(capability="risk_sensitive")
    assert len(risk_templates) >= 2
    risk_ids = {t.template_id for t in risk_templates}
    assert "inventory_replenishment" in risk_ids
    assert "fraud_credit" in risk_ids

    # Filter by maturity
    stable_templates = list_domain_templates(maturity="stable")
    assert len(stable_templates) >= 2
    stable_ids = {t.template_id for t in stable_templates}
    assert "churn_retention" in stable_ids
    assert "market_entry" in stable_ids


def test_domain_template_dataclass_round_trip() -> None:
    template = get_domain_template("market_entry")
    data = template.to_dict()
    assert data["template_id"] == "market_entry"
    assert data["domain"] == "business_strategy"

    reconstructed = DomainTemplate.from_dict(data)
    assert reconstructed == template


def test_domain_template_error_handling() -> None:
    with pytest.raises(InputError, match="must be a dictionary"):
        DomainTemplate.from_dict("not_a_dict")  # type: ignore[arg-type]

    non_existent_path = Path("specs/domain-templates/does_not_exist_registry.json")
    with pytest.raises(InputError, match="not found"):
        load_domain_template_registry(registry_path=non_existent_path)
