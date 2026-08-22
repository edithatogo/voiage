"""Curated Domain Template and Adapter Registry for Enterprise VOI (#577).

This module provides typed access, schema validation, and discovery for curated
industry decision templates across customer success, pricing, marketing,
supply chain, maintenance, risk, finance, strategy, and people analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jsonschema

from voiage.exceptions import raise_input_error, raise_value_error

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REGISTRY_PATH = _ROOT / "specs" / "domain-templates" / "registry.json"
_DEFAULT_SCHEMA_PATH = (
    _ROOT
    / "specs"
    / "domain-templates"
    / "schemas"
    / "v1"
    / "domain-template.schema.json"
)


@dataclass(frozen=True)
class DomainTemplate:
    """Represents a validated industry decision template.

    Attributes
    ----------
    template_id : str
        Unique identifier for the domain template (snake_case).
    version : str
        Semantic version string.
    domain : str
        Business domain category.
    title : str
        Human-readable title.
    description : str
        Detailed description of decision context.
    decisions : list[str]
        List of candidate intervention strategies.
    information_actions : list[str]
        List of candidate data collection or experiment actions.
    required_fields : list[str]
        Required parameter or sample columns.
    capabilities : list[str]
        VOI capabilities supported (e.g., evpi, evppi, evsi, enbs).
    maturity : str
        Maturity disposition ("candidate", "experimental", "stable").
    rights : str
        Data rights and IP permissions statement.
    privacy : str
        Privacy classification.
    provenance : str
        Origin and methodological foundation.
    license : str
        Software license identifier.
    maintainers : list[str]
        Designated maintainers.
    compatibility : str
        voiage version constraint string.
    review_date : str
        ISO 8601 review date (YYYY-MM-DD).
    optional_dependencies : list[str]
        Optional package dependencies.
    examples : list[str]
        Paths to reference scripts or notebooks.
    """

    template_id: str
    version: str
    domain: str
    title: str
    description: str
    decisions: list[str]
    information_actions: list[str]
    required_fields: list[str]
    capabilities: list[str]
    maturity: str
    rights: str
    privacy: str
    provenance: str
    license: str
    maintainers: list[str]
    compatibility: str
    review_date: str
    optional_dependencies: list[str]
    examples: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainTemplate:
        """Instantiate DomainTemplate from raw dictionary."""
        if not isinstance(data, dict):
            raise_input_error("DomainTemplate data must be a dictionary.")
        return cls(
            template_id=str(data["template_id"]),
            version=str(data["version"]),
            domain=str(data["domain"]),
            title=str(data["title"]),
            description=str(data["description"]),
            decisions=list(data["decisions"]),
            information_actions=list(data["information_actions"]),
            required_fields=list(data["required_fields"]),
            capabilities=list(data["capabilities"]),
            maturity=str(data["maturity"]),
            rights=str(data.get("rights", "")),
            privacy=str(data.get("privacy", "")),
            provenance=str(data.get("provenance", "")),
            license=str(data.get("license", "Apache-2.0 OR MIT")),
            maintainers=list(data.get("maintainers", [])),
            compatibility=str(data.get("compatibility", ">=2.0.0")),
            review_date=str(data.get("review_date", "2026-08-22")),
            optional_dependencies=list(data.get("optional_dependencies", [])),
            examples=list(data.get("examples", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize DomainTemplate to a dictionary."""
        return {
            "template_id": self.template_id,
            "version": self.version,
            "domain": self.domain,
            "title": self.title,
            "description": self.description,
            "decisions": list(self.decisions),
            "information_actions": list(self.information_actions),
            "required_fields": list(self.required_fields),
            "optional_dependencies": list(self.optional_dependencies),
            "capabilities": list(self.capabilities),
            "maturity": self.maturity,
            "examples": list(self.examples),
            "rights": self.rights,
            "privacy": self.privacy,
            "provenance": self.provenance,
            "license": self.license,
            "maintainers": list(self.maintainers),
            "compatibility": self.compatibility,
            "review_date": self.review_date,
        }


def load_domain_template_registry(
    registry_path: Path | None = None,
) -> list[DomainTemplate]:
    """Load and parse all domain templates from the JSON registry manifest.

    Parameters
    ----------
    registry_path : Path, optional
        Custom path to registry.json file. Defaults to specs/domain-templates/registry.json.

    Returns
    -------
    list[DomainTemplate]
        List of parsed DomainTemplate objects.
    """
    path = registry_path or _DEFAULT_REGISTRY_PATH
    if not path.is_file():
        raise_input_error(f"Domain template registry not found at {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    templates_data = raw.get("templates", [])
    return [DomainTemplate.from_dict(item) for item in templates_data]


def list_domain_templates(
    domain: str | None = None,
    capability: str | None = None,
    maturity: str | None = None,
    registry_path: Path | None = None,
) -> list[DomainTemplate]:
    """Filter and discover domain templates.

    Parameters
    ----------
    domain : str, optional
        Filter by business domain (e.g. "customer_success", "pricing_revenue").
    capability : str, optional
        Filter by supported VOI capability (e.g. "evpi", "evsi", "enbs").
    maturity : str, optional
        Filter by maturity level ("candidate", "experimental", "stable").
    registry_path : Path, optional
        Path to registry.json file.

    Returns
    -------
    list[DomainTemplate]
        Filtered list of domain templates.
    """
    templates = load_domain_template_registry(registry_path=registry_path)

    if domain is not None:
        templates = [t for t in templates if t.domain == domain]
    if capability is not None:
        templates = [t for t in templates if capability in t.capabilities]
    if maturity is not None:
        templates = [t for t in templates if t.maturity == maturity]

    return templates


def get_domain_template(
    template_id: str, registry_path: Path | None = None
) -> DomainTemplate:
    """Retrieve a specific domain template by its template_id.

    Parameters
    ----------
    template_id : str
        Unique template ID.
    registry_path : Path, optional
        Path to registry.json file.

    Returns
    -------
    DomainTemplate
        The requested domain template.

    Raises
    ------
    ValueError
        If template_id is not found in the registry.
    """
    templates = load_domain_template_registry(registry_path=registry_path)
    for template in templates:
        if template.template_id == template_id:
            return template

    raise_value_error(
        f"Domain template '{template_id}' not found in registry. "
        f"Available: {[t.template_id for t in templates]}"
    )
    return None  # type: ignore[unreachable]


def validate_domain_template_registry(
    registry_path: Path | None = None,
    schema_path: Path | None = None,
) -> bool:
    """Validate all registry entries against the domain-template JSON schema.

    Parameters
    ----------
    registry_path : Path, optional
        Path to registry.json.
    schema_path : Path, optional
        Path to domain-template.schema.json.

    Returns
    -------
    bool
        True if all templates pass JSON schema validation.
    """
    r_path = registry_path or _DEFAULT_REGISTRY_PATH
    s_path = schema_path or _DEFAULT_SCHEMA_PATH

    raw_registry = json.loads(r_path.read_text(encoding="utf-8"))
    schema = json.loads(s_path.read_text(encoding="utf-8"))

    for template_dict in raw_registry.get("templates", []):
        jsonschema.validate(instance=template_dict, schema=schema)

    return True
