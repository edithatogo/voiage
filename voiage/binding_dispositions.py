"""Polyglot ABI and Binding Parity: Industry Decision Contracts (#579).

This module exposes the disposition of industry decision contracts (DecisionProblem,
Decision Cards, Enterprise Adapters, and Domain Templates) across Rust, Python,
R, Julia, and Mojo.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from voiage.exceptions import raise_input_error, raise_value_error

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST_PATH = (
    _ROOT / "specs" / "abi" / "industry-decision-binding-dispositions.json"
)


@dataclass(frozen=True)
class LanguageBindingDisposition:
    """Disposition of a specific contract in a target language environment.

    Attributes
    ----------
    language : str
        Target language ("python", "rust", "r", "julia", "mojo").
    status : str
        Implementation state ("implemented", "contract_only", "adapter", "unsupported", "upstream_blocked").
    symbol : str
        Exported symbol, type, or schema reference.
    interchange : str
        Supported data interchange format (e.g. "json", "json_and_arrow", "c_abi").
    reason : str
        Explanatory rationale for unsupported or upstream-blocked states.
    """

    language: str
    status: str
    symbol: str = ""
    interchange: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ContractBindingParity:
    """Binding parity record for a canonical industry decision contract."""

    contract_id: str
    schema_path: str
    dispositions: dict[str, LanguageBindingDisposition]

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract parity record to dictionary."""
        return {
            "contract_id": self.contract_id,
            "schema_path": self.schema_path,
            "dispositions": {
                lang: {
                    "status": disp.status,
                    "symbol": disp.symbol,
                    "interchange": disp.interchange,
                    "reason": disp.reason,
                }
                for lang, disp in self.dispositions.items()
            },
        }


def load_industry_decision_binding_dispositions(
    manifest_path: Path | None = None,
) -> dict[str, ContractBindingParity]:
    """Load and parse binding parity dispositions for all industry contracts."""
    path = manifest_path or _DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise_input_error(f"Binding dispositions manifest not found at {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    contracts_data = raw.get("contracts", {})

    results: dict[str, ContractBindingParity] = {}
    for cid, cdata in contracts_data.items():
        disps: dict[str, LanguageBindingDisposition] = {}
        for lang, ldata in cdata.get("dispositions", {}).items():
            disps[lang] = LanguageBindingDisposition(
                language=lang,
                status=str(ldata.get("status", "unsupported")),
                symbol=str(ldata.get("symbol", "")),
                interchange=str(ldata.get("interchange", "")),
                reason=str(ldata.get("reason", "")),
            )
        results[cid] = ContractBindingParity(
            contract_id=cid,
            schema_path=str(cdata.get("schema", "")),
            dispositions=disps,
        )

    return results


def get_binding_disposition(
    contract_id: str,
    language: str,
    manifest_path: Path | None = None,
) -> LanguageBindingDisposition:
    """Retrieve the disposition of a specific contract in a given language."""
    all_contracts = load_industry_decision_binding_dispositions(
        manifest_path=manifest_path
    )
    if contract_id not in all_contracts:
        raise_value_error(
            f"Contract '{contract_id}' not found in manifest. "
            f"Available: {list(all_contracts.keys())}"
        )

    contract = all_contracts[contract_id]
    if language not in contract.dispositions:
        raise_value_error(
            f"Language '{language}' not configured for contract '{contract_id}'. "
            f"Configured: {list(contract.dispositions.keys())}"
        )

    return contract.dispositions[language]


def validate_binding_dispositions_manifest(
    manifest_path: Path | None = None,
) -> bool:
    """Validate that the manifest has required keys and valid status values."""
    path = manifest_path or _DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise_input_error(f"Binding dispositions manifest not found at {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    valid_statuses = {
        "implemented",
        "contract_only",
        "adapter",
        "unsupported",
        "upstream_blocked",
    }
    valid_languages = {"python", "rust", "r", "julia", "mojo"}

    contracts = raw.get("contracts", {})
    if not contracts:
        raise_input_error("Manifest contains no contracts.")

    for cid, cdata in contracts.items():
        disps = cdata.get("dispositions", {})
        for lang, ldata in disps.items():
            if lang not in valid_languages:
                raise_input_error(f"Unknown language '{lang}' in contract '{cid}'")
            if ldata.get("status") not in valid_statuses:
                raise_input_error(
                    f"Invalid status '{ldata.get('status')}' for {lang} in {cid}"
                )

    return True
