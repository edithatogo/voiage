"""Project resolved Python/Cargo identities without claiming R/Julia coverage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote
import uuid


def project_inventory(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain exact supported identities and report the explicit coverage boundary."""
    source = json.loads(raw)
    if source.get("bomFormat") != "CycloneDX" or source.get("specVersion") != "1.6":
        raise ValueError("expected the complete CycloneDX 1.6 release inventory")
    components, excluded = [], []
    ecosystems: set[str] = set()
    for component in source.get("components", []):
        purl = component.get("purl", "")
        if purl.startswith(("pkg:cran/", "pkg:julia/")):
            excluded.append(
                {
                    "name": component["name"],
                    "purl": purl,
                    "reason": "R/Julia are outside this resolved vulnerability inventory",
                }
            )
            continue
        if not purl.startswith(("pkg:pypi/", "pkg:cargo/")):
            raise ValueError(
                "unrecognized component requires an explicit coverage decision"
            )
        version = component.get("version")
        identity = purl.split("?", 1)[0].split("#", 1)[0]
        if (
            not isinstance(version, str)
            or not version
            or re.search(r"[\s~^<>=*|,\[\]()]", version)
            or "@" not in identity
            or unquote(identity.rsplit("@", 1)[1]) != version
        ):
            raise ValueError("supported component lacks an exact resolved version")
        ecosystems.add(purl.split("/", 1)[0])
        components.append(copy.deepcopy(component))
    if ecosystems != {"pkg:pypi", "pkg:cargo"}:
        raise ValueError("expected both resolved Python and Cargo package inventories")
    parent_digest = hashlib.sha256(raw).hexdigest()
    metadata = copy.deepcopy(source["metadata"])
    metadata["properties"] = [
        item
        for item in metadata.get("properties", [])
        if item["name"].startswith("voiage:source:")
    ] + [
        {"name": "voiage:grype:scope", "value": "resolved-python-and-cargo-only"},
        {"name": "voiage:grype:parent-sbom-sha256", "value": parent_digest},
        {"name": "voiage:grype:excluded-ecosystems", "value": "r,julia"},
    ]
    references = {item["bom-ref"] for item in components}
    references.add(metadata["component"]["bom-ref"])
    dependencies = [
        {
            "ref": entry["ref"],
            "dependsOn": [
                reference
                for reference in entry.get("dependsOn", [])
                if reference in references
            ],
        }
        for entry in source.get("dependencies", [])
        if entry["ref"] in references
    ]
    result = {
        "$schema": source["$schema"],
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'grype:' + parent_digest)}",
        "metadata": metadata,
        "components": components,
        "dependencies": dependencies,
    }
    receipt = {
        "parent_sbom_sha256": parent_digest,
        "included_components": len(components),
        "excluded_components": excluded,
    }
    return result, receipt


def main() -> None:
    """Write a separately scoped inventory and its coverage receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("the full release inventory must remain unchanged")
    inventory, receipt = project_inventory(args.source.read_bytes())
    for path, document in ((args.output, inventory), (args.coverage, receipt)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
