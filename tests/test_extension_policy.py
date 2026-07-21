"""Ensure every methods module has an explicit v1 extension disposition."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_extension_policy_covers_every_methods_module() -> None:
    policy = json.loads(
        (ROOT / "specs/v1/extension-policy.json").read_text(encoding="utf-8")
    )
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "voiage/methods").glob("*.py")
        if path.name != "__init__.py"
    }
    declared = set(policy["modules"]) | set(policy["stable_kernel_facades"])
    assert declared == actual
    assert set(policy["stable_kernel_facades"]).isdisjoint(set(policy["modules"]))
    assert set(policy["modules"].values()) == {"optional_extension", "experimental"}


def test_stable_kernel_facades_are_explicitly_rust_owned() -> None:
    policy = json.loads(
        (ROOT / "specs/v1/extension-policy.json").read_text(encoding="utf-8")
    )
    assert set(policy["stable_kernel_facades"]) == {
        "voiage/methods/basic.py",
        "voiage/methods/ceaf.py",
        "voiage/methods/dominance.py",
        "voiage/methods/sample_information.py",
    }
    assert "Rust owns stable numerical policy" in policy["dispositions"]["stable_kernel_facade"]
