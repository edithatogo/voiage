"""Ensure every methods module has an explicit v1 extension disposition."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]


def test_import_voiage_does_not_eagerly_load_optional_extensions() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, voiage; "
                "assert 'voiage.backends' not in sys.modules; "
                "assert not any(name.startswith('voiage.methods.') and "
                "name.rsplit('.', 1)[-1] in {'ai_assisted_evidence_triage', 'perspective'} "
                "for name in sys.modules)"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


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
    assert (
        "Rust owns stable numerical policy"
        in policy["dispositions"]["stable_kernel_facade"]
    )


def test_extension_surface_policy_covers_every_runtime_python_file() -> None:
    policy = json.loads(
        (ROOT / "specs/v1/extension-surface-policy.json").read_text(encoding="utf-8")
    )
    files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "voiage").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    patterns = policy["patterns"]
    for file in files:
        matches = [
            category
            for category, category_patterns in patterns.items()
            if any(fnmatch.fnmatch(file, pattern) for pattern in category_patterns)
        ]
        assert len(matches) == 1, f"{file} has {len(matches)} extension dispositions"
