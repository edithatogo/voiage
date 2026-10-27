"""Enforce legacy-free source imports and a clean installed base package."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tokenize

FORBIDDEN = frozenset({"pandas", "xarray"})

# Executed in a fresh isolated interpreter, never the lint process or pytest.
IMPORT_PROBE = r"""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
assert sys.version_info[:3] == (3, 15, 1), sys.version
forbidden = {"pandas", "xarray"}
installed = {d.metadata["Name"].lower().replace("_", "-") for d in metadata.distributions()}
assert not (installed & forbidden), f"Forbidden installed distributions: {installed & forbidden}"
import voiage
loaded = sorted(name for name in sys.modules if name.split(".", 1)[0] in forbidden)
assert not loaded, f"Forbidden imported modules: {loaded}"
expected = Path(sys.argv[1]).resolve()
location = Path(voiage.__file__).resolve()
assert location.is_relative_to(expected), f"Source-shadowed package: {location}; expected {expected}"
assert metadata.version("voiage") == voiage.__version__
print(json.dumps({"python": sys.version.split()[0], "installed_package": str(location),
                  "version": voiage.__version__, "forbidden_distributions": [], "forbidden_modules": []}))
"""


def scan_source(source: Path) -> list[dict[str, object]]:
    """Reject direct legacy imports in every Python file, regardless of pragmas."""
    files = sorted(source.rglob("*.py")) if source.is_dir() else []
    if not files:
        raise ValueError(f"No Python sources found: {source}")
    violations = []
    for path in files:
        with tokenize.open(path) as handle:
            tree = ast.parse(handle.read(), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            violations.extend(
                {"path": str(path), "line": node.lineno, "module": module}
                for module in modules
                if module.split(".", 1)[0] in FORBIDDEN
            )
    return violations


def probe_installed(expected_prefix: Path) -> subprocess.CompletedProcess[str]:
    """Import only the installed wheel in a fresh, isolated Python subprocess."""
    return subprocess.run(  # noqa: S603 - fixed code and selected interpreter, no shell
        [sys.executable, "-I", "-c", IMPORT_PROBE, str(expected_prefix)],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def main() -> int:
    """Run the source lint or the no-extras installed-package probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("voiage"))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--expected-prefix", type=Path)
    args = parser.parse_args()
    if args.probe:
        if args.expected_prefix is None:
            parser.error("--probe requires --expected-prefix")
        result = probe_installed(args.expected_prefix)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    violations = scan_source(args.source)
    print(json.dumps({"passed": not violations, "violations": violations}, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
