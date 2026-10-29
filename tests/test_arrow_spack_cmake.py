"""Exercise the Arrow recipe's CMake arguments without an HPC build."""

import importlib.util
from pathlib import Path
import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "packaging/current/spack/packages/arrow/package.py"
)


def test_arrow_26_shared_and_static_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    class Base:
        def cmake_args(self):
            return ["-DARROW_DEPENDENCY_SOURCE=SYSTEM"]

        def define(self, name, value):
            return f"-D{name}={value}"

        def define_from_variant(self, name, variant):
            return self.define(name, "ON" if self.shared else "OFF")

    package = types.ModuleType("spack.package")
    package.EnvironmentModifications = object
    package.depends_on = package.resource = package.version = lambda *args, **kwargs: (
        None
    )
    builtin = types.ModuleType("spack_repo.builtin.packages.arrow.package")
    builtin.Arrow = Base
    monkeypatch.setitem(sys.modules, "spack.package", package)
    monkeypatch.setitem(
        sys.modules, "spack_repo.builtin.packages.arrow.package", builtin
    )
    module_spec = importlib.util.spec_from_file_location("test_arrow_recipe", RECIPE)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    arrow = module.Arrow()

    class Spec:
        def __init__(self, current):
            self.current = current

        def satisfies(self, constraint):
            return constraint == "@=26.0.0" and self.current

    arrow.spec = Spec(True)
    for shared, expected in ((True, "ON"), (False, "OFF")):
        arrow.shared = shared
        args = arrow.cmake_args()
        assert f"-DARROW_BUILD_SHARED={expected}" in args
        assert "-DARROW_BUILD_STATIC=ON" in args
        assert "-DARROW_DEPENDENCY_SOURCE=SYSTEM" in args
    arrow.spec = Spec(False)
    assert arrow.cmake_args() == ["-DARROW_DEPENDENCY_SOURCE=SYSTEM"]


def test_hook_body_drift_is_rejected() -> None:
    import ast
    import hashlib
    from pathlib import Path

    from scripts import check_current_arrow_providers as checker

    source = Path("packaging/current/spack/packages/arrow/package.py").read_text()
    mutated = source.replace("args = super().cmake_args()", "args = []")
    assert mutated != source
    tree = ast.parse(mutated)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    digests = {
        n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest()
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }
    assert digests != checker._ARROW_HOOK_AST_SHA256
