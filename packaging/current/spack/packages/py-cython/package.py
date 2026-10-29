# Copyright Spack Project Developers. See COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Python 3.15 source provider for the Cython compiler."""

from spack.package import (
    EnvironmentModifications,
    Executable,
    depends_on,
    license,  # noqa: A004
    version,
)
from spack_repo.builtin.build_systems.python import PythonPackage


class PyCython(PythonPackage):
    """Build Cython with its compiled accelerators and regeneration hook."""

    homepage = "https://github.com/cython/cython"
    pypi = "cython/cython-3.3.0.tar.gz"
    tags = ["build-tools"]  # noqa: RUF012
    license("Apache-2.0")
    version(
        "3.3.0",
        sha256="eed0d93fbca7087f143b42c34b05a825849bdf17f101572c2105acfa49aa88b8",
        url="https://files.pythonhosted.org/packages/a9/d8/4981ef716ad0e3ff0d3ef383aefc6b03c4a88dee33b272bf8e0d833001ca/cython-3.3.0.tar.gz",
    )
    depends_on("python@3.15.1", type=("build", "link", "run"))
    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("py-setuptools@66:", type=("build", "run"))
    import_modules = ["Cython", "Cython.Compiler.Scanning"]  # noqa: RUF012

    @property
    def command(self):
        """Return the installed Cython executable for dependent packages."""
        return Executable(self.prefix.bin.cython)

    def setup_dependent_build_environment(
        self, env: EnvironmentModifications, dependent_spec
    ) -> None:
        """Regenerate bundled C files for the selected Python interpreter."""
        env.set("CYTHON_FORCE_REGEN", "1")
