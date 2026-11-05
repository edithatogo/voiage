# SPDX-License-Identifier: Apache-2.0
"""Source-bound LibCST build prerequisite for PyArrow on Python 3.15."""

from spack_repo.builtin.build_systems.python import PythonPackage
from spack.package import EnvironmentModifications, depends_on, license, version


class PyLibcst(PythonPackage):
    """Preserve the upstream native parser and Python-specific requirements."""

    homepage = "https://github.com/Instagram/LibCST"
    pypi = "libcst/libcst-1.9.0.tar.gz"
    license("MIT AND PSF-2.0")
    version(
        "1.9.0",
        sha256="087b58a9afe076bb08e2d726478e1f16cb928d67ffa9092817e033c335de522a",
        url="https://files.pythonhosted.org/packages/02/c0/098e5c91ff1537f00c85a6438b6cb1863d17144680cc91f47c87f104a200/libcst-1.9.0.tar.gz",
    )
    depends_on("python@3.15.1", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-setuptools-scm", type="build")
    depends_on("py-setuptools-rust", type="build")
    depends_on("py-wheel", type="build")
    depends_on("rust@1.85:1", type="build")
    depends_on("py-pyyaml@6.0.3:", type="run")
    import_modules = ["libcst", "libcst.native"]

    def setup_build_environment(self, env: EnvironmentModifications) -> None:
        """Use the selected stable compiler while retaining upstream features."""
        env.set("RUSTC", str(self.spec["rust"].prefix.bin.rustc))
        env.set("CARGO", str(self.spec["rust"].prefix.bin.cargo))
