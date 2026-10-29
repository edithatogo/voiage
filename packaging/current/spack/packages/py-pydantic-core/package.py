# SPDX-License-Identifier: Apache-2.0
"""Source-bound pydantic-core provider for Python 3.15."""

from spack.package import EnvironmentModifications, depends_on, license, version  # noqa: A004
from spack_repo.builtin.build_systems.python import PythonPackage


class PyPydanticCore(PythonPackage):
    """Build the upstream native validator with the selected stable compiler."""

    homepage = "https://github.com/pydantic/pydantic-core"
    pypi = "pydantic_core/pydantic_core-2.46.4.tar.gz"
    license("MIT")
    version(
        "2.46.4",
        sha256="62f875393d7f270851f20523dd2e29f082bcc82292d66db2b64ea71f64b6e1c1",
        url="https://files.pythonhosted.org/packages/9d/56/921726b776ace8d8f5db44c4ef961006580d91dc52b803c489fafd1aa249/pydantic_core-2.46.4.tar.gz",
    )
    depends_on("python@3.15.1", type=("build", "run"))
    depends_on("rust@1.88:1", type="build")
    depends_on("py-maturin@1.10:1", type="build")
    depends_on("py-typing-extensions@4.14.1:", type="run")
    import_modules = ["pydantic_core", "pydantic_core._pydantic_core"]  # noqa: RUF012

    def setup_build_environment(self, env: EnvironmentModifications) -> None:
        """Select the stable compiler without changing upstream build features."""
        env.set("RUSTC", str(self.spec["rust"].prefix.bin.rustc))
        env.set("CARGO", str(self.spec["rust"].prefix.bin.cargo))
