# SPDX-License-Identifier: Apache-2.0
"""Source-bound native runtime for the current Polars frontend."""

from spack_repo.builtin.build_systems.python import PythonPackage
from spack.package import EnvironmentModifications, depends_on, license, version


class PyPolarsRuntime32(PythonPackage):
    """Preserve upstream full/nightly defaults and the dated compiler."""

    homepage = "https://pola.rs"
    pypi = "polars-runtime-32/polars_runtime_32-2.1.0.tar.gz"
    license("MIT")
    version(
        "2.1.0",
        sha256="c575f5f3cdb8a5f9be65477b9b4202b062593687fe62a0082add87fbd4571067",
        url="https://files.pythonhosted.org/packages/e0/a6/46e09af693326735b1fba3a8bf5a725b57136fb1342e707e75d3113c826b/polars_runtime_32-2.1.0.tar.gz",
    )
    depends_on("python@3.15.1", type=("build", "run"))
    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("py-maturin@1.9:1", type="build")
    depends_on("rust@=nightly-2026-09-01", type="build")
    import_modules = ["_polars_runtime_32"]

    def setup_build_environment(self, env: EnvironmentModifications) -> None:
        """Use the selected Spack compiler, without rustup fallback downloads."""
        env.set("RUSTC", str(self.spec["rust"].prefix.bin.rustc))
        env.set("CARGO", str(self.spec["rust"].prefix.bin.cargo))
