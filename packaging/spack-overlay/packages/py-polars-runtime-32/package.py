# SPDX-License-Identifier: Apache-2.0
from spack_repo.builtin.build_systems.python import PythonPackage
from spack.package import depends_on, license, version


class PyPolarsRuntime32(PythonPackage):
    """Native runtime required by the split Polars Python distribution."""

    homepage = "https://pola.rs"
    pypi = "polars-runtime-32/polars_runtime_32-1.42.1.tar.gz"
    license("MIT")
    version("1.42.1", sha256="4d4809e1c1b9a6611f6944f27b24abea902b5159e6b6fa262fd716e947af5afd")
    depends_on("python@3.10:", type=("build", "run"))
    depends_on("py-maturin@1.3.2:", type="build")
    # Source rust-toolchain.toml: no unversioned nightly or stable substitution.
    depends_on("rust@nightly-2026-04-01", type="build")
    import_modules = ["_polars_runtime_32"]
