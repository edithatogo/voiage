from spack_repo.builtin.build_systems.python import PythonPackage
from spack.package import depends_on, license, version


class PyVoiage(PythonPackage):
    """Value of Information analysis with a Rust-native Python extension."""

    homepage = "https://github.com/edithatogo/voiage"
    pypi = "voiage/voiage-2.2.0.tar.gz"

    license("Apache-2.0")
    version("2.2.0", sha256="e4edfd41011891a94cbc2b144ff1d20340fcc32481e7a2b24157494b7490a16b")

    depends_on("python@3.12:3.14", type=("build", "run"))
    depends_on("rust@1.85:", type="build")
    depends_on("py-maturin@1.9:1", type="build")
    depends_on("py-click@8.5:8", type="run")
    depends_on("py-numpy@2.2.6:2", type="run")
    depends_on("py-scipy@1.16.3:1.16", type="run")
    depends_on("py-pandas@1.3:2", type="run")
    depends_on("py-xarray@0.19:2024", type="run")
    depends_on("py-scikit-learn@1.7.2:1", type="run")
    depends_on("py-pyarrow@25:25", type="run")
    depends_on("py-polars@1.42.1:1", type="run")
    depends_on("py-pydantic@2.13.4:2", type="run")
    depends_on("py-jsonschema@4.26:4", type="run")
    depends_on("py-typing-extensions@4.16:", type="run")
    depends_on("py-typer@0.27.2:0", type="run")

    import_modules = ["voiage", "voiage._core"]
