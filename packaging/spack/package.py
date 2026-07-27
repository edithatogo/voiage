from spack.package import PythonPackage


class PyVoiage(PythonPackage):
    """Value of Information analysis with a Rust-native Python extension."""

    homepage = "https://github.com/edithatogo/voiage"
    git = "https://github.com/edithatogo/voiage.git"

    version("2.0.0", commit="5e92151fc87afefbb411c992fb9f82fc4b8c049f")

    depends_on("python@3.12:", type=("build", "run"))
    depends_on("rust", type="build")
    depends_on("py-maturin@1.9:1", type="build")
    depends_on("py-click@8.3:", type="run")
    depends_on("py-numpy@2.2:", type="run")
    depends_on("py-scipy@1.16:", type="run")
    depends_on("py-pandas@1.3:", type="run")
    depends_on("py-xarray@0.19:", type="run")
    depends_on("py-scikit-learn@1.7:", type="run")
    depends_on("py-pyarrow@25:", type="run")
    depends_on("py-polars@1.42:", type="run")
    depends_on("py-pydantic@2.13:", type="run")
    depends_on("py-typing-extensions@4.16:", type="run")
    depends_on("py-typer@0.9:", type="run")
