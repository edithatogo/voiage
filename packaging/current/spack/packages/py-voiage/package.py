"""Current-runtime candidate recipe; native resolution/build remains gated."""

from spack_repo.builtin.build_systems.python import PythonPackage
from spack.package import depends_on, license, version


class PyVoiage(PythonPackage):
    """Build the immutable migrated source candidate, never the old release sdist."""

    homepage = "https://github.com/edithatogo/voiage"
    git = "https://github.com/edithatogo/voiage.git"
    license("Apache-2.0")
    version("develop", commit="3df3cb49e5c927075c0609a48a2535b4457424d4")
    depends_on("python@3.15.1", type=("build", "run"))
    depends_on("rust@1.85:", type="build")
    depends_on("py-maturin@1.9:1", type="build")
    depends_on("py-click@8.3.3:8", type="run")
    depends_on("py-numpy@2.5.4:2", type="run")
    depends_on("py-scipy@1.18.1:1.18", type="run")
    depends_on("py-scikit-learn@1.9.1:1", type="run")
    depends_on("py-pyarrow@26.0.0:26", type="run")
    depends_on("py-polars@2.1.0:2", type="run")
    depends_on("py-pydantic@2.13.4:2", type="run")
    depends_on("py-jsonschema@4.26.0:4", type="run")
    depends_on("py-typing-extensions@4.16.0:", type="run")
    depends_on("py-typer@0.9:0", type="run")
    import_modules = ["voiage", "voiage._core"]
