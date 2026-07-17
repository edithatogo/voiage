from spack.package import PythonPackage


class PyVoiage(PythonPackage):
    """Draft Spack recipe for upstream review."""

    homepage = "https://github.com/edithatogo/voiage"
    git = "https://github.com/edithatogo/voiage.git"

    version("0.2.0", commit="653bd313af341cba65b84409b7abfb3af77f1407")

    depends_on("python@3.10:", type=("build", "run"))
    depends_on("py-numpy@1.20:2", type="run")
    depends_on("py-scipy@1.7:", type="run")
    depends_on("py-pandas@1.3:", type="run")
    depends_on("py-xarray@0.19:", type="run")
