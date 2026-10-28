# SPDX-License-Identifier: Apache-2.0
"""Current PyArrow frontend with explicit native I/O capabilities."""

from spack_repo.builtin.build_systems.python import PythonPackage
from spack.package import depends_on, license, version


class PyPyarrow(PythonPackage):
    """Retain the upstream backend and the current core I/O/codec surface."""

    homepage = "https://arrow.apache.org"
    pypi = "pyarrow/pyarrow-26.0.0.tar.gz"
    license("Apache-2.0")
    version(
        "26.0.0",
        sha256="0cccd36e00ea3afeb52ded61f2721ce71f604853d70c45365c58324eb773d6ae",
        url="https://files.pythonhosted.org/packages/ec/34/17c34cb38e5d940e38f0f0d9fdfa0e8a506676409ea9b85aff7e3079f831/pyarrow-26.0.0.tar.gz",
    )
    depends_on("python@3.15.1", type=("build", "run"))
    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("cmake@3.25:", type="build")
    depends_on("pkgconfig", type="build")
    depends_on("py-scikit-build-core@1:", type="build")
    depends_on("py-cython@3.1:", type="build")
    depends_on("py-libcst@1.9:", type="build")
    depends_on("py-numpy@2:", type="build")
    depends_on("py-setuptools-scm@8:+toml", type="build")
    depends_on(
        "arrow@=26.0.0+brotli+bz2+compute+csv+dataset+filesystem+ipc+lz4+mimalloc+parquet+python+shared+snappy+zlib+zstd",
        type=("build", "link", "run"),
    )
    import_modules = [
        "pyarrow",
        "pyarrow.compute",
        "pyarrow.csv",
        "pyarrow.dataset",
        "pyarrow.ipc",
        "pyarrow.parquet",
    ]
