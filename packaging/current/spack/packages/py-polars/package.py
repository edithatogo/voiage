# SPDX-License-Identifier: Apache-2.0
"""Current core frontend for the upstream split Polars distribution."""

from spack_repo.builtin.build_systems.python import PythonPackage
from spack.package import depends_on, license, version


class PyPolars(PythonPackage):
    """Install the pure Python frontend with its exact native runtime."""

    homepage = "https://pola.rs"
    pypi = "polars/polars-2.1.0.tar.gz"
    license("MIT")
    version(
        "2.1.0",
        sha256="9eee488d6e10773c028753dfc8dac647fd60a708108dcb818a733e85d33df253",
        url="https://files.pythonhosted.org/packages/b9/da/c27daf205a824f89f71172a75ad0098c5837a5a31ddeaac7f8737ea8574b/polars-2.1.0.tar.gz",
    )
    depends_on("python@3.15.1", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-polars-runtime-32@=2.1.0", type="run")
    import_modules = ["polars"]
