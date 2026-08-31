# SPDX-License-Identifier: Apache-2.0
from spack_repo.builtin.build_systems.python import PythonPackage
from spack.package import depends_on, license, version


class PyAnnotatedDoc(PythonPackage):
    """Documentation metadata for Python annotations."""

    homepage = "https://github.com/fastapi/annotated-doc"
    pypi = "annotated-doc/annotated_doc-0.0.5.tar.gz"
    license("MIT")
    version("0.0.5", sha256="c7e58ce09192557605d8bbd92836d7e1d520ac9580096042c0bfd197efacf1bb")
    depends_on("python@3.9:", type=("build", "run"))
    depends_on("py-pdm-backend", type="build")
