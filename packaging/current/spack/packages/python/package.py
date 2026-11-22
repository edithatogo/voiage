# SPDX-License-Identifier: Apache-2.0
"""Checksum-bound CPython source for the current Spack overlay."""

from spack.package import version
from spack_repo.builtin.packages.python.package import Python as BuiltinPython


class Python(BuiltinPython):
    """Extend the pinned upstream recipe with the qualified Python release."""

    version(
        "3.15.2",
        sha256="8b5536d280142ecf6cdfe15d12a21ba1c65e300bbf7b28a8d0cdb849575797de",
        url="https://www.python.org/ftp/python/3.15.2/Python-3.15.2.tar.xz",
    )
