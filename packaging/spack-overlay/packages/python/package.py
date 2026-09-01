# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""Preserve the pinned catalogue build with a security-maintained source."""

from spack.package import depends_on, version
from spack_repo.builtin.packages.python.package import Python as BuiltinPython


class Python(BuiltinPython):
    """Security-maintained source with the pinned catalogue's build semantics."""

    version(
        "3.12.14",
        sha256="6c6df908d2c3fd24e6d76869e92542abd0f33aec9dfc18df8875f89660286d43",
    )
    depends_on("c", type="build")
    depends_on("openssl@3.6.4:", when="@3.12.14: +ssl")
    depends_on("expat@2.8.3:", when="@3.12.14: +pyexpat")
