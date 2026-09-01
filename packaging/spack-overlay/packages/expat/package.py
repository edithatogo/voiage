# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""Preserve the pinned catalogue build with a security-maintained source."""

from spack.package import depends_on, version
from spack_repo.builtin.packages.expat.package import Expat as BuiltinExpat


class Expat(BuiltinExpat):
    """Security-maintained source with the pinned catalogue's build semantics."""

    version(
        "2.8.3",
        sha256="b4cc2483927d5e90bf8c40b44a6b95b368b42a8a96e25883fce188b48a92b670",
    )
    depends_on("c", type="build")
