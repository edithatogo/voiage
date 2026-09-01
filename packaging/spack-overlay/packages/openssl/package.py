# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""Preserve the pinned catalogue build with a security-maintained source."""

from spack.package import depends_on, version
from spack_repo.builtin.packages.openssl.package import Openssl as BuiltinOpenssl


class Openssl(BuiltinOpenssl):
    """Security-maintained source with the pinned catalogue's build semantics."""

    version(
        "3.6.4",
        sha256="9bffaa1ad1e07b354c21bd3324ec02fa15579f45a7d0494b3e74bc449b7333ef",
    )
    depends_on("c", type="build")
