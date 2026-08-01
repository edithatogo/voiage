# P10 reconciliation — 2026-08-01

This additive record reconciles the strict-local endpoint after PRs #716,
#719, #721, #751, and #754 merged onto `main`.

| PR | merge commit | hosted result |
| --- | --- | --- |
| #716 | `0507ce7138b966667e2518ad4c84c5a8ac8e02ac` | merged; required checks passed |
| #719 | `17f719f1d359e57fcc412a09770993535b5a43dd` | merged; required checks passed |
| #721 | `53eabb8583dff1645a91cdb3591f37930d6724b8` | merged; required checks passed |
| #751 | `e8aaba82848819e8aff2779356d95a01b3d376c7` | merged; required checks passed |
| #754 | `4919e8b94e1fd6ba7a04bf36808265975b2ed792` | merged; required checks passed |

The current endpoint is `4919e8b9`. The strict-local profile retains the
normalized contract, direct preparation, optional providers, offline Croissant
and Frictionless profiles, descriptor-only inspection, receipts, source-policy
enforcement, deterministic conformance fixtures, SDK/DataFrame contracts, and
synthetic ML, engineering/operations, and business examples.

General remote transport, DNS/redirect policy, archives, mutable live sources,
authoritative external probes, and third-party parser parity are successor
scope in [#752](https://github.com/edithatogo/voiage/issues/752) and
[#753](https://github.com/edithatogo/voiage/issues/753), linked to parent
[#325](https://github.com/edithatogo/voiage/issues/325). Their rights/authority
and security-policy gates remain pending and are not hidden gaps in this track.

P10-T4 remains active until final Conductor review records this bounded closure,
updates metadata/registry status, and makes the archive decision. No
publication, authenticated source acquisition, or security-policy relaxation
is implied.
