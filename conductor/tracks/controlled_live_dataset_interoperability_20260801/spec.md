# Specification: Controlled Live Standardized-Dataset Interoperability

## Overview

Provide opt-in interoperability probes for one authoritative Croissant and one
authoritative Frictionless dataset without expanding the strict-local provider
profile or general remote transport policy.

## Requirements

- Each probe MUST use a rights-cleared public descriptor and resource with
  pinned content digests, licence, citation, usage terms, and selection record.
- Network access MUST be opt-in, bounded, and disabled by default.
- A probe MUST create a materialization receipt and prove deterministic offline
  replay from the verified local materialization.
- The probe MUST preserve the existing canonical path:
  `provider -> NormalizedInputBundle -> prepare_analysis_inputs`.
- A source change, missing terms, unavailable content, or checksum mismatch
  MUST fail closed and never be silently substituted.

## Acceptance criteria

- **AC-01:** Approved source packet and use authority are recorded before I/O.
- **AC-02:** Croissant and Frictionless probes are explicit, digest-pinned,
  receipt-producing, and offline-replayable.
- **AC-03:** Probes demonstrate no alternate numerical or preparation path.
- **AC-04:** Documentation accurately limits support to the selected probes.

## External gates

Authoritative source identity, rights/terms, and explicit authority to retrieve
the selected resources are human/external gates. They are not satisfiable from
repository fixtures or CI alone.

## Out of scope

General remote sources, redirects, DNS resolution, archive extraction, mutable
datasets, credentials, and parser-library feature parity belong to #753 or a
future parser-specific track.
