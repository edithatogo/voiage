# C14 design: Assurance Frontier

## Architecture

```mermaid
flowchart LR
    subgraph Producer["VOP producer boundary"]
        Models["Typed domain models"] --> Generator["Deterministic bundle generator"]
        References["Independent analytical references"] --> Generator
        Generator --> Bundle["Versioned schemas + Arrow identity + fixtures"]
        Bundle --> Manifest["SHA-256 manifest + provenance"]
    end

    subgraph Consumer["VOIAGE independent consumer"]
        Pin["Pinned producer commit + aggregate digest"] --> Verify["Fail-closed verifier"]
        Manifest --> Verify
        Verify --> Differential["JSON / Arrow / Parquet and NumPy / JAX checks"]
        Verify --> Bindings["Available native binding conformance"]
    end

    subgraph Assurance["Continuous assurance"]
        Stable["Frozen stable lane"] --> Promotion["Evidence-based promotion"]
        Frontier["Isolated experimental frontier lane"] --> Promotion
        Mutation["Mutation ratchets"] --> Evidence["Retained evidence"]
        Perf["CPU / memory / allocation / serialization / Scalene"] --> Evidence
        Supply["Reproducible build / SBOM / audit / release attestation"] --> Evidence
        Logs["Correlated redacted analysis logs"] --> Evidence
    end

    subgraph Governance["Human-governed reconciliation"]
        Drift["Read-only issue and Project drift audit"] --> Proposal["Approval-ready artifact"]
        Proposal --> Human{"Human approval"}
        Human -->|approved external action| Remote["GitHub issue / Project"]
    end

    Differential --> Evidence
    Bindings --> Evidence
    Promotion --> Evidence
    Evidence --> Drift
```

## Trust boundaries and invariants

1. Bundle identity is content-addressed; paths, symlinks, inventory and producer
   provenance are verified before data is parsed.
2. VOIAGE consumes copied, pinned artifacts only. It never imports VOP runtime
   code, so producer and consumer tests remain genuinely differential.
3. Experimental dependencies and optional accelerators are isolated from the
   frozen stable lane and cannot convert missing capability into a false pass.
4. Pull requests use read-only permissions. OIDC and artifact attestation exist
   only on the release path, which remains human-triggered by a signed tag or an
   explicitly selected existing release tag.
5. Logging context owns reserved correlation fields and recursively redacts
   credential-shaped values before human or JSON output.
6. Governance audits never issue mutations. Missing Project credentials produce
   `not_checked`, not `clean`; reconciliation remains a human decision.

## Failure model

Contract mismatch, unsupported provenance, budget regression, mutation debt,
non-reproducible artifacts, unsafe log fields and governance ambiguity all fail
closed. Optional toolchain absence is recorded as a capability gate and cannot
satisfy a required test.

