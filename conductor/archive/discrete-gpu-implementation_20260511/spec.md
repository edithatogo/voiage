# Track Specification: Discrete GPU Implementation

## Overview

This track converts the discrete GPU follow-on into an execution track that can
reuse the same result envelope as the CPU and Apple Metal paths.

## Functional Requirements

1. Choose the discrete backend engine(s) and deployment assumptions.
2. Preserve the public API and the deterministic summary envelope.
3. Keep CPU fallback authoritative.

## Acceptance Criteria

1. A discrete GPU backend is implemented behind the shared abstraction.
2. The benchmark packet compares GPU and CPU results.
3. No contract change is required for the public API.
