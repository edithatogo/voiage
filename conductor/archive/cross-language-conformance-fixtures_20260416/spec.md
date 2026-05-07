# Spec: Cross-Language Conformance Fixtures

## Overview

Define a deterministic fixture and manifest layout that future Python, R, and Julia implementations can use to validate behavior against a shared contract.

## Goals

- Establish a versioned fixture directory.
- Distinguish normative fixtures from illustrative examples.
- Keep fixture naming and manifests stable enough for multi-language runners.

## Non-Goals

- Implementing language-specific execution engines.
- Defining statistical algorithms beyond the contract needed for fixture comparison.
- Introducing compatibility shims for unsupported historical layouts.

## Acceptance Criteria

- The fixture structure is versioned and documented.
- Normative and illustrative fixtures are separated clearly.
- A manifest or index file can be consumed by runner implementations without ambiguity.
- The contract is testable and stable across supported languages.
