# Track Specification: Apple Metal Implementation

## Overview

This track hardens the Apple Metal backend from prototype/reviewable status into
the first production-style accelerator lane. It keeps the CPU fallback
authoritative and preserves the public API shape.

## Functional Requirements

1. Complete device-backed Apple Silicon comparison and packaging guidance.
2. Keep the backend contract-preserving and optional.
3. Maintain benchmark evidence and reproducible handoff artifacts.

## Acceptance Criteria

1. Metal-backed execution is implemented and reviewed.
2. CPU fallback remains the reference path.
3. No public API change is required to use the backend.
