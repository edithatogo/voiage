# HEOR Process Mining Outline

This directory reserves the ecosystem contract outline for health economics and
outcomes research process-mining use cases. It is the home for PM4Py-style
pathway analysis in the ecosystem, but it is not a commitment to add process
mining into the current `voiage` package.

## Scope

- event-log ingestion for HEOR pathways
- pathway discovery and conformance checking
- bottleneck and variant analysis
- conformance metrics for care-flow and implementation pathways
- portable event-log and trace summaries for sibling modules

## Non-Goals

- replacing the current `voiage` VOI API
- adding process mining as a core dependency
- tying the outline to private Python objects

## Contract Notes

- CLI support should be planned before implementation.
- MCP should be considered only if the module becomes agent-queryable or
  workflow-orchestration heavy.
- PM4Py is the current reference candidate, but not a required dependency.
