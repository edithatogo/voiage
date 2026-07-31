# Ingestion Provider SDK v1

`INGESTION_PROVIDER_SDK_VERSION == "1"` freezes the public provider boundary.
It is intentionally small so external parser packages remain decoupled from the
VOI calculation runtime.

## Required provider surface

An entry point in the `voiage.ingestion.providers` group exports one initialized
object with:

- a non-empty `provider_id` string;
- a `ProviderCapabilities` instance whose `provider_id` is identical;
- `can_handle(descriptor: dict[str, object]) -> bool`, a side-effect-free
  recognizer; and
- `ingest(descriptor_path: Path, *, policy: SourceAccessPolicy) ->
  NormalizedInputBundle`.

`can_handle` must not access resources, import parser stacks, or perform network
I/O. `ingest` must resolve every declared resource through the supplied policy
and return a valid normalized bundle. Providers must not infer VOI roles,
strategies, outcomes, or decision semantics.

## Capability and compatibility rules

`ProviderCapabilities` is the provider's conservative support declaration:
format versions, media types, transformations, projection, filtering,
streaming, and random access. A provider must declare unsupported operations as
false or empty. Adding an optional capability is compatible only when the
existing profile remains unchanged; changing a declared capability, provider
identifier, error category, or normalized manifest meaning requires a new SDK
major version.

Discovery is opt-in and allow-listed. Base imports and descriptor probing never
load third-party entry points. The machine-readable consumer fixture is
`specs/core-api/fixtures/v2/ingestion-provider-sdk-v1.json`; changes to its
version, entry-point group, capability fields, or protocol surface require an
SDK-major review. A provider must be tested in a clean base install and with
only its named optional extra installed.

The repository's `croissant`, `frictionless`, and aggregate `ingestion` extras
are currently dependency-neutral reservations. They install the same runtime as
the base package because the built-in local CSV profiles use only the base Arrow
and JSON stack. This is not a claim that `mlcroissant` or `frictionless` are
installed, nor that their remote, archive, or cache behaviours are supported.

## Consumer contract

`from_dataframe` is the supported generic DataFrame entry point. It accepts a
producer that Arrow can convert through `__dataframe__`, preserves the resulting
Arrow column schema, excludes producer-specific indexes, honours `allow_copy`,
and returns the same `NormalizedInputBundle` preparation path as providers.
Nullable values, categories, and timezone-aware values are supported when Arrow
can materialize them. Nested values and conversions requiring a disallowed copy
fail with a stable `ValueError`; no alternate calculation path is used.

The resulting manifest records source-neutral conversion diagnostics and a
`voiage.dev:dataframe-interchange` extension. They state the requested copy
policy, index exclusion, and per-field Arrow dtype, nullability, categorical,
and timezone decisions. A successful `allow_copy=False` conversion is recorded
as zero-copy. When copying is permitted, Arrow's public interchange API does
not reveal whether a copy actually occurred, so the outcome is recorded as
`not_observable` rather than guessed.

## Publication checklist

1. Declare an entry point that returns an initialized provider instance.
2. Add supported and rejected descriptor fixtures, source-policy tests, and
   deterministic provenance/receipt assertions.
3. Verify capability identity, import isolation, and explicit allow-listed
   discovery.
4. Document parser dependencies behind a named optional extra; never add them
   to the base install.
5. Publish the supported profile and unsupported boundaries before release.
