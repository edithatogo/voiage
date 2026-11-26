# Simulation-Outcome Input Profile v1 (`voiage.simulation-outcome.v1`)

## Scope and Ownership

`voiage.simulation-outcome.v1` defines the versioned, producer-agnostic input
profile for admitting already-valued outcomes and optional draw-aligned
parameter samples into `voiage-domain`.

- **Voiage owns:** profile version validation, complete strategy/draw pairing,
  explicit value/objective declaration checks, draw-key parameter alignment,
  and downstream VOI computation.
- **Producers (Kairos, game solvers, HEOR models) own:** simulation execution,
  RNG/common-random-number coupling, valuation, units, perspective, currency/price
  year, population basis, horizon, and discounting semantics.

Voiage never infers a sign flip, currency conversion, unit rescaling, or
row-position draw alignment.

## Profile Contract

1. **Profile version (`profile_version`)**:
   Must equal `"voiage.simulation-outcome.v1"` (exact match, no leading or
   trailing whitespace). Unknown, empty, or unsupported versions fail explicitly
   with `UnsupportedProfileVersion`.
2. **Producer provenance (`provenance`)**:
   Non-empty, untrimmed `producer_id`, `producer_version`, `model_revision`,
   `valuation_source`, and `artifact_digest`.
3. **Strategy & draw identity (`strategy_ids`, `draw_namespace`, `outcomes`)**:
   - All identifier and key strings (`strategy_ids`, `draw_namespace`,
     `draw_key`, `parameter_id`, `unit`, `perspective`, and optional scope
     fields) must be nonempty and free of leading or trailing whitespace so
     distinct raw keys never coalesce silently.
   - `strategy_ids` must be nonempty and unique.
   - `draw_namespace` must be nonempty; every outcome record must carry the same
     `draw_namespace` and a nonempty `draw_key`.
   - Every `(draw_key, strategy_id)` pair must appear at most once, every
     `strategy_id` in outcomes must be declared in `strategy_ids`, and every
     observed `draw_key` must supply all declared `strategy_ids`.
   - Every outcome `value` and `weight` must be finite (`weight >= 0`), and all
     strategy rows sharing a `draw_key` must declare the exact same `weight`.
   - `weight_semantics` must be one of `"equal"`, `"probability"`, or
     `"relative_weight"` and match the observed draw weights (`"equal"` requires
     positive identical draw weights; `"probability"` requires weights summing to
     `1.0` within `1e-9`; `"relative_weight"` requires positive total weight).
   - Validated profiles sort draws canonically by `draw_key` and retain the
     deterministic key mapping receipt (`draw_keys`, `strategy_ids`).
4. **Value & objective declarations (`value_declaration`)**:
   - Required nonempty `unit` and `perspective`.
   - `convention` (`"utility"`, `"net_benefit"`, `"loss"`) must match `objective`
     (`"maximize"` for `"utility"`/`"net_benefit"`, `"minimize"` for `"loss"`).
   - Optional `currency_price_year`, `population_basis`, `horizon`, and
     `discounting_status` must be nonempty when supplied.
   - If an outcome record carries a per-record `declaration`, every field must
     match the profile-level `value_declaration` exactly.
5. **Optional draw-aligned parameters (`parameter_samples`)**:
   - When present, every parameter sample record must match the profile's
     `draw_namespace`, supply a unique `draw_key` matching an outcome draw
     one-for-one (no missing or unmatched keys, even when row counts match),
     declare a consistent nonempty set of parameter names across all draws, and
     carry finite `f64` values.

## Arrow/Parquet Interchange (`O06`)

Python Arrow/Parquet interchange (`voiage.contracts.interchange`) serializes
paired outcome rows while failing closed on missing metadata or structural
tampering:

- **Required table columns:** `draw_namespace` (`utf8`), `draw_key` (`utf8`),
  `strategy_id` (`utf8`), `value` (`float64`), `weight` (`float64`), plus
  optional parameter columns prefixed with `param::<parameter_name>` (`float64`).
- **Required Arrow schema metadata keys:**
  - `voiage.contract` = `"simulation-outcome"`
  - `voiage.simulation_outcome.profile_version` = `"voiage.simulation-outcome.v1"`
  - `voiage.simulation_outcome.metadata` = canonical JSON containing
    `provenance`, `draw_namespace`, `strategy_ids`, `value_declaration`,
    `weight_semantics`, `draw_keys`, and optional `parameter_names`
  - `vop_voiage.schema_fingerprint` = logical schema SHA-256 fingerprint
  - `vop_voiage.interchange` (`"apache-arrow"`, `"apache-arrow-ipc"`, or
    `"apache-arrow-parquet"`)
- Shuffled input rows and multi-chunk / multi-row-group files must round-trip to
  identical keyed `(draw_key, strategy_id)` values and declarations. Dropping
  any required metadata key or required column raises `ValueError`.
