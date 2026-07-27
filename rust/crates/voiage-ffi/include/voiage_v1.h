#ifndef VOIAGE_V1_H
#define VOIAGE_V1_H

/* Portable voiage v1 ABI and Rust-owned scalar operations. */

#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(VOIAGE_BUILD_SHARED)
#    define VOIAGE_V1_API __declspec(dllexport)
#  elif defined(VOIAGE_USE_SHARED)
#    define VOIAGE_V1_API __declspec(dllimport)
#  else
#    define VOIAGE_V1_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define VOIAGE_V1_API __attribute__((visibility("default")))
#else
#  define VOIAGE_V1_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VOIAGE_V1_ABI_MAJOR UINT32_C(1)
#define VOIAGE_V1_ABI_MINOR UINT32_C(12)
#define VOIAGE_V1_CAPABILITIES_STRUCT_VERSION UINT32_C(1)
#define VOIAGE_V1_CAPABILITY_VERSION_NEGOTIATION (UINT64_C(1) << 0)
#define VOIAGE_V1_CAPABILITY_QUERY (UINT64_C(1) << 1)
#define VOIAGE_V1_CAPABILITY_EVPI (UINT64_C(1) << 2)
#define VOIAGE_V1_CAPABILITY_EVPI_RESULT (UINT64_C(1) << 3)
#define VOIAGE_V1_CAPABILITY_DOCUMENT (UINT64_C(1) << 4)
#define VOIAGE_V1_CAPABILITY_EXPECTED_LOSS_RESULT (UINT64_C(1) << 5)
#define VOIAGE_V1_CAPABILITY_ENBS (UINT64_C(1) << 6)
#define VOIAGE_V1_CAPABILITY_DOMINANCE_RESULT (UINT64_C(1) << 7)
#define VOIAGE_V1_CAPABILITY_CEAF_RESULT (UINT64_C(1) << 8)
#define VOIAGE_V1_CAPABILITY_STRUCTURAL_VOI_RESULT (UINT64_C(1) << 9)
#define VOIAGE_V1_CAPABILITY_EVPPI_REGRESSION_RESULT (UINT64_C(1) << 10)
#define VOIAGE_V1_CAPABILITY_EVSI_APPROXIMATION_RESULT (UINT64_C(1) << 11)
#define VOIAGE_V1_CAPABILITY_DECISION_PROBLEM_JSON (UINT64_C(1) << 12)
#define VOIAGE_V1_CAPABILITY_EVPI_RESULT_JSON (UINT64_C(1) << 13)
#define VOIAGE_V1_CAPABILITY_SCALAR_RESULT_JSON (UINT64_C(1) << 14)
#define VOIAGE_V1_EVPPI_ASSURANCE_INCOMPLETE UINT32_C(0)
#define VOIAGE_V1_EVSI_ASSURANCE_INCOMPLETE UINT32_C(0)
#define VOIAGE_V1_EVSI_ESTIMATOR_REGRESSION UINT32_C(1)
#define VOIAGE_V1_EVSI_ESTIMATOR_MOMENT_MATCHING UINT32_C(2)
#define VOIAGE_V1_NULL_HANDLE UINT64_C(0)

typedef int32_t voiage_v1_status;
enum {
    VOIAGE_V1_STATUS_OK = 0,
    VOIAGE_V1_STATUS_INVALID_ARGUMENT = 1,
    VOIAGE_V1_STATUS_DIMENSION_MISMATCH = 2,
    VOIAGE_V1_STATUS_BACKEND_UNAVAILABLE = 3,
    VOIAGE_V1_STATUS_NUMERICAL_FAILURE = 4,
    VOIAGE_V1_STATUS_SERIALIZATION_FAILURE = 5,
    VOIAGE_V1_STATUS_BUFFER_TOO_SMALL = 6,
    VOIAGE_V1_STATUS_PANIC = 7,
    VOIAGE_V1_STATUS_INTERNAL_ERROR = 255
};

typedef struct VoiageAbiVersionV1 {
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
} VoiageAbiVersionV1;

typedef struct VoiageAbiCapabilitiesV1 {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t capability_bits;
} VoiageAbiCapabilitiesV1;

typedef struct VoiageEvpiResultV1 {
    uint32_t struct_size;
    uint32_t struct_version;
    double value;
    uint64_t sample_count;
    uint64_t strategy_count;
    uint32_t has_assurance;
    uint32_t reserved;
    double opportunity_loss_variance;
    double monte_carlo_standard_error;
} VoiageEvpiResultV1;

typedef struct VoiageExpectedLossResultV1 {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t optimal_strategy_index;
    uint64_t sample_count;
    uint64_t strategy_count;
    double minimum_expected_opportunity_loss;
    uint32_t has_assurance;
    uint32_t reserved;
    double opportunity_loss_variance;
    double monte_carlo_standard_error;
} VoiageExpectedLossResultV1;

typedef struct VoiageDominanceResultV1 {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t strategy_count;
    uint64_t frontier_count;
    uint64_t strongly_dominated_count;
    uint64_t extended_dominated_count;
    uint64_t transition_count;
} VoiageDominanceResultV1;

typedef struct VoiageCeafResultV1 {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t sample_count;
    uint64_t strategy_count;
    uint64_t threshold_count;
} VoiageCeafResultV1;

typedef struct VoiageStructuralVoiResultV1 {
    uint32_t struct_size;
    uint32_t struct_version;
    double value;
    uint64_t structure_count;
    uint64_t sample_count;
    uint64_t strategy_count;
    uint32_t has_assurance;
    uint32_t reserved;
    double informed_value_variance;
    double monte_carlo_standard_error;
} VoiageStructuralVoiResultV1;

typedef struct VoiageEvppiRegressionResultV1 {
    uint32_t struct_size;
    uint32_t struct_version;
    double value;
    uint64_t sample_count;
    uint64_t strategy_count;
    uint64_t parameter_count;
    uint32_t assurance_state;
    uint32_t reserved;
} VoiageEvppiRegressionResultV1;

typedef struct VoiageEvsiApproximationResultV1 {
    uint32_t struct_size;
    uint32_t struct_version;
    double evsi;
    double expected_current_value;
    double expected_sample_value;
    double expected_perfect_information;
    double information_fraction;
    uint64_t sample_count;
    uint64_t strategy_count;
    uint64_t parameter_count;
    uint64_t trial_sample_size;
    uint32_t estimator_kind;
    uint32_t assurance_state;
} VoiageEvsiApproximationResultV1;

/* A handle is an opaque process-local token, never an address. Zero is null. */
typedef uint64_t VoiageHandleV1;

VOIAGE_V1_API VoiageAbiVersionV1 voiage_v1_abi_version(void);
VOIAGE_V1_API VoiageAbiCapabilitiesV1 voiage_v1_capabilities(void);
/* Canonical UTF-8 JSON plus trailing NUL. Query with a null, zero-capacity
 * buffer. required_size is mandatory; no partial document is written. */
VOIAGE_V1_API voiage_v1_status voiage_v1_capabilities_json(
    char *buffer,
    uint64_t capacity,
    uint64_t *required_size);
/* Validate through the stable Rust Decision Problem contract and return
 * compact UTF-8 JSON plus trailing NUL. Query with null buffer and zero
 * capacity. Invalid input writes neither required_size nor buffer. */
VOIAGE_V1_API voiage_v1_status voiage_v1_decision_problem_json(
    const uint8_t *input,
    uint64_t input_length,
    char *buffer,
    uint64_t capacity,
    uint64_t *required_size);
/* Validate the canonical EVPI v1 result envelope and return compact UTF-8 JSON
 * plus trailing NUL through the same query/copy ownership contract. */
VOIAGE_V1_API voiage_v1_status voiage_v1_evpi_result_json(
    const uint8_t *input,
    uint64_t input_length,
    char *buffer,
    uint64_t capacity,
    uint64_t *required_size);
VOIAGE_V1_API voiage_v1_status voiage_v1_evppi_result_json(
    const uint8_t *input,
    uint64_t input_length,
    char *buffer,
    uint64_t capacity,
    uint64_t *required_size);
VOIAGE_V1_API voiage_v1_status voiage_v1_evsi_result_json(
    const uint8_t *input,
    uint64_t input_length,
    char *buffer,
    uint64_t capacity,
    uint64_t *required_size);
VOIAGE_V1_API voiage_v1_status voiage_v1_enbs_result_json(
    const uint8_t *input,
    uint64_t input_length,
    char *buffer,
    uint64_t capacity,
    uint64_t *required_size);
VOIAGE_V1_API voiage_v1_status voiage_v1_evpi(
    const double *values,
    uint64_t rows,
    uint64_t columns,
    double *out_value);
VOIAGE_V1_API voiage_v1_status voiage_v1_enbs(
    double evsi_result,
    double research_cost,
    double *out_value);
VOIAGE_V1_API voiage_v1_status voiage_v1_evpi_result(
    const double *values,
    uint64_t rows,
    uint64_t columns,
    VoiageEvpiResultV1 *out_result);
VOIAGE_V1_API voiage_v1_status voiage_v1_expected_loss_result(
    const double *values,
    uint64_t rows,
    uint64_t columns,
    double *out_expected_net_benefit,
    double *out_expected_opportunity_loss,
    uint64_t array_capacity,
    VoiageExpectedLossResultV1 *out_result);
/* Status values: 0 frontier, 1 strongly dominated, 2 extended dominated. */
VOIAGE_V1_API voiage_v1_status voiage_v1_dominance_result(
    const double *costs,
    const double *effects,
    uint64_t strategy_count,
    int32_t *out_status,
    uint64_t *out_frontier_indices,
    uint64_t strategy_capacity,
    double *out_incremental_costs,
    double *out_incremental_effects,
    double *out_icers,
    uint64_t transition_capacity,
    VoiageDominanceResultV1 *out_result);
VOIAGE_V1_API voiage_v1_status voiage_v1_ceaf_result(
    const double *values,
    uint64_t sample_count,
    uint64_t strategy_count,
    uint64_t threshold_count,
    const double *thresholds,
    double confidence_level,
    uint64_t *out_optimal_strategy_indices,
    double *out_acceptability_probabilities,
    double *out_probability_lower,
    double *out_probability_upper,
    double *out_expected_net_benefit,
    uint32_t *out_has_assurance,
    double *out_probability_variance,
    double *out_probability_standard_error,
    uint64_t threshold_capacity,
    VoiageCeafResultV1 *out_result);
/* Net benefit is row-major [structure][sample][strategy]. */
VOIAGE_V1_API voiage_v1_status voiage_v1_structural_evpi_result(
    const double *values,
    uint64_t structure_count,
    uint64_t sample_count,
    uint64_t strategy_count,
    const double *structure_probabilities,
    VoiageStructuralVoiResultV1 *out_result);
/* structures_of_interest may be null only when its count is zero. */
VOIAGE_V1_API voiage_v1_status voiage_v1_structural_evppi_result(
    const double *values,
    uint64_t structure_count,
    uint64_t sample_count,
    uint64_t strategy_count,
    const double *structure_probabilities,
    const uint64_t *structures_of_interest,
    uint64_t structures_of_interest_count,
    VoiageStructuralVoiResultV1 *out_result);
/* Stable full-sample linear estimator. Assurance state remains incomplete. */
VOIAGE_V1_API voiage_v1_status voiage_v1_evppi_regression_result(
    const double *net_benefit,
    uint64_t sample_count,
    uint64_t strategy_count,
    const double *parameter_samples,
    uint64_t parameter_sample_count,
    uint64_t parameter_count,
    VoiageEvppiRegressionResultV1 *out_result);
/* Promoted Rust-native deterministic approximations. A single call does not
 * establish replicate assurance. */
VOIAGE_V1_API voiage_v1_status voiage_v1_evsi_regression_result(
    const double *net_benefit,
    uint64_t sample_count,
    uint64_t strategy_count,
    const double *parameter_samples,
    uint64_t parameter_sample_count,
    uint64_t parameter_count,
    uint64_t trial_sample_size,
    VoiageEvsiApproximationResultV1 *out_result);
VOIAGE_V1_API voiage_v1_status voiage_v1_evsi_moment_matching_result(
    const double *net_benefit,
    uint64_t sample_count,
    uint64_t strategy_count,
    const double *parameter_samples,
    uint64_t parameter_sample_count,
    uint64_t parameter_count,
    uint64_t trial_sample_size,
    VoiageEvsiApproximationResultV1 *out_result);
/* R-compatible dimension-width adapter for the same Rust EVPI kernel. */
VOIAGE_V1_API voiage_v1_status voiage_v1_evpi_i32(
    const double *values,
    int32_t rows,
    int32_t columns,
    double *out_value);
VOIAGE_V1_API void voiage_v1_evpi_i32_r(
    const double *values,
    const int32_t *rows,
    const int32_t *columns,
    double *out_value,
    int32_t *out_status);
VOIAGE_V1_API voiage_v1_status voiage_v1_handle_create(
    VoiageHandleV1 *out_handle);
VOIAGE_V1_API voiage_v1_status voiage_v1_handle_free(
    VoiageHandleV1 handle);

/* Lengths are fixed-width. required_size includes the trailing NUL. A null
 * buffer with zero capacity queries the size. No partial message is written. */
VOIAGE_V1_API voiage_v1_status voiage_v1_error_message(
    char *buffer,
    uint64_t capacity,
    uint64_t *required_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VOIAGE_V1_H */
