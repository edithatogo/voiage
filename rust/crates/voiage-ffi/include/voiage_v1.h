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
#define VOIAGE_V1_ABI_MINOR UINT32_C(5)
#define VOIAGE_V1_CAPABILITIES_STRUCT_VERSION UINT32_C(1)
#define VOIAGE_V1_CAPABILITY_VERSION_NEGOTIATION (UINT64_C(1) << 0)
#define VOIAGE_V1_CAPABILITY_QUERY (UINT64_C(1) << 1)
#define VOIAGE_V1_CAPABILITY_EVPI (UINT64_C(1) << 2)
#define VOIAGE_V1_CAPABILITY_EVPI_RESULT (UINT64_C(1) << 3)
#define VOIAGE_V1_CAPABILITY_DOCUMENT (UINT64_C(1) << 4)
#define VOIAGE_V1_CAPABILITY_EXPECTED_LOSS_RESULT (UINT64_C(1) << 5)
#define VOIAGE_V1_CAPABILITY_ENBS (UINT64_C(1) << 6)
#define VOIAGE_V1_CAPABILITY_DOMINANCE_RESULT (UINT64_C(1) << 7)
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
