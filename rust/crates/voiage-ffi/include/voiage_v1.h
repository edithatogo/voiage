#ifndef VOIAGE_V1_H
#define VOIAGE_V1_H

/* Portable voiage v1 ABI infrastructure. Numerical operations are deferred. */

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
#define VOIAGE_V1_ABI_MINOR UINT32_C(0)
#define VOIAGE_V1_CAPABILITIES_STRUCT_VERSION UINT32_C(1)
#define VOIAGE_V1_CAPABILITY_VERSION_NEGOTIATION (UINT64_C(1) << 0)
#define VOIAGE_V1_CAPABILITY_QUERY (UINT64_C(1) << 1)
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

/* A handle is an opaque process-local token, never an address. Zero is null. */
typedef uint64_t VoiageHandleV1;

VOIAGE_V1_API VoiageAbiVersionV1 voiage_v1_abi_version(void);
VOIAGE_V1_API VoiageAbiCapabilitiesV1 voiage_v1_capabilities(void);
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
