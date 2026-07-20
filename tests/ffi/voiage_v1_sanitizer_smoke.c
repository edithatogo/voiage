#include "voiage_v1.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { ITERATIONS = 10000 };

static int exercise_error_transport(void) {
    uint64_t required = 0;
    if (voiage_v1_error_message(NULL, 0, &required) != VOIAGE_V1_STATUS_OK ||
        required < 2) {
        return 10;
    }

    char *message = malloc((size_t)required);
    if (message == NULL) {
        return 11;
    }
    const voiage_v1_status status =
        voiage_v1_error_message(message, required, &required);
    const int result =
        status == VOIAGE_V1_STATUS_OK && message[required - 1] == '\0' &&
                strlen(message) + 1 == required
            ? 0
            : 12;
    free(message);
    return result;
}

int main(void) {
    const VoiageAbiVersionV1 version = voiage_v1_abi_version();
    const VoiageAbiCapabilitiesV1 capabilities = voiage_v1_capabilities();
    if (version.struct_size != sizeof(version) || version.abi_major != 1 ||
        capabilities.struct_size != sizeof(capabilities)) {
        return 1;
    }

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        VoiageHandleV1 handle = VOIAGE_V1_NULL_HANDLE;
        if (voiage_v1_handle_create(&handle) != VOIAGE_V1_STATUS_OK ||
            handle == VOIAGE_V1_NULL_HANDLE) {
            return 2;
        }
        if (voiage_v1_handle_free(handle) != VOIAGE_V1_STATUS_OK) {
            return 3;
        }
        if (voiage_v1_handle_free(handle) != VOIAGE_V1_STATUS_INVALID_ARGUMENT) {
            return 4;
        }
    }

    if (voiage_v1_handle_free(VOIAGE_V1_NULL_HANDLE) != VOIAGE_V1_STATUS_OK) {
        return 5;
    }
    return exercise_error_transport();
}
