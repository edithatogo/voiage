#include "voiage_v1.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(std::is_standard_layout<VoiageAbiVersionV1>::value,
              "version response must have standard layout");
static_assert(sizeof(VoiageAbiVersionV1) == 12, "version layout drift");
static_assert(sizeof(VoiageAbiCapabilitiesV1) == 16,
              "capabilities layout drift");
static_assert(offsetof(VoiageAbiCapabilitiesV1, capability_bits) == 8,
              "capability bit offset drift");
static_assert(sizeof(VoiageHandleV1) == 8, "handle width drift");
static_assert(sizeof(voiage_v1_status) == 4, "status width drift");

static int exercise_contract() {
    VoiageAbiVersionV1 version = voiage_v1_abi_version();
    VoiageAbiCapabilitiesV1 capabilities = voiage_v1_capabilities();
    VoiageHandleV1 handle = VOIAGE_V1_NULL_HANDLE;
    std::uint64_t required_size = 0;
    voiage_v1_status status = voiage_v1_handle_create(&handle);
    if (status != VOIAGE_V1_STATUS_OK || handle == VOIAGE_V1_NULL_HANDLE) {
        return 1;
    }

    status = voiage_v1_error_message(nullptr, 0, &required_size);
    if (status != VOIAGE_V1_STATUS_OK || required_size == 0) {
        return 2;
    }
    status = voiage_v1_handle_free(handle);
    if (status != VOIAGE_V1_STATUS_OK) {
        return 3;
    }

    static_cast<void>(version);
    static_cast<void>(capabilities);
    return 0;
}

int main() {
    return exercise_contract();
}
