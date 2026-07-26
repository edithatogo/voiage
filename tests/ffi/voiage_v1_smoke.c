#include "voiage_v1.h"

#include <stddef.h>
#include <stdint.h>

_Static_assert(sizeof(VoiageAbiVersionV1) == 12, "version layout drift");
_Static_assert(offsetof(VoiageAbiVersionV1, struct_size) == 0,
               "struct_size must be first");
_Static_assert(sizeof(VoiageAbiCapabilitiesV1) == 16,
               "capabilities layout drift");
_Static_assert(offsetof(VoiageAbiCapabilitiesV1, capability_bits) == 8,
               "capability bit offset drift");
_Static_assert(sizeof(VoiageHandleV1) == 8, "handle width drift");
_Static_assert(sizeof(voiage_v1_status) == 4, "status width drift");
_Static_assert(sizeof(VoiageEvpiResultV1) == 56, "EVPI result layout drift");
_Static_assert(sizeof(VoiageExpectedLossResultV1) == 64,
               "expected-loss result layout drift");
_Static_assert(sizeof(VoiageDominanceResultV1) == 48,
               "dominance result layout drift");

static int exercise_contract(void) {
    VoiageAbiVersionV1 version = voiage_v1_abi_version();
    VoiageAbiCapabilitiesV1 capabilities = voiage_v1_capabilities();
    uint64_t capability_document_size = 0;
    if (voiage_v1_capabilities_json(NULL, 0, &capability_document_size) !=
            VOIAGE_V1_STATUS_OK ||
        capability_document_size <= 1) {
        return 6;
    }
    const double values[] = {10.0, 1.0, 2.0, 8.0};
    VoiageEvpiResultV1 evpi_result = {0};
    if (voiage_v1_evpi_result(values, 2, 2, &evpi_result) !=
            VOIAGE_V1_STATUS_OK ||
        evpi_result.struct_size != sizeof(evpi_result) ||
        evpi_result.sample_count != 2 || evpi_result.strategy_count != 2) {
        return 5;
    }
    double expected_benefits[2] = {0};
    double expected_losses[2] = {0};
    VoiageExpectedLossResultV1 expected_loss_result = {0};
    if (voiage_v1_expected_loss_result(
            values, 2, 2, expected_benefits, expected_losses, 2,
            &expected_loss_result) != VOIAGE_V1_STATUS_OK ||
        expected_loss_result.struct_size != sizeof(expected_loss_result) ||
        expected_loss_result.strategy_count != 2) {
        return 7;
    }
    double enbs = 0.0;
    if (voiage_v1_enbs(12.5, 3.0, &enbs) != VOIAGE_V1_STATUS_OK ||
        enbs != 9.5) {
        return 8;
    }
    const double costs[] = {100.0, 120.0, 150.0};
    const double effects[] = {1.0, 0.9, 2.0};
    int32_t dominance_status[3] = {-1, -1, -1};
    uint64_t frontier[3] = {0};
    double incremental_costs[2] = {0};
    double incremental_effects[2] = {0};
    double icers[2] = {0};
    VoiageDominanceResultV1 dominance_result = {0};
    if (voiage_v1_dominance_result(
            costs, effects, 3, dominance_status, frontier, 3,
            incremental_costs, incremental_effects, icers, 2,
            &dominance_result) != VOIAGE_V1_STATUS_OK ||
        dominance_result.frontier_count != 2 ||
        dominance_result.strongly_dominated_count != 1) {
        return 9;
    }
    VoiageHandleV1 handle = VOIAGE_V1_NULL_HANDLE;
    uint64_t required_size = 0;
    voiage_v1_status status = voiage_v1_handle_create(&handle);
    if (status != VOIAGE_V1_STATUS_OK || handle == VOIAGE_V1_NULL_HANDLE) {
        return 1;
    }

    status = voiage_v1_error_message(NULL, 0, &required_size);
    if (status != VOIAGE_V1_STATUS_OK || required_size == 0) {
        return 2;
    }
    status = voiage_v1_handle_free(handle);
    if (status != VOIAGE_V1_STATUS_OK) {
        return 3;
    }

    (void)version;
    (void)capabilities;
    return 0;
}

int main(void) {
    return exercise_contract();
}
