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
    uint64_t capability_document_size = 0;
    if (voiage_v1_capabilities_json(NULL, 0, &capability_document_size) !=
            VOIAGE_V1_STATUS_OK ||
        capability_document_size <= 1) {
        return 9;
    }

    const double values[] = {10.0, 1.0, 2.0, 8.0};
    VoiageEvpiResultV1 result = {0};
    if (voiage_v1_evpi_result(values, 2, 2, &result) != VOIAGE_V1_STATUS_OK ||
        result.struct_size != sizeof(result) || result.has_assurance != 1) {
        return 8;
    }
    double expected_benefits[2] = {0};
    double expected_losses[2] = {0};
    VoiageExpectedLossResultV1 expected_loss_result = {0};
    if (voiage_v1_expected_loss_result(
            values, 2, 2, expected_benefits, expected_losses, 2,
            &expected_loss_result) != VOIAGE_V1_STATUS_OK ||
        expected_loss_result.struct_size != sizeof(expected_loss_result)) {
        return 10;
    }
    double enbs = 0.0;
    if (voiage_v1_enbs(12.5, 3.0, &enbs) != VOIAGE_V1_STATUS_OK ||
        enbs != 9.5) {
        return 11;
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
        dominance_result.frontier_count != 2) {
        return 12;
    }
    const double ceaf_values[] = {10.0, 1.0, 5.0, 8.0,
                                  2.0,  3.0, 7.0, 4.0};
    const double thresholds[] = {0.0, 100.0};
    uint64_t optimal[2] = {0};
    double probability[2] = {0};
    double lower[2] = {0};
    double upper[2] = {0};
    double selected_benefit[2] = {0};
    uint32_t has_assurance[2] = {0};
    double probability_variance[2] = {0};
    double probability_error[2] = {0};
    VoiageCeafResultV1 ceaf_result = {0};
    if (voiage_v1_ceaf_result(
            ceaf_values, 2, 2, 2, thresholds, 0.95, optimal, probability,
            lower, upper, selected_benefit, has_assurance,
            probability_variance, probability_error, 2,
            &ceaf_result) != VOIAGE_V1_STATUS_OK ||
        ceaf_result.threshold_count != 2) {
        return 13;
    }
    const double structural_values[] = {10.0, 8.0, 11.0, 7.0,
                                        6.0, 12.0, 5.0,  13.0};
    const double structure_probabilities[] = {0.5, 0.5};
    const uint64_t structures_of_interest[] = {0, 1};
    VoiageStructuralVoiResultV1 structural_result = {0};
    if (voiage_v1_structural_evpi_result(
            structural_values, 2, 2, 2, structure_probabilities,
            &structural_result) != VOIAGE_V1_STATUS_OK ||
        structural_result.value != 1.5) {
        return 14;
    }
    if (voiage_v1_structural_evppi_result(
            structural_values, 2, 2, 2, structure_probabilities,
            structures_of_interest, 2,
            &structural_result) != VOIAGE_V1_STATUS_OK ||
        structural_result.value != 1.5) {
        return 15;
    }
    const double evppi_net_benefit[] = {5.0, 1.0, 4.0, 2.0,
                                        1.0, 5.0, 2.0, 4.0};
    const double evppi_parameters[] = {0.0, 0.0, 0.0, 1.0,
                                       1.0, 0.0, 1.0, 1.0};
    VoiageEvppiRegressionResultV1 evppi_result = {0};
    if (voiage_v1_evppi_regression_result(
            evppi_net_benefit, 4, 2, evppi_parameters, 4, 2,
            &evppi_result) != VOIAGE_V1_STATUS_OK ||
        evppi_result.assurance_state != VOIAGE_V1_EVPPI_ASSURANCE_INCOMPLETE) {
        return 16;
    }
    VoiageEvsiApproximationResultV1 evsi_result = {0};
    if (voiage_v1_evsi_regression_result(
            evppi_net_benefit, 4, 2, evppi_parameters, 4, 2, 3,
            &evsi_result) != VOIAGE_V1_STATUS_OK ||
        evsi_result.estimator_kind != VOIAGE_V1_EVSI_ESTIMATOR_REGRESSION ||
        evsi_result.assurance_state != VOIAGE_V1_EVSI_ASSURANCE_INCOMPLETE) {
        return 17;
    }
    if (voiage_v1_evsi_moment_matching_result(
            evppi_net_benefit, 4, 2, evppi_parameters, 4, 2, 3,
            &evsi_result) != VOIAGE_V1_STATUS_OK ||
        evsi_result.estimator_kind !=
            VOIAGE_V1_EVSI_ESTIMATOR_MOMENT_MATCHING) {
        return 18;
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
