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
_Static_assert(sizeof(VoiageCeafResultV1) == 32, "CEAF result layout drift");
_Static_assert(sizeof(VoiageStructuralVoiResultV1) == 64,
               "structural VOI result layout drift");
_Static_assert(sizeof(VoiageEvppiRegressionResultV1) == 48,
               "EVPPI regression result layout drift");
_Static_assert(sizeof(VoiageEvsiApproximationResultV1) == 88,
               "EVSI approximation result layout drift");

static int exercise_contract(void) {
    VoiageAbiVersionV1 version = voiage_v1_abi_version();
    VoiageAbiCapabilitiesV1 capabilities = voiage_v1_capabilities();
    uint64_t capability_document_size = 0;
    if (voiage_v1_capabilities_json(NULL, 0, &capability_document_size) !=
            VOIAGE_V1_STATUS_OK ||
        capability_document_size <= 1) {
        return 6;
    }
    const char decision_problem[] =
        "{\"decision_problem_id\":\"screening-001\","
        "\"title\":\"Screening programme\","
        "\"analysis_type\":\"net-benefit-first\",\"currency\":\"AUD\","
        "\"willingness_to_pay\":50000.0,"
        "\"interventions\":[{\"intervention_id\":\"usual-care\","
        "\"name\":\"Usual care\",\"is_reference\":true}]}";
    uint64_t decision_problem_size = 0;
    if (voiage_v1_decision_problem_json(
            (const uint8_t *)decision_problem,
            (uint64_t)(sizeof(decision_problem) - 1), NULL, 0,
            &decision_problem_size) != VOIAGE_V1_STATUS_OK ||
        decision_problem_size <= 1) {
        return 16;
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
        return 10;
    }
    const double structural_values[] = {10.0, 8.0, 11.0, 7.0,
                                        6.0, 12.0, 5.0,  13.0};
    const double structure_probabilities[] = {0.5, 0.5};
    const uint64_t structures_of_interest[] = {0, 1};
    VoiageStructuralVoiResultV1 structural_result = {0};
    if (voiage_v1_structural_evpi_result(
            structural_values, 2, 2, 2, structure_probabilities,
            &structural_result) != VOIAGE_V1_STATUS_OK ||
        structural_result.struct_size != sizeof(structural_result) ||
        structural_result.value != 1.5) {
        return 11;
    }
    if (voiage_v1_structural_evppi_result(
            structural_values, 2, 2, 2, structure_probabilities,
            structures_of_interest, 2,
            &structural_result) != VOIAGE_V1_STATUS_OK ||
        structural_result.value != 1.5) {
        return 12;
    }
    const double evppi_net_benefit[] = {5.0, 1.0, 4.0, 2.0,
                                        1.0, 5.0, 2.0, 4.0};
    const double evppi_parameters[] = {0.0, 0.0, 0.0, 1.0,
                                       1.0, 0.0, 1.0, 1.0};
    VoiageEvppiRegressionResultV1 evppi_result = {0};
    if (voiage_v1_evppi_regression_result(
            evppi_net_benefit, 4, 2, evppi_parameters, 4, 2,
            &evppi_result) != VOIAGE_V1_STATUS_OK ||
        evppi_result.struct_size != sizeof(evppi_result) ||
        evppi_result.assurance_state != VOIAGE_V1_EVPPI_ASSURANCE_INCOMPLETE) {
        return 13;
    }
    VoiageEvsiApproximationResultV1 evsi_result = {0};
    if (voiage_v1_evsi_regression_result(
            evppi_net_benefit, 4, 2, evppi_parameters, 4, 2, 3,
            &evsi_result) != VOIAGE_V1_STATUS_OK ||
        evsi_result.struct_size != sizeof(evsi_result) ||
        evsi_result.estimator_kind != VOIAGE_V1_EVSI_ESTIMATOR_REGRESSION ||
        evsi_result.assurance_state != VOIAGE_V1_EVSI_ASSURANCE_INCOMPLETE) {
        return 14;
    }
    if (voiage_v1_evsi_moment_matching_result(
            evppi_net_benefit, 4, 2, evppi_parameters, 4, 2, 3,
            &evsi_result) != VOIAGE_V1_STATUS_OK ||
        evsi_result.estimator_kind !=
            VOIAGE_V1_EVSI_ESTIMATOR_MOMENT_MATCHING) {
        return 15;
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
