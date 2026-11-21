use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use voiage_domain::SampleMatrix;
use voiage_numerics::{
    evaluate_signed_social_information, SignedInformationDesign, SignedInformationInput,
    SignedInformationSelector, SignedWelfareStage,
};

use super::{invalid, kernel_error};

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(super) struct Request {
    world_probabilities: Vec<f64>,
    policy_payoffs: Vec<Vec<Vec<f64>>>,
    welfare_weights: Vec<f64>,
    role_agents: Vec<Vec<usize>>,
    post_transfer_welfare: bool,
    designs: Vec<DesignRequest>,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct DesignRequest {
    policies: Vec<usize>,
    selected_policy: Option<usize>,
    selector_agent: Option<usize>,
    comparator: usize,
    transfers: Vec<(usize, usize, f64)>,
    costs: Vec<(usize, f64)>,
}

impl DesignRequest {
    fn as_kernel_input(&self) -> SignedInformationDesign<'_> {
        SignedInformationDesign {
            policy_indices: &self.policies,
            selected_policy: self.selected_policy,
            selector: self.selector_agent.map_or(
                SignedInformationSelector::SocialWelfare,
                SignedInformationSelector::Agent,
            ),
            comparator: self.comparator,
            transfers: &self.transfers,
            costs: &self.costs,
        }
    }
}

/// Evaluates the numeric projection of the validated portable v1 contract.
#[pyfunction]
pub(super) fn compute_signed_social_information(
    py: Python<'_>,
    request: Request,
) -> PyResult<Bound<'_, PyList>> {
    let evaluations = py.detach(|| {
        let payoffs = request
            .policy_payoffs
            .into_iter()
            .map(SampleMatrix::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(invalid)?;
        let designs = request
            .designs
            .iter()
            .map(DesignRequest::as_kernel_input)
            .collect::<Vec<_>>();
        let input = SignedInformationInput {
            world_probabilities: &request.world_probabilities,
            policy_payoffs: &payoffs,
            welfare_weights: &request.welfare_weights,
            role_agents: &request.role_agents,
            welfare_stage: if request.post_transfer_welfare {
                SignedWelfareStage::PostTransfer
            } else {
                SignedWelfareStage::PreTransfer
            },
            designs: &designs,
        };
        evaluate_signed_social_information(&input).map_err(kernel_error)
    })?;
    let results = PyList::empty(py);
    for value in evaluations {
        let design = PyDict::new(py);
        let ledger = PyDict::new(py);
        ledger.set_item("pre_transfer", value.ledger.pre_transfer)?;
        ledger.set_item("transfer", value.ledger.transfer)?;
        ledger.set_item("cost", value.ledger.cost)?;
        ledger.set_item("post_transfer", value.ledger.post_transfer)?;
        design.set_item("ledger", ledger)?;
        design.set_item("selector_values", value.selector_values)?;
        design.set_item("selected_policy", value.selected_policy)?;
        design.set_item("social_pre_transfer", value.social_pre_transfer)?;
        design.set_item("social_post_transfer", value.social_post_transfer)?;
        design.set_item("signed_by_agent", value.signed_by_agent)?;
        design.set_item("signed_by_role", value.signed_by_role)?;
        design.set_item("signed_welfare", value.signed_welfare)?;
        results.append(design)?;
    }
    Ok(results)
}
