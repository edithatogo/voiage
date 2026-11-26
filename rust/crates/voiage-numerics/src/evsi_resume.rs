//! Bounded, checkpointed execution of streaming EVSI reduction (`F05`).
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use voiage_diagnostics::{
    CancellationToken, CheckpointIdentity, CheckpointStore, ExecutionBudget, ExecutionCheckpoint,
    ReplayContract,
};
use voiage_domain::SampleMatrix;

/// Result of a bounded, resumable EVSI streaming run.
#[derive(Debug, PartialEq)]
pub enum ResumableEvsiOutcome {
    /// All outer datasets were evaluated and final EVSI is available.
    Complete {
        /// Calculated EVSI.
        evsi: f64,
        /// Fully processed outer datasets.
        evaluations: u64,
    },
    /// Budget exhaustion stopped execution at a batch boundary.
    Partial {
        /// Fully processed outer datasets.
        evaluations: u64,
    },
    /// Cancellation stopped execution at a batch boundary.
    Cancelled {
        /// Fully processed outer datasets.
        evaluations: u64,
    },
}

fn sha256_prefixed(bytes: &[u8]) -> String {
    let mut value = String::from("sha256:");
    for byte in Sha256::digest(bytes) {
        write!(value, "{byte:02x}").expect("string write");
    }
    value
}

fn evsi_input_hash(net_benefit: &SampleMatrix, log_weights: &SampleMatrix) -> String {
    let mut hasher = Sha256::new();
    let [rows, cols] = net_benefit.shape();
    let [ds, wcols] = log_weights.shape();
    hasher.update((rows as u64).to_le_bytes());
    hasher.update((cols as u64).to_le_bytes());
    hasher.update((ds as u64).to_le_bytes());
    hasher.update((wcols as u64).to_le_bytes());
    for row in net_benefit.rows() {
        for v in row {
            hasher.update(v.to_bits().to_le_bytes());
        }
    }
    for row in log_weights.rows() {
        for v in row {
            hasher.update(v.to_bits().to_le_bytes());
        }
    }
    let mut hex = String::new();
    for b in hasher.finalize() {
        write!(hex, "{b:02x}").expect("string write");
    }
    hex
}

fn bound_evsi_identity(
    net_benefit: &SampleMatrix,
    log_weights: &SampleMatrix,
    identity: &CheckpointIdentity,
) -> Result<CheckpointIdentity, String> {
    let h = evsi_input_hash(net_benefit, log_weights);
    CheckpointIdentity::new(
        format!("{}:{h}", identity.model_input),
        &identity.rng,
        format!("{}:streaming-evsi-is-v1", identity.algorithm),
    )
    .map_err(|e| e.to_string())
}

/// Executes streaming importance-weighted EVSI with deterministic checkpoint/restart and cancellation.
///
/// # Errors
///
/// Returns a descriptive error string on shape mismatch, non-finite weights, or corrupt checkpoints.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn evsi_streaming_resumable(
    net_benefit: &SampleMatrix,
    log_weights: &SampleMatrix,
    store: &CheckpointStore,
    identity: &CheckpointIdentity,
    max_evaluations: u64,
    batch_size: usize,
    cancellation: &CancellationToken,
    resume: bool,
) -> Result<ResumableEvsiOutcome, String> {
    const MAX_PAYLOAD: usize = 64 * 1024 * 1024;
    let [sample_count, strategy_count] = net_benefit.shape();
    let [dataset_count, weight_cols] = log_weights.shape();
    if sample_count == 0 || strategy_count == 0 || dataset_count == 0 || weight_cols != sample_count
    {
        return Err("invalid matrix dimensions for streaming EVSI".to_string());
    }
    if batch_size == 0 {
        return Err("batch size must be positive".to_string());
    }

    let bound = bound_evsi_identity(net_benefit, log_weights, identity)?;
    let (mut completed, mut batches, mut payload, budget) = if resume {
        let (cp, payload, budget) = store
            .load_for_resume(MAX_PAYLOAD, &bound, max_evaluations)
            .map_err(|e| e.to_string())?;
        let completed =
            usize::try_from(cp.evaluations).map_err(|_| "invalid evaluation count".to_string())?;
        if completed > dataset_count
            || payload.len() != completed * 8
            || cp.replay != ReplayContract::default()
        {
            return Err("invalid EVSI checkpoint".to_string());
        }
        (completed, cp.completed_batches, payload, budget)
    } else {
        let initial = ExecutionCheckpoint::new(bound.clone(), 0, 0, sha256_prefixed(&[]))
            .map_err(|e| e.to_string())?;
        store
            .save_with_payload(&initial, &[])
            .map_err(|e| e.to_string())?;
        (0, 0, Vec::new(), ExecutionBudget::new(max_evaluations))
    };

    let dataset_rows: Vec<&[f64]> = log_weights.rows().collect();
    while completed < dataset_count {
        let n = batch_size.min(dataset_count - completed);
        match cancellation.reserve_batch(&budget, n as u64) {
            Err(_) => {
                return Ok(ResumableEvsiOutcome::Cancelled {
                    evaluations: completed as u64,
                })
            }
            Ok(false) => {
                return Ok(ResumableEvsiOutcome::Partial {
                    evaluations: completed as u64,
                })
            }
            Ok(true) => {}
        }

        for lw in &dataset_rows[completed..completed + n] {
            let max_lw = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !max_lw.is_finite() {
                return Err("importance log-weights are all -inf".to_string());
            }
            let raw: Vec<f64> = lw.iter().map(|&v| (v - max_lw).exp()).collect();
            let sum_raw: f64 = raw.iter().sum();
            if !sum_raw.is_finite() || sum_raw <= 0.0 {
                return Err("invalid importance weight sum".to_string());
            }
            let mut post_means = vec![0.0; strategy_count];
            for (nb_row, &r) in net_benefit.rows().zip(raw.iter()) {
                let w = r / sum_raw;
                for (k, &nb) in nb_row.iter().enumerate() {
                    post_means[k] += w * nb;
                }
            }
            let row_best = post_means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            payload.extend_from_slice(&row_best.to_bits().to_le_bytes());
        }

        completed += n;
        batches = batches
            .checked_add(1)
            .ok_or("invalid checkpoint batch count")?;
        let cp = ExecutionCheckpoint::new(
            bound.clone(),
            batches,
            completed as u64,
            sha256_prefixed(&payload),
        )
        .map_err(|e| e.to_string())?;
        store
            .save_with_payload(&cp, &payload)
            .map_err(|e| e.to_string())?;
    }

    if cancellation.is_cancelled() {
        return Ok(ResumableEvsiOutcome::Cancelled {
            evaluations: completed as u64,
        });
    }

    let d_f64 = f64::from(u32::try_from(dataset_count).map_err(|_| "dataset count overflow")?);
    let mut post_study_mean = 0.0_f64;
    for chunk in payload.chunks_exact(8) {
        let arr: [u8; 8] = chunk
            .try_into()
            .map_err(|_| "corrupt EVSI checkpoint chunk".to_string())?;
        post_study_mean += f64::from_bits(u64::from_le_bytes(arr)) / d_f64;
    }
    let s_f64 = f64::from(u32::try_from(sample_count).map_err(|_| "sample count overflow")?);
    let mut prior_means = vec![0.0; strategy_count];
    for row in net_benefit.rows() {
        for (k, &v) in row.iter().enumerate() {
            prior_means[k] += v / s_f64;
        }
    }
    let current_best = prior_means
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let evsi = (post_study_mean - current_best).max(0.0);
    Ok(ResumableEvsiOutcome::Complete {
        evsi,
        evaluations: completed as u64,
    })
}
