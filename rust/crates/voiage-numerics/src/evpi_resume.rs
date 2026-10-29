//! Bounded, checkpointed execution of the existing EVPI reduction.
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use voiage_diagnostics::{
    CancellationToken, CheckpointIdentity, CheckpointStore, ExecutionBudget, ExecutionCheckpoint,
    ReplayContract,
};
use voiage_domain::SampleMatrix;

/// Result of a bounded EVPI run.
#[derive(Debug, PartialEq)]
pub enum ResumableEvpiOutcome {
    /// All rows were evaluated and the final value is available.
    Complete {
        /// The calculated EVPI.
        value: f64,
        /// Fully processed rows.
        evaluations: u64,
    },
    /// Budget exhaustion stopped execution at a batch boundary.
    Partial {
        /// Fully processed rows.
        evaluations: u64,
    },
    /// Cancellation stopped execution at a batch boundary.
    Cancelled {
        /// Fully processed rows.
        evaluations: u64,
    },
}

fn hash(bytes: &[u8]) -> String {
    let mut value = String::from("sha256:");
    for byte in Sha256::digest(bytes) {
        write!(value, "{byte:02x}").expect("string write");
    }
    value
}

fn input_hash(input: &SampleMatrix, rows: usize, columns: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update((rows as u64).to_le_bytes());
    hasher.update((columns as u64).to_le_bytes());
    for row in input.rows() {
        for value in row {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    let mut input_hash = String::new();
    for byte in hasher.finalize() {
        write!(input_hash, "{byte:02x}").expect("string write");
    }
    input_hash
}

fn finish_evpi(payload: &[u8], rows: usize, columns: usize) -> Result<f64, String> {
    let mut strategies = Vec::new();
    strategies
        .try_reserve_exact(columns)
        .map_err(|_| "allocation failed")?;
    for _ in 0..columns {
        let mut values = Vec::new();
        values
            .try_reserve_exact(rows)
            .map_err(|_| "allocation failed")?;
        strategies.push(values);
    }
    let mut perfect = Vec::new();
    perfect
        .try_reserve_exact(rows)
        .map_err(|_| "allocation failed")?;
    for record in payload.chunks_exact((columns + 1) * 8) {
        for (i, values) in strategies.iter_mut().enumerate() {
            values.push(f64::from_bits(u64::from_le_bytes(
                record[i * 8..i * 8 + 8]
                    .try_into()
                    .map_err(|_| "invalid payload")?,
            )));
        }
        perfect.push(f64::from_bits(u64::from_le_bytes(
            record[columns * 8..columns * 8 + 8]
                .try_into()
                .map_err(|_| "invalid payload")?,
        )));
    }
    perfect.sort_by(f64::total_cmp);
    let perfect_sum: f64 = perfect.into_iter().sum();
    let current = strategies
        .into_iter()
        .map(|mut v| {
            v.sort_by(f64::total_cmp);
            v.into_iter().sum::<f64>()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    Ok((perfect_sum - current).max(0.0))
}

fn check_memory_budget(rows: usize, columns: usize) -> Result<(), String> {
    const MAX_PAYLOAD: usize = 64 * 1024 * 1024;
    let payload_size = rows
        .checked_mul(columns + 1)
        .and_then(|n| n.checked_mul(8))
        .ok_or("input too large")?;
    if payload_size > MAX_PAYLOAD {
        return Err("checkpoint exceeds memory limit".into());
    }
    // The completed reduction retains the payload and decoded contributions,
    // plus one vector descriptor per strategy. Account for these separately.
    let working_bytes = payload_size
        .checked_mul(2)
        .and_then(|size| {
            columns
                .checked_mul(std::mem::size_of::<Vec<f64>>())
                .and_then(|extra| size.checked_add(extra))
        })
        .ok_or("input too large")?;
    if working_bytes > 256 * 1024 * 1024 {
        return Err("EVPI working memory estimate exceeds limit".into());
    }
    Ok(())
}

fn bound_identity(
    input: &SampleMatrix,
    rows: usize,
    columns: usize,
    identity: &CheckpointIdentity,
) -> Result<CheckpointIdentity, String> {
    let input_hash = input_hash(input, rows, columns);
    CheckpointIdentity::new(
        format!("{}:{input_hash}", identity.model_input),
        &identity.rng,
        format!("{}:evpi-sorted-v1", identity.algorithm),
    )
    .map_err(|e| e.to_string())
}

fn initialize_checkpoint(
    store: &CheckpointStore,
    identity: &CheckpointIdentity,
) -> Result<(), String> {
    let initial =
        ExecutionCheckpoint::new(identity.clone(), 0, 0, hash(&[])).map_err(|e| e.to_string())?;
    store
        .save_with_payload(&initial, &[])
        .map_err(|e| e.to_string())
}

/// Run or resume EVPI in bounded batches, persisting each completed batch.
///
/// The evaluation budget counts newly processed input rows. Input hashing and
/// validation of previously saved contributions revisit input without consuming
/// that budget. Cancellation is cooperative at batch boundaries; hashing,
/// checkpoint I/O, validation and final reduction are not interruptible.
/// The checkpoint payload is limited to 64 MiB and estimated working storage to
/// 256 MiB, excluding the input, allocator overhead and unrelated process memory.
/// A fresh run first writes an empty checkpoint, preventing a previous run's
/// checkpoint from being mistaken for the new one when no batch is admitted.
///
/// # Errors
/// Returns an error for invalid parameters, incompatible checkpoints, or failed persistence.
#[allow(clippy::too_many_arguments)]
pub fn evpi_resumable(
    input: &SampleMatrix,
    store: &CheckpointStore,
    identity: &CheckpointIdentity,
    max_evaluations: u64,
    batch_size: usize,
    cancellation: &CancellationToken,
    resume: bool,
) -> Result<ResumableEvpiOutcome, String> {
    const MAX_PAYLOAD: usize = 64 * 1024 * 1024;
    let [rows, columns] = input.shape();
    let count = u32::try_from(rows).map_err(|_| "too many samples")?;
    if batch_size == 0 {
        return Err("batch size must be positive".into());
    }
    check_memory_budget(rows, columns)?;
    let bound = bound_identity(input, rows, columns, identity)?;
    let (mut completed, mut batches, mut payload, budget) = if resume {
        let (checkpoint, payload, budget) = store
            .load_for_resume(MAX_PAYLOAD, &bound, max_evaluations)
            .map_err(|e| e.to_string())?;
        let completed =
            usize::try_from(checkpoint.evaluations).map_err(|_| "invalid evaluation count")?;
        if completed > rows
            || checkpoint.completed_batches > checkpoint.evaluations
            || (checkpoint.completed_batches == 0) != (completed == 0)
            || payload.len()
                != completed
                    .checked_mul(columns + 1)
                    .and_then(|n| n.checked_mul(8))
                    .ok_or("invalid payload")?
            || checkpoint.replay != ReplayContract::default()
        {
            return Err("invalid EVPI checkpoint".into());
        }
        for (row, record) in input
            .rows()
            .take(completed)
            .zip(payload.chunks_exact((columns + 1) * 8))
        {
            let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for (value, encoded) in row
                .iter()
                .copied()
                .chain(std::iter::once(maximum))
                .zip(record.chunks_exact(8))
            {
                if encoded != (value / f64::from(count)).to_bits().to_le_bytes() {
                    return Err("checkpoint payload does not match EVPI input".into());
                }
            }
        }
        (completed, checkpoint.completed_batches, payload, budget)
    } else {
        initialize_checkpoint(store, &bound)?;
        (0, 0, Vec::new(), ExecutionBudget::new(max_evaluations))
    };
    while completed < rows {
        let n = batch_size.min(rows - completed);
        match cancellation.reserve_batch(&budget, n as u64) {
            Err(_) => {
                return Ok(ResumableEvpiOutcome::Cancelled {
                    evaluations: completed as u64,
                })
            }
            Ok(false) => {
                return Ok(ResumableEvpiOutcome::Partial {
                    evaluations: completed as u64,
                })
            }
            Ok(true) => {}
        }
        payload
            .try_reserve(n * (columns + 1) * 8)
            .map_err(|_| "allocation failed")?;
        for row in input.rows().skip(completed).take(n) {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for value in row {
                payload.extend_from_slice(&(value / f64::from(count)).to_bits().to_le_bytes());
            }
            payload.extend_from_slice(&(max / f64::from(count)).to_bits().to_le_bytes());
        }
        completed += n;
        batches = batches
            .checked_add(1)
            .ok_or("invalid checkpoint batch count")?;
        let checkpoint =
            ExecutionCheckpoint::new(bound.clone(), batches, completed as u64, hash(&payload))
                .map_err(|e| e.to_string())?;
        store
            .save_with_payload(&checkpoint, &payload)
            .map_err(|e| e.to_string())?;
    }
    if cancellation.is_cancelled() {
        return Ok(ResumableEvpiOutcome::Cancelled {
            evaluations: completed as u64,
        });
    }
    let value = finish_evpi(&payload, rows, columns)?;
    if cancellation.is_cancelled() {
        return Ok(ResumableEvpiOutcome::Cancelled {
            evaluations: completed as u64,
        });
    }
    Ok(ResumableEvpiOutcome::Complete {
        value,
        evaluations: completed as u64,
    })
}
