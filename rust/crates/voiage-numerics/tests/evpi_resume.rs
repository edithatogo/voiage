//! Interrupted EVPI execution must resume without changing the result.
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use voiage_diagnostics::{CancellationToken, CheckpointIdentity, CheckpointStore};
use voiage_domain::SampleMatrix;
use voiage_numerics::{evpi, evpi_resumable, ResumableEvpiOutcome};

#[test]
fn interrupted_evpi_resumes_and_rejects_changed_input() {
    let input =
        SampleMatrix::try_from(vec![vec![10.0, 1.0], vec![2.0, 8.0], vec![4.0, 5.0]]).unwrap();
    let changed =
        SampleMatrix::try_from(vec![vec![10.0, 1.0], vec![2.0, 9.0], vec![4.0, 5.0]]).unwrap();
    let directory = std::env::temp_dir().join(format!("voiage-evpi-resume-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let store = CheckpointStore::new(directory.join("checkpoint.bin"));
    let identity = CheckpointIdentity::new("model", "deterministic", "evpi").unwrap();
    let cancellation = CancellationToken::new();
    let partial = evpi_resumable(&input, &store, &identity, 2, 2, &cancellation, false).unwrap();
    assert!(matches!(
        partial,
        ResumableEvpiOutcome::Partial { evaluations: 2 }
    ));
    let (original_checkpoint, original_payload) = store.load_with_payload(4096).unwrap();
    let invalid_identity =
        CheckpointIdentity::new("x".repeat(1_048_577), "deterministic", "evpi").unwrap();
    assert!(evpi_resumable(
        &input,
        &store,
        &invalid_identity,
        3,
        1,
        &cancellation,
        false
    )
    .is_err());
    assert_eq!(
        store.load_with_payload(4096).unwrap(),
        (original_checkpoint.clone(), original_payload.clone())
    );
    let mut wrong_replay = original_checkpoint.clone();
    wrong_replay.replay.cpu_tolerance_bits = 1.0_f64.to_bits();
    store
        .save_with_payload(&wrong_replay, &original_payload)
        .unwrap();
    assert!(evpi_resumable(&input, &store, &identity, 3, 2, &cancellation, true).is_err());
    store
        .save_with_payload(&original_checkpoint, &original_payload)
        .unwrap();
    let mut forged_batches = original_checkpoint.clone();
    forged_batches.completed_batches = u64::MAX;
    store
        .save_with_payload(&forged_batches, &original_payload)
        .unwrap();
    assert!(evpi_resumable(&input, &store, &identity, 3, 2, &cancellation, true).is_err());
    store
        .save_with_payload(&original_checkpoint, &original_payload)
        .unwrap();
    let mut forged_payload = original_payload.clone();
    forged_payload[0] ^= 1;
    let mut forged_digest = String::from("sha256:");
    for byte in Sha256::digest(&forged_payload) {
        write!(forged_digest, "{byte:02x}").unwrap();
    }
    let mut forged_checkpoint = original_checkpoint.clone();
    forged_checkpoint.payload_digest = forged_digest;
    store
        .save_with_payload(&forged_checkpoint, &forged_payload)
        .unwrap();
    assert!(evpi_resumable(&input, &store, &identity, 3, 2, &cancellation, true).is_err());
    store
        .save_with_payload(&original_checkpoint, &original_payload)
        .unwrap();
    let stopped = CancellationToken::new();
    stopped.cancel();
    assert!(matches!(
        evpi_resumable(&input, &store, &identity, 3, 2, &stopped, true).unwrap(),
        ResumableEvpiOutcome::Cancelled { evaluations: 2 }
    ));
    assert_eq!(
        store.load_with_payload(4096).unwrap(),
        (original_checkpoint, original_payload)
    );
    assert!(evpi_resumable(&changed, &store, &identity, 3, 2, &cancellation, true).is_err());
    let resumed = evpi_resumable(&input, &store, &identity, 3, 2, &cancellation, true).unwrap();
    let ResumableEvpiOutcome::Complete { value, evaluations } = resumed else {
        panic!("expected complete result")
    };
    assert_eq!(evaluations, 3);
    assert!((value - evpi(&input).unwrap()).abs() <= 1e-12);
    assert!(matches!(
        evpi_resumable(&input, &store, &identity, 3, 2, &stopped, true).unwrap(),
        ResumableEvpiOutcome::Cancelled { evaluations: 3 }
    ));
    std::fs::remove_file(directory.join("checkpoint.bin")).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn zero_budget_and_cancellation_never_complete() {
    let input = SampleMatrix::try_from(vec![vec![10.0, 1.0], vec![2.0, 8.0]]).unwrap();
    let directory =
        std::env::temp_dir().join(format!("voiage-evpi-boundary-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let store = CheckpointStore::new(directory.join("checkpoint.bin"));
    let identity = CheckpointIdentity::new("model", "deterministic", "evpi").unwrap();
    let token = CancellationToken::new();
    assert!(matches!(
        evpi_resumable(&input, &store, &identity, 0, 1, &token, false).unwrap(),
        ResumableEvpiOutcome::Partial { evaluations: 0 }
    ));
    assert_eq!(store.load_with_payload(0).unwrap().0.evaluations, 0);
    token.cancel();
    assert!(matches!(
        evpi_resumable(&input, &store, &identity, 2, 1, &token, false).unwrap(),
        ResumableEvpiOutcome::Cancelled { evaluations: 0 }
    ));
    assert_eq!(store.load_with_payload(0).unwrap().0.evaluations, 0);
    std::fs::remove_file(directory.join("checkpoint.bin")).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn new_zero_budget_run_does_not_inherit_stale_checkpoint() {
    let directory = std::env::temp_dir().join(format!("voiage-evpi-stale-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let store = CheckpointStore::new(directory.join("checkpoint.bin"));
    let input = SampleMatrix::try_from(vec![vec![10.0, 1.0], vec![2.0, 8.0]]).unwrap();
    let identity = CheckpointIdentity::new("model", "deterministic", "evpi").unwrap();
    let token = CancellationToken::new();
    assert!(matches!(
        evpi_resumable(&input, &store, &identity, 1, 1, &token, false).unwrap(),
        ResumableEvpiOutcome::Partial { evaluations: 1 }
    ));
    assert!(matches!(
        evpi_resumable(&input, &store, &identity, 0, 1, &token, false).unwrap(),
        ResumableEvpiOutcome::Partial { evaluations: 0 }
    ));
    assert!(matches!(
        evpi_resumable(&input, &store, &identity, 0, 1, &token, true).unwrap(),
        ResumableEvpiOutcome::Partial { evaluations: 0 }
    ));
    std::fs::remove_file(directory.join("checkpoint.bin")).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn oversized_checkpoint_is_rejected_before_execution() {
    let input = SampleMatrix::try_from(vec![vec![1.0; 8_388_608]]).unwrap();
    let directory = std::env::temp_dir().join(format!("voiage-evpi-limit-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let store = CheckpointStore::new(directory.join("checkpoint.bin"));
    let identity = CheckpointIdentity::new("model", "deterministic", "evpi").unwrap();
    let result = evpi_resumable(
        &input,
        &store,
        &identity,
        1,
        1,
        &CancellationToken::new(),
        false,
    );
    assert!(result.unwrap_err().contains("memory limit"));
    assert!(!directory.join("checkpoint.bin").exists());
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn excessive_strategy_vector_overhead_is_rejected() {
    let input = SampleMatrix::try_from(vec![vec![1.0; 7_000_000]]).unwrap();
    let directory =
        std::env::temp_dir().join(format!("voiage-evpi-overhead-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let store = CheckpointStore::new(directory.join("checkpoint.bin"));
    let identity = CheckpointIdentity::new("model", "deterministic", "evpi").unwrap();
    let result = evpi_resumable(
        &input,
        &store,
        &identity,
        1,
        1,
        &CancellationToken::new(),
        false,
    );
    assert!(result.unwrap_err().contains("working memory estimate"));
    assert!(!directory.join("checkpoint.bin").exists());
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn resumed_evpi_matches_existing_reduction_for_varied_scales() {
    let directory = std::env::temp_dir().join(format!("voiage-evpi-scales-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("checkpoint.bin");
    let store = CheckpointStore::new(&path);
    let identity = CheckpointIdentity::new("model", "deterministic", "evpi").unwrap();
    for scale in [1e-200, 1.0, 1e200] {
        let input = SampleMatrix::try_from(vec![
            vec![10.0 * scale, scale],
            vec![2.0 * scale, 8.0 * scale],
            vec![-3.0 * scale, 5.0 * scale],
        ])
        .unwrap();
        let token = CancellationToken::new();
        assert!(matches!(
            evpi_resumable(&input, &store, &identity, 1, 1, &token, false).unwrap(),
            ResumableEvpiOutcome::Partial { evaluations: 1 }
        ));
        let outcome = evpi_resumable(&input, &store, &identity, 3, 1, &token, true).unwrap();
        let ResumableEvpiOutcome::Complete { value, .. } = outcome else {
            panic!("expected completion")
        };
        assert_eq!(value.to_bits(), evpi(&input).unwrap().to_bits());
    }
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
