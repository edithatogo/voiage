//! Payload binding and atomic replacement for resumable checkpoints.

use voiage_diagnostics::{
    CancellationToken, CheckpointError, CheckpointIdentity, CheckpointStore, ExecutionCheckpoint,
};

#[test]
fn checkpoint_payload_is_bound_and_failed_replacement_preserves_pair() {
    let directory =
        std::env::temp_dir().join(format!("voiage-checkpoint-payload-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("checkpoint.json");
    let store = CheckpointStore::new(&path);
    let identity = CheckpointIdentity::new("model", "seed", "algorithm").unwrap();
    let checkpoint = ExecutionCheckpoint::new(
        identity.clone(),
        1,
        3,
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )
    .unwrap();
    store.save_with_payload(&checkpoint, b"abc").unwrap();
    for changed in [
        CheckpointIdentity::new("other", "seed", "algorithm").unwrap(),
        CheckpointIdentity::new("model", "other", "algorithm").unwrap(),
        CheckpointIdentity::new("model", "seed", "other").unwrap(),
    ] {
        assert!(store.load_for_resume(3, &changed, 3).is_err());
    }
    assert!(store.load_for_resume(3, &checkpoint.identity, 2).is_err());
    let (loaded, payload, budget) = store.load_for_resume(3, &checkpoint.identity, 4).unwrap();
    assert_eq!((loaded, payload), (checkpoint.clone(), b"abc".to_vec()));
    assert_eq!(budget.remaining(), 1);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        cancellation.reserve_batch(&budget, 1),
        Err(CheckpointError::Cancelled)
    );
    assert_eq!(budget.remaining(), 1);
    assert!(!CancellationToken::new().reserve_batch(&budget, 2).unwrap());
    assert_eq!(budget.remaining(), 1);
    assert_eq!(
        store.load_with_payload(3).unwrap(),
        (checkpoint.clone(), b"abc".to_vec())
    );
    let wrong = ExecutionCheckpoint::new(identity, 2, 4, "sha256:incorrect").unwrap();
    assert!(store.save_with_payload(&wrong, b"other").is_err());
    assert_eq!(
        store.load_with_payload(3).unwrap(),
        (checkpoint.clone(), b"abc".to_vec())
    );
    let mut malformed = checkpoint.clone();
    malformed.identity.model_input.clear();
    assert!(store.save_with_payload(&malformed, b"abc").is_err());
    malformed.identity.model_input = "x".repeat(1_048_577);
    assert!(store.save_with_payload(&malformed, b"abc").is_err());
    assert_eq!(
        store.load_with_payload(3).unwrap(),
        (checkpoint.clone(), b"abc".to_vec())
    );
    assert!(store.load_with_payload(2).is_err());
    let mut bytes = std::fs::read(&path).unwrap();
    let position = bytes
        .windows(3)
        .position(|window| window == b"abc")
        .unwrap();
    bytes[position] = b'z';
    std::fs::write(&path, bytes).unwrap();
    assert!(store.load_with_payload(3).is_err());
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn checkpoint_payload_rejects_truncated_trailing_and_forged_lengths() {
    let directory =
        std::env::temp_dir().join(format!("voiage-checkpoint-invalid-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("checkpoint.bin");
    let store = CheckpointStore::new(&path);
    let identity = CheckpointIdentity::new("model", "seed", "algorithm").unwrap();
    let checkpoint = ExecutionCheckpoint::new(
        identity,
        1,
        1,
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )
    .unwrap();
    store.save_with_payload(&checkpoint, b"abc").unwrap();
    let original = std::fs::read(&path).unwrap();
    for bytes in [
        &original[..original.len() - 1],
        &[original.as_slice(), b"x"].concat()[..],
    ] {
        std::fs::write(&path, bytes).unwrap();
        assert!(store.load_with_payload(usize::MAX).is_err());
    }
    let mut forged = original;
    forged[17..25].copy_from_slice(&u64::MAX.to_le_bytes());
    std::fs::write(&path, forged).unwrap();
    assert!(store.load_with_payload(usize::MAX).is_err());
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
