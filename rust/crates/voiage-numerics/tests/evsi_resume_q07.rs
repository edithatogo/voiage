//! Q07 streaming EVSI checkpoint/restart and cancellation exact-bit replay tests.

use voiage_diagnostics::{CancellationToken, CheckpointIdentity, CheckpointStore};
use voiage_domain::SampleMatrix;
use voiage_numerics::{evsi_importance_sampling_raw, evsi_streaming_resumable, ResumableEvsiOutcome};

#[test]
fn q07_streaming_evsi_resumes_exact_bit_and_respects_cancellation() {
    let nb = SampleMatrix::try_from(vec![
        vec![0.0, 1.0],
        vec![0.0, 2.0],
        vec![0.0, 3.0],
        vec![0.0, -1.0],
        vec![0.0, -2.0],
    ])
    .unwrap();
    let lw = SampleMatrix::try_from(vec![
        vec![0.1, 0.2, 0.3, -0.1, -0.2],
        vec![-0.1, -0.2, -0.3, 0.1, 0.2],
        vec![0.2, 0.1, 0.0, -0.1, -0.2],
        vec![0.0, 0.1, 0.2, 0.0, -0.1],
    ])
    .unwrap();

    let expected = evsi_importance_sampling_raw(&nb, &lw).unwrap().evsi;
    let dir = std::env::temp_dir().join(format!("voiage-evsi-resume-q07-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store = CheckpointStore::new(dir.join("checkpoint.bin"));
    let id = CheckpointIdentity::new("q07-evsi-run", "deterministic", "evsi").unwrap();
    let cancel = CancellationToken::new();

    let part = evsi_streaming_resumable(&nb, &lw, &store, &id, 2, 2, &cancel, false).unwrap();
    assert_eq!(part, ResumableEvsiOutcome::Partial { evaluations: 2 });

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let c_out = evsi_streaming_resumable(&nb, &lw, &store, &id, 10, 2, &cancelled, true).unwrap();
    assert_eq!(c_out, ResumableEvsiOutcome::Cancelled { evaluations: 2 });

    let done = evsi_streaming_resumable(&nb, &lw, &store, &id, 10, 2, &cancel, true).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    match done {
        ResumableEvsiOutcome::Complete { evsi, evaluations } => {
            assert_eq!(evaluations, 4);
            assert_eq!(evsi.to_bits(), expected.to_bits());
        }
        other => panic!("expected Complete, got {other:?}"),
    }
}
