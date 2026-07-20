//! Contract tests for validated dimension-aware collections.

use voiage_domain::{DomainError, SampleCube, SampleMatrix, SampleVector, StrategyCollection};

#[test]
fn sample_vector_validates_and_exposes_values() {
    let vector = SampleVector::try_from(vec![1.0, -2.5, 0.0]).unwrap();
    assert_eq!(vector.len(), 3);
    assert_eq!(vector.as_slice(), &[1.0, -2.5, 0.0]);

    assert_eq!(SampleVector::try_from(Vec::new()), Err(DomainError::Empty));
    assert_eq!(
        SampleVector::try_from(vec![f64::NAN]),
        Err(DomainError::NonFinite { index: 0 })
    );
}

#[test]
fn sample_matrix_rejects_empty_ragged_and_non_finite_rows() {
    let matrix = SampleMatrix::try_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    assert_eq!(matrix.shape(), [2, 2]);
    assert_eq!(matrix.row(1), Some(&[3.0, 4.0][..]));
    assert_eq!(matrix.get(1, 0), Some(3.0));

    assert_eq!(SampleMatrix::try_from(Vec::new()), Err(DomainError::Empty));
    assert_eq!(
        SampleMatrix::try_from(vec![Vec::new()]),
        Err(DomainError::EmptyDimension { dimension: 1 })
    );
    assert_eq!(
        SampleMatrix::try_from(vec![vec![1.0], vec![2.0, 3.0]]),
        Err(DomainError::Ragged {
            dimension: 1,
            expected: 1,
            actual: 2,
            index: 1,
        })
    );
    assert_eq!(
        SampleMatrix::try_from(vec![vec![1.0, f64::INFINITY]]),
        Err(DomainError::NonFinite { index: 1 })
    );
}

#[test]
fn sample_cube_rejects_mismatched_planes_and_preserves_shape() {
    let cube = SampleCube::try_from(vec![
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![5.0, 6.0], vec![7.0, 8.0]],
    ])
    .unwrap();
    assert_eq!(cube.shape(), [2, 2, 2]);
    assert_eq!(cube.get(1, 0, 1), Some(6.0));

    assert_eq!(
        SampleCube::try_from(vec![vec![vec![1.0]], vec![vec![2.0], vec![3.0]]]),
        Err(DomainError::Ragged {
            dimension: 1,
            expected: 1,
            actual: 2,
            index: 1,
        })
    );
    assert_eq!(
        SampleCube::try_from(vec![vec![vec![1.0], vec![2.0, 3.0]]]),
        Err(DomainError::Ragged {
            dimension: 2,
            expected: 1,
            actual: 2,
            index: 1,
        })
    );
}

#[test]
fn strategy_collection_rejects_blank_and_duplicate_names() {
    let strategies = StrategyCollection::try_from(vec!["standard care", "intervention"]).unwrap();
    assert_eq!(strategies.len(), 2);
    assert_eq!(strategies.get(1), Some("intervention"));
    let normalized =
        StrategyCollection::try_from(vec![" standard care ", " intervention "]).unwrap();
    assert_eq!(normalized.get(0), Some("standard care"));
    assert_eq!(normalized.get(1), Some("intervention"));

    assert_eq!(
        StrategyCollection::try_from(Vec::<&str>::new()),
        Err(DomainError::Empty)
    );
    assert_eq!(
        StrategyCollection::try_from(vec!["standard care", "  "]),
        Err(DomainError::BlankStrategy { index: 1 })
    );
    assert_eq!(
        StrategyCollection::try_from(vec!["standard care", "standard care"]),
        Err(DomainError::DuplicateStrategy { index: 1 })
    );
    assert_eq!(
        StrategyCollection::try_from(vec!["standard care", " standard care "]),
        Err(DomainError::DuplicateStrategy { index: 1 })
    );
}

#[test]
fn serde_deserialization_is_fail_closed() {
    let vector: SampleVector = serde_json::from_str("[1.0, 2.0]").unwrap();
    assert_eq!(serde_json::to_string(&vector).unwrap(), "[1.0,2.0]");
    assert!(serde_json::from_str::<SampleVector>("[1.0, null]").is_err());
    assert!(serde_json::from_str::<SampleVector>("[]").is_err());

    let matrix: SampleMatrix = serde_json::from_str("[[1.0,2.0],[3.0,4.0]]").unwrap();
    assert_eq!(matrix.shape(), [2, 2]);
    assert!(serde_json::from_str::<SampleMatrix>("[[1.0],[2.0,3.0]]").is_err());
    assert!(serde_json::from_str::<SampleMatrix>("[[1e400]]").is_err());

    let cube: SampleCube = serde_json::from_str("[[[1.0]],[[2.0]]]").unwrap();
    assert_eq!(cube.shape(), [2, 1, 1]);
    assert!(serde_json::from_str::<SampleCube>("[[[1.0]],[[2.0],[3.0]]]").is_err());

    let strategies: StrategyCollection =
        serde_json::from_str(r#"["standard care","intervention"]"#).unwrap();
    assert_eq!(strategies.get(0), Some("standard care"));
    assert!(serde_json::from_str::<StrategyCollection>(r#"["same","same"]"#).is_err());
}
