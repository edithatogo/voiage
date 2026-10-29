use voiage_domain::SampleMatrix;

use crate::NumericalInputError;

/// Computes expected value of perfect information from `[sample][strategy]`
/// net-benefit values.
///
/// # Errors
///
/// The matrix is validated by [`SampleMatrix`]. This result-bearing signature
/// preserves a uniform numerical-kernel boundary for future checked failures.
pub fn evpi(net_benefit: &SampleMatrix) -> Result<f64, NumericalInputError> {
    let [sample_count, strategy_count] = net_benefit.shape();
    let divisor = f64::from(u32::try_from(sample_count).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "sample count exceeds the supported numerical range",
        )
    })?);
    let mut strategy_values = vec![Vec::with_capacity(sample_count); strategy_count];
    let mut perfect_information_values = Vec::with_capacity(sample_count);

    for row in net_benefit.rows() {
        let mut row_maximum = f64::NEG_INFINITY;
        for (strategy_index, value) in row.iter().copied().enumerate() {
            strategy_values[strategy_index].push(value / divisor);
            row_maximum = row_maximum.max(value);
        }
        perfect_information_values.push(row_maximum / divisor);
    }

    perfect_information_values.sort_by(f64::total_cmp);
    let perfect_information_mean: f64 = perfect_information_values.into_iter().sum();
    let current_information_mean = strategy_values
        .into_iter()
        .map(|mut values| {
            values.sort_by(f64::total_cmp);
            values.into_iter().sum::<f64>()
        })
        .fold(f64::NEG_INFINITY, f64::max);

    Ok((perfect_information_mean - current_information_mean).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::evpi;
    use voiage_domain::SampleMatrix;

    fn matrix(values: Vec<Vec<f64>>) -> SampleMatrix {
        values.try_into().expect("valid matrix")
    }

    fn registered_tests_are_live(
        value: &serde_json::Value,
        listing: &str,
        ignored: &str,
    ) -> Result<(), String> {
        if value["schema_version"] != "1.0" {
            return Err("unsupported case register".into());
        }
        let cases = value["independent_references"]
            .as_array()
            .filter(|cases| !cases.is_empty())
            .ok_or("case list must not be empty")?;
        let live: std::collections::HashSet<_> = listing
            .lines()
            .filter_map(|line| line.strip_suffix(": test"))
            .collect();
        let ignored: std::collections::HashSet<_> = ignored
            .lines()
            .filter_map(|line| line.strip_suffix(": test"))
            .collect();
        for case in cases {
            let id = case["case_id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .ok_or("missing case id")?;
            let target = case["implementation_test"]
                .as_str()
                .ok_or("missing test target")?;
            let name = target
                .strip_prefix("rust/crates/voiage-numerics/src/evpi.rs::")
                .filter(|name| !name.is_empty() && !name.contains("::"))
                .ok_or_else(|| format!("unsupported test target: {target}"))?;
            let qualified = format!("evpi::tests::{name}");
            if !live.contains(qualified.as_str()) || ignored.contains(qualified.as_str()) {
                return Err(format!("case {id} names missing test {qualified}"));
            }
        }
        Ok(())
    }

    #[test]
    fn numerical_case_register_names_compiled_tests() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        if !root.join("rust/Cargo.toml").is_file() {
            // Published standalone crates do not contain the repository register.
            eprintln!("repository case register unavailable in standalone crate");
            return;
        }
        let candidates = ["tracks", "archive"].map(|location| {
            root.join("conductor")
                .join(location)
                .join("agent_safe_engineering_20260907/numerical-case-register.json")
        });
        let present: Vec<_> = candidates.iter().filter(|path| path.is_file()).collect();
        assert_eq!(
            present.len(),
            1,
            "exactly one active or archived numerical case register must be present"
        );
        let bytes = std::fs::read(present[0]).expect("numerical case register must be readable");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON register");
        let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .arg("--list")
            .output()
            .expect("enumerate compiled tests");
        assert!(output.status.success(), "test enumeration failed");
        let listing = String::from_utf8(output.stdout).expect("UTF-8 test listing");
        let ignored = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--list", "--ignored"])
            .output()
            .expect("enumerate ignored tests");
        assert!(ignored.status.success(), "ignored test enumeration failed");
        let ignored = String::from_utf8(ignored.stdout).expect("UTF-8 ignored test listing");
        registered_tests_are_live(&value, &listing, &ignored)
            .expect("every registered case must name a live test");
    }

    #[test]
    fn numerical_case_register_rejects_missing_and_malformed_targets() {
        let case = serde_json::json!({"schema_version": "1.0", "independent_references": [{
            "case_id": "example", "implementation_test": "rust/crates/voiage-numerics/src/evpi.rs::exists"
        }]});
        assert!(registered_tests_are_live(&case, "evpi::tests::exists: test\n", "").is_ok());
        assert!(registered_tests_are_live(&case, "evpi::tests::other: test\n", "").is_err());
        assert!(registered_tests_are_live(&case, "evpi::tests::exists: benchmark\n", "").is_err());
        assert!(registered_tests_are_live(
            &case,
            "evpi::tests::exists: test\n",
            "evpi::tests::exists: test\n"
        )
        .is_err());
        let mut wrong_path = case;
        wrong_path["independent_references"][0]["implementation_test"] =
            "elsewhere.rs::exists".into();
        assert!(registered_tests_are_live(&wrong_path, "evpi::tests::exists: test\n", "").is_err());
    }

    #[test]
    fn affine_invariance_holds_for_constant_shift_and_positive_scaling() {
        let base = matrix(vec![vec![1.0, 3.0], vec![4.0, 2.0], vec![5.0, 6.0]]);
        let shifted = matrix(vec![vec![11.0, 13.0], vec![14.0, 12.0], vec![15.0, 16.0]]);
        let scaled = matrix(vec![vec![2.0, 6.0], vec![8.0, 4.0], vec![10.0, 12.0]]);
        let baseline = evpi(&base).expect("baseline");
        assert!(
            (evpi(&shifted).expect("shifted") - baseline).abs() < 1.0e-12 * baseline.max(1.0),
            "baseline={baseline:?} shifted={:?}",
            evpi(&shifted)
        );
        assert!(
            (evpi(&scaled).expect("scaled") - 2.0 * baseline).abs() < 1.0e-12 * baseline.max(1.0)
        );
    }

    #[test]
    fn matrix_contract_rejects_non_finite_inputs_before_evaluation() {
        assert!(SampleMatrix::try_from(vec![vec![1.0, f64::NAN]]).is_err());
        assert!(SampleMatrix::try_from(vec![vec![1.0, f64::INFINITY]]).is_err());
    }

    #[test]
    fn extreme_finite_scale_preserves_the_analytic_evpi() {
        let scale = 0.75 * f64::MAX;
        let values = matrix(vec![vec![scale, 0.0], [0.5 * scale, scale].to_vec()]);
        let expected = 0.25 * scale;
        let actual = evpi(&values).expect("finite extreme-scale result");
        assert!(
            (actual - expected).abs() / expected < 1.0e-12,
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn large_magnitude_translation_preserves_the_analytic_reference() {
        let base = matrix(vec![vec![1.0e300, 1.1e300], vec![1.2e300, 1.05e300]]);
        let shifted = matrix(vec![
            vec![1.000_000_000_1e300, 1.100_000_000_1e300],
            vec![1.200_000_000_1e300, 1.050_000_000_1e300],
        ]);
        let baseline = evpi(&base).expect("finite large-magnitude result");
        let translated = evpi(&shifted).expect("translated large-magnitude result");
        assert!(baseline.is_finite() && baseline > 0.0);
        assert!((translated - baseline).abs() / baseline < 1.0e-10);
    }
}
