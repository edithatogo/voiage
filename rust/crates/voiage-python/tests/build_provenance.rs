#![allow(dead_code, missing_docs)]

#[path = "../build.rs"]
mod build_script;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new(label: &str) -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "voiage-build-provenance-{label}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create temporary directory");
        Self(path)
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove temporary directory");
    }
}

fn git(directory: &Path, arguments: &[&str]) -> String {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(directory)
        .output()
        .expect("run git");
    assert!(output.status.success(), "git command failed: {output:?}");
    String::from_utf8(output.stdout)
        .expect("UTF-8 git output")
        .trim()
        .to_owned()
}

fn repository() -> (TemporaryDirectory, String, String) {
    let directory = TemporaryDirectory::new("repository");
    git(&directory.0, &["init", "--quiet"]);
    git(&directory.0, &["config", "user.name", "Voiage Test"]);
    git(
        &directory.0,
        &["config", "user.email", "voiage@example.invalid"],
    );
    fs::write(directory.0.join("tracked"), "content").expect("write tracked file");
    git(&directory.0, &["add", "tracked"]);
    git(&directory.0, &["commit", "--quiet", "-m", "fixture"]);
    let revision = git(&directory.0, &["rev-parse", "HEAD"]);
    let tree = git(&directory.0, &["rev-parse", "HEAD^{tree}"]);
    (directory, revision, tree)
}

#[test]
fn gitless_release_accepts_complete_valid_workflow_identity() {
    let directory = TemporaryDirectory::new("archive");
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let tree = "89abcdef0123456789abcdef0123456789abcdef";

    assert_eq!(
        build_script::resolve_source_identity(
            &directory.0,
            Some(revision),
            Some(tree),
            Some("true"),
            true,
        ),
        Ok((revision.to_owned(), tree.to_owned(), false))
    );
}

#[test]
fn release_without_git_or_workflow_identity_fails_closed() {
    let directory = TemporaryDirectory::new("unidentified");
    let error = build_script::resolve_source_identity(&directory.0, None, None, None, true)
        .expect_err("release identity must be required");
    assert!(error.contains("release builds require"));
    assert_eq!(
        build_script::resolve_source_identity(&directory.0, None, None, None, false),
        Ok(("unknown".to_owned(), "unknown".to_owned(), true))
    );
}

#[test]
fn malformed_or_incomplete_workflow_identity_is_rejected() {
    let directory = TemporaryDirectory::new("malformed");
    let tree = "89abcdef0123456789abcdef0123456789abcdef";
    for revision in [
        "0123456789abcdef0123456789abcdef0123456",
        "0123456789ABCDEF0123456789ABCDEF01234567",
        "g123456789abcdef0123456789abcdef01234567",
    ] {
        assert!(build_script::resolve_source_identity(
            &directory.0,
            Some(revision),
            Some(tree),
            Some("true"),
            true,
        )
        .is_err());
    }
    assert!(build_script::resolve_source_identity(
        &directory.0,
        Some("0123456789abcdef0123456789abcdef01234567"),
        Some(tree),
        Some("True"),
        true,
    )
    .is_err());
    assert!(build_script::resolve_source_identity(
        &directory.0,
        Some("0123456789abcdef0123456789abcdef01234567"),
        None,
        Some("true"),
        true,
    )
    .is_err());
}

#[test]
fn supplied_identity_must_match_git_revision_tree_and_cleanliness() {
    let (directory, revision, tree) = repository();
    assert_eq!(
        build_script::resolve_source_identity(
            &directory.0,
            Some(&revision),
            Some(&tree),
            Some("true"),
            true,
        ),
        Ok((revision.clone(), tree.clone(), false))
    );

    for (candidate_revision, candidate_tree, clean) in [
        (
            "0123456789abcdef0123456789abcdef01234567",
            tree.as_str(),
            "true",
        ),
        (
            revision.as_str(),
            "89abcdef0123456789abcdef0123456789abcdef",
            "true",
        ),
        (revision.as_str(), tree.as_str(), "false"),
    ] {
        assert!(build_script::resolve_source_identity(
            &directory.0,
            Some(candidate_revision),
            Some(candidate_tree),
            Some(clean),
            true,
        )
        .is_err());
    }
}

#[test]
fn untracked_files_make_git_identity_dirty() {
    let (directory, revision, tree) = repository();
    fs::write(directory.0.join("untracked"), "content").expect("write untracked file");
    assert_eq!(
        build_script::resolve_source_identity(&directory.0, None, None, None, true),
        Ok((revision, tree, true))
    );
}

#[test]
fn distinct_dirty_source_states_from_the_same_commit_have_distinct_digests() {
    let (directory, _, tree) = repository();
    fs::write(directory.0.join("tracked"), "first dirty state").expect("modify tracked file");
    let first = build_script::source_state_sha256(&directory.0, &tree, true)
        .expect("hash first dirty state");
    fs::write(directory.0.join("tracked"), "second dirty state").expect("modify tracked file");
    let second = build_script::source_state_sha256(&directory.0, &tree, true)
        .expect("hash second dirty state");
    assert_ne!(first, second);
    let build_id = |state: &str| {
        build_script::build_id_from_parts(
            "0123456789abcdef0123456789abcdef01234567",
            &tree,
            true,
            state,
            "aarch64-apple-darwin",
            "rustc 1.96.0 (fixture 2026-01-01)",
            "release",
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            Some("1780000000"),
        )
    };
    assert_ne!(build_id(&first), build_id(&second));

    fs::write(directory.0.join("new-source.rs"), "fn first() {}").expect("write source input");
    let with_untracked = build_script::source_state_sha256(&directory.0, &tree, true)
        .expect("hash untracked source state");
    assert_ne!(second, with_untracked);
}

#[test]
fn clean_source_state_is_deterministic_and_ignores_generated_outputs() {
    let (directory, _, tree) = repository();
    let expected =
        build_script::source_state_sha256(&directory.0, &tree, false).expect("hash clean state");
    fs::create_dir(directory.0.join("target")).expect("create target directory");
    fs::write(directory.0.join("target/generated"), "unstable").expect("write generated file");
    assert_eq!(
        build_script::source_state_sha256(&directory.0, &tree, false).expect("rehash clean state"),
        expected
    );
}

#[test]
fn clean_source_state_matches_the_independent_length_prefixed_vector() {
    let directory = TemporaryDirectory::new("clean-vector");
    let tree = "89abcdef0123456789abcdef0123456789abcdef";
    assert_eq!(
        build_script::source_state_sha256(&directory.0, tree, false)
            .expect("hash committed tree identity"),
        "671d87c6d79f057a0d79c65fc970d626a48cfbb598dbf6b937b8dc4bc1dfaae6"
    );
}

#[cfg(unix)]
#[test]
fn relevant_untracked_symlinks_fail_closed() {
    use std::os::unix::fs::symlink;

    let (directory, _, tree) = repository();
    symlink("tracked", directory.0.join("linked-source.rs")).expect("create source symlink");
    let error = build_script::source_state_sha256(&directory.0, &tree, true)
        .expect_err("source symlink must be rejected");
    assert!(error.contains("not a regular file"));
}

#[test]
fn rustc_version_must_have_a_valid_probe_shape() {
    assert!(build_script::parse_rustc_version("rustc 1.96.0 (abc 2026-01-01)").is_ok());
    for malformed in ["", "rustc unknown", "cargo 1.96.0", "rustc 1.96"] {
        assert!(build_script::parse_rustc_version(malformed).is_err());
    }
}

#[test]
fn embedded_archive_identity_is_exact_and_fail_closed() {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let tree = "89abcdef0123456789abcdef0123456789abcdef";
    assert_eq!(
        build_script::parse_embedded_source_identity(&format!(
            "revision={revision}\ntree={tree}\nclean=true\n"
        )),
        Ok((revision.to_owned(), tree.to_owned(), false))
    );
    for malformed in [
        format!("revision={revision}\ntree={tree}\n"),
        format!("revision={revision}\ntree={tree}\nclean=false\n"),
        format!("revision=UNKNOWN\ntree={tree}\nclean=true\n"),
        format!("tree={tree}\nrevision={revision}\nclean=true\n"),
    ] {
        assert!(build_script::parse_embedded_source_identity(&malformed).is_err());
    }
}
