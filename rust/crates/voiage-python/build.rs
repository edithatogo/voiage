//! Embed deterministic, source-bound provenance in the private Python extension.

use sha2::{Digest, Sha256};
use std::env;
use std::fmt::Write;
use std::fs;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::Command;

const BUILD_ID_ALGORITHM: &str = "length-prefixed-sha256-v2";
const SOURCE_STATE_ALGORITHM: &str = "git-diff-and-untracked-sha256-v1";
const EMBEDDED_PROVENANCE_FILE: &str = "source-provenance.txt";

fn digest_hex<D: AsRef<[u8]>>(digest: D) -> String {
    let bytes = digest.as_ref();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        hex.push(char::from(b"0123456789abcdef"[(byte >> 4) as usize]));
        hex.push(char::from(b"0123456789abcdef"[(byte & 0xf) as usize]));
    }
    hex
}

fn sha256_hex(bytes: &[u8]) -> String {
    digest_hex(Sha256::digest(bytes))
}

fn command_output(program: &str, args: &[&str], directory: &Path) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(directory)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn command_bytes(program: &str, args: &[&str], directory: &Path) -> Result<Vec<u8>, String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(directory)
        .output()
        .map_err(|error| format!("failed to execute {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!("{program} {} failed", args.join(" ")));
    }
    Ok(output.stdout)
}

fn valid_git_oid(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_clean(value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err("VOIAGE_SOURCE_CLEAN must be exactly 'true' or 'false'".to_owned()),
    }
}

fn git_identity(repository: &Path) -> Option<(String, String, bool)> {
    let revision = command_output(
        "git",
        &["rev-parse", "--verify", "HEAD^{commit}"],
        repository,
    )?;
    let tree = command_output("git", &["rev-parse", "--verify", "HEAD^{tree}"], repository)?;
    if !valid_git_oid(&revision) || !valid_git_oid(&tree) {
        return None;
    }
    let status = command_output(
        "git",
        &["status", "--porcelain=v1", "--untracked-files=normal"],
        repository,
    )?;
    Some((revision, tree, !status.is_empty()))
}

fn excluded_untracked(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(component, Component::Normal(name) if matches!(
            name.to_str(),
            Some(".git" | ".astro" | ".coverage" | ".mypy_cache" | ".pytest_cache" | ".ruff_cache" | ".tox" | "__pycache__" | "dist" | "target")
        ))
    })
}

fn append_field(hasher: &mut Sha256, value: &[u8]) {
    let length = u64::try_from(value.len()).expect("source-state field length fits u64");
    hasher.update(length.to_be_bytes());
    hasher.update(value);
}

/// Hash the exact source state that can affect a dirty checkout build.
///
/// Clean and archive builds use only the committed Git tree OID. Dirty Git
/// checkouts additionally bind the build to Git's binary tracked diff and all
/// non-ignored untracked regular files except known generated/cache trees.
/// Paths are sorted, length-prefixed UTF-8 repository-relative names. Symlinks
/// and unreadable/non-regular relevant entries fail closed.
pub(crate) fn source_state_sha256(
    repository: &Path,
    tree_oid: &str,
    dirty: bool,
) -> Result<String, String> {
    let mut hasher = Sha256::new();
    append_field(&mut hasher, SOURCE_STATE_ALGORITHM.as_bytes());
    append_field(&mut hasher, tree_oid.as_bytes());
    if !dirty {
        return Ok(digest_hex(hasher.finalize()));
    }

    let tracked_diff = command_bytes(
        "git",
        &[
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            "--",
        ],
        repository,
    )?;
    append_field(&mut hasher, &tracked_diff);

    let untracked = command_bytes(
        "git",
        &["ls-files", "--others", "--exclude-standard", "-z"],
        repository,
    )?;
    let mut paths = untracked
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            std::str::from_utf8(path)
                .map(PathBuf::from)
                .map_err(|_| "untracked source path is not valid UTF-8".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| !excluded_untracked(path));
    paths.sort();

    for relative in paths {
        if relative.is_absolute()
            || relative.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(format!(
                "unsafe untracked source path: {}",
                relative.display()
            ));
        }
        let path = repository.join(&relative);
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            format!(
                "cannot inspect relevant untracked file {}: {error}",
                relative.display()
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!(
                "relevant untracked input is not a regular file: {}",
                relative.display()
            ));
        }
        let contents = fs::read(&path).map_err(|error| {
            format!(
                "cannot read relevant untracked file {}: {error}",
                relative.display()
            )
        })?;
        append_field(&mut hasher, relative.to_string_lossy().as_bytes());
        append_field(&mut hasher, &contents);
    }
    Ok(digest_hex(hasher.finalize()))
}

pub(crate) fn resolve_source_identity(
    repository: &Path,
    supplied_revision: Option<&str>,
    supplied_tree: Option<&str>,
    supplied_clean: Option<&str>,
    release: bool,
) -> Result<(String, String, bool), String> {
    let supplied = match (supplied_revision, supplied_tree, supplied_clean) {
        (Some(revision), Some(tree), Some(clean)) => {
            if !valid_git_oid(revision) {
                return Err(
                    "VOIAGE_SOURCE_REVISION must be exactly 40 lowercase hexadecimal characters"
                        .to_owned(),
                );
            }
            if !valid_git_oid(tree) {
                return Err(
                    "VOIAGE_SOURCE_TREE_GIT_OID must be exactly 40 lowercase hexadecimal characters"
                        .to_owned(),
                );
            }
            Some((revision.to_owned(), tree.to_owned(), !parse_clean(clean)?))
        }
        (None, None, None) => None,
        _ => {
            return Err(
                "VOIAGE_SOURCE_REVISION, VOIAGE_SOURCE_TREE_GIT_OID, and VOIAGE_SOURCE_CLEAN must be supplied together".to_owned(),
            );
        }
    };

    match (git_identity(repository), supplied) {
        (Some(git), Some(environment)) if git != environment => Err(format!(
            "supplied source identity does not match Git: env={}:{}:{} git={}:{}:{}",
            environment.0, environment.1, environment.2, git.0, git.1, git.2
        )),
        (Some(git), Some(_) | None) => Ok(git),
        (None, Some(environment)) => Ok(environment),
        (None, None) if release => Err(
            "release builds require a valid Git identity or complete VOIAGE source identity variables"
                .to_owned(),
        ),
        // Development builds remain possible from ad-hoc source snapshots, but
        // an unidentified source is conservatively reported as dirty.
        (None, None) => Ok(("unknown".to_owned(), "unknown".to_owned(), true)),
    }
}

pub(crate) fn parse_embedded_source_identity(
    contents: &str,
) -> Result<(String, String, bool), String> {
    let lines: Vec<_> = contents.lines().collect();
    if lines.len() != 3 {
        return Err("embedded source provenance must contain exactly three lines".to_owned());
    }
    let revision = lines[0]
        .strip_prefix("revision=")
        .ok_or_else(|| "embedded source provenance is missing revision".to_owned())?;
    let tree = lines[1]
        .strip_prefix("tree=")
        .ok_or_else(|| "embedded source provenance is missing tree".to_owned())?;
    let clean = lines[2]
        .strip_prefix("clean=")
        .ok_or_else(|| "embedded source provenance is missing clean state".to_owned())?;
    if !valid_git_oid(revision) || !valid_git_oid(tree) || clean != "true" {
        return Err(
            "embedded source provenance must contain lowercase Git OIDs and clean=true".to_owned(),
        );
    }
    Ok((revision.to_owned(), tree.to_owned(), false))
}

pub(crate) fn parse_rustc_version(version: &str) -> Result<&str, String> {
    let mut fields = version.split_ascii_whitespace();
    let number = fields
        .next()
        .filter(|field| *field == "rustc")
        .and_then(|_| fields.next());
    let valid_number = number.is_some_and(|number| {
        let components: Vec<_> = number.split('.').collect();
        components.len() == 3
            && components
                .iter()
                .all(|component| component.parse::<u64>().is_ok())
    });
    if !valid_number {
        return Err(format!("malformed Rust compiler version: {version}"));
    }
    Ok(version)
}

fn probe_rustc_version(rustc: &str, manifest_dir: &Path) -> Result<String, String> {
    let version = command_output(rustc, &["--version"], manifest_dir)
        .ok_or_else(|| format!("failed to execute Rust compiler probe: {rustc} --version"))?;
    parse_rustc_version(&version)?;
    Ok(version)
}

fn embedded_source_identity(
    manifest_dir: &Path,
    repository: &Path,
) -> Option<(String, String, bool)> {
    if env::var_os("VOIAGE_SOURCE_REVISION").is_some()
        || env::var_os("VOIAGE_SOURCE_TREE_GIT_OID").is_some()
        || env::var_os("VOIAGE_SOURCE_CLEAN").is_some()
    {
        return None;
    }
    let path = manifest_dir.join(EMBEDDED_PROVENANCE_FILE);
    let contents = fs::read_to_string(&path).ok()?;
    let embedded = parse_embedded_source_identity(&contents)
        .unwrap_or_else(|error| panic!("invalid embedded source provenance: {error}"));
    match git_identity(repository) {
        None => Some(embedded),
        Some((revision, tree, true))
            if revision == embedded.0
                && tree == embedded.1
                && only_embedded_provenance_is_dirty(repository) =>
        {
            Some(embedded)
        }
        Some(_) => None,
    }
}

fn only_embedded_provenance_is_dirty(repository: &Path) -> bool {
    let Some(status) = command_output(
        "git",
        &["status", "--porcelain=v1", "--untracked-files=normal"],
        repository,
    ) else {
        return false;
    };
    status.lines().all(|line| {
        let path = line.get(3..).unwrap_or_default();
        path == "rust/crates/voiage-python/source-provenance.txt"
    })
}

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.join("../..");
    let repository_dir = workspace_dir.join("..");
    let git_dir = command_output("git", &["rev-parse", "--absolute-git-dir"], &repository_dir)
        .map_or_else(|| repository_dir.join(".git"), PathBuf::from);
    let lock_path = workspace_dir.join("Cargo.lock");
    let lock_digest = sha256_hex(&fs::read(&lock_path).expect("read workspace Cargo.lock"));

    let profile = env::var("PROFILE").expect("Cargo PROFILE");
    let embedded_identity = embedded_source_identity(&manifest_dir, &repository_dir);
    let supplied_revision = env::var("VOIAGE_SOURCE_REVISION").ok();
    let supplied_tree = env::var("VOIAGE_SOURCE_TREE_GIT_OID").ok();
    let supplied_clean = env::var("VOIAGE_SOURCE_CLEAN").ok();
    let embedded_revision = embedded_identity
        .as_ref()
        .map(|identity| identity.0.as_str());
    let embedded_tree = embedded_identity
        .as_ref()
        .map(|identity| identity.1.as_str());
    let embedded_clean = embedded_identity.as_ref().map(|_| "true");
    let (revision, tree_oid, dirty) = resolve_source_identity(
        &repository_dir,
        supplied_revision.as_deref().or(embedded_revision),
        supplied_tree.as_deref().or(embedded_tree),
        supplied_clean.as_deref().or(embedded_clean),
        profile == "release",
    )
    .unwrap_or_else(|error| panic!("invalid source provenance: {error}"));
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_version = probe_rustc_version(&rustc, &manifest_dir)
        .unwrap_or_else(|error| panic!("invalid compiler provenance: {error}"));
    let target = env::var("TARGET").expect("Cargo TARGET");
    let source_date_epoch = env::var("SOURCE_DATE_EPOCH").ok();
    if let Some(value) = &source_date_epoch {
        value
            .parse::<i64>()
            .expect("SOURCE_DATE_EPOCH must be a signed integer");
    }

    let source_state = source_state_sha256(&repository_dir, &tree_oid, dirty)
        .unwrap_or_else(|error| panic!("invalid source state provenance: {error}"));
    let build_id = build_id_from_parts(
        &revision,
        &tree_oid,
        dirty,
        &source_state,
        &target,
        &rustc_version,
        &profile,
        &lock_digest,
        source_date_epoch.as_deref(),
    );

    println!("cargo:rerun-if-changed={}", lock_path.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("Cargo.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src").display()
    );
    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
    println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
    println!(
        "cargo:rerun-if-changed={}",
        git_dir.join("packed-refs").display()
    );
    if let Some(reference) = command_output("git", &["symbolic-ref", "-q", "HEAD"], &repository_dir)
    {
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join(reference).display()
        );
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={EMBEDDED_PROVENANCE_FILE}");
    for variable in [
        "SOURCE_DATE_EPOCH",
        "VOIAGE_SOURCE_REVISION",
        "VOIAGE_SOURCE_TREE_GIT_OID",
        "VOIAGE_SOURCE_CLEAN",
        "RUSTC",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    println!("cargo:rustc-env=VOIAGE_SOURCE_REVISION={revision}");
    println!("cargo:rustc-env=VOIAGE_SOURCE_TREE_GIT_OID={tree_oid}");
    println!("cargo:rustc-env=VOIAGE_SOURCE_DIRTY={dirty}");
    println!("cargo:rustc-env=VOIAGE_SOURCE_STATE_SHA256={source_state}");
    println!("cargo:rustc-env=VOIAGE_SOURCE_STATE_ALGORITHM={SOURCE_STATE_ALGORITHM}");
    println!("cargo:rustc-env=VOIAGE_TARGET_TRIPLE={target}");
    println!("cargo:rustc-env=VOIAGE_RUSTC_VERSION={rustc_version}");
    println!("cargo:rustc-env=VOIAGE_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=VOIAGE_CARGO_LOCK_SHA256={lock_digest}");
    println!("cargo:rustc-env=VOIAGE_BUILD_ID={build_id}");
    println!("cargo:rustc-env=VOIAGE_BUILD_ID_ALGORITHM={BUILD_ID_ALGORITHM}");
    println!(
        "cargo:rustc-env=VOIAGE_SOURCE_DATE_EPOCH={}",
        source_date_epoch.unwrap_or_default()
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_id_from_parts(
    revision: &str,
    tree_oid: &str,
    dirty: bool,
    source_state: &str,
    target: &str,
    rustc: &str,
    profile: &str,
    lock_digest: &str,
    source_date_epoch: Option<&str>,
) -> String {
    sha256_hex(
        serde_free_identity(
            revision,
            tree_oid,
            dirty,
            source_state,
            target,
            rustc,
            profile,
            lock_digest,
            source_date_epoch,
        )
        .as_bytes(),
    )
}

#[allow(clippy::too_many_arguments)]
fn serde_free_identity(
    revision: &str,
    tree_oid: &str,
    dirty: bool,
    source_state: &str,
    target: &str,
    rustc: &str,
    profile: &str,
    lock_digest: &str,
    source_date_epoch: Option<&str>,
) -> String {
    [
        BUILD_ID_ALGORITHM,
        revision,
        tree_oid,
        if dirty { "true" } else { "false" },
        source_state,
        target,
        rustc,
        profile,
        lock_digest,
        source_date_epoch.unwrap_or(""),
    ]
    .into_iter()
    .fold(String::new(), |mut identity, value| {
        write!(identity, "{}:{value}", value.len()).expect("writing to String is infallible");
        identity
    })
}
