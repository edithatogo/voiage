"""Validate the fail-closed pre-closeout checkpoint for backlog delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

PLACEHOLDER = "REPLACE_AFTER_NATIVE_SPACK_TERMINAL_EVIDENCE"
EXPECTED_BASE = "b25ae916dc9ef40a7ec94a39e7c3d4ce11f4563a"
EXPECTED_NATIVE_CANDIDATE = "067fec9d5cf2baf2abea5f78a7a41fdfe4503617"
EXPECTED_NATIVE_CANDIDATES = {EXPECTED_NATIVE_CANDIDATE}
EXPECTED_NATIVE_RELEVANT_PATHS = [
    "packaging/spack",
    "scripts/hpc_package_smoke.py",
    "docs/release/hpc-distribution-handoff.md",
]
EXPECTED_NATIVE_RELEVANT_MANIFEST_SHA256 = (
    "15989288f186cf7ed6a68b6e4ddca504f87fc8af2aa3d8fb8284d7907e43ae8a"
)
EXPECTED_NATIVE_COMMANDS = {"spack_install", "installed_module_build"}
EXPECTED_NATIVE_PROBES = {
    "numerical",
    "arrow_pyarrow",
    "polars",
    "linkage",
    "non_main_thread_import",
    "module_load",
}
EXPECTED_NATIVE_QUALIFICATIONS = EXPECTED_NATIVE_COMMANDS | EXPECTED_NATIVE_PROBES
GIT = shutil.which("git")
if GIT is None:  # pragma: no cover - repository validation requires Git
    raise RuntimeError("git executable is required")
EXPECTED_MERGES = {
    "1087": (
        "614224cf4cab2514ece333345ff25f7441fddeba",
        "fea90d41898ac31c970b0c2b7a8a80ef3366ab96",
        "44a79dedb8cf4c8ce6a62f12d549bb6e7585b2ef",
        40,
    ),
    "1088": (
        "d33e0baaf0ef8b337d4ab7eb4f2d2756148e2dab",
        "9307d9ec7fcdc808ed7931afc298fa3bebac36e8",
        "658afdcf9f8fa15f10f4e5d56245a08cd248351e",
        40,
    ),
}
EXPECTED_CLOSED = {
    "1024": "2026-09-03T09:06:09Z",
    "614": "2026-09-03T10:50:22Z",
    "615": "2026-09-03T10:52:35Z",
}
REQUIRED_EXTERNAL = {
    296,
    298,
    312,
    471,
    555,
    620,
    850,
    853,
    876,
    1023,
    1026,
    1037,
    1045,
}
EXPECTED_OPEN_ISSUES = [
    {"number": 296, "title": "track: Research software registry readiness"},
    {"number": 298, "title": "Registry: assess RRID eligibility and evidence"},
    {
        "number": 312,
        "title": "Paper: complete arXiv preprint readiness and author review",
    },
    {
        "number": 471,
        "title": "JOSS: document research use and independent validation evidence",
    },
    {
        "number": 555,
        "title": "Registry: publish Julia binding through BinaryBuilder and General",
    },
    {"number": 620, "title": "Track externally gated OpenSSF Scorecard improvements"},
    {
        "number": 850,
        "title": "#570: Scope value of information with stochastic sampling-acquisition harm",
    },
    {
        "number": 853,
        "title": "#850: Bind sampling-harm candidate review and accountable disposition",
    },
    {
        "number": 876,
        "title": "#853: Commission eligible independent review for generic-kernel exclusion",
    },
    {
        "number": 1023,
        "title": "Julia: publish Voiage.jl to General Registry and JuliaHealth ecosystem",
    },
    {
        "number": 1025,
        "title": "HPC: publish Spack py-voiage recipe and EasyBuild easyconfigs",
    },
    {
        "number": 1026,
        "title": "Sustainability: activate Open Source Collective, complete OpenSSF Passing badge, and track SciCrunch RRID",
    },
    {
        "number": 1037,
        "title": "Release v2.2.0 and pursue pyOpenSci-first venue submissions",
    },
    {"number": 1045, "title": "Dependency Dashboard"},
    {
        "number": 1053,
        "title": "Complete remaining backlog delivery and verified repository cleanup",
    },
]
EXPECTED_NOTES_REFS = {
    "refs/notes/commits": {
        "tip": "809c94f9936a995af89a607bb95dfcda1d4813b3",
        "entry_count": 1468,
    },
    "refs/notes/conductor": {
        "tip": "9f8dbe3705278cf6a8853b1ebc911eaa1bdc6ba1",
        "entry_count": 16,
    },
    "refs/notes/origin": {
        "tip": "3e605b08481579862c8e551d63b338f3e17790cf",
        "entry_count": 1259,
    },
    "refs/notes/origin-commits": {
        "tip": "d3f4b22b6ff83fa4d56a9761ce032c05c01f2d32",
        "entry_count": 797,
    },
    "refs/notes/remote-commits": {
        "tip": "80e57ff3858ab1d902d0f36541cc11217d1821e8",
        "entry_count": 758,
    },
    "refs/notes/remote-merge": {
        "tip": "faf7a8e5afedb17262c05e2bcae03b9d24267dcf",
        "entry_count": 964,
    },
    "refs/notes/remote-merge-pr715": {
        "tip": "8cd9b7dfa9207e8c5fbc9e388ac170101e49a6c0",
        "entry_count": 965,
    },
    "refs/notes/remote-merge-pr717": {
        "tip": "244b2fce26b47ebee85d1775ada36db27afa30a3",
        "entry_count": 966,
    },
    "refs/notes/remote-merge-pr721": {
        "tip": "49c2d7e8a988ae8486e20b03d7e0bcdf54b2e4a9",
        "entry_count": 967,
    },
    "refs/notes/remote-merge-pr722": {
        "tip": "8df1a7e421428bfc7ce0253910ae13982693f002",
        "entry_count": 968,
    },
}
EXPECTED_CUSTOM_REF_COUNT = 104
EXPECTED_CUSTOM_REF_MANIFEST_SHA256 = (
    "9740eca74b6c9e02c9e044ae83bbfd015b5248d6ae0b5f7741ac3f5bc0a7802c"
)
EXPECTED_TRANSIENT_REFS = {
    "prefix": "refs/codex/turn-diffs/",
    "durable_invariant": False,
    "excluded_from_custom_ref_manifest": True,
    "reason": "Codex app turn-diff refs rotate between task turns and are not repository recovery state.",
}
EXPECTED_STASH_REFLOG = [
    {"oid": "e55b3bf2b7f9b61b291e6e5951807153bc044180", "selector": "stash@{0}", "subject": "On codex/spack-polars-url-20260905: codex-spack-polars-url-rebase"},
    {"oid": "2980d5a1ce2bfe3d49d89728aaae87dc1ecee52f", "selector": "stash@{1}", "subject": "autostash"},
    {"oid": "6cffb178a04204fe59986e93097083012268073f", "selector": "stash@{2}", "subject": "On codex/goal-test-acceleration-20260831: preserve borrowed PR1057 integration before acceleration rebase 20260901"},
    {"oid": "ecd50d8ebfe5a5b52419e81607cad9302a1573ff", "selector": "stash@{3}", "subject": "On codex/repair-actions-1056-20260831: voiage-actions1056-preserved-review-repair-20260831"},
    {"oid": "2d814e758ce307933a27b975df66424dc6c328cf", "selector": "stash@{4}", "subject": "On codex/issue-318-frontier-programme: codex-preserve-dsa-surface-wip-after-723-rebase"},
    {"oid": "fc8691befe363f37c183516c53bd78428e993ec7", "selector": "stash@{5}", "subject": "On codex/issue-318-frontier-programme: codex-preserve-dsa-surface-wip-before-723-rebase"},
    {"oid": "4381c04fc341b350bb8df142c40643ab038e0c97", "selector": "stash@{6}", "subject": "On main: codex-v1-programme-transplant"},
    {"oid": "623cc0af9e18eec9194564c91ff1cff5eabba8fa", "selector": "stash@{7}", "subject": "WIP on paper: a8d34da Update documentation link to GitHub Pages"},
    {"oid": "ebf789a31d4831ce8a00ff381963844da5ed274a", "selector": "stash@{8}", "subject": "WIP on paper-development: e15bbfa Finalize voiage Enhancement Project with comprehensive automation and tooling"},
]

EXPECTED_RECOVERY_PRESERVATION = {
    "retired_branch": "codex/recovery/easybuild-2024a-arrow-integration-18e076b4",
    "branch_removed": True,
    "recovery_ref": "refs/codex/recovery/20260903-final-cleanup/easybuild-2024a-arrow-integration-18e076b4",
    "commit": "18e076b4c00bf0f9bc5664de4ff12040088db65d",
    "tree": "e49a881b6aa2e9f27343f2cbfc63ea0cc72e52ba",
    "bundle_path": "/Volumes/PortableSSD/GitHub/voiage-recovery/20260903-final-cleanup/easybuild-2024a-arrow-integration-18e076b4.bundle",
    "bundle_sha256": "5f5beb6f9926792fb1424c933e6bdb5908034c7f7c86f25c587af2f64a9c5196",
    "bundle_complete_history": True,
    "restore_verification_path": "/Volumes/PortableSSD/GitHub/voiage-recovery/20260903-final-cleanup/restore-verification.txt",
    "restore_verification_sha256": "606f9df8d29313b4f25ee6daef9f2a560104545ec680b8da0bc764ad85e247a0",
    "restored_commit": "18e076b4c00bf0f9bc5664de4ff12040088db65d",
    "restored_tree": "e49a881b6aa2e9f27343f2cbfc63ea0cc72e52ba",
    "restored_fsck_full_strict": "passed",
}
EXPECTED_INVENTORY = {
    "open_pull_requests": [],
    "worktrees": [
        {
            "path": "/Volumes/PortableSSD/GitHub/voiage",
            "head": EXPECTED_BASE,
            "branch": "main",
        },
        {
            "path": "/Volumes/PortableSSD/GitHub/.worktrees/voiage-remaining-backlog-final-closeout-20260903",
            "head": "9307d9ec7fcdc808ed7931afc298fa3bebac36e8",
            "branch": "codex/remaining-backlog-final-closeout-20260903",
        },
        {
            "path": "/Volumes/PortableSSD/GitHub/.worktrees/voiage-ruff-preview-repair-20260905",
            "head": "9307d9ec7fcdc808ed7931afc298fa3bebac36e8",
            "branch": "codex/ruff-preview-repair-20260905",
        },
    ],
    "local_branches": {
        "codex/remaining-backlog-final-closeout-20260903": "9307d9ec7fcdc808ed7931afc298fa3bebac36e8",
        "codex/ruff-preview-repair-20260905": "9307d9ec7fcdc808ed7931afc298fa3bebac36e8",
        "main": EXPECTED_BASE,
    },
    "remote_branches": {
        "origin/2.0.x": "73c92eebdf581c763f1afb5b5196f687a6d33575",
        "origin/main": EXPECTED_BASE,
    },
    "stash_reflog": EXPECTED_STASH_REFLOG,
    "notes_refs": EXPECTED_NOTES_REFS,
    "transient_refs": EXPECTED_TRANSIENT_REFS,
}


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"external recovery artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed executable and controlled arguments
        [GIT, "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"git recovery verification failed: {' '.join(args)}")
    return (result.stdout + result.stderr).strip()


def _validate_recovery(recovery: dict[str, Any], root: Path) -> None:
    if recovery != EXPECTED_RECOVERY_PRESERVATION:
        raise ValueError("recovery ref or external bundle evidence is stale")
    bundle = Path(recovery["bundle_path"])
    record = Path(recovery["restore_verification_path"])
    if _sha256(bundle) != recovery["bundle_sha256"]:
        raise ValueError("external recovery bundle hash does not match")
    if _sha256(record) != recovery["restore_verification_sha256"]:
        raise ValueError("external restore verification hash does not match")
    if record.read_text(encoding="utf-8").splitlines() != [
        f"restore_commit={recovery['restored_commit']}",
        f"restore_tree={recovery['restored_tree']}",
        "fsck=passed",
    ]:
        raise ValueError("external restore verification content is invalid")
    ref = recovery["recovery_ref"]
    if _git(root, "rev-parse", ref) != recovery["commit"]:
        raise ValueError("recovery ref commit has drifted")
    if _git(root, "rev-parse", f"{ref}^{{tree}}") != recovery["tree"]:
        raise ValueError("recovery ref tree has drifted")
    heads = _git(root, "bundle", "list-heads", str(bundle)).splitlines()
    if heads != [f"{recovery['commit']} {ref}"]:
        raise ValueError("recovery bundle does not contain the exact sole ref")
    verification = _git(root, "bundle", "verify", str(bundle))
    if "The bundle records a complete history." not in verification:
        raise ValueError("recovery bundle is not complete history")


def _validate_custom_refs(snapshot: dict[str, Any], root: Path) -> None:
    manifest = snapshot.pop("custom_refs", None)
    if not isinstance(manifest, dict) or set(manifest) != {
        "count",
        "manifest_sha256",
        "refs",
    }:
        raise ValueError("custom ref manifest is incomplete")
    refs = manifest["refs"]
    if not isinstance(refs, list):
        raise TypeError("custom ref manifest is invalid")
    canonical = json.dumps(refs, sort_keys=True, separators=(",", ":")).encode()
    if (
        manifest["count"] != EXPECTED_CUSTOM_REF_COUNT
        or manifest["manifest_sha256"] != EXPECTED_CUSTOM_REF_MANIFEST_SHA256
        or hashlib.sha256(canonical).hexdigest() != manifest["manifest_sha256"]
    ):
        raise ValueError("custom ref manifest hash or count is stale")
    output = _git(root, "for-each-ref", "--format=%(refname) %(objectname)")
    live = [
        {"name": name, "oid": oid}
        for name, oid in (line.split() for line in output.splitlines())
        if not name.startswith(
            (
                "refs/heads/",
                "refs/remotes/",
                "refs/tags/",
                "refs/notes/",
                "refs/codex/turn-diffs/",
            )
        )
    ]
    if refs != live:
        raise ValueError("custom ref name or OID inventory has drifted")


def _validate_native_receipt(native: dict[str, Any], root: Path) -> None:
    """Validate exact, transcript-bound terminal native evidence."""
    required = {
        "path",
        "sha256",
        "outcome",
        "platform",
        "host",
        "run_id",
        "candidate_commit",
        "base_main",
        "relevant_path_manifest_sha256",
    }
    if not isinstance(native, dict) or set(native) != required:
        raise ValueError("terminal native receipt binding is incomplete")
    relative = Path(str(native["path"]))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("native receipt path is unsafe")
    root_resolved = root.resolve(strict=True)

    def read_bound(binding: Any, kind: str) -> dict[str, Any]:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"native {kind} transcript binding is incomplete")
        rel = Path(str(binding["path"]))
        if rel.is_absolute() or not rel.parts or ".." in rel.parts:
            raise ValueError(f"native {kind} transcript path is unsafe")
        unresolved = root_resolved / rel
        if unresolved.is_symlink():
            raise ValueError(f"native {kind} transcript must not be a symlink")
        try:
            target = unresolved.resolve(strict=True)
            target.relative_to(root_resolved)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"native {kind} transcript is missing or escaped"
            ) from error
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError(f"native {kind} transcript hash does not match")
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"native {kind} transcript is not JSON") from error
        if not isinstance(parsed, dict):
            raise TypeError(f"native {kind} transcript is invalid")
        return parsed

    receipt_path = root_resolved / relative
    if receipt_path.is_symlink():
        raise ValueError("native receipt path must not be a symlink")
    evidence = read_bound(
        {"path": relative.as_posix(), "sha256": native["sha256"]}, "receipt"
    )
    identity = evidence.get("identity")
    expected_identity = {
        key: native[key]
        for key in ("platform", "host", "run_id", "candidate_commit", "base_main")
    }
    if (
        evidence.get("schema_version") != "voiage.native-spack-terminal-receipt.v1"
        or evidence.get("terminal") is not True
        or identity != expected_identity
    ):
        raise ValueError("native receipt schema or identity is invalid")
    if evidence.get("outcome") != native["outcome"] or native["outcome"] not in {
        "passed",
        "failed_terminal",
    }:
        raise ValueError("native receipt outcome is not exact and terminal")
    if (
        native["candidate_commit"] not in EXPECTED_NATIVE_CANDIDATES
        or native["base_main"] != EXPECTED_BASE
    ):
        raise ValueError(
            "native receipt candidate is not explicitly reviewed for current main"
        )
    for key in ("platform", "host", "run_id"):
        if not isinstance(native[key], str) or not native[key].strip():
            raise ValueError(f"native receipt {key} identity is empty")
    _git(root, "cat-file", "-e", f"{native['candidate_commit']}^{{commit}}")
    _git(root, "merge-base", "--is-ancestor", native["candidate_commit"], EXPECTED_BASE)
    manifests = []
    for commit in (native["candidate_commit"], EXPECTED_BASE):
        listing = _git(
            root, "ls-tree", "-r", commit, "--", *EXPECTED_NATIVE_RELEVANT_PATHS
        )
        rows = []
        for line in listing.splitlines():
            metadata, path = line.split("\t", 1)
            mode, object_type, oid = metadata.split()
            rows.append({"mode": mode, "type": object_type, "oid": oid, "path": path})
        manifests.append(rows)
    canonical = json.dumps(manifests[0], sort_keys=True, separators=(",", ":")).encode()
    if (
        manifests[0] != manifests[1]
        or hashlib.sha256(canonical).hexdigest()
        != EXPECTED_NATIVE_RELEVANT_MANIFEST_SHA256
        or native["relevant_path_manifest_sha256"]
        != EXPECTED_NATIVE_RELEVANT_MANIFEST_SHA256
    ):
        raise ValueError("native candidate relevant paths do not equal current main")
    if evidence.get("relevant_path_manifest") != {
        "paths": EXPECTED_NATIVE_RELEVANT_PATHS,
        "sha256": EXPECTED_NATIVE_RELEVANT_MANIFEST_SHA256,
        "entries": manifests[0],
    }:
        raise ValueError("native wrapper relevant-path manifest is stale")
    raw_bindings = evidence.get("raw_bindings")
    if not isinstance(raw_bindings, dict) or set(raw_bindings) != {
        "guest_receipt",
        "input",
        "candidate_commit_file",
    }:
        raise ValueError("native wrapper raw bindings are incomplete")
    read_bound(raw_bindings["guest_receipt"], "guest receipt")
    read_bound(raw_bindings["input"], "input")
    commit_binding = raw_bindings["candidate_commit_file"]
    commit_file = root_resolved / Path(commit_binding.get("path", ""))
    if (
        _sha256(commit_file) != commit_binding.get("sha256")
        or commit_file.read_text(encoding="utf-8").strip() != native["candidate_commit"]
    ):
        raise ValueError("native candidate commit file is inconsistent")

    if native["outcome"] == "passed":
        matrix = evidence.get("qualification_matrix")
        commands, probes = evidence.get("commands"), evidence.get("probes")
        if matrix != dict.fromkeys(sorted(EXPECTED_NATIVE_QUALIFICATIONS), "passed"):
            raise ValueError("passed native qualification matrix is incomplete")
        if (
            not isinstance(commands, list)
            or {x.get("identity") for x in commands if isinstance(x, dict)}
            != EXPECTED_NATIVE_COMMANDS
        ):
            raise ValueError("passed native command identities are incomplete")
        if (
            not isinstance(probes, list)
            or {x.get("identity") for x in probes if isinstance(x, dict)}
            != EXPECTED_NATIVE_PROBES
        ):
            raise ValueError("passed native probe identities are incomplete")
        for item in commands:
            if (
                set(item) != {"identity", "argv", "exit_code", "transcript"}
                or not isinstance(item["argv"], list)
                or not item["argv"]
                or item["exit_code"] != 0
            ):
                raise ValueError("passed native command evidence is invalid")
            if read_bound(item["transcript"], "command") != {
                "identity": item["identity"],
                "argv": item["argv"],
                "exit_code": 0,
                "outcome": "passed",
            }:
                raise ValueError("native command transcript outcome does not match")
        for item in probes:
            if (
                set(item)
                != {"identity", "command", "exit_code", "result", "transcript"}
                or not isinstance(item["command"], list)
                or not item["command"]
                or item["exit_code"] != 0
                or item["result"] != "passed"
            ):
                raise ValueError("passed native probe evidence is invalid")
            if read_bound(item["transcript"], "probe") != {
                "identity": item["identity"],
                "command": item["command"],
                "exit_code": 0,
                "result": "passed",
                "outcome": "passed",
            }:
                raise ValueError("native probe transcript outcome does not match")
    else:
        failure = evidence.get("failure")
        if not isinstance(failure, dict) or set(failure) != {
            "stage",
            "identity",
            "argv",
            "exit_code",
            "transcript",
        }:
            raise ValueError("failed terminal receipt lacks bound failure evidence")
        if (
            failure["stage"] not in EXPECTED_NATIVE_QUALIFICATIONS
            or failure["identity"] != failure["stage"]
            or not isinstance(failure["argv"], list)
            or not failure["argv"]
            or not isinstance(failure["exit_code"], int)
            or failure["exit_code"] == 0
        ):
            raise ValueError("failed terminal command or stage is invalid")
        expected = {
            "stage": failure["stage"],
            "identity": failure["identity"],
            "argv": failure["argv"],
            "exit_code": failure["exit_code"],
            "outcome": "failed_terminal",
        }
        if read_bound(failure["transcript"], "failure") != expected:
            raise ValueError("failure transcript is inconsistent with terminal outcome")


def validate_closeout(path: Path, root: Path) -> dict[str, Any]:
    """Validate exact merged delivery, inventory, and pending gate boundaries."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "voiage.remaining-backlog-pre-closeout.v1"
    ):
        raise ValueError("invalid pre-closeout schema")
    if payload.get("stage") != "pre_closeout":
        raise ValueError("receipt must identify itself as pre-closeout")
    if payload.get("base_main") != EXPECTED_BASE:
        raise ValueError("final closeout base must be exact merged main")
    delivery = payload.get("merged_delivery")
    if not isinstance(delivery, dict) or set(delivery) != set(EXPECTED_MERGES):
        raise ValueError("merged delivery set is incomplete")
    for number, (head, merge, tree, checks) in EXPECTED_MERGES.items():
        record = delivery[number]
        if record != {
            "head_sha": head,
            "merge_sha": merge,
            "reviewed_tree": tree,
            "merged_tree": tree,
            "terminal_checks": checks,
        }:
            raise ValueError(f"PR #{number} merge evidence is stale")
    if payload.get("closed_repository_issues") != EXPECTED_CLOSED:
        raise ValueError("closed repository issue evidence is stale")
    inventory = payload.get("inventory_snapshot")
    if not isinstance(inventory, dict):
        raise TypeError("pre-closeout inventory is missing")
    inventory_without_custom = dict(inventory)
    _validate_custom_refs(inventory_without_custom, root)
    if inventory_without_custom != EXPECTED_INVENTORY:
        raise ValueError("pre-closeout branch, worktree, or PR inventory is stale")
    stash_lines = _git(root, "stash", "list", "--format=%H%x09%gd%x09%gs")
    live_stashes = [
        dict(zip(("oid", "selector", "subject"), line.split("\t", 2), strict=True))
        for line in stash_lines.splitlines()
    ]
    if live_stashes != EXPECTED_STASH_REFLOG:
        raise ValueError("ordered stash reflog inventory has drifted")
    _validate_recovery(payload.get("recovery_preservation"), root)
    if payload.get("open_issues") != EXPECTED_OPEN_ISSUES:
        raise ValueError("open issue inventory is stale or incomplete")
    native = payload.get("native_spack_receipt")
    state = payload.get("state")
    if state == "awaiting_native_spack_receipt":
        if native != {
            "path": None,
            "sha256": None,
            "outcome": None,
            "platform": None,
            "host": None,
            "run_id": None,
            "candidate_commit": None,
            "base_main": None,
            "relevant_path_manifest_sha256": None,
            "placeholder": PLACEHOLDER,
        }:
            raise ValueError("pending closeout must retain exact native placeholder")
    elif state == "ready_for_final_validation":
        if not isinstance(native, dict):
            raise ValueError("terminal native receipt binding is incomplete")
        _validate_native_receipt(native, root)
    else:
        raise ValueError("invalid final closeout state")
    boundaries = payload.get("open_issue_boundaries")
    if (
        not isinstance(boundaries, dict)
        or "1025" not in boundaries
        or "1037" not in boundaries
    ):
        raise ValueError("#1025 and #1037 must remain explicit")
    if set(boundaries.get("human_external_issue_numbers", [])) != REQUIRED_EXTERNAL:
        raise ValueError("human and external issue inventory is incomplete")
    if "does not establish" not in str(payload.get("boundary")):
        raise ValueError("closeout boundary is missing")
    post = payload.get("post_closeout_requirement")
    if (
        not isinstance(post, dict)
        or post.get("required") is not True
        or post.get("issue_1053_may_close_only_after_post_closeout_receipt") is not True
        or "separate auditable repository change" not in str(post.get("artifact"))
    ):
        raise ValueError("authoritative post-closeout stage is not required")
    return {
        "state": state,
        "merge_count": len(delivery),
        "external_issue_count": len(REQUIRED_EXTERNAL),
        "open_issue_count": len(EXPECTED_OPEN_ISSUES),
    }


def main() -> int:
    """Validate the canonical closeout receipt from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "receipt",
        type=Path,
        nargs="?",
        default=Path(
            "conductor/tracks/remaining_backlog_delivery_20260831/pre-closeout-20260903.json"
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = validate_closeout(args.receipt, args.root)
    print(f"Remaining backlog pre-closeout: PASS ({result['state']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
