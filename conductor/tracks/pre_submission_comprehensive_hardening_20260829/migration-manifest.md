# Active-track migration manifest

Generated from the clean `main` baseline at `7992dcee` and approved for
consolidation on 2026-08-29. The source `plan.md` SHA-256 values freeze the
meaning of each line reference before archival. A reference such as
`track_id:19` means the pending or in-progress checkbox beginning at line 19 of
that exact, hash-bound plan. Parent-phase and child-task checkboxes are both
retained, so the 161 references are an accountability inventory rather than a
claim that 161 independent implementations remain.

Completed source tasks and their evidence remain byte-preserved in the archived
source tracks. Pending source work is consolidated by outcome into the new
plan; no pending checkbox is being represented as implemented merely because
its source track is superseded.

| Source track | Pending/in-progress refs | Source plan lines | Plan SHA-256 |
| --- | ---: | --- | --- |
| `controlled_live_dataset_interoperability_20260801` | 4 | `5,18,19,21` | `2362738bf18247e5c1997e16fcc1b91c3ce3e57cc798c5c4a9f4ed519cb07737` |
| `datasets_worked_examples_20260723` | 11 | `19,21,23,25,30,32,34,36,41,43,45` | `98918e577974116a9c03ffc636c1d15d5b0e8671020df0fdc484cc37c0e6abdd` |
| `estimation_focused_variance_voi_20260727` | 5 | `84,93,114,132,135` | `aa67b24c59c2947e5095a6dca2678163a9b27a95dc3732164c8f540a69195ae1` |
| `external_voi_library_feature_parity_20260723` | 10 | `16,18,22,26,28,30,32,37,39,41` | `63d7dd5d4070e79cbf1bd4074ec508c9e7550749c4e83da810abb12378927e94` |
| `hpc_spack_easybuild_distribution_20260823` | 9 | `5,6,7,8,9,10,11,12,13` | `fa36fda28e34cc0cc5be01384e74f1147b28c4fa38cd1df413df3b3b985243a5` |
| `information_source_portfolio_voi_20260801` | 1 | `28` | `70423964a072bc71034a7c5ba326a69049d073c999ad4974549fe66aa450528e` |
| `julia_general_registry_ecosystem_20260823` | 9 | `5,6,7,8,9,10,11,12,13` | `40c6f49118e3aca2a32833f4b71c7209b4ce676beaa0786517068164397d6ae0` |
| `ml_llm_agent_voi_20260723` | 10 | `16,18,22,26,28,30,32,37,39,41` | `d9950a3421a09c8f2fc69d737a1b0461da15bc94865c4c2cf5107b607edf85ff` |
| `polyglot_abi_binding_parity_20260723` | 10 | `16,18,22,34,36,38,40,45,47,49` | `cc8ef605345044cb81f97f37ea08e013bb5360bacbbc694fb7e8561be404828b` |
| `quality_release_automation_20260723` | 9 | `18,22,26,28,30,32,37,49,55` | `d1cfc13523c088f0509a50a5f82f0a8f618618e4b003c540fa3184972e967c6a` |
| `remote_dataset_ingestion_security_20260801` | 7 | `7,11,13,15,19,21,23` | `923d525f045cbb98fb6bca0dcb8a771b378da52296d6e2cee3bd1a5a5b3cb896` |
| `research_contribution_ai_transparency_20260723` | 10 | `12,19,21,23,27,29,31,33,38,44` | `f954b8ad9ce4bce1575d7a3f199e87e21b4264aa7f8a0b8c890f6eb581c752d6` |
| `ropensci_cran_readiness_20260823` | 9 | `5,6,7,8,9,10,11,12,13` | `7c8ce14ec0629480a1544b4c0b49adbcb7ed3c58f56462a4efd6638873c0b91a` |
| `rust_polyglot_voi_completion_20260723` | 5 | `51,55,58,60,63` | `e42573e794094fbd43c401dabc3d27c0eae448a7e7634ce481f581399805d6f5` |
| `sampling_acquisition_harm_voi_20260802` | 8 | `77,167,186,196,201,206,210,213` | `96863abacca0679d1ba50c7fc0e3772f306a85adc81c8d83160fbfa45a001998` |
| `stable_voi_rust_core_completion_20260723` | 10 | `18,20,22,24,28,30,32,34,39,47` | `072e3ebfdd3ad5e4dea4a944148e68d5a2bc301aed54615a01100b7b985bae82` |
| `supported_frontier_method_completion_20260723` | 11 | `144,169,276,301,568,590,595,599,605,610,615` | `077d2c7314a061f8c94ec712c5d43f34e3be71a5b28a345f0c5573ad0bbada30` |
| `sustainability_badging_governance_20260823` | 9 | `5,6,7,8,9,10,11,12,13` | `29d70acc3742198d575e10b2788650dea80f4cb9e241836a93d344b27cf9e07f` |
| `uncertainty_modelling_value_20260801` | 1 | `23` | `7328ec43235e7c074774b5d621bd577b4dd7ced62c20d601870943f76b48be40` |
| `value_of_perspective_completion_20260723` | 3 | `12,42,68` | `cd98e28dcf319439e94637aaa4f9ba9b98af58a370447d079d07aaf18b05668d` |
| `voi_method_census_contract_reconciliation_20260723` | 10 | `24,26,28,30,35,37,39,41,46,52` | `6ae9264008c45f771f117b2b18bb21b505f2444f7dfe182b1904a809f91a24bb` |

## Consolidated destination

- Core and frontier scientific work maps to Phases 1–3.
- Data, remote-ingestion, rights, integration, examples, and domain work maps
  to Phases 1–3, with approval-dependent network activity remaining a gate.
- Rust/Python ownership, API, ABI, R, Julia, HPC packaging, and installed parity
  map to Phases 2–4.
- Dependencies, preview features, profiling, CI/CD, security, release, and
  automation map to Phases 1, 2, and 5.
- rOpenSci maps to Phase 4 and Phase 6; pyOpenSci and JOSS map to Phase 6.
- Sustainability, contribution/AI transparency, registries, badges, archives,
  and external handoffs map to Phase 6, with external actions excluded.
- Exact-head assurance and source-track closeout map to Phases 0 and 7.

