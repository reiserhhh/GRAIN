# F17 claim-level reconciliation: protocol V1 (packet 29)

Frozen 2026-09-27 07:06 JST, before building any table. Metadata only: 0 fits, forwards, API calls or reruns.
- **Authority.** `NEXT_PACKET_29_CLAIM_RECONCILIATION.md`; `SEMINAR_FABLE_21_DECISION.md`; `SUPERVISOR_CHECK_20260927_0654.md`.
- **Output root.** `results/grain_gptpro_foundation_v3/F17/claim_level_reconciliation_r1/` (x-mode).
- **Builder.** `scripts/build_grain_f17_claim_reconciliation.py` (new). Tests `tests/test_grain_f17_claim_reconciliation.py`.

## Clause units

1. Every "still unmet" item and every overstatement flag in audit r3 (AUDIT_A, AUDIT_B), split on ";". A multi-part
   item is split further when its parts now have different statuses. Each clause keeps its verbatim text.
2. Every ledger row NOT audited in r3 gets one clause built from its current remaining-gap text (verbatim, without the
   F17 tail). Where statuses differ inside it, it is split.
3. All 50 ledger rows (44 computational + 6 human preparation) must appear in the coverage table. All 100 cards are
   checked through the existing `SOURCE_TASK_CROSSWALK.csv`, with no blanket acceptance.

## Status vocabulary (exactly these)

- RESOLVED_FINITE_WITH_AUTHORITY: a named current root plus a named acceptance authority (supervisor / seminar decision
  applied to the row, or the audit itself judging a flag adequate).
- CANDIDATE_FINITE_PENDING_REVIEW: a named current root answers the clause; acceptance of that mapping is not recorded.
- EXECUTABLE_CPU_CACHE: no current answer; CPU, cache or declared-world work suffices.
- BLOCKED_MODEL_BUDGET: needs forwards, generation, LLM readers or saved weights not authorized now.
- MISSING_MATERIAL_OR_METADATA: needs material that does not exist in the caches (exposure logs, primary texts,
  calibration cores) or a metadata correction.
- HUMAN_VALIDATION: needs real persons or human judgement by the source clause itself; not-authorized is not human.
- OPTIONAL_EXTENSION: only with an existing source-scope or decision justification pointer.
- SUPERSEDED_HISTORICAL: only with an explicit replacing root.
- UNADJUDICATED: not settled within the cap; never assumed resolved.

## Rules

- Every RESOLVED / CANDIDATE / SUPERSEDED label names a registered current root; the builder checks existence of its
  decision document and its typed evidence level through `revision_stage.evidence_level`. Missing or failing
  pointers are reported, not repaired.
- Self-reference is refused: this root cannot be evidence for any clause.
- Tests passing or files existing are never scientific support.
- Ledger `science_resolution`, `protocol` and `acceptance_decision` stay unchanged (21/44; 3/6). The only ledger
  edit is replacing the generic "see AUDIT_x_REPORT.md" tail in the CURRENT gap field with this table's itemized
  remainder; the old tail is kept verbatim in the diff file.
- No repairs duplicating `F00/record_reconciliation_r1`; no wholesale registration; no re-snapshotting.

## Read-only input archive (hashes before any update)

| file | sha256 |
|---|---|
| `results/grain_gptpro_foundation_v3/F17/source_audit_r3/AUDIT_A_REPORT.md` | `8dbf5b619d3209335e18e1a43fe916e46f2d308f43b911595ffe4d05a77b5553` |
| `results/grain_gptpro_foundation_v3/F17/source_audit_r3/AUDIT_B_REPORT.md` | `d25ce80ae3ec583e9da97d69392156e51ae532e57e7a9551d6ca1cab91bfb6f9` |
| `results/grain_gptpro_foundation_v3/F00/record_reconciliation_r1/decision.md` | `0a922abb8418add590c41a2d7696c72f53610223728b30eba2643ec2dd3b031b` |
| `results/grain_gptpro_foundation_v3/F00/r35_twelve_row_reconciliation_r1/decision.md` | `95f4e04cae218a1e5ae917e895335206ab7c04731cf7f6b17fd29862fe5a470e` |
| `configs/grain_v3_revision_registry.json` | `8a2a1cf64d50e25593ef800b852a3df3335cf1fb99b6e1eb7229ef031e1ba8c0` |

Parent-repo inputs (hashed at build time into `input_archive.json`): ledger CSV 64e2e8dadd1ee115a60e400591480ccbf2f3e1ba48389eb81298d4554cdd1da4, crosswalk f29feccff9b7d65a0f92e7fe22b2b7fac603f12a1210fd4da6a97fdca220f715.
