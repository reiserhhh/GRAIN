#!/usr/bin/env python
"""Build the additive revision/supersession registry (packet 12 Part C2). Reads only; writes the registry JSON.

Sources: the frozen 100-card plan (legacy r1 roots), configs/grain_v3_card_registry.json (canonical recovery/review
locations per card), the requirement ledger (roots cited as evidence per requirement), common-record metadata, and a
small declared table for packets whose target, authority, relations, amendments and adapters were curated by hand.
Nothing here closes a card: a root "contributes" evidence; card acceptance stays with the ledger and review cards.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/grain_gptpro_foundation_v3"
PERSONA = Path("/Volumes/mobile3/projects/project persona")
PLAN = PERSONA / "configs/grain_gptpro_atomic_v3.json"
CARD_REG = REPO / "configs/grain_v3_card_registry.json"
LEDGER_CSV = PERSONA / "reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv"
OUT = REPO / "configs/grain_v3_revision_registry.json"
LOOP = PERSONA / "reports/claude_research_loop_20260926"
REF_RE = re.compile(r"\b((?:F\d\d|P05|R14|R17)/[A-Za-z0-9_]+_r\d+)\b")
EVIDENCE_DOCS = ("decision.md", "DECISION.md", "summary.json", "REPORT.md", "INVENTORY.md", "MAPPING.md", "decision.json", "REACCEPTANCE.md")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel_res(path: str) -> str:
    return path.split("results/grain_gptpro_foundation_v3/", 1)[-1].rstrip("/")


AUTH = {
    "ACCEPTANCE_AUDIT_20260915": REPO / "reports/grain_measurement_foundation/GPTPRO_V3_EXECUTION_AUDIT_20260915.md",
    "SEMINAR_06": LOOP / "SEMINAR_FABLE_06_DECISION.md", "SEMINAR_07": LOOP / "SEMINAR_FABLE_07_DECISION.md",
    "SEMINAR_08": LOOP / "SEMINAR_FABLE_08_DECISION.md", "SEMINAR_09": LOOP / "SEMINAR_FABLE_09_DECISION.md",
    "SEMINAR_10": LOOP / "SEMINAR_FABLE_10_DECISION.md", "PACKET_11": LOOP / "NEXT_PACKET_11_PERSONAL_LAW_AND_SHARED_SHOCK.md",
    "PACKET_12": LOOP / "NEXT_PACKET_12_TARGET_REPAIR_AND_INTEGRATION.md", "SEMINAR_11": LOOP / "SEMINAR_FABLE_11_DECISION.md",
    "PACKET_13": LOOP / "NEXT_PACKET_13_PROVENANCE_AND_EQUIVALENCE.md", "PACKET_14": LOOP / "NEXT_PACKET_14_PRAGMATIC_IDENTIFIABILITY.md", "SUPERVISOR_1439": LOOP / "SUPERVISOR_CHECK_20260926_1439.md",
    "SEMINAR_13": LOOP / "SEMINAR_FABLE_13_DECISION.md", "PACKET_15": LOOP / "NEXT_PACKET_15_HISTORY_AND_PRAGMATIC_READER.md",
    "SEMINAR_14": LOOP / "SEMINAR_FABLE_14_DECISION.md", "PACKET_16": LOOP / "NEXT_PACKET_16_CHANNEL_UTILIZATION_AND_NATURAL_OOM.md",
    "SEMINAR_15": LOOP / "SEMINAR_FABLE_15_DECISION.md", "PACKET_17": LOOP / "NEXT_PACKET_17_APPLICATION_J_WORDING_AND_PROVENANCE.md",
    "SEMINAR_16": LOOP / "SEMINAR_FABLE_16_DECISION.md", "PACKET_18": LOOP / "NEXT_PACKET_18_PRODUCT_SPACE_AND_RELIABILITY.md",
    "SUPERVISOR_2042": LOOP / "SUPERVISOR_CHECK_20260926_2042.md", "PACKET_19": LOOP / "NEXT_PACKET_19_RECONSTRUCTION_AND_REACTIVITY.md",
    "SUPERVISOR_2143": LOOP / "SUPERVISOR_CHECK_20260926_2143.md", "PACKET_20": LOOP / "NEXT_PACKET_20_ERROR_BUDGET_AND_HISTORY.md",
    "SUPERVISOR_2245": LOOP / "SUPERVISOR_CHECK_20260926_2245.md", "SEMINAR_18": LOOP / "SEMINAR_FABLE_18_DECISION.md", "PACKET_21": LOOP / "NEXT_PACKET_21_FOUR_COMPONENT_IDENTIFICATION.md",
    "SUPERVISOR_2347": LOOP / "SUPERVISOR_CHECK_20260926_2347.md", "PACKET_22": LOOP / "NEXT_PACKET_22_HISTORY_INFORMATION.md",
    "SUPERVISOR_0048": LOOP / "SUPERVISOR_CHECK_20260927_0048.md", "PACKET_23": LOOP / "NEXT_PACKET_23_RAW_GRAIN_HISTORY.md",
    "SUPERVISOR_0149": LOOP / "SUPERVISOR_CHECK_20260927_0149.md", "SEMINAR_19": LOOP / "SEMINAR_FABLE_19_DECISION.md", "PACKET_24": LOOP / "NEXT_PACKET_24_SOURCE_CROSSED_ACTION.md",
    "SUPERVISOR_0251": LOOP / "SUPERVISOR_CHECK_20260927_0251.md", "PACKET_25": LOOP / "NEXT_PACKET_25_RECOVERY_DESIGN.md",
    "SUPERVISOR_0352": LOOP / "SUPERVISOR_CHECK_20260927_0352.md", "SEMINAR_20": LOOP / "SEMINAR_FABLE_20_DECISION.md", "PACKET_26": LOOP / "NEXT_PACKET_26_COMMITTED_DESIGN.md",
    "SUPERVISOR_0452": LOOP / "SUPERVISOR_CHECK_20260927_0452.md", "PACKET_27": LOOP / "NEXT_PACKET_27_AFFINE_ACTION_DIAGNOSIS.md",
    "SUPERVISOR_0553": LOOP / "SUPERVISOR_CHECK_20260927_0553.md", "PACKET_28": LOOP / "NEXT_PACKET_28_RELIABILITY_INFORMATION.md",
    "SUPERVISOR_0654": LOOP / "SUPERVISOR_CHECK_20260927_0654.md", "SEMINAR_21": LOOP / "SEMINAR_FABLE_21_DECISION.md", "PACKET_29": LOOP / "NEXT_PACKET_29_CLAIM_RECONCILIATION.md",
    "SUPERVISOR_0754": LOOP / "SUPERVISOR_CHECK_20260927_0754.md", "PACKET_30": LOOP / "NEXT_PACKET_30_FREEDOM_COMPONENTS.md",
    "SUPERVISOR_0855": LOOP / "SUPERVISOR_CHECK_20260927_0855.md", "SEMINAR_22": LOOP / "SEMINAR_FABLE_22_DECISION.md", "PACKET_31": LOOP / "NEXT_PACKET_31_INFORMATIVE_CONTEXTS.md",
    "SUPERVISOR_0955": LOOP / "SUPERVISOR_CHECK_20260927_0955.md", "PACKET_32": LOOP / "NEXT_PACKET_32_GENERATION_QUERY.md",
    "SUPERVISOR_1056": LOOP / "SUPERVISOR_CHECK_20260927_1056.md", "SEMINAR_23": LOOP / "SEMINAR_FABLE_23_DECISION.md", "PACKET_33": LOOP / "NEXT_PACKET_33_RATIO_INFERENCE_DOMAIN.md",
    "SUPERVISOR_1157": LOOP / "SUPERVISOR_CHECK_20260927_1157.md", "PACKET_34": LOOP / "NEXT_PACKET_34_ALIRT_REFERENCE.md",
}

# Declared append-only amendment authorities (file -> authority); the byte prefix is verified automatically.
AMEND_AUTH = {
    "F05/common_shock_episodes_r1": "PACKET_12", "F13/personal_law_r1": "PACKET_12",
    "F12/fixed_core_dif_r1": "PACKET_11", "F13/covariance_identification_r1": "PACKET_11",
    "F12/intervention_transport_r1": "SEMINAR_10", "R14/soft_channel_trace_r1": "SEMINAR_10",
    "P05/c2_dedup_sensitivity_r1": "SEMINAR_07", "F15/text_behaviour_quotient_r1": "PACKET_14", "P05/c2_venue_dependence_r1": "SEMINAR_06+SEMINAR_07",
    "F17/speaker_identifiability_r1": "PACKET_15", "F15/controlled_psr_witness_r1": "PACKET_16",
    "F14/pandora_venue_oom_r1": "PACKET_17", "F17/hedge_channel_utilization_r1": "PACKET_17",
    "F12/designed_wording_facet_r1": "PACKET_18",
}

DECLARED = {
    "F13/personal_law_r1": {"claims": {"support": "complete-history personal-law results; B2 .25 variance is width not heterogeneity", "cannot_infer": "integrated-target calibration (see personal_law_r2)", "missing_evidence": "input roots have snapshots only, no run-time manifest"}, "target": "binary personal-law next-p (T_latent) uncertainty; W0/H1/H2", "requirements": ["R24", "R25"], "authority": "PACKET_11",
                            "tests": ["tests/test_grain_r24_personal_law.py"], "depends_on_revisions": ["F13/binary_source_full_r2", "F13/recovery3_r1"],
                            "relations": [{"type": "SUPPLEMENTS", "other": "F13/recovery3_r1", "scope": "adds personal-law worlds; shared-law evidence retained"}],
                            "add_to_cards": ["F13-04"]},
    "F13/personal_law_r2": {"claims": {"support": "exact legacy replay; selection-aware MNAR target; w'Cw companion with full B2/M covariance", "cannot_infer": "calibrated uncertainty for the true integrated target under MNAR; human validity", "missing_evidence": "selection model for MAR/MNAR weights; hierarchical shared-offset law"}, "target": "same design; selection-aware integrated target, fixed-working-weight companion w'Cw", "requirements": ["R24", "R25"],
                            "authority": "PACKET_12", "tests": ["tests/test_grain_r24_target_repair.py"],
                            "depends_on_revisions": ["F13/personal_law_r1", "F13/binary_source_full_r2"],
                            "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "F13/personal_law_r1",
                                           "scope": "cover_int reading and MNAR integrated-target metrics; every r1 number retained by exact replay"}],
                            "interval_contract": "docs/GRAIN_R24_TARGET_REPAIR_PROTOCOL_V2.md (Intervals and their contracts)", "add_to_cards": ["F13-04", "F13-02"]},
    "F05/common_shock_episodes_r1": {"claims": {"support": "joint filtering lowers state error -.0344 [-.0383,-.0306] over 32 independent episodes", "cannot_infer": "persistent/clustered shocks; natural text; unconditional coverage (source refit not propagated)", "missing_evidence": "crossed stable-author episodes"}, "target": "joint vs independent state filtering over 32 independent episodes per world", "requirements": ["R10", "R29"],
                                     "authority": "PACKET_11", "tests": ["tests/test_grain_r10_common_shock_episodes.py"],
                                     "depends_on_revisions": ["F05/source_full_r2", "F05/application_full_r1", "F05/joint_full_r1"],
                                     "relations": [{"type": "SUPPLEMENTS", "other": "F05/joint_full_r1", "scope": "32 independent episodes; 3-episode result retained"}],
                                     "add_to_cards": ["F05-05"]},
    "F02/collision_provenance_r1": {"claims": {"support": "GP01 bytes match the 2026-09-15 snapshot; exact collision equations; byte-identical regeneration", "cannot_infer": "immutability since creation; psychological equivalence", "missing_evidence": "A_C vs B separation and non-trivial G (see F15/text_behaviour_quotient_r1)"}, "target": "legacy GP01 collision-world hash provenance and exact equation verification", "requirements": ["R08", "R35"],
                                    "authority": "PACKET_12", "tests": ["tests/test_grain_revision_stage.py"], "depends_on_revisions": ["F02/recovery_full_r2"],
                                    "relations": [{"type": "SUPPLEMENTS", "other": "F02/recovery_full_r2", "scope": "provenance of legacy GP01 evidence"}],
                                    "add_to_cards": ["F02-01"]},
    "F16/serialize_full_r1": {"target": "public calibrated bundles and sealed cold scorer", "requirements": ["R22", "R23", "R24", "R25", "R33", "R34", "R35"],
                              "authority": "ACCEPTANCE_AUDIT_20260915", "tests": ["tests/test_grain_gptpro_f16_serialize.py"], "add_to_cards": ["F16-03"],
                              "evidence_doc": "reports/grain_measurement_foundation/GPTPRO_V3_F16_SERIALIZE_20260926.md"},
    "F13/covariance_identification_r1": {"target": "R29 covariance identification and duplicate information", "requirements": ["R29"], "authority": "SEMINAR_10",
                                         "tests": ["tests/test_grain_r29_covariance_identification.py"]},
    "F12/fixed_core_dif_r1": {"target": "fixed-core frame/topic/day response variation (not paraphrase)", "requirements": ["R23", "R25"], "authority": "SEMINAR_10",
                              "tests": ["tests/test_grain_r23_fixed_core_dif.py"]},
    "F12/intervention_transport_r1": {"target": "level vs intervention-signature transport NLI->Qwen", "requirements": ["R21", "R23"], "authority": "SEMINAR_09",
                                      "tests": ["tests/test_grain_r21_intervention_transport.py"]},
    "R14/soft_channel_trace_r1": {"target": "soft-channel companion and event trace", "requirements": ["R14", "R16"], "authority": "SEMINAR_09",
                                  "tests": ["tests/test_grain_r14_soft_channel_trace.py"]},
    "F17/h02_recovery_r2": {"target": "H02 encompassing mechanism-set design rule", "requirements": ["H02"], "authority": "SEMINAR_06",
                            "relations": [{"type": "SUPERSEDES_SCIENTIFIC_USE", "other": "F17/h02_recovery_r1",
                                           "scope": "design rule and N (single-winner rule wrong under compound truths)"}]},
    "P05/c2_event_version_sensitivity_r1": {"target": "C2 FIRST/LAST one-version-per-key views", "requirements": ["R01", "R27"], "authority": "SEMINAR_08",
                                            "tests": ["tests/test_grain_c2_event_version.py"],
                                            "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "P05/c2_venue_dependence_r1",
                                                           "scope": "np_mcd B-C venue dependence (row-matched arm was not event-matched); r1 numbers retained"}]},
    "F00/stage_orchestrator_r1": {"target": "legacy-integrity orchestrator (verification and dry-run only)", "requirements": ["R35"], "authority": "SEMINAR_06",
                                  "evidence_doc": "results/grain_gptpro_foundation_v3/F00/stage_orchestrator_r1/stage_integrity_summary.json",
                                  "tests": ["tests/test_grain_stage_orchestrator.py"], "add_to_cards": ["F00-04"]},
    "F00/revision_stage_r1": {"target": "revision registry, fail-closed stage modes, bounded replays and coverage table", "requirements": ["R35", "R08"],
                              "authority": "PACKET_12", "tests": ["tests/test_grain_revision_stage.py"]},
    "F00/baseline_inventory_r2": {"target": "current F00-01: source/plan hashes, legacy inventory, actual GP03 B1 replay, corrected audit", "requirements": ["R35", "R36"],
                                  "authority": "PACKET_13", "tests": ["tests/test_grain_f00_revisions.py"], "add_to_cards": ["F00-01"],
                                  "claims": {"support": "5/5 sources and both plans match recorded hashes; 80/80 legacy GP/PA/PV files match the 2026-09-15 snapshot; GP03 B1 metrics and 64,800 scores replay byte-identically",
                                             "cannot_infer": "immutability before 2026-09-15; psychological validity; any B2/M property",
                                             "missing_evidence": "same-reader B2/M controls, evidence cards, reader calibration (F08)"}},
    "F00/repair_routing_r2": {"target": "current F00-05: gap routing regenerated from registry and ledger", "requirements": ["R35", "R36"], "authority": "PACKET_13",
                              "tests": ["tests/test_grain_f00_revisions.py"], "add_to_cards": ["F00-05"], "depends_on_revisions": ["F00/baseline_inventory_r2"],
                              "claims": {"support": "every ledger gap and legacy GP/PA/PV issue is routed to named cards with their current evidence and readiness",
                                         "cannot_infer": "that any routed gap is solved; scientific acceptance of any card",
                                         "missing_evidence": "cards without current evidence remain listed as explicit gaps"}},
    "F00/integrity_snapshots_r1": {"target": "dated current integrity snapshots for roots without run-time manifests", "requirements": ["R35"], "authority": "PACKET_13",
                                   "tests": ["tests/test_grain_f00_revisions.py"],
                                   "claims": {"support": "detects any change after 2026-09-26 in 120 snapshotted roots", "cannot_infer": "historical provenance, pre-registration or immutability since creation",
                                              "missing_evidence": "run-time manifests for the snapshotted roots never existed"}},
    "F00/revision_stage_r2": {"target": "repaired preflight: pinned inputs/code, authorization, dependency contract before any child", "requirements": ["R35", "R08"],
                              "authority": "PACKET_13", "tests": ["tests/test_grain_revision_stage.py", "tests/test_grain_revision_preflight.py"], "add_to_cards": ["F00-04"],
                              "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "F00/revision_stage_r1",
                                             "scope": "r1 claimed all execution modes fail closed; its run_adapter launched children without integrity/dependency preflight. r1 replay results retained."}],
                              "claims": {"support": "preflight refuses missing/changed/unmapped inputs, stale authorization, unready scientific dependencies with zero child launches",
                                         "cannot_infer": "scientific validity of any replayed result; diagnostic PASS grants no readiness",
                                         "missing_evidence": "adapters exist for 6 roots only"}},
    "F15/text_behaviour_quotient_r1": {"target": "R08 finite text/behaviour quotient: automorphism-induced lumpability, conflict and unknown-transform counterexamples",
                                       "requirements": ["R08", "A01"], "authority": "PACKET_13", "tests": ["tests/test_grain_r08_text_behaviour_quotient.py"], "add_to_cards": ["F15-01"],
                                       "relations": [{"type": "SUPPLEMENTS", "other": "F15/bisimulation_metric_r1", "scope": "adds Y/B separation, non-trivial G, conflicting and unknown transformations"}],
                                       "claims": {"support": "G-automorphism with invariant construct view gives a lumpable orbit quotient; one-step invariance insufficient (W3); register->B conflict loses information (W2); passive text blind to trait while B/probe resolve (W6); unknown relabeling non-identifiable from text (W4); d<eps non-transitive (W5)",
                                                  "cannot_infer": "human construct validity; any real-text quotient; estimated laws from real histories",
                                                  "missing_evidence": "real histories (H01); natural-text predictive-state falsifier (needs neural authorization)"}},
    "F15/equivalence_hierarchy_r1": {"target": "R08 amendment companion: implication hierarchy counterexamples and Bayes-error identity", "requirements": ["R08"], "authority": "PACKET_14",
                                     "tests": ["tests/test_grain_r08_equivalence_hierarchy.py"], "add_to_cards": ["F15-01"],
                                     "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "F15/text_behaviour_quotient_r1", "scope": "construction-rule wording 'only when/equivalently'; six-world numbers retained"}],
                                     "claims": {"support": "symmetry sufficient not necessary; predictive equivalence coarser than lumpability; single-observation error (1-TV)/2", "cannot_infer": "anything about real texts", "missing_evidence": "none for this finite scope"}},
    "F15/conditional_ib_r1": {"target": "R08 conditional information bottleneck finite witness", "requirements": ["R08"], "authority": "PACKET_14", "tests": ["tests/test_grain_r08_conditional_ib.py"],
                              "add_to_cards": ["F15-01"],
                              "claims": {"support": "deterministic-encoder frontier over 8 observable histories; behaviour carries most retained information; content summaries merge futures", "cannot_infer": "stochastic-IB optimum; human minimality; real histories", "missing_evidence": "observed-history PSR vs static comparison (manifest frozen)"}},
    "F17/speaker_identifiability_r1": {"target": "R15-M1 shared-utility speaker identifiability", "requirements": ["R15"], "authority": "PACKET_14", "tests": ["tests/test_grain_r15_speaker_identifiability.py"],
                                       "claims": {"support": "derived commitment-profile aliasing; probe-dependent Fisher rank; exact equivalences; sample-size bounds", "cannot_infer": "that natural hedges have these causes; the probe-to-gate mapping (stipulated)", "missing_evidence": "reader detection of commitment scope; real speakers"}},
    "F14/sequence_inventory_r1": {"target": "existing PANDORA sequence inventory and frozen next-packet manifests", "requirements": ["R08", "R15"], "authority": "PACKET_14",
                                  "claims": {"support": "624 authors x 36 events, no ties, source/held split, aligned cached vectors", "cannot_infer": "any comparison result (not run)", "missing_evidence": "execution of the frozen manifests"}},
    "F00/common_record_adapter_r2": {"target": "fail-closed COMMON record adapter v2", "requirements": ["R35"], "authority": "SEMINAR_07",
                                     "relations": [{"type": "SUPERSEDES_SCIENTIFIC_USE", "other": "F00/common_record_adapter_r1", "scope": "v1 run_tests ignored pytest exit codes; v2 fails closed (declared packet 18 Part D)"}],
                                     "evidence_doc": "results/grain_gptpro_foundation_v3/F00/common_record_adapter_r2/adapter_report.json",
                                     "tests": ["tests/test_grain_common_records_v2.py"], "add_to_cards": ["F00-02", "F00-03"]},
    "F15/controlled_psr_witness_r1": {"target": "R08 controlled source-fitted finite predictive-state witness (existing W1/W6 kernels)", "requirements": ["R08"], "authority": "PACKET_15",
                                      "tests": ["tests/test_grain_r08_controlled_psr_witness.py"], "add_to_cards": ["F15-01"],
                                      "relations": [{"type": "SUPPLEMENTS", "other": "F15/text_behaviour_quotient_r1", "scope": "fits a spectral PSR from source sequences only; oracle as reference"}],
                                      "claims": {"support": "support-gated arms (3/6 + 2 variants): source-fitted spectral PSR beats static/order-one/history-PCA on held controlled continuations, .005-.007 nats short of oracle; novel combination transports; novel primitive refused",
                                                 "cannot_infer": "natural-text PSR; human histories; do(C) on real contexts; the 3 support-failed arms",
                                                 "missing_evidence": "natural controlled contexts (not in held corpora); real histories (H01)"}},
    "F14/pandora_history_comparison_r1": {"target": "R08 observational PANDORA history comparison (amended frozen manifest; 30 IPIP-NEO-300 keyed facets)", "requirements": ["R08"], "authority": "PACKET_15",
                                          "tests": ["tests/test_grain_r08_pandora_history.py"],
                                          "relations": [{"type": "SUPPLEMENTS", "other": "F14/sequence_inventory_r1", "scope": "executes the amended frozen R08 manifest"}],
                                          "claims": {"support": "prequential held-author scores: last-venue blend beats static venue law (-.025 nats); population AR(1) and PCA5 give no keyed-facet gain; novel venues favour order-one",
                                                     "cannot_infer": "that predictive-state theory is refuted; do(C); general PSR estimation on natural text",
                                                     "missing_evidence": "controlled natural contexts; richer history models under separate authorization"}},
    "F17/hedge_commitment_reader_r1": {"target": "R15 local NLI reader of observable commitment, scope and ownership (576 synthetic pairs)", "requirements": ["R15"], "authority": "PACKET_15",
                                       "tests": ["tests/test_grain_r15_hedge_commitment_reader.py"],
                                       "relations": [{"type": "SUPPLEMENTS", "other": "F17/speaker_identifiability_r1", "scope": "reader detection for the identified utterance-law functionals; no cause assignment"}],
                                       "claims": {"support": "ownership detected by speaker-directed queries; uncertainty/deference cues detected vs bare; own-commitment query frame-gated and fails frozen source form check; bare-proposition query credits quoted content",
                                                  "cannot_infer": "speaker causes; truth; human validity; other readers",
                                                  "missing_evidence": "matched-bank reader comparison; audience frames without an opinion holder; real speakers (H02/H04)"}},
    "F17/hedge_channel_utilization_r1": {"target": "R15 hedged-evidence utilization through a source-calibrated reader channel (576 frozen readings; text as unit)", "requirements": ["R15"], "authority": "PACKET_16",
                                         "tests": ["tests/test_grain_r15_hedge_channel_utilization.py"],
                                         "relations": [{"type": "SUPPLEMENTS", "other": "F17/hedge_commitment_reader_r1", "scope": "reuses readings unchanged as a calibrated observation channel"}],
                                         "claims": {"support": "merged non-commitment share of a collection recovered in all knowledge states with calibrated sets (width .07-.11, coverage 1.0 on held core); single forms and owner coarse; two-pole channel incompatible with the 3-level speaker alphabet (only P(assert|C0) under declared mapping)",
                                                    "cannot_infer": "speaker causes; real speakers; natural text; other readers; independent semantic samples",
                                                    "missing_evidence": "matched-bank cross-reader comparison; compositional re-read (neural authorization); H02/H04"}},
    "F14/pandora_venue_oom_r1": {"target": "R08 source-fitted natural venue-sequence spectral OOM vs static/order-one/blend (prequential)", "requirements": ["R08"], "authority": "PACKET_16",
                                 "tests": ["tests/test_grain_r08_pandora_venue_oom.py"],
                                 "relations": [{"type": "SUPPLEMENTS", "other": "F14/pandora_history_comparison_r1", "scope": "same roles, dictionary and windows; baselines reproduced exactly"}],
                                 "claims": {"support": "rank-20 OOM beats order-one (-.075) but loses to personal static composition (+.145) and blend (+.169); coverage 1.0; loss concentrated on known top-20 venues",
                                            "cannot_infer": "refutation of predictive-state theory; do(C); personality dynamics; rare-venue prediction",
                                            "missing_evidence": "keyed-score (not embedding/GRAIN) falsifier arm (CPU); do(C) contexts (new data); H01"}},
    "P05/application_j_diagnostic_r1": {"target": "R01 native application-J tangent diagnostic (frozen site L12.H0, 72 application pairs x 2 queries)", "requirements": ["R01", "A05"], "authority": "PACKET_17",
                                        "tests": ["tests/test_grain_r01_application_j.py"],
                                        "relations": [{"type": "SUPPLEMENTS", "other": "P05/native_action_r1", "scope": "measures the application tangent the source-only packet lacked; cached effects reused unchanged"},
                                                      {"type": "SUPPLEMENTS", "other": "P05/native_geometry_r1", "scope": "tests the gradient non-transfer candidate directly"}],
                                        "claims": {"support": "application tangent reproduces donor action (.9985/.9988) and lies 96% in the source span; compiled errors are first-order direction mismatch; E24 overshoot is first-order; J24 unrelated direction inside the shared span; nonlinearity minor except J4 on E",
                                                   "cannot_infer": "a remedy; a unique cause of the z->action failure; human personality; LLM-free inference",
                                                   "missing_evidence": "which z->action property fails (overfitting/extrapolation/input insufficiency)"}},
    "F12/designed_wording_facet_r1": {"target": "R25 designed wording-perturbation facet on cached reader outputs (96 texts, 48 pairs)", "requirements": ["R25", "R09"], "authority": "PACKET_17",
                                      "tests": ["tests/test_grain_r25_wording_facet.py"],
                                      "relations": [{"type": "SUPPLEMENTS", "other": "F17/hedge_commitment_reader_r1", "scope": "reuses readings unchanged"}],
                                      "claims": {"support": "identifiable balanced decomposition; query-specific wording shares concentrated in template and context interaction; merged functional wording-robust",
                                                 "cannot_infer": "natural paraphrase facet; inter-reader variation; person-level reliability",
                                                 "missing_evidence": "natural paraphrases; other readers"}},
    "F00/r35_twelve_row_reconciliation_r1": {"target": "R35 typed snapshot evidence levels, contract permission at preflight, and 12-row dependency reconciliation", "requirements": ["R35"], "authority": "PACKET_17",
                                             "tests": ["tests/test_grain_r35_snapshot_contracts.py"],
                                             "claims": {"support": "snapshot integrity typed as LIMITED; fail-closed contract permission; per-row current roots, levels, blocks and missing computation",
                                                        "cannot_infer": "historical provenance; preregistration; scientific or protocol acceptance from readiness",
                                                        "missing_evidence": "reviewed re-record of 2 legacy test records; 4 decision documents; A02 current-evidence mapping"}},
    "P05/product_space_identification_r1": {"target": "R01 product-space identification of the frozen z->action estimators (cached; exact reconstructions and null witnesses)", "requirements": ["R01", "A05"], "authority": "PACKET_18",
                                            "tests": ["tests/test_grain_r01_product_space.py"],
                                            "relations": [{"type": "SUPPLEMENTS", "other": "P05/application_j_diagnostic_r1", "scope": "product-row exposure beyond J-span coverage"}],
                                            "claims": {"support": "z24 product-row exposure .674/.743 despite J-span .96 and Z-span 1.0; source LOO similar (design-level); exact null witnesses carry large application effects; z4 exposure .92 yet J4 failed",
                                                       "cannot_infer": "overfitting, irreducible input loss or unique cause; human accuracy", "missing_evidence": "source-internal design varying observation rows"}},
    "F04/reliability_spine_r1": {"target": "R09 level/response/state/crossed-wording reliability spine (new-draw simulation)", "requirements": ["R09", "R12", "R29"], "authority": "PACKET_18",
                                 "tests": ["tests/test_grain_r09_reliability_spine.py"],
                                 "claims": {"support": "REML-estimated crossed variance; known-model R=1-E Var/Var with MC check; shared- vs disjoint-wording retest vs true recovery; same-time state recovery; held-context vector prediction with curved counterexample; threshold bracket stability",
                                            "cannot_infer": "real person-level reliability (H01); natural-wording facet; continuous threshold", "missing_evidence": "H01; natural paraphrases; denser threshold grid; propagated source uncertainty"}},
    "F16/target_structure_inventory_r1": {"target": "R34 per-target structural object inventory and executable specification", "requirements": ["R34"], "authority": "PACKET_18",
                                          "tests": ["tests/test_grain_r34_target_inventory.py"],
                                          "claims": {"support": "structural objects, gauges, assets, subset-capable builders and declared comparisons for T1/T2/T3 and text routes",
                                                     "cannot_infer": "any structural reliability value (not run)", "missing_evidence": "halves driver; template variant; text row assembly"}},
    "F00/record_reconciliation_r1": {"target": "R35 typed test-record and decision-document reconciliation; A02 mapping proposal", "requirements": ["R35", "A02"], "authority": "PACKET_18",
                                     "tests": ["tests/test_grain_r35_record_reconciliation.py"],
                                     "claims": {"support": "hash-pinned re-records reconcile 2 legacy test-format false negatives; 4 documentation records; adapter v1 superseded by v2; A02 mapping proposed",
                                                "cannot_infer": "scientific or protocol acceptance; historical provenance", "missing_evidence": "supervisor acceptance of the A02 mapping; registration of the F05 near-root audit summary"}},
    "F16/source_half_reconstruction_r1": {"target": "R34 actual source-half structural reconstruction of T1/T2/T3 (POINT_FIT_PLUGIN half refits)", "requirements": ["R34"], "authority": "PACKET_19",
                                          "tests": ["tests/test_grain_r34_source_halves.py"],
                                          "relations": [{"type": "SUPPLEMENTS", "other": "F16/target_structure_inventory_r1", "scope": "executes the inventory's halves driver specification"}],
                                          "claims": {"support": "target posteriors move 2-7% of conditional SD between disjoint 150-author half refits; covariance varies .11-.22; text-channel halves degenerate (2 cores/stratum)",
                                                     "cannot_infer": "real-data or small-source stability; parameter-structure identity; text author-level scores", "missing_evidence": "subset template variant; text half-channel bundle path; more calibration cores"}},
    "F06/reactive_head_to_head_r1": {"target": "R26 continuous-target head-to-head with declared reactivity and common random numbers", "requirements": ["R26", "A07"], "authority": "PACKET_19",
                                     "tests": ["tests/test_grain_r26_reactive_head_to_head.py"],
                                     "relations": [{"type": "SUPPLEMENTS", "other": "F06/head_to_head_r1", "scope": "same bank/grids/policies with a declared reaction state and paired streams"}],
                                     "claims": {"support": "ignored reaction biases probe-heavy trait estimates; known-law handling restores accuracy; handling changes the preferred policy",
                                                "cannot_infer": "real reactivity; unknown-law performance; a universal policy ranking", "missing_evidence": "reaction-state estimation without a known law; H04 parameters"}},
    "F05/near_root_identification_r1": {"target": "A02 evidence repair: verified invariants certificates and per-cell near-root re-derivation (registers the application audit summary)", "requirements": ["A02"], "authority": "PACKET_19",
                                        "tests": ["tests/test_grain_a02_near_root_evidence.py"], "requirement_roles": {"A02": "primary"},
                                        "relations": [{"type": "SUPPLEMENTS", "other": "F05/application_full_r1", "scope": "per-cell re-derivation from its state_parameter_metrics.csv"},
                                                      {"type": "SUPPLEMENTS", "other": "F05/invariants_r1", "scope": "independent certificate verification"}],
                                        "claims": {"support": "certificates verified; near-root weak practical identification is estimator-level state/intercept confounding (89/90 cells)",
                                                   "cannot_infer": "real-system identification", "missing_evidence": "none at finite simulated scope"}},
    "F00/supervisor2042_scope_note_r1": {"target": "supervisor-2042 interpretation corrections for packet-18 R01 and R09", "requirements": ["R01", "R09"], "authority": "SUPERVISOR_2042",
                                         "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "P05/product_space_identification_r1", "scope": "exposure metric; norm assumption; descriptive weights; model-internal witness; numbers retained"},
                                                       {"type": "SUPERSEDES_INTERPRETATION", "other": "F04/reliability_spine_r1", "scope": "criterion recovery vs reliability; DGP-specific masquerade; unpropagated source uncertainty; numbers retained"}],
                                         "claims": {"support": "interpretation corrections only", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F16/error_budget_reconciliation_r1": {"target": "R34 cached error-budget reconciliation: component identity of archived T1/T2/T3 readouts for the implemented mixtures", "requirements": ["R34"], "authority": "PACKET_20",
                                           "tests": ["tests/test_grain_r34_error_budget.py"],
                                           "relations": [{"type": "SUPPLEMENTS", "other": "F16/serialize_full_r1", "scope": "component identity of its cold readouts (aliases resolved, not double counted)"},
                                                         {"type": "SUPPLEMENTS", "other": "F16/t3_application_full_r1", "scope": "per-bootstrap reconstruction incl. state-carried vs future-innovation split"}],
                                           "claims": {"support": "45,600 readouts: stored components sum to totals and per-component reconstruction within 1e-12; two one-person text cold replays expose E[V]+Var(mean); combined source-reader terms kept combined",
                                                      "cannot_infer": "exact psychological posterior; separation of source from reader; accuracy improvement", "missing_evidence": "per-component text parts beyond the replayed persons; text comparator-arm parts not saved"}},
    "F16/text_half_channel_sensitivity_r1": {"target": "R34 text half-channel calibration sensitivity pilot (one-component diagnostic views)", "requirements": ["R34"], "authority": "PACKET_20",
                                             "tests": ["tests/test_grain_r34_text_half_channel.py"],
                                             "relations": [{"type": "SUPPLEMENTS", "other": "F16/source_half_reconstruction_r1", "scope": "text half-channel score effect omitted in packet 19"}],
                                             "claims": {"support": "NLI channel core-sensitive (T3 mean |A-B| .57, .83 view-SD; T1 .30; T2 .13); hybrid half channels identical so scores cannot differ (insensitivity)",
                                                        "cannot_infer": "split-half reliability; additive reader variance; production-ensemble behaviour", "missing_evidence": "more calibration cores per stratum (material design)"}},
    "F06/fixed_history_inference_r1": {"target": "R26 fixed-history aware rescore of naive histories and isolated reaction-cost control", "requirements": ["R26", "A07"], "authority": "PACKET_20",
                                       "tests": ["tests/test_grain_r26_fixed_history.py"],
                                       "relations": [{"type": "SUPPLEMENTS", "other": "F06/reactive_head_to_head_r1", "scope": "same streams; 2,400 histories rescored without planner calls; 800 cost-0 runs paired to verified replays"}],
                                       "claims": {"support": "O/S known-law bias reduction is an inference effect on fixed histories; OC aware choice loss (+.026) comes from the .10 reaction-cost term (cost-0 recovers -.024 T1 with more probes and burden)",
                                                  "cannot_infer": "individual causal state effect; real reactivity; policy preference without a declared utility", "missing_evidence": "H04 real parameters"}},
    "F04/generalizability_coefficients_r1": {"target": "R09 design-specific G coefficients of declared composites from saved reliability-spine REML moments", "requirements": ["R09"], "authority": "PACKET_20",
                                             "tests": ["tests/test_grain_r09_g_coefficients.py"],
                                             "relations": [{"type": "SUPPLEMENTS", "other": "F04/reliability_spine_r1", "scope": "closed-form G from its saved moments; no refit"}],
                                             "claims": {"support": "random- vs fixed-wording Eρ² per target; HIGH B12 half .24-.30 vs .55-.58 matches disjoint/shared retest; Φ_random not estimated",
                                                        "cannot_infer": "human reliability; propagated source uncertainty; EB posterior reliability", "missing_evidence": "H01; natural paraphrases"}},
    "F00/supervisor2143_corrections_r1": {"target": "supervisor-2143 append-only corrections for packet-19 R26, R34 and A02 wording", "requirements": ["R26", "R34", "A02"], "authority": "SUPERVISOR_2143",
                                          "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "F06/reactive_head_to_head_r1", "scope": "12,800 person-runs; probe share vs count; regime not isolated inference; numbers retained"},
                                                        {"type": "SUPERSEDES_INTERPRETATION", "other": "F16/source_half_reconstruction_r1", "scope": "sensitivity not structure/interval validation; stability ranking withdrawn; TV0 insensitivity; numbers retained"},
                                                        {"type": "SUPERSEDES_INTERPRETATION", "other": "F05/near_root_identification_r1", "scope": "oracle .133->.151 not unaffected; numbers retained"}],
                                          "claims": {"support": "interpretation corrections only", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F16/r34_finite_candidate_r1": {"target": "R34 finite-acceptance candidate: original-clause map (compiler, uncertainty, source-half reconstruction), scope and human rebuild separate", "requirements": ["R34"], "authority": "PACKET_20",
                                    "claims": {"support": "clause-to-evidence map for supervisor adjudication", "cannot_infer": "acceptance or promotion", "missing_evidence": "supervisor adjudication; H06; text-route material design"}},
    "F00/supervisor2245_corrections_r1": {"target": "supervisor-2245 append-only interpretation corrections for packet-20 roots", "requirements": ["R09", "R26", "R34", "A07"], "authority": "SUPERVISOR_2245",
                                          "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "F16/error_budget_reconciliation_r1", "scope": "43,800/12,600/45,600/1,800 check classes; T3 shares route-specific; numbers retained"},
                                                        {"type": "SUPERSEDES_INTERPRETATION", "other": "F16/text_half_channel_sensitivity_r1", "scope": "hybrid halves invariant to this core substitution, not globally insensitive (rank 4, det 1/6); numbers retained"},
                                                        {"type": "SUPERSEDES_INTERPRETATION", "other": "F04/generalizability_coefficients_r1", "scope": "fixed-wording G is a changed universe (half 1 wording, full 3); numbers retained"},
                                                        {"type": "SUPERSEDES_INTERPRETATION", "other": "F06/fixed_history_inference_r1", "scope": "cost term accounts for most, not all, of the OC choice loss; model accounting; numbers retained"}],
                                          "claims": {"support": "interpretation corrections with recount/geometry replay", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F03/paired_four_component_identification_r1": {"target": "R05 paired selection/interpretation/style/realization identification with declared loadings, null and latent-coordinate witnesses", "requirements": ["R05"], "authority": "PACKET_21",
                                                    "tests": ["tests/test_grain_r05_four_component.py"],
                                                    "claims": {"support": "A/B/C rank 2 (exact null witness), A-E rank 4 only relative to declared idealized D/E; undeclared basis-row changes (leaky clamp, changed D style loading) are absorbed by an invertible coordinate change with zero misfit and wrong named components; only the C=B restriction is observable (W4 detected, cause not located); paired Var(A-B) recovers choice where naive variance subtraction fails; corrects historical FU3-f (shared material does not delete interpretation)",
                                                               "cannot_infer": "real task exclusions; human interpretation variance; selection's incremental external validity; natural-text separation", "missing_evidence": "HUMAN task/loading validation and external criterion; supervisor adjudication of the R05 candidate"}},
    "F00/supervisor2347_notes_r1": {"target": "supervisor-2347 R05 finite acceptance and three append-only scope notes", "requirements": ["R05"], "authority": "SUPERVISOR_2347",
                                    "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "F03/paired_four_component_identification_r1", "scope": "'only C=B observable' = only over-identifying between-person covariance restrictions of this design; W4 zero-eigen direction; source-fixed application bootstrap; numbers retained"}],
                                    "claims": {"support": "interpretation notes only", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F14/history_increment_r1": {"target": "R08 natural observational history: nested venue OOM mixture and coupled keyed semantic-history readouts with prefix-permutation control", "requirements": ["R08"], "authority": "PACKET_22",
                                 "tests": ["tests/test_grain_r08_history_increment.py"],
                                 "relations": [{"type": "SUPPLEMENTS", "other": "F14/pandora_venue_oom_r1", "scope": "nested convex contrast with its saved rank-20 operators (no refit)"},
                                               {"type": "SUPPLEMENTS", "other": "F14/pandora_history_comparison_r1", "scope": "coupled keyed readouts on its roles and targets"}],
                                 "claims": {"support": "no reliably detectable held increment beyond composition / prefix mean in these estimators (venue mix -.0016 [-.019,.016]; keyed history increments within ~.18% of MSE; order control inconclusive)",
                                            "cannot_infer": "absence of all dynamic information; do(C); raw-embedding/GRAIN equivalence; construct validity; confirmatory status (held authors reused)", "missing_evidence": "raw-embedding/GRAIN comparator; controlled contexts; human histories"}},
    "F14/r08_comparator_inventory_r1": {"target": "R08 raw-embedding/GRAIN comparator cache inventory (metadata only)", "requirements": ["R08"], "authority": "PACKET_22",
                                        "claims": {"support": "exact-cohort event-level e5-large-v2 comment cache identified (author order verified); GRAIN only user-level and leaking",
                                                   "cannot_infer": "any comparator result", "missing_evidence": "row-order bridge; prefix GRAIN recomputation"}},
    "F00/supervisor0048_qualification_r1": {"target": "supervisor-0048 append-only qualification: packet-22 .18% wording and GRAIN per-comment cache inventory correction", "requirements": ["R08"], "authority": "SUPERVISOR_0048",
                                            "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "F14/history_increment_r1", "scope": ".18% = scaled CI endpoint for a fixed contrast, not a universal bound; numbers retained"},
                                                          {"type": "SUPERSEDES_INTERPRETATION", "other": "F14/r08_comparator_inventory_r1", "scope": "GRAIN per-comment units exist in vm05_preflight_r1; numbers retained"}],
                                            "claims": {"support": "wording/inventory qualification only", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F14/raw_grain_history_r1": {"target": "R08 exact-cache raw-vector (E5 PCA30) and VM05 lexical/SVD mean-std GRAIN prefix comparators at the common keyed next-event target", "requirements": ["R08"], "authority": "PACKET_23",
                                 "tests": ["tests/test_grain_r08_raw_grain_history.py"],
                                 "relations": [{"type": "SUPPLEMENTS", "other": "F14/history_increment_r1", "scope": "same roles/targets; frozen references replayed"}],
                                 "claims": {"support": "bridges verified (raw exact; VM05 identities + hash chain; 0 representation overlap); keyed static beats raw PCA30 and VM05 GRAIN summaries at the keyed target on all 30 coordinates; raw order increment inconclusive",
                                            "cannot_infer": "absence of information for raw/behavioural targets; whole-framework GRAIN; PSR minimality; do(C); construct validity; confirmatory status", "missing_evidence": "controlled contexts; human validation; per-comment GRAIN content order check"}},
    "F14/raw_grain_history_repairs_r1": {"target": "packet-23 acceptance repairs: coordinate-CI wording, machine-readable hash chain, metadata-only comment-ID order, standalone sealed cold scoring", "requirements": ["R08"], "authority": "PACKET_24",
                                         "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "F14/raw_grain_history_r1", "scope": "per-coordinate POINT (not CI) wording; order now comment-ID verified; numbers retained"}],
                                         "claims": {"support": "hash chain 4/4; 624x36 comment-ID order verified in both sources; fresh sealed process reproduces all held losses (max 0) from bundle + declared inputs only",
                                                    "cannot_infer": "new scientific result", "missing_evidence": "n/a"}},
    "P05/source_crossed_action_r1": {"target": "R01 source-crossed derivative design: Kronecker exposure, fixed-linear-map column share, 12-cell held-out D/C/R/F/E/zero comparison", "requirements": ["R01"], "authority": "PACKET_24",
                                     "tests": ["tests/test_grain_r01_source_crossed.py"],
                                     "relations": [{"type": "SUPPLEMENTS", "other": "P05/product_space_identification_r1", "scope": "crossed source design on the same frozen caches"}],
                                     "claims": {"support": "full-cross rank 864/144 and held exposure .97 without held derivative gain; fixed linear z map misfits source crossed derivatives (P_Z share .10-.68); zero beats all designs at z24",
                                                "cannot_infer": "finite native crossed effects; unique cause (misspecification vs representation vs coverage); independent new observations", "missing_evidence": "authorized finite crossed patches; intercept/nonlinear map test"}},
    "F00/supervisor0251_corrections_r1": {"target": "supervisor-0251 append-only scope corrections for packet-24 R01/R08 wording", "requirements": ["R01", "R08"], "authority": "SUPERVISOR_0251",
                                          "relations": [{"type": "SUPERSEDES_INTERPRETATION", "other": "P05/source_crossed_action_r1", "scope": "32%/44% residual not most; coverage alone did not fix these estimators; no exported operator bundle; numbers retained"},
                                                        {"type": "SUPERSEDES_INTERPRETATION", "other": "F14/raw_grain_history_repairs_r1", "scope": "material timestamps/view index not separately tested; no-refit replay is not source-row-unavailable or OS isolation; numbers retained"}],
                                          "claims": {"support": "interpretation corrections only", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F06/recovery_target_design_r1": {"target": "R26/A07 recovery-parameter observation design: fixed vs scalar-level adaptive reference vs recovery-target adaptive at matched observation and intervention quota", "requirements": ["R26", "A07"], "authority": "PACKET_25",
                                      "tests": ["tests/test_grain_r26_recovery_design.py"],
                                      "relations": [{"type": "SUPPLEMENTS", "other": "F06/gaussian_full_r1", "scope": "same world, streams, prior and inference; new quota-matched arms"}],
                                      "claims": {"support": "see decision.md (finite simulated comparison at matched burden)", "cannot_infer": "psychological recovery; real burden; published ALBA/IRT behaviour; longer-horizon optimal design",
                                                 "missing_evidence": "human recovery/burden parameters (H04)"}},
    "F06/recovery_committed_design_r1": {"target": "R26/A07 committed two-observation recovery design: scored design = executed design for both adaptive arms (odd-index epochs), plus beta0 schedule replay on fixed z", "requirements": ["R26", "A07"], "authority": "PACKET_26",
                                         "tests": ["tests/test_grain_r26_committed_design.py"],
                                         "relations": [{"type": "SUPPLEMENTS", "other": "F06/recovery_target_design_r1", "scope": "same world/streams/inference/quota; corrected commitment rule; fixed arm replayed exactly"},
                                                       {"type": "SUPERSEDES_INTERPRETATION", "other": "F06/recovery_target_design_r1", "scope": "packet-25 scalar reference degenerate from scored-vs-executed mismatch; front-loading/action-count causal readings not isolated; numbers retained"}],
                                         "claims": {"support": "see decision.md", "cannot_infer": "published ALBA/IRT; longer-horizon optimal design; psychological recovery; real burden", "missing_evidence": "H04"}},
    "P05/source_affine_action_r1": {"target": "R01 common donor shift versus z-dependent action: ZERO/COMMON/LINEAR/AFFINE (unpenalized intercept) on cached crossed derivatives, 12-cell held-out, source projection shares, origin/orientation checks", "requirements": ["R01"], "authority": "PACKET_27",
                                    "tests": ["tests/test_grain_r01_source_affine.py"],
                                    "relations": [{"type": "SUPPLEMENTS", "other": "P05/source_crossed_action_r1", "scope": "same frozen caches/folds/loss; LINEAR replays packet 24 exactly; adds COMMON and AFFINE"}],
                                    "claims": {"support": "common donor shift carries 8-9% of source crossed-derivative energy; AFFINE improves source in-sample fit but not held transfer (AFFINE-LINEAR inconclusive x4; AFFINE-COMMON worse at z24 on both panels); COMMON indistinguishable from zero",
                                               "cannot_infer": "finite native crossed effects; unique cause among representation/nonlinearity/finite-action/coverage; human personality accuracy; equivalence from CIs containing 0", "missing_evidence": "authorized finite crossed patches or a new representation hypothesis (not authorized)"}},
    "F04/reliability_information_r1": {"target": "R12 reliability vs readings and vs condition diversity in the declared scalar world: analytic sigma2/Szz explanation, paired 3-seed CRN simulation, equal-Szz diagnostic, stratified author-bootstrap CIs and log-grid curvature; FU2-c W1/D1 exact replay with person-level CIs", "requirements": ["R12"], "authority": "PACKET_28",
                                       "tests": ["tests/test_grain_r12_reliability_information.py"],
                                       "relations": [{"type": "SUPPLEMENTS", "other": "F04/excitation_reliability_r2", "scope": "same world and old design grid; analytic explanation reproduces its curve; adds CIs, curvature and equal-Szz diagnostic; old numbers retained"},
                                                     {"type": "SUPPLEMENTS", "other": "F09/rival_person_simulation_r1", "scope": "exact W1/D1 replay (1.1e-16) saving person arrays; supplies FU2-c omitted mean-effect/retest CIs"}],
                                       "claims": {"support": "old diversity decline = Szz decline; equal-Szz K contrasts inconclusive; spread alone reproduces the effect; FU2-c person-level CIs recovered in simulation",
                                                  "cannot_infer": "real-token or real-person reliability; absence of context-specific nonlinear information; monotonically rising diversity curves", "missing_evidence": "H01 human reliability/validity; context-specific informative-context world (not authorized)"}},
    "F17/claim_level_reconciliation_r1": {"target": "R35/F17 claim-level reconciliation: every audit-r3 clause and non-audited row gap traced to current evidence with typed status, live pointer checks, stale generic ledger tails itemized (bookkeeping only)", "requirements": ["R35"], "authority": "PACKET_29",
                                          "tests": ["tests/test_grain_f17_claim_reconciliation.py"],
                                          "relations": [{"type": "SUPPLEMENTS", "other": "F00/r35_twelve_row_reconciliation_r1", "scope": "extends the 12-row reconciliation to all 50 rows at clause level; no readiness rule changed"},
                                                        {"type": "SUPPLEMENTS", "other": "F00/record_reconciliation_r1", "scope": "links its 4 decision documents and 2 test re-records; no duplicate repair"}],
                                          "claims": {"support": "bookkeeping: 201 clauses mapped, 50 rows and 100 cards covered, 0 pointer failures, ledger statuses unchanged",
                                                     "cannot_infer": "any scientific result; any status promotion; resolution of the 64 UNADJUDICATED clauses", "missing_evidence": "supervisor adjudication of 15 candidate mappings and of R12 finite review"}},
    "F17/claim_level_reconciliation_r2": {"target": "union-of-current-gaps supplement: audit clauses plus CURRENT gaps of all 50 rows; mandatory R12/R26/mixed-row fixes; overrides for R21/R23/R36/R03 (bookkeeping only)", "requirements": ["R35"], "authority": "PACKET_30",
                                          "tests": ["tests/test_grain_f17_claim_reconciliation_r2.py"],
                                          "relations": [{"type": "SUPPLEMENTS", "other": "F17/claim_level_reconciliation_r1", "scope": "adds current-gap rows for audited rows; r1 immutable"},
                                                        {"type": "SUPERSEDES_INTERPRETATION", "other": "F17/claim_level_reconciliation_r1", "scope": "withdraws r1 R12 human-only remainder inference and r1 package-1/2 wording; r1 rows retained"}],
                                          "claims": {"support": "bookkeeping: every current ledger gap represented; sentinel R12/R26 clauses present; statuses unchanged",
                                                     "cannot_infer": "any scientific result or promotion", "missing_evidence": "semantic mapping of UNADJUDICATED current-gap rows"}},
    "F07/freedom_components_r1": {"target": "R13 declared-world crossed author x condition x occasion x reader component estimation at five paired freedom levels, three targets, two reader regimes, aliased negative control", "requirements": ["R13"], "authority": "PACKET_30",
                                  "tests": ["tests/test_grain_r13_freedom_components.py"],
                                  "relations": [{"type": "SUPPLEMENTS", "other": "F07/openness_mechanism_r1", "scope": "replaces declared-curve information with components estimated from crossed observations; old numbers retained"}],
                                  "claims": {"support": "components identifiable with repeats (aliased control flagged); freedom decision flips with target definition while G rises; repeatability != fidelity != recovery",
                                             "cannot_infer": "real texts/readers/persons; interior optimum; psychological meaning of synthetic targets", "missing_evidence": "empirical five-level study (generation authorization or H04)"}},
    "F04/informative_contexts_r1": {"target": "R12 informative contexts vs prompt counts: known-loading information geometry with persistent context errors (D1/D2/D3/relabel), source-only noise estimation, GLS target risk, paired contrasts", "requirements": ["R12"], "authority": "PACKET_31",
                                    "tests": ["tests/test_grain_r12_informative_contexts.py"],
                                    "relations": [{"type": "SUPPLEMENTS", "other": "F04/reliability_information_r1", "scope": "extends scalar Szz to a 3-D information matrix with persistent context errors; packet-28 numbers retained"}],
                                    "claims": {"support": "direction vs persistent-error independence vs relabel separated; unidentified targets refused; exact tau2=0 identity; naive label counting under-covers",
                                               "cannot_infer": "real tokens; estimated loadings; novel-context variance; human persons; general diversity theorem", "missing_evidence": "real-token inventory; estimated-loadings extension (deferred); H01"}},
    "F04/informative_contexts_provenance_s1": {"target": "regenerated provenance for F04/informative_contexts_r1: source observations, stable author IDs, role/seed bindings verified by deterministic replay (not new research)", "requirements": ["R12"], "authority": "PACKET_32",
                                               "relations": [{"type": "SUPPLEMENTS", "other": "F04/informative_contexts_r1", "scope": "provenance only; r1 untouched"}],
                                               "claims": {"support": "6/6 calibration rows and 24/24 application arrays reproduce", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F01/generation_query_r1": {"target": "R04 generating causes (goal, audience, language, proficiency) vs query-specific sensors in a declared static subcase: crossed vs confounded G/A, anchors vs none, distinct vs replica sensors, exact observational-equivalence witnesses", "requirements": ["R04"], "authority": "PACKET_32",
                                "tests": ["tests/test_grain_r04_generation_query.py"],
                                "relations": [{"type": "SUPPLEMENTS", "other": "F01/combined_world_r1", "scope": "adds G/A/L/K on the generation side and genuinely distinct sensors; combined-world numbers retained"}],
                                "claims": {"support": "recorded crossed contrast cancels realization gain; bilingual anchors fix language offset; exact rivals for unrecorded G/A, constant-K reader-side, no-anchor language; distinct sensors raise rank not precision",
                                           "cannot_infer": "natural language generation; real proficiency/language effects; dynamic feedback jointly with G/A/L/K; psychological meaning", "missing_evidence": "exposure/condition logs; human validation"}},
    "F01/generation_query_provenance_s1": {"target": "regenerated provenance for F01/generation_query_r1: observations, per-author sufficient statistics, fitted calibration bundles, stable author IDs, simulator-only truth kept separate (not new research)", "requirements": ["R04"], "authority": "PACKET_33",
                                           "relations": [{"type": "SUPPLEMENTS", "other": "F01/generation_query_r1", "scope": "provenance only; r1 untouched; 45/45 metric rows replayed"}],
                                           "claims": {"support": "deterministic replay reproduces r1 metrics to 3.6e-15", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F01/ratio_inference_r1": {"target": "R04 output domain of the contrast-normalized score: exact per-query t Fieller sets, Bonferroni joint intersection, fixed-known calibration reference and plug-in calibration diagnostic on frozen packet-32 observations", "requirements": ["R04"], "authority": "PACKET_33",
                               "tests": ["tests/test_grain_ratio_sets.py"],
                               "relations": [{"type": "SUPPLEMENTS", "other": "F01/generation_query_r1", "scope": "set-valued inference on the same frozen observations; point estimates retained"},
                                             {"type": "SUPERSEDES_INTERPRETATION", "other": "F01/generation_query_r1", "scope": "packet-32 wording corrections (orientation convention, sufficiency not necessity, descriptive distinct-vs-replica, negative scale scored not refused, ratio moments); numbers retained"}],
                               "claims": {"support": "fixed-calibration Fieller coverage >= .95 within MC error; unbounded sets predicted by noncentral t; plug-in calibration remains below nominal; empty joint sets expose the negative replica calibration",
                                          "cannot_infer": "exact source-calibration uncertainty; unknown loadings; natural text; psychological polarity", "missing_evidence": "joint source-calibration procedure; exposure logs; human validation"}},
    "F01/ratio_inference_summary_s1": {"target": "derived per-seed domain proportions and finite widths from saved ratio_inference_r1 per-author sets (no new fit)", "requirements": ["R04"], "authority": "PACKET_34",
                                       "relations": [{"type": "SUPPLEMENTS", "other": "F01/ratio_inference_r1", "scope": "reporting only"}],
                                       "claims": {"support": "90 per-seed summary rows; empty sets excluded from width, kept in coverage denominators", "cannot_infer": "any new result", "missing_evidence": "n/a"}},
    "F06/alirt_reference_r1": {"target": "R26 direct ALIRT published-method reference: paper-equation SciPy implementation (per-item source mapping, quartile polytomization, fitted graded response model, MFI CAT with MAP) vs source-chosen fixed order vs operator-aware policy at matched items/burden, common and own scorers", "requirements": ["R26", "A07"], "authority": "PACKET_34",
                               "tests": ["tests/test_grain_alirt_reference.py"],
                               "relations": [{"type": "SUPPLEMENTS", "other": "F06/recovery_committed_design_r1", "scope": "adds the published-method scalar reference distinct from the accepted Gaussian reference"}],
                               "claims": {"support": "positive control passes (ALIRT works in scalar world; operator adds nothing); in condition-response world operator selection+estimator reduce new-condition error greatly (RMSE .60 vs 1.30-1.39) while scalar target ties/trades off",
                                          "cannot_infer": "authors' code or mirt EM behaviour; natural language; clinical benchmark; temporal/recovery targets; human validity", "missing_evidence": "authors' code (repo 404); real text; human validation"}},
}

R = "results/grain_gptpro_foundation_v3/"
P11 = R + "F13/personal_law_r1/"
P12 = R + "F13/personal_law_r2/"
EP = R + "F05/common_shock_episodes_r1/"
F16S = R + "F16/serialize_full_r1/"
F16_PICKS = ("numeric_t1_20260915_identity_global_mixture_T1", "numeric_dynamic_20260915_identity_global_mixture_T2", "text_t1_20260915_mixture_T1")
GP01_ROOT = "results/grain_measurement_discovery_v1/gp01/r1"
R24_CODE = ["scripts/run_grain_r24_personal_law.py", "grain_research/gptpro_foundation_v3/f02.py", "grain_research/gptpro_foundation_v3/f02_recovery.py",
            "grain_research/gptpro_foundation_v3/f13_binary_ensemble.py", "grain_research/gptpro_foundation_v3/f13_binary_source.py",
            "grain_research/gptpro_foundation_v3/f08_estimators.py", "grain_research/measurement_discovery/estimators.py"]
ADAPTER_CODE = ["grain_research/gptpro_foundation_v3/revision_adapters.py"]
ADAPTERS = {
    "r24_r1_analysis": {"root": "F13/personal_law_r1", "mode": "reproduce", "dependency_mode": "scientific", "compute_cap_seconds": 240,
                        "what": "rerun r1 analysis from stored author rows; byte-compare 7 outputs",
                        "inputs": [("author_metrics", P11 + "author_metrics.csv", "file"), ("source_validation", P11 + "source_validation_calibration.csv", "file"),
                                   ("source_side", P11 + "source_side.json", "file"), ("bootstrap_dir", R + "F13/binary_source_full_r2", "tree")]
                                  + [(f"stored_{f}", P11 + f, "file") for f in ("summary.csv", "paired_contrasts.csv", "coverage_intervals.csv", "source_validation_summary.csv",
                                                                                  "homogeneous_reduction.json", "tau2_truth_vs_source.csv", "analysis.json")],
                        "code": ADAPTER_CODE + ["scripts/analyze_grain_r24_personal_law.py"] + R24_CODE, "requires_ready": ["F13/personal_law_r1"]},
    "r24_r2_cold_verify": {"root": "F13/personal_law_r2", "mode": "reproduce", "dependency_mode": "scientific", "compute_cap_seconds": 240,
                           "what": "independent cold verification from stored event trace",
                           "inputs": [("event_trace", P12 + "event_trace.parquet", "file"), ("histories", P12 + "histories.parquet", "file"),
                                      ("truth", P12 + "truth_evaluation_only.parquet", "file"), ("author_metrics_r2", P12 + "author_metrics_r2.csv", "file"),
                                      ("stored_cold_verification", P12 + "cold_verification.json", "file")],
                           "code": ADAPTER_CODE + ["scripts/analyze_grain_r24_personal_law_r2.py"], "requires_ready": ["F13/personal_law_r2"]},
    "r10_episode_analysis": {"root": "F05/common_shock_episodes_r1", "mode": "reproduce", "dependency_mode": "scientific", "compute_cap_seconds": 180,
                             "what": "rerun episode analysis from stored episodes; byte-compare 6 outputs",
                             "inputs": [("episodes", EP + "episodes", "tree"), ("joint_full_r1_metrics", R + "F05/joint_full_r1/joint_state_metrics.csv", "file")]
                                       + [(f"stored_{f}", EP + f, "file") for f in ("levels.csv", "paired_contrasts.csv", "source_component.csv", "source_component.json",
                                                                                    "archived_three_episode_reference.csv", "analysis.json")],
                             "code": ADAPTER_CODE + ["scripts/analyze_grain_r10_common_shock_episodes.py"], "requires_ready": ["F05/common_shock_episodes_r1"]},
    "r24_r1_canonical_replay": {"root": "F13/personal_law_r1", "mode": "execute", "dependency_mode": "scientific", "compute_cap_seconds": 180,
                                "what": "regenerate W0 seed 20260915 slice; compare to stored replay check",
                                "inputs": [("recovery3_author_metrics", R + "F13/recovery3_r1/author_metrics.csv", "file"), ("bootstrap_dir", R + "F13/binary_source_full_r2", "tree"),
                                           ("stored_replay_check", P11 + "replay_check.json", "file")],
                                "code": ADAPTER_CODE + R24_CODE, "requires_ready": ["F13/personal_law_r1", "F13/recovery3_r1", "F13/binary_source_full_r2"]},
    "r10_one_episode_replay": {"root": "F05/common_shock_episodes_r1", "mode": "execute", "dependency_mode": "scientific", "compute_cap_seconds": 180,
                               "what": "regenerate and refit seed 20261001 in both worlds; compare author metrics",
                               "inputs": [(f"stored_{w}_{k}", EP + f"episodes/{w}_20261001/{f}", "file") for w in ("core", "shared_shock")
                                          for k, f in (("author_metrics", "author_metrics.csv"), ("episode", "episode.json"))],
                               "code": ADAPTER_CODE + ["scripts/run_grain_r10_common_shock_episodes.py", "scripts/run_grain_gptpro_f05_joint.py",
                                                       "grain_research/gptpro_foundation_v3/f05.py", "grain_research/gptpro_foundation_v3/f05_recovery.py",
                                                       "grain_research/gptpro_foundation_v3/f05_joint.py"],
                               "requires_ready": ["F05/common_shock_episodes_r1"]},
    "f16_sealed_cold_subset": {"root": "F16/serialize_full_r1", "mode": "execute", "dependency_mode": "scientific", "compute_cap_seconds": 240,
                               "what": "sealed cold scorer on 3 bundles (numeric_t1, numeric_dynamic, text_t1); compare scores",
                               "inputs": [("bundle_index", F16S + "bundle_index.json", "file")]
                                         + [(f"{kind}__{n}", F16S + path.format(n=n), "file") for n in F16_PICKS
                                            for kind, path in (("bundle", "bundles/{n}.json"), ("requests", "cold/{n}.requests.jsonl"), ("stored_scores", "cold/{n}.scores.jsonl"))],
                               "code": ADAPTER_CODE + ["F16_FILES_OPENED"], "requires_ready": ["F16/serialize_full_r1"]},
    "r08_gp01_collision": {"root": "F02/collision_provenance_r1", "mode": "execute", "dependency_mode": "diagnostic", "compute_cap_seconds": 180,
                           "what": "diagnostic replay of legacy GP01 history: re-hash, exact equations, regeneration; compare to stored provenance",
                           "inputs": [("gp01_dir", GP01_ROOT, "tree"), ("snapshot_20260915", str(PERSONA / "reports/gptpro_seminar_20260915/full_coverage_audit/frozen_artifacts_snapshot.json"), "file", True),
                                      ("f00_01_legacy_input_manifest", R + "F00/r1/F00-01/input_manifest.json", "file"),
                                      ("future_equivalence", R + "F02/recovery_full_r2/future_equivalence.csv", "file"),
                                      ("stored_provenance", R + "F02/collision_provenance_r1/provenance.json", "file")],
                           "code": ADAPTER_CODE + ["scripts/verify_grain_r08_collision_provenance.py", "grain_research/measurement_discovery/collision_worlds.py",
                                                   "scripts/run_grain_collision_worlds.py"],
                           "history_roots": [GP01_ROOT, "F00/r1/F00-01"]},
}


def evidence_doc(root: Path) -> str | None:
    for name in EVIDENCE_DOCS:
        if (root / name).exists():
            return str((root / name).relative_to(REPO))
    return None


def common_meta(root: Path) -> dict:
    cr = root / "common_record_r1"
    out = {"common_record": cr.is_dir()}
    if (cr / "summary.json").exists():
        out["status"] = json.loads((cr / "summary.json").read_text()).get("status")
    if (cr / "protocol.json").exists():
        out["protocols"] = json.loads((cr / "protocol.json").read_text()).get("protocol_docs", [])
    elif (root / "summary.json").exists():
        try:
            out["status"] = json.loads((root / "summary.json").read_text()).get("status")
        except Exception:  # noqa: BLE001
            pass
    return out


def amendments(key: str, root: Path) -> list[dict]:
    im = root / "common_record_r1/input_manifest.json"
    if not im.exists():
        return []
    out = []
    for e in json.loads(im.read_text()).get("packet_files", []):
        p = root / e["path"]
        if not p.exists() or sha(p) == e["sha256"]:
            continue
        data = p.read_bytes()
        cut = next((i for i in range(len(data)) if hashlib.sha256(data[:i]).hexdigest() == e["sha256"]), None)
        out.append({"file": e["path"], "recorded_sha256": e["sha256"], "recorded_prefix_bytes": cut, "current_sha256": sha(p),
                    "append_only_prefix_found": cut is not None, "authority": AMEND_AUTH.get(key),
                    "first_appended_line": data[cut:cut + 120].decode(errors="replace").strip().splitlines()[0] if cut is not None else None})
    return out


def pin(path: str, kind: str, revisions: dict) -> dict:
    import sys
    sys.path.insert(0, str(REPO))
    from grain_research.gptpro_foundation_v3 import revision_stage as rs
    p = rs.resolve_root(path)
    if not p.exists():
        return {"path": path, "kind": kind, "sha256": None, "pin_source": "MISSING_AT_REGISTRY_BUILD"}
    digest = rs.tree_sha(p) if kind == "tree" else rs.sha(p)
    owner = rs.owning_root({"revisions": revisions}, p)
    source = f"current_hash_at_registry_build_{datetime.now(timezone.utc).date()}"
    if owner is not None and kind == "file":
        recorded = {str(f): d for _s, f, d in rs.manifest_entries(rs.resolve_root(owner))}
        if recorded.get(str(p)) == digest:
            source = "runtime_manifest"
        elif rs.snapshot_path(owner).exists() and json.loads(rs.snapshot_path(owner).read_text())["files"].get(str(p.relative_to(rs.resolve_root(owner)))) == digest:
            source = "current_integrity_snapshot_" + json.loads(rs.snapshot_path(owner).read_text())["snapshot_date"]
    return {"path": path, "kind": kind, "sha256": digest, "pin_source": source}


# Packet 17 Part D: each consuming contract declares the integrity evidence it accepts (preflight checks live hashes).
_SNAP_REASON = ("current-bytes replay/verification contract; its roots rest on dated current integrity snapshots (typed LIMITED "
                "inputs): not historical run-time provenance, not preregistration, no scientific or protocol acceptance")
EVIDENCE_PERMISSION = {
    "r24_r1_analysis": {"evidence_levels_permitted": ["runtime_manifest", "current_snapshot_limited"], "evidence_level_reason": _SNAP_REASON},
    "r24_r2_cold_verify": {"evidence_levels_permitted": ["runtime_manifest", "current_snapshot_limited"], "evidence_level_reason": _SNAP_REASON},
    "r24_r1_canonical_replay": {"evidence_levels_permitted": ["runtime_manifest", "current_snapshot_limited"], "evidence_level_reason": _SNAP_REASON},
    "r10_episode_analysis": {"evidence_levels_permitted": ["runtime_manifest"]},
    "r10_one_episode_replay": {"evidence_levels_permitted": ["runtime_manifest"]},
    "f16_sealed_cold_subset": {"evidence_levels_permitted": ["runtime_manifest"]},
    "r08_gp01_collision": {"evidence_levels_permitted": ["runtime_manifest"]},
}
assert set(EVIDENCE_PERMISSION) == set(ADAPTERS), "every adapter must declare its permitted evidence levels"
for _k, _v in ADAPTERS.items():
    _v.update(EVIDENCE_PERMISSION[_k])


# Packet 18 Part D: typed record reconciliation (new root F00/record_reconciliation_r1; legacy payloads untouched)
_RR = "results/grain_gptpro_foundation_v3/F00/record_reconciliation_r1"
DOC_RECONCILIATION = {k: f"{_RR}/decision_documents/{k.replace('/', '__')}.md" for k in
                      ("F00/common_record_adapter_r1", "F13/recovery2_r1", "P05/contrast_ledger_r1", "F16/construction_briefs_r2")}
TEST_RECORD_RECONCILIATION = {k: f"{_RR}/test_records/{k.replace('/', '__')}.tests.txt" for k in ("P05/native_action_r1", "P05/c2_venue_dependence_r1")}


# Packet 19 Part C: per-requirement evidence roles (primary vs supporting); filled only after verification passes
REQUIREMENT_ROLES: dict = {"F05/invariants_r1": {"A02": "primary"}, "F04/information_full_r1": {"A02": "supporting"}, "F00/record_reconciliation_r1": {"A02": "provenance_record"}, "F00/r35_twelve_row_reconciliation_r1": {"A02": "provenance_record"}}


def build_adapter(spec: dict, auth: dict, revisions: dict) -> dict:
    inputs = []
    for item in spec["inputs"]:
        role, path, kind = item[:3]
        external = len(item) > 3 and item[3]
        inputs.append({"role": role, **pin(path, kind, revisions), **({"external": True} if external else {})})
    code = []
    for c in spec["code"]:
        if c == "F16_FILES_OPENED":
            opened = set()
            for n in F16_PICKS:
                m = json.loads((RES / f"F16/serialize_full_r1/cold/{n}.scores.jsonl.manifest.json").read_text())
                opened |= {str(Path(x).relative_to(REPO)) for x in m.get("files_opened", []) if str(x).startswith(str(REPO)) and str(x).endswith(".py")}
            code += [{"path": x, **{k: v for k, v in pin(x, "file", revisions).items() if k != "path"}} for x in sorted(opened)]
        else:
            code.append({"path": c, **{k: v for k, v in pin(c, "file", revisions).items() if k != "path"}})
    out = {k: v for k, v in spec.items() if k not in ("inputs", "code")}
    out.update({"inputs": inputs, "code": code, "authorization": str(AUTH["PACKET_13"]), "authorization_sha256": auth["PACKET_13"]["sha256"]})
    return out


def main() -> int:
    plan = {t["id"]: t for t in json.loads(PLAN.read_text())["tasks"]}
    card_reg = json.loads(CARD_REG.read_text())["cards"]
    ledger = list(csv.DictReader(LEDGER_CSV.open()))
    req_of = {}
    for r in ledger:
        for ref in set(REF_RE.findall(r["implementation_evidence"])):
            req_of.setdefault(ref, set()).add(r["requirement"])
    revisions, cards = {}, {}
    for cid, task in plan.items():
        legacy = rel_res(task["output_root"])
        revisions[legacy] = {"kind": "REJECTED_LEGACY", "curation": "PLAN_OUTPUT_ROOT", "cards": [cid], "requirements": task["requirements"],
                             "authority": "ACCEPTANCE_AUDIT_20260915", "status": "REJECTED_LEGACY_READABLE_HISTORY",
                             "evidence": {"decision": evidence_doc(RES / legacy)}, "scientific_readiness_eligible": False}
        current = []
        for a in card_reg.get(cid, {}).get("artifacts", []):
            if not a.startswith("results/grain_gptpro_foundation_v3/"):
                continue
            k = rel_res(a)
            if k != legacy and (RES / k).is_dir():
                current.append(k)
        for k, d in DECLARED.items():
            if cid in d.get("add_to_cards", []) and k not in current and (RES / k).is_dir():
                current.append(k)
        cards[cid] = {"legacy_root": legacy, "current_evidence": current, "depends_on": task["depends_on"],
                      "mapping_source": "configs/grain_v3_card_registry.json + declared packet additions"}
    candidates = set(req_of) | {k for c in cards.values() for k in c["current_evidence"]} | set(DECLARED)
    for d in DECLARED.values():
        candidates |= set(d.get("depends_on_revisions", [])) | {r["other"] for r in d.get("relations", [])}
    for k in sorted(candidates):
        root = RES / k
        if k in revisions or not root.is_dir():
            continue
        d = DECLARED.get(k, {})
        meta = common_meta(root)
        entry = {"kind": "CURRENT_REVISION", "curation": "DECLARED" if d else "AUTO_LEDGER_OR_CARD_REGISTRY",
                 "cards": sorted(c for c, v in cards.items() if k in v["current_evidence"]),
                 "requirements": sorted(set(d.get("requirements", [])) | req_of.get(k, set())),
                 "target": d.get("target"), "authority": d.get("authority"), "status": meta.get("status", "UNSTATED"),
                 "protocols": meta.get("protocols", []), "evidence": {"decision": d.get("evidence_doc") or DOC_RECONCILIATION.get(k) or evidence_doc(root), "common_record": meta["common_record"]},
                 **({"test_record_reconciliation": {"path": TEST_RECORD_RECONCILIATION[k], "sha256": sha(REPO / TEST_RECORD_RECONCILIATION[k]),
                                                    "recorded_at": (REPO / TEST_RECORD_RECONCILIATION[k]).read_text().splitlines()[2].split("=", 1)[1].split(" ")[0]}}
                    if k in TEST_RECORD_RECONCILIATION else {}),
                 "tests": d.get("tests", []), "depends_on_revisions": d.get("depends_on_revisions", []), "relations": d.get("relations", []),
                 "amendments": amendments(k, root), "scientific_readiness_eligible": True}
        if d.get("interval_contract"):
            entry["interval_contract"] = d["interval_contract"]
        if d.get("claims"):
            entry["claims"] = d["claims"]
        roles = {**REQUIREMENT_ROLES.get(k, {}), **d.get("requirement_roles", {})}
        if roles:
            entry["requirement_roles"] = roles
        revisions[k] = entry
    # legacy cards replaced for scientific use by their current evidence (several packets may jointly replace one root)
    for cid, c in cards.items():
        for k in c["current_evidence"]:
            if k in revisions:
                revisions[k].setdefault("relations", [])
                rel = {"type": "REPLACES_REJECTED_LEGACY", "other": c["legacy_root"], "scope": f"evidence for {cid} only; card acceptance per ledger/review card"}
                if rel not in revisions[k]["relations"]:
                    revisions[k]["relations"].append(rel)
    revisions["results/grain_measurement_discovery_v1/gp01/r1"] = {
        "kind": "LEGACY_EXTERNAL_EVIDENCE", "curation": "DECLARED", "cards": [], "requirements": ["R08"], "authority": "PACKET_12",
        "status": "LEGACY_HISTORY_HASH_PROVENANCE_ONLY", "evidence": {"decision": "results/grain_measurement_discovery_v1/gp01/r1/decision.md"},
        "scientific_readiness_eligible": False, "note": "audited by F02/collision_provenance_r1; hash match is provenance, not scientific truth"}
    auth = {k: {"path": str(p), "sha256": sha(p) if p.exists() else None} for k, p in AUTH.items()}
    adapters = {k: build_adapter(v, auth, revisions) for k, v in ADAPTERS.items()}
    reg = {"schema": "grain-v3-revision-registry-v1", "built_at": datetime.now(timezone.utc).isoformat(),
           "builder_sha256": sha(Path(__file__)), "plan_sha256": sha(PLAN), "card_registry_sha256": sha(CARD_REG), "ledger_sha256": sha(LEDGER_CSV),
           "authorities": auth, "revisions": revisions, "cards": cards, "adapters": adapters,
           "semantics": {"REJECTED_LEGACY": "readable history; never grants readiness",
                         "SUPERSEDES_SCIENTIFIC_USE": "dependents must use the superseding revision; two such claims on one root are ambiguous and fail closed",
                         "SUPERSEDES_INTERPRETATION": "numbers retained; interpretation corrected by the superseding revision",
                         "SUPPLEMENTS": "additional evidence; nothing superseded", "REPLACES_REJECTED_LEGACY": "evidence used instead of a rejected legacy card root; several may coexist"}}
    OUT.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
    kinds = {}
    for e in revisions.values():
        kinds[e["kind"] + "/" + e["curation"]] = kinds.get(e["kind"] + "/" + e["curation"], 0) + 1
    print(json.dumps({"revisions": len(revisions), "kinds": kinds, "cards": len(cards),
                      "cards_with_current_evidence": sum(bool(c["current_evidence"]) for c in cards.values()),
                      "amendments": {k: [(a["file"], a["append_only_prefix_found"], a["authority"]) for a in e.get("amendments", [])] for k, e in revisions.items() if e.get("amendments")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
