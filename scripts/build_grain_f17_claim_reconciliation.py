#!/usr/bin/env python
"""Packet 29: F17 claim-level reconciliation (docs/GRAIN_F17_CLAIM_RECONCILIATION_PROTOCOL_V1.md).
Metadata only. Maps every audit-r3 clause and every non-audited ledger row's current gap to current evidence with a typed
status; checks cited roots live through the registry. Never changes ledger statuses; this root is never evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_F17_CLAIM_RECONCILIATION_PROTOCOL_V1.md"
PROTOCOL_SHA = "1081b617ac37b49083d76049eb45de4cfbc76523c9a440307020d4b4a9f70399"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F17/claim_level_reconciliation_r1"
SELF = "F17/claim_level_reconciliation_r1"
PARENT = Path("/Volumes/mobile3/projects/project persona")
LEDGER = PARENT / "reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv"
CROSSWALK = PARENT / "reports/gptpro_seminar_20260915/atomic_v3/SOURCE_TASK_CROSSWALK.csv"
AUD = {"A": V3 / "F17/source_audit_r3/AUDIT_A_REPORT.md", "B": V3 / "F17/source_audit_r3/AUDIT_B_REPORT.md"}
STATUSES = ("RESOLVED_FINITE_WITH_AUTHORITY", "CANDIDATE_FINITE_PENDING_REVIEW", "EXECUTABLE_CPU_CACHE", "BLOCKED_MODEL_BUDGET",
            "MISSING_MATERIAL_OR_METADATA", "HUMAN_VALIDATION", "OPTIONAL_EXTENSION", "SUPERSEDED_HISTORICAL", "UNADJUDICATED")
RES, CAN, EXE, BLK, MAT, HUM, OPT, SUP, UNA = STATUSES
NEED_ROOT = {RES, CAN, SUP}
A18 = "SEMINAR_FABLE_18_DECISION (applied in packet 21; ledger SUPERVISOR_SOURCE_ACCEPTANCE)"
A10 = "SEMINAR_FABLE_10_DECISION (ledger SUPERVISOR_SOURCE_ACCEPTANCE)"
A2347 = "SUPERVISOR_CHECK_20260926_2347 (applied in packet 22)"
A0654 = "SUPERVISOR_CHECK_20260927_0654 (packet-28 finite work accepted; no R12 promotion)"
A15 = "SEMINAR_FABLE_15_DECISION + NEXT_PACKET_17 Part A (ledger SUPERVISOR_SOURCE_ACCEPTANCE)"
AAUD = "F17/source_audit_r3 (the auditor itself judged this adequate/disclosed)"
WFLAG = "wording check: read the cited decision/ledger text; if the overstatement is still present, append a dated correction (no number changes)"


def M(status, roots=(), pointer="", authority="", answer="", action="", resource="", reason="", part=""):
    return dict(status=status, roots=list(roots), pointer=pointer, authority=authority, answer=answer, action=action, resource=resource, reason=reason, part=part)


# ---------------------------------------------------------------- hand classification (audit-r3 clauses)
MAP = {
 "R01.u1": M(CAN, ["P05/native_action_r1", "P05/source_crossed_action_r1", "P05/source_affine_action_r1"], "decision.md answer sections", "numbers accepted (SUPERVISOR_0251 corrections; SUPERVISOR_0553 addendum); clause mapping not adjudicated", "finite NEGATIVE: no tested z->action estimator beats the zero reference on held source cells", "supervisor: adjudicate whether the finite negative answers 'M* as operation beating a baseline'", "none", "a current root answers it; acceptance of the mapping is not recorded"),
 "R01.u2": M(CAN, ["P05/c2_venue_dependence_r1"], "decision.md", "executed after seminar-06 go-ahead; clause acceptance not recorded", "MCD increment is venue-dependent (B-C -.024 [-.034,-.016]); dependence, not identified leakage", "supervisor adjudication of the clause mapping", "none", "executed after the audit"),
 "R01.u3": M(SUP, ["P05/native_action_r1", "P05/application_j_diagnostic_r1"], "native source J and application J acquired", "", "the J remedy is no longer blocked: source and application Jacobians exist at L12.H0", "none", "none", "explicit replacing roots"),
 "R01.u4": M(UNA, [], "", "", "", "identify 'C2 rival (b)' in the r2 audit text and check P05/c2_* roots", "doc", "rival (b) not identifiable from the r3 abridgement within the cap"),
 "R01.u5": M(EXE, [], "", "", "", "write one R01 mechanism synthesis unifying native_action, product-space, application-J, crossed and affine results (no new fit)", "CPU/doc", "no current root unifies the findings"),
 "R01.f1": M(RES, ["P05/contrast_ledger_r1"], "CONTRAST_LEDGER.md", AAUD, "hedging adequate per auditor", "none", "none", "audit judged adequate"),
 "R02.u1": M(UNA, [], "", "", "", "compare T1-T3 passport fields (F16/passports_r2) with F16/target_structure_inventory_r1; list fields still identical", "CPU/doc", "later inventory may address it; not verified within cap"),
 "R02.u2": M(HUM, [], "", "", "", "real B/X/A behaviour, criterion and informant data (H02)", "human participants", "source clause requires real behaviour/criteria"),
 "R02.f1": M(UNA, [], "", "", "", WFLAG + " (F16/passports_r2 'shared infrastructure')", "doc", "not checked within cap"),
 "R02.f2": M(UNA, [], "", "", "", WFLAG + " (GRAIN-readout passport module reuse)", "doc", "not checked within cap"),
 "R02.f3": M(UNA, [], "", "", "", WFLAG + " (selection NEGATIVE basis)", "doc", "not checked within cap"),
 "R02.f4": M(UNA, [], "", "", "", "state that N is the weakest GRAIN criterion trait where N results are cited", "doc", "not checked within cap"),
 "R02.f5": M(UNA, [], "", "", "", "state that the ladder is not yet shown to discriminate (all BELOW_L0)", "doc", "not checked within cap"),
 "R03.f1": M(UNA, [], "", "", "", "restamp PC27 as ABSTRACT_PAGE in the contribution ledger if still READ_OPEN_ACCESS", "doc", "not checked within cap"),
 "R03.f2": M(UNA, [], "", "", "", "count abstracts separately from primaries", "doc", "not checked within cap"),
 "R03.f3": M(UNA, [], "", "", "", "disclose summarizing-fetch access for each primary", "doc", "not checked within cap"),
 "R04.f1": M(UNA, [], "", "", "", WFLAG + " (two readers = one sensor)", "doc", "not checked within cap; substantive fix is R04.u2"),
 "R04.f2": M(UNA, [], "", "", "", WFLAG + " (proficiency scales reader noise)", "doc", "not checked within cap"),
 "R04.f3": M(UNA, [], "", "", "", "disclose grid resolution of true xi", "doc", "not checked within cap"),
 "R04.f4": M(UNA, [], "", "", "", "disclose MCAR opportunity assumption", "doc", "not checked within cap"),
 "R03.u1": M(MAT, [], "", "", "", "add per-item unread reasons to the contribution ledger; retrieve texts only if the user provides access", "primary texts", "primaries paywalled/403 or unread; material absent"),
 "R03.u2": M(MAT, [], "", "", "", "retrieve AERA Standards 2014 and Peters (PNAS Nexus) open texts when web retrieval is authorized", "primary texts (web retrieval)", "open material not yet retrieved; no retrieval in these packets"),
 "R03.u3": M(MAT, [], "", "", "", "retrieve full texts of Littman, ALBA, Geiger 2025 when authorized", "primary texts (web retrieval)", "material not retrieved"),
 "R04.u1": M(EXE, [], "", "", "", "add G/A/L generation-chain nodes in the declared F14 world and test identification", "CPU/declared world", "no later root"),
 "R04.u2": M(EXE, [], "", "", "", "implement genuinely query-specific readers (distinct sensors, not one sensor at two noise SDs) in the declared world", "CPU/declared world", "no later root"),
 "R04.u3": M(MAT, [], "", "", "", "natural exposure frames / edges need exposure logs (new data) and human validation", "new controlled data", "exposure logs do not exist in any cache"),
 "R05.u1": M(RES, ["F03/paired_four_component_identification_r1", "F00/supervisor2347_notes_r1"], "realization arm and latent-coordinate counterexample", A2347, "finite identification answer accepted", "none", "none", "row accepted with this root"),
 "R05.u2": M(HUM, [], "", "", "", "real task exclusion/loading validity", "human participants", "source clause requires a real test"),
 "R05.u3": M(MAT, [], "", "", "", "exposure data do not exist; new controlled data", "new controlled data", "material absent"),
 "R05.f1": M(RES, ["F03/paired_four_component_identification_r1", "F00/supervisor2347_notes_r1"], "null and latent-coordinate counterexample; scope notes", A2347, "construction dependence stated in accepted scope notes", "none", "none", "accepted scope notes"),
 "R05.f2": M(CAN, ["F03/paired_four_component_identification_r1"], "decision.md intervals", "", "paired design reports uncertainty", "confirm an SE/CI exists for the between-arm variance difference; else append", "doc", "not verified within cap"),
 "R05.f3": M(CAN, ["F00/supervisor2347_notes_r1"], "scope notes", "", "arm-removal assumption is a declared scope item", "confirm wording", "doc", "not verified within cap"),
 "R08.u1": M(UNA, [], "", "", "", "check F15/text_behaviour_quotient_r1 for a separated A_C vs B outcome table", "doc", "not verified within cap"),
 "R08.u2": M(CAN, ["F15/text_behaviour_quotient_r1"], "unknown-transform / conflict counterexamples", "", "non-trivial nuisance-group cases given as finite counterexamples", "supervisor adjudication of the mapping", "none", "later root addresses nuisance G cases"),
 "R08.u3": M(HUM, [], "", "", "", "real histories and validation (H01)", "human participants", "source clause requires real people"),
 "R08.f1": M(UNA, [], "", "", "", WFLAG + " (FU1-f)", "doc", "not checked within cap"),
 "R08.f2": M(UNA, [], "", "", "", WFLAG + " (FU1-f protocol deviation disclosure)", "doc", "not checked within cap"),
 "R09.u1a": M(RES, ["F04/reliability_spine_r1", "F04/generalizability_coefficients_r1", "F12/designed_wording_facet_r1"], "decision.md", A18, "ICC(alpha_u), reliability(s_ut), multidimensional B_u, PxCxfacet and held-out prediction covered in the simulated crossed designs", "none", "none", "row accepted", "ICC(alpha_u); reliability(s_ut); multi-dimensional B_u; PxCxfacet; multi-dimensional held-out prediction"),
 "R09.u1b": M(OPT, [], "PACKET20_SOURCE_SCOPE_RECONCILIATION.md R09 section; ledger R09 gap", "", "", "optional dense threshold grid", "CPU", "declared optional (not an original P02 p.23 clause) in packet-20 scope reconciliation", "threshold"),
 "R09.u2": M(HUM, [], "", "", "", "incremental validity over questionnaires (H01)", "human participants", "source clause requires real criteria"),
 "R09.f1": M(SUP, ["F04/reliability_spine_r1"], "crossed author x condition x wording x time new-draw design", A18, "replaced by a world with the missing facets", "none", "none", "explicit replacing root"),
 "R09.f2": M(SUP, ["F12/designed_wording_facet_r1"], "measured wording facet on cached reader outputs", A18, "replaced by a measured (not declared) facet", "none", "none", "explicit replacing root"),
 "R09.f3": M(SUP, ["F04/reliability_spine_r1", "F04/generalizability_coefficients_r1"], "G coefficients", A18, "informative design-specific reliability now reported", "none", "none", "explicit replacing root"),
 "R11.u1": M(EXE, [], "", "", "", "gamma sweep, H1-vs-H2 trigger and paired-CRN accuracy comparison in the declared reactivity world", "CPU/declared world", "no later root"),
 "R11.u2": M(HUM, [], "", "", "", "real reactivity/burden parameters (H04)", "human participants", "source clause requires real reactivity"),
 "R12.u1": M(HUM, [], "", "", "", "real person-level beta/gamma retest and external validity (H01)", "human participants", "source clause requires real people"),
 "R12.u2": M(RES, ["F04/reliability_information_r1"], "decision.md curve-shape table", A0654, "curvature with paired CIs delivered; one pointwise CI misses the analytic value (retained)", "none", "none", "accepted finite work"),
 "R12.f1": M(RES, ["F04/reliability_information_r1"], "fu2c_replay/", A0654, "exact W1/D1 replay recovers person arrays (1.1e-16)", "none", "none", "accepted finite work"),
 "R12.f2": M(RES, ["F04/reliability_information_r1"], "fu2c_replay/fu2c_uncertainty.csv", A0654, "mean-effect and retest CIs supplied", "none", "none", "accepted finite work"),
 "R13.u1": M(EXE, [], "", "", "", "estimate per-level PxC variance components separately for >2 toy constructs in the declared freedom world (estimation, not closed form)", "CPU/declared world", "no later root; theory gap is executable, not human"),
 "R13.u2": M(BLK, [], "", "", "", "five response-freedom levels need generated texts (generation budget) or human collection (H04)", "generation budget or human data", "material bank lacks the levels; API not authorized"),
 "R13.f1": M(EXE, [], "", "", "", "replace FU2-b closed form by estimation (same package as R13.u1)", "CPU/declared world", "fix requires estimation"),
 "R13.f2": M(EXE, [], "", "", "", "vary the assumed reader variance .1*o in the R13.u1 package", "CPU/declared world", "assumption untested"),
 "R13.f3": M(UNA, [], "", "", "", "append the declared-form caveat to the R13 ledger evidence (bookkeeping, next packet)", "doc", "wording edit outside this packet's authorized bookkeeping"),
 "R14.u1a": M(SUP, ["R14/counterfactual_reader_r1"], "decision.md", A10, "counterfactual insert executed at finite scope (2 readers x 40 units)", "none", "none", "explicit replacing root", "counterfactual insert"),
 "R14.u1b": M(BLK, [], "", "", "", "internal path and language/pragmatics edits need forwards/generation", "forward/generation budget", "not authorized", "internal path; language/pragmatics"),
 "R14.u2": M(UNA, [], "", "", "", "enumerate which evidence types are 'absent' and classify each (BLOCKED vs HUMAN)", "doc", "clause too generic to classify within cap"),
 "R14.f1": M(UNA, [], "", "", "", "record as a permanent provenance limitation (thresholds not pre-frozen); no retro-preregistration", "doc", "cannot be repaired retroactively; wording not checked"),
 "R14.f2": M(UNA, [], "", "", "", WFLAG + " (GRAIN M3=PASS vs occupation/stereotype BLOCKED)", "doc", "not checked within cap"),
 "R14.f3": M(UNA, [], "", "", "", WFLAG + " (E5 M3 vs FU1-c)", "doc", "not checked within cap"),
 "R14.f4": M(UNA, [], "", "", "", WFLAG + " (F14 language-edit label; comprehensiveness 1.00)", "doc", "not checked within cap"),
 "R14.f5": M(UNA, [], "", "", "", "state that the shared checker verifies paths/strings only", "doc", "not checked within cap"),
 "R15.u1": M(RES, ["F17/hedge_channel_utilization_r1", "F17/hedge_commitment_reader_r1", "F17/speaker_identifiability_r1"], "decision.md", A15, "availability/detection/attribution/utilization chain for a machine reader at finite scope", "none", "none", "row accepted with these roots"),
 "R15.u2": M(CAN, ["F17/speaker_identifiability_r1"], "decision.md", "", "politeness vs impression management identified within the declared model; other P02 p.13 rivals partly", "list which P02 p.13 hedge rivals remain unmodelled", "doc", "coverage of all rivals not verified"),
 "R15.u3": M(HUM, [], "", "", "", "real speakers (human stage)", "human participants", "source clause requires real people"),
 "R15.f1": M(UNA, [], "", "", "", WFLAG + " (15-point simplex vs 3 levels)", "doc", "not checked within cap"),
 "R15.f2": M(UNA, [], "", "", "", WFLAG + " (discrete grid vs continuous confounding)", "doc", "not checked within cap"),
 "R15.f3": M(SUP, ["F17/hedge_channel_utilization_r1"], "ledger R15 rows rewritten after packets 14-17", A15, "stale ledger text replaced", "none", "none", "ledger rewritten with later roots"),
 "R16.u1": M(BLK, [], "", "", "", "version robustness, alignment and LLM readers need forwards", "LLM readers", "not authorized"),
 "R16.u2": M(HUM, [], "", "", "", "human readers", "human participants", "source clause requires human readers"),
 "R16.u3": M(UNA, [], "", "", "", "check whether R14/same_object_channels_r1 source/pilot calibration defines the source promotion rule", "doc", "not verified within cap"),
 "R16.f1": M(UNA, [], "", "", "", WFLAG + " (ledger '+ residual')", "doc", "not checked within cap"),
 "R16.f2": M(UNA, [], "", "", "", WFLAG + " (interaction not separable from error)", "doc", "not checked within cap"),
 "R16.f3": M(UNA, [], "", "", "", WFLAG + " (two readers, no CI)", "doc", "not checked within cap"),
 "A07.u1": M(EXE, [], "", "", "", "per-turn trait+state prototype in the declared F06 world", "CPU/declared world", "no later root"),
 "A07.u2": M(HUM, [], "", "", "", "real ADO/CAT parameters (H04)", "human participants", "source requires real parameters"),
 "A07.f1": M(SUP, ["F06/reactive_head_to_head_r1", "F06/fixed_history_inference_r1"], "paired CRN head-to-head with CIs", "", "unpaired comparison replaced by paired CRN streams with CIs", "none", "none", "explicit replacing roots"),
 "A07.f2": M(SUP, ["F06/reactive_head_to_head_r1"], "paired contrasts", "", "small-gap reading replaced by paired contrasts", "none", "none", "explicit replacing root"),
 "A07.f3": M(UNA, [], "", "", "", WFLAG + " (error at stop vs fixed budget)", "doc", "not checked within cap"),
 "A08.u1": M(EXE, [], "", "", "", "sparse autoencoder on existing cached sentence embeddings (no LLM activations)", "CPU/cache", "audit lists EXECUTABLE; no root since"),
 "A08.u2": M(BLK, [], "", "", "", "vector/LLM RDM needs forwards", "forward budget", "not authorized"),
 "A08.u3": M(HUM, [], "", "", "", "behaviour RDM needs human behaviour data", "human participants", "source requires behaviour"),
 "A08.u4": M(EXE, [], "", "", "", "train a real concept-bottleneck model on cached features", "CPU/cache", "no root since"),
 "A08.f1": M(UNA, [], "current ledger A08 gap already reads '(not a trained concept-bottleneck model)'", "", "", "confirm the A08 decision document wording matches the corrected ledger text", "doc", "no root; ledger text alone is not replacing evidence"),
 "A08.f2": M(UNA, [], "", "", "", WFLAG + " (GRAIN readout outside frozen protocol)", "doc", "not checked within cap"),
 "A08.f3": M(UNA, [], "", "", "", WFLAG + " (pole R2 near-tautological)", "doc", "not checked within cap"),
 "A08.f4": M(UNA, [], "current ledger A08 gap text", "", "", "confirm the A08 gap lists SAE and RDM items as current", "doc", "no root"),
 "R18.u1a": M(EXE, [], "", "", "", "add time-pressure and public-visibility factors to the declared rival world", "CPU/declared world", "no later root", "time pressure/public factors"),
 "R18.u1b": M(EXE, [], "", "", "", "wording/topic stability of person mechanism parameters in the declared world", "CPU/declared world", "no later root", "wording/topic stability"),
 "R18.u1c": M(EXE, [], "", "", "", "Bayes-factor rival comparison (FU2-d used held-out cells, not BF)", "CPU/declared world", "no later root", "BF comparison"),
 "R18.u2": M(HUM, [], "", "", "", "real phenomena and trait data (H01/H02/H04)", "human participants", "source requires real people"),
 "R18.f1": M(UNA, [], "", "", "", WFLAG + " (D4 by construction)", "doc", "not checked within cap"),
 "R18.f2": M(UNA, [], "", "", "", WFLAG + " (construction not stated)", "doc", "not checked within cap"),
 "R18.f3": M(UNA, [], "", "", "", WFLAG + " (D1 fits generating family)", "doc", "not checked within cap"),
 "R21.u1": M(EXE, [], "", "", "", "random-matched construct baseline and significance for transported signatures (cached F12 outputs)", "CPU/cache", "no later root"),
 "R21.u2": M(CAN, ["F12/criterion_preservation_r1"], "decision.md", "", "criterion preservation under fitted g tested: inconclusive (n=50)", "supervisor adjudication; an inconclusive answer may be the finite answer", "none", "executed after the audit"),
 "R21.u3": M(BLK, [], "", "", "", "paraphrase/translation/platform swaps need generation", "generation budget", "not authorized"),
 "R21.u4": M(BLK, [], "", "", "", "LLM-state gauge needs forwards", "forward budget", "not authorized"),
 "R21.f1": M(UNA, [], "", "", "", WFLAG + " (pole flip by construction)", "doc", "not checked within cap"),
 "R21.f2": M(UNA, [], "", "", "", "state reuse count of F16 items in R21 claims", "doc", "not checked within cap"),
 "R23.u1": M(CAN, ["F12/criterion_preservation_r1"], "decision.md", "", "criterion after g tested: inconclusive (n=50)", "supervisor adjudication", "none", "executed after the audit"),
 "R23.u2": M(HUM, [], "", "", "", "cross-language person slopes (H03)", "human participants", "source requires bilingual persons"),
 "R23.u3": M(BLK, [], "", "", "", "multilingual encoders / LLM states need forwards", "forward budget", "not authorized"),
 "R23.f1": M(UNA, [], "", "", "", WFLAG + " (encoder deltas vs person slopes)", "doc", "not checked within cap"),
 "R23.f2": M(UNA, [], "", "", "", WFLAG + " (version transport vs slope preservation)", "doc", "not checked within cap"),
 "R23.f3": M(EXE, [], "", "", "", "random-map baseline for transport (same package as R21.u1)", "CPU/cache", "no later root"),
 "R25.u1": M(EXE, [], "", "", "", "held-out contexts beyond 1-D and new situation types in the declared world", "CPU/declared world", "no later root"),
 "R25.u2": M(HUM, [], "", "", "", "real invariance across natural paraphrase/situations", "human participants / natural material", "source requires natural data"),
 "R25.f1": M(SUP, ["F12/designed_wording_facet_r1"], "measured wording facet", "", "declared-input recovery replaced by a measured facet", "none", "none", "explicit replacing root"),
 "R25.f2": M(SUP, ["F12/designed_wording_facet_r1", "F04/reliability_spine_r1"], "measured facet / crossed design", "", "algebraic ICC(B) replaced", "none", "none", "explicit replacing roots"),
 "R26.u1": M(CAN, ["F06/reactive_head_to_head_r1", "F06/fixed_history_inference_r1"], "decision.md; PACKET20_SOURCE_SCOPE_RECONCILIATION R26 rows", "numbers corrected by SUPERVISOR_2143/2245; clause acceptance not recorded", "reactivity with paired CRN streams and CIs delivered", "supervisor adjudication", "none", "delivered after the audit"),
 "R26.u2": M(HUM, [], "", "", "", "external criterion and real recovery/burden (H04)", "human participants", "source requires real criteria"),
 "R26.f1": M(SUP, ["F06/reactive_head_to_head_r1"], "paired CRN", "", "unpaired CAT comparison replaced", "none", "none", "explicit replacing root"),
 "R26.f2": M(SUP, ["F06/reactive_head_to_head_r1"], "paired CRN", "", "replaced", "none", "none", "explicit replacing root"),
 "R26.f3": M(SUP, ["F06/reactive_head_to_head_r1"], "paired CRN", "", "replaced", "none", "none", "explicit replacing root"),
 "R26.f4": M(UNA, [], "", "", "", WFLAG + " (error at stop)", "doc", "not checked within cap"),
 "R26.f5": M(RES, ["F06/reactive_head_to_head_r1"], "upper bound disclosed", AAUD, "disclosed", "none", "none", "audit judged disclosed"),
 "R27.u1": M(EXE, [], "", "", "", "map of opportunity/choice equivalence conditions via calibrated mechanism pairs (declared world)", "CPU/declared world", "no later root"),
 "R27.u2a": M(MAT, ["F14/metadata_full_r2"], "metadata inventory only", "", "", "public-corpus text assessment needs reading public text (available corpora exist; no text read so far)", "public text read (authorization)", "material exists but has not been read; not human", "public corpora"),
 "R27.u2b": M(BLK, [], "", "", "", "response planting needs generation", "generation budget", "not authorized", "response planting"),
 "R27.u2c": M(MAT, [], "", "", "", "exposure logs absent", "new controlled data", "material absent", "exposure"),
 "R27.u2d": M(SUP, ["P05/native_action_r1"], "native J acquired", "", "J no longer blocked", "none", "none", "explicit replacing root", "J"),
 "R27.u3": M(CAN, ["P05/c2_venue_dependence_r1"], "decision.md", "", "C2 ablation executed", "supervisor adjudication", "none", "executed after the audit"),
 "R27.f1": M(UNA, [], "", "", "", WFLAG + " ('on real text')", "doc", "not checked within cap"),
 "R27.f2": M(UNA, [], "", "", "", WFLAG + " (Bernoulli-offer form)", "doc", "not checked within cap"),
 "R27.f3": M(UNA, [], "", "", "", WFLAG + " (one parametrization)", "doc", "not checked within cap"),
 "R29.u1": M(CAN, ["F04/reliability_spine_r1", "F17/r29_source_mapping_r1"], "new-draw replicates", "R29 row accepted (NEXT_PACKET_11 mapping); this clause's mapping not adjudicated", "replicate-based coverage available from new-draw spine", "confirm Monte Carlo coverage and facet-level precision curve are reported; else EXECUTABLE", "doc", "not verified within cap"),
 "R29.u2": M(HUM, [], "", "", "", "real facets", "human participants", "source requires real data"),
 "R29.f1": M(CAN, ["F04/reliability_spine_r1"], "replicates", "", "single-realization claim may be replaced", "confirm", "doc", "not verified"),
 "R29.f2": M(UNA, [], "", "", "", "record chi-square method adoption order as a provenance limitation", "doc", "cannot be undone"),
 "R29.f3": M(EXE, [], "", "", "", "Satterthwaite intervals for difference-of-MS components", "CPU", "no later root"),
 "R29.f4": M(UNA, [], "", "", "", WFLAG + " (intervals uninformative)", "doc", "not checked within cap"),
 "R31.u1": M(BLK, [], "", "", "", "multi-reader metric needs further readers' Jacobians (forwards); single-reader J exists (P05/native_action_r1)", "forward budget", "not authorized"),
 "R31.u2": M(EXE, [], "", "", "", "Sigma_N-selective modes on cached readouts", "CPU/cache", "no later root"),
 "R31.u3": M(UNA, [], "", "", "", "determine whether occupation/stereotype is blocked by labels (material) or readers (budget)", "doc", "reason not stated in abridged audit"),
 "R31.u4": M(EXE, [], "", "", "", "validity of residualized readout against logged labels", "CPU/cache", "no later root"),
 "R31.f1": M(UNA, [], "", "", "", WFLAG + " ('controlled')", "doc", "not checked within cap"),
 "R31.f2": M(UNA, [], "", "", "", WFLAG + " (removed 20% may be signal)", "doc", "not checked within cap"),
 "R33.u1": M(HUM, [], "", "", "", "comprehensibility needs human readers", "human participants", "source requires humans"),
 "R33.u2": M(EXE, [], "", "", "", "psychological elicitation bank for one declared target (design artefact)", "CPU/doc", "no later root"),
 "R33.u3": M(EXE, [], "", "", "", "freeze step for explanation units", "CPU/doc", "no later root"),
 "R33.u4": M(SUP, ["R14/counterfactual_reader_r1"], "decision.md", "", "counterfactual insert executed", "none", "none", "explicit replacing root"),
 "R34.u1a": M(RES, ["F16/source_half_reconstruction_r1", "F16/r34_finite_candidate_r1", "F16/error_budget_reconciliation_r1"], "decision.md", A18, "per-target structural reliability via source-half reconstruction (numeric route)", "none", "none", "row accepted", "numeric route"),
 "R34.u1b": M(MAT, ["F16/text_half_channel_sensitivity_r1"], "pilot only", "", "", "text-route structural reliability needs more calibration cores per stratum", "material (calibration cores)", "two cores per stratum make split dispersion degenerate", "text route"),
 "R34.u2": M(CAN, ["F16/target_structure_inventory_r1"], "per-target specification", "", "per-target structural objects specified", "confirm which P03 rows remain identical", "doc", "not verified within cap"),
 "R34.u3": M(HUM, [], "", "", "", "H06 external team rebuild", "human participants", "source requires external team"),
 "R34.f1": M(UNA, [], "", "", "", WFLAG + " (distinct negative facts)", "doc", "not checked within cap"),
 "R34.f2": M(UNA, [], "", "", "", WFLAG + " (selection NEGATIVE basis)", "doc", "not checked within cap"),
 "R34.f3": M(UNA, [], "", "", "", WFLAG + " (borrowed invariance entries)", "doc", "not checked within cap"),
 "R34.f4": M(UNA, [], "", "", "", WFLAG + " (T1 rival evidence world)", "doc", "not checked within cap"),
 "R35.u1a": M(CAN, ["F00/common_record_adapter_r2"], "fail-closed record contract", "", "record contract implemented (separate stores not verified)", "confirm separate raw stores", "doc", "partially addressed", "raw-record contract and separate stores"),
 "R35.u1b": M(EXE, [], "", "", "", "state decisive rival predictions for headline claims before tests", "CPU/doc", "no later root", "decisive rival predictions"),
 "R35.u1c": M(EXE, [], "", "", "", "multiverse for other headline claims (one exists)", "CPU/cache", "no later root", "multiverse for other headline claims"),
 "R35.u1d": M(EXE, [], "", "", "", "append the four unlogged hygiene touches and fix the FU1-c path (log append)", "doc", "bookkeeping not in this packet's authorized edits", "hygiene-log gaps"),
 "R35.f1": M(UNA, [], "", "", "", "fix typo '-.52..30' if still present", "doc", "not checked within cap"),
 "R35.f2": M(UNA, [], "", "", "", WFLAG + " ('.20 mid-range')", "doc", "not checked within cap"),
 "R36.u1a": M(EXE, [], "", "", "", "take one anonymous GRAIN phenomenon through rivals on existing data", "CPU/cache", "no later root", "anonymous GRAIN phenomenon"),
 "R36.u1b": M(EXE, [], "", "", "", "per-phenomenon structural reliability", "CPU/cache", "no later root", "per-phenomenon structural reliability"),
 "R36.u1c": M(EXE, [], "", "", "", "P2 gauge", "CPU/cache", "no later root", "P2 gauge"),
 "R36.u1d": M(EXE, [], "", "", "", "positive control on an old construct", "CPU/cache", "no later root", "old-construct positive control"),
 "R36.u2": M(BLK, [], "", "", "", "P1 gauge needs forwards", "forward budget", "not authorized"),
 "R36.u3": M(HUM, [], "", "", "", "interventions/prospective data (H02)", "human participants", "source requires new data"),
 "R36.f1": M(UNA, [], "", "", "", "add PASS criteria to the stage gate function or state it is a lookup", "doc", "not checked within cap"),
 "R36.f2": M(UNA, [], "", "", "", "replace hard-coded statuses with computed ones or disclose", "doc", "not checked within cap"),
 "H01.u1": M(EXE, [], "", "", "", "rerun parameter recovery under the final design (A/B/C arms, paraphrase, manipulation, missing days)", "CPU/declared world", "no later root"),
 "H01.u2": M(EXE, [], "", "", "", "add the realization-constrained arm to the H01 contract/recovery (now defined by F03/paired_four_component_identification_r1)", "CPU/doc", "contract not updated"),
 "H01.u3": M(MAT, [], "", "", "", "remove the stale '(if any)' column (metadata)", "metadata", "metadata correction"),
 "H01.f1": M(UNA, [], "", "", "", "label the 250-300 override against the frozen rule's 200", "doc", "not checked within cap"),
 "H01.f2": M(UNA, [], "", "", "", "same as H01.f1", "doc", "not checked within cap"),
 "H02.u1": M(CAN, ["F17/h02_recovery_r1", "F17/h02_recovery_r2"], "decision.md", "", "H02-specific identification design N 450 under a declared world", "confirm it replaces the borrowed 300", "doc", "later roots compute an H02 design"),
 "H02.u2": M(EXE, [], "", "", "", "write the coding manual (replace placeholder)", "doc", "no later root"),
 "H02.u3": M(EXE, [], "", "", "", "add interpersonal-cost factor for M3 (P02 p.24) to the contract", "doc", "no later root"),
 "H02.u4": M(EXE, [], "", "", "", "add realization-constrained arm", "doc", "no later root"),
 "H03.u1": M(EXE, [], "", "", "", "name P03 p.24 manipulation factors in the contract", "doc", "no later root"),
 "H03.u2": M(EXE, [], "", "", "", "put B into the distance estimand and recovery", "CPU/declared world", "no later root"),
 "H03.u3": M(EXE, [], "", "", "", "DIF benchmark beyond 6 pairs", "CPU/declared world", "no later root"),
 "H03.u4": M(EXE, [], "", "", "", "frozen rerun of the post-hoc recovery", "CPU/declared world", "no later root"),
 "H03.f1": M(EXE, [], "", "", "", "same as H03.u4", "CPU/declared world", "sizing provenance"),
 "H04.f1": M(UNA, [], "", "", "", "replace conventional d=.3 with the recovery rule or disclose", "doc", "not checked within cap"),
 "H04.f2": M(MAT, [], "", "", "", "remove stale 88 from planning_inputs (metadata)", "metadata", "metadata correction"),
 "H04.f3": M(UNA, [], "", "", "", "label G-study provenance as post hoc", "doc", "not checked within cap"),
}
# ---------------------------------------------------------------- non-audited rows: current gap clause(s)
ROWMAP = {
 "R06": M(MAT, ["F00/r35_twelve_row_reconciliation_r1"], "row accepted; residual", "", "", "persistent-excitation prediction on real text needs controlled contexts (do(C)); cite F04 rank certificate", "new controlled data", "public text lacks controlled condition variation"),
 "R07": M(EXE, [], "row PROPOSED_FINITE_CLOSED", "", "", "write the per-target practical identification summary from existing roots", "CPU/doc", "summary missing; no root"),
 "R10": M(EXE, ["F05/common_shock_episodes_r1"], "row accepted; residual", "", "", "persistent/clustered common shocks and crossed stable-author designs in the declared world; propagate source-refit uncertainty", "CPU/declared world", "not run"),
 "R17": M(BLK, [], "", "", "", "construct-changing edits under a construct contract need generation; LLM readers need forwards; edit-neutrality check HUMAN", "generation/forward budget + human check", "not authorized"),
 "R19": M(EXE, [], "row SCIENCE_RESOLVED; residual limitation", "", "", "declare rivals so out-of-family power is targeted", "CPU/doc", "residual stated limitation; row status unchanged"),
 "R20": M(BLK, [], "", "", "", "LLM readers, out-of-reader transfer, version robustness, >2-reader T_C need forwards; human readers H-stage", "LLM readers + human", "not authorized"),
 "R22": M(HUM, [], "row PROPOSED_FINITE_CLOSED", "", "", "H05 data contract", "human participants", "real group data"),
 "R24": M(EXE, ["F13/personal_law_r2"], "row accepted; residual", "", "", "hierarchical shared-offset personal law; MAR/MNAR selection model", "CPU/declared world", "not run"),
 "R28": M(HUM, [], "", "", "", "real group data (H05)", "human participants", "source requires groups"),
 "R30": M(EXE, [], "row PROPOSED_FINITE_CLOSED", "", "", "optimize the IB curve", "CPU", "stated not optimized"),
 "R32": M(HUM, ["F17/h02_recovery_r1"], "", "", "", "external criteria (H02); F17-02 claim levels", "human participants", "source requires criteria"),
 "A01": M(UNA, ["F15/bisimulation_metric_r1"], "", "", "", "split: estimated laws/real histories H01 vs nuisance-transform invariance clause (check executability)", "doc", "mixed; not split within cap"),
 "A02": M(HUM, ["F05/near_root_identification_r1", "F00/record_reconciliation_r1"], "row accepted via A02 mapping; this clause is the residual", "", "", "real-system identification", "human participants", "residual requires real systems; row status unchanged"),
 "A03": M(EXE, [], "row PROPOSED_FINITE_CLOSED", "", "", "learned iVAE", "CPU", "not executed"),
 "A04": M(EXE, [], "row PROPOSED_FINITE_CLOSED", "", "", "IB curve", "CPU", "not executed"),
 "A05a": M(SUP, ["P05/application_j_diagnostic_r1"], "application-side J acquired", "", "application J no longer absent", "none", "none", "explicit replacing root", "application-side J"),
 "A05b": M(BLK, [], "", "", "", "Sigma_P from psychological probes and multi-reader G_psych need forwards / probes", "forward budget", "not authorized", "Sigma_P; multi-reader G_psych"),
 "A06": M(MAT, [], "", "", "", "primary texts unavailable (R03)", "primary texts", "material absent"),
 "H05": M(HUM, [], "", "", "", "re-audit of the r3 contract against the sources (doc), then human stage", "doc + human participants", "human-stage preparation"),
 "H06": M(HUM, [], "", "", "", "re-audit of the r3 contract against the sources (doc), then human stage", "doc + human participants", "human-stage preparation"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_audits():
    rows = []
    for tag, path in AUD.items():
        for line in path.read_text().splitlines():
            if not line.startswith("| ") or line.startswith("| req") or line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            req, verdict, closed, unmet, flags = cells[:5]
            for kind, txt in (("u", unmet), ("f", flags)):
                if txt in ("-", "none", "none new", "none (honest)", "none of the r2 items"):
                    continue
                for i, p in enumerate([x.strip() for x in re.split(r";\s*", txt) if x.strip()], 1):
                    rows.append({"audit_id": f"{req}.{kind}{i}", "requirement": req, "clause_type": "audit_r3_unmet" if kind == "u" else "audit_r3_overstatement_flag",
                                 "verbatim": p, "audit_locator": f"F17/source_audit_r3/AUDIT_{tag}_REPORT.md row {req} ({'still unmet' if kind == 'u' else 'overstatement flags'} #{i})", "audit_verdict": verdict})
    return rows


def main():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"archived input changed: {rel}")
    if OUT.exists():
        raise FileExistsError("claim_level_reconciliation_r1 exists")
    OUT.mkdir(parents=True)
    ledger = list(csv.DictReader(LEDGER.open()))
    lrow = {r["requirement"]: r for r in ledger}
    cross = {r["requirement"]: r for r in csv.DictReader(CROSSWALK.open())}
    reg = rs.load_registry()
    inputs = {"ledger": sha(LEDGER), "crosswalk": sha(CROSSWALK), "registry": sha(ROOT / "configs/grain_v3_revision_registry.json"),
              "audit_A": sha(AUD["A"]), "audit_B": sha(AUD["B"]), "protocol": PROTOCOL_SHA}
    (OUT / "input_archive.json").write_text(json.dumps(inputs, indent=1) + "\n")
    audit = parse_audits()
    clauses = []
    for a in audit:
        kids = [k for k in MAP if k == a["audit_id"] or (k.startswith(a["audit_id"]) and k[len(a["audit_id"]):].isalpha() and len(k) == len(a["audit_id"]) + 1)]
        if not kids:
            raise KeyError(f"unclassified audit clause {a['audit_id']}")
        for k in kids:
            clauses.append({**a, "clause_id": k, **MAP[k]})
    audited = {a["requirement"] for a in audit}
    for req, r in lrow.items():
        if req in audited:
            continue
        gap = r["remaining_executable_gap"].split(" | F17-01 unmet:")[0]
        keys = [k for k in ROWMAP if k == req or (k.startswith(req) and k[len(req):].isalpha())]
        if not keys:
            raise KeyError(f"unclassified non-audited row {req}")
        for k in keys:
            clauses.append({"audit_id": "", "requirement": req, "clause_type": "current_ledger_gap", "verbatim": gap, "audit_locator": f"ledger {req} remaining_executable_gap (not audited in r3)",
                            "audit_verdict": "", "clause_id": f"{k}.g", **ROWMAP[k]})
    # ---- live pointer checks
    icache, checks = {}, {}
    for c in clauses:
        for root in c["roots"]:
            if root in checks:
                continue
            e = reg["revisions"].get(root)
            dec = (e or {}).get("evidence", {}).get("decision") if isinstance((e or {}).get("evidence"), dict) else None
            decp = ROOT / dec if dec else None
            checks[root] = {"registered": e is not None, "kind": (e or {}).get("kind"), "curation": (e or {}).get("curation"), "authority_field": (e or {}).get("authority"),
                            "decision_doc": dec, "decision_exists": bool(decp and decp.exists()), "decision_sha256": sha(decp) if decp and decp.exists() else None,
                            "evidence_level": rs.evidence_level(reg, root, icache=icache) if e is not None else "none", "self_reference": root == SELF}
    out = []
    for c in clauses:
        cr = [checks[r] for r in c["roots"]]
        ptr_ok = all(x["registered"] and x["decision_exists"] and x["kind"] == "CURRENT_REVISION" for x in cr) if cr else None
        req = c["requirement"]; lr = lrow[req]
        out.append({"clause_id": c["clause_id"], "requirement": req, "cards": cross.get(req, {}).get("atomic_tasks", ""), "sources": lr["sources"], "clause_type": c["clause_type"],
                    "origin": "ORIGINAL_SOURCE_AUDIT" if c["clause_type"] == "audit_r3_unmet" else ("WORDING_FLAG" if c["clause_type"].endswith("flag") else "CURRENT_GAP"),
                    "part": c["part"], "verbatim": c["verbatim"], "audit_locator": c["audit_locator"], "status": c["status"], "evidence_roots": ";".join(c["roots"]),
                    "evidence_pointer": c["pointer"], "authority": c["authority"],
                    "integrity_basis": ";".join(f"{r}:{checks[r]['evidence_level']}" for r in c["roots"]), "pointer_ok": ptr_ok, "scientific_answer": c["answer"],
                    "remaining_action": c["action"], "resource": c["resource"], "reason": c["reason"],
                    "row_science_resolution": lr["science_resolution"], "row_acceptance_decision": lr["acceptance_decision"]})
    with (OUT / "clause_reconciliation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    (OUT / "evidence_check.json").write_text(json.dumps(checks, indent=1) + "\n")
    # ---- coverage: 50 rows, 100 cards
    cov = []
    for req, r in lrow.items():
        cs = [o for o in out if o["requirement"] == req]
        cnt = Counter(o["status"] for o in cs)
        cov.append({"requirement": req, "title": r["title"], "scope": r["scope"], "clauses": len(cs), **{s: cnt.get(s, 0) for s in STATUSES},
                    "science_resolution": r["science_resolution"], "acceptance_decision": r["acceptance_decision"], "protocol": r["protocol"]})
    with (OUT / "coverage_50_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0])); w.writeheader(); w.writerows(cov)
    card_reg = sorted(reg.get("cards", {}).keys()) if isinstance(reg.get("cards"), dict) else sorted(reg.get("cards", []))
    xcards = defaultdict(list)
    for req, r in cross.items():
        for c in [x.strip() for x in r["atomic_tasks"].split(";") if x.strip()]:
            xcards[c].append(req)
    card_rows = [{"card": c, "requirements": ";".join(xcards.get(c, [])), "in_crosswalk": c in xcards} for c in card_reg]
    with (OUT / "coverage_100_cards.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["card", "requirements", "in_crosswalk"]); w.writeheader(); w.writerows(card_rows)
    # ---- fresh metadata-only legacy counts
    dirs = sorted(f"{p.parent.name}/{p.name}" for p in V3.glob("*/*") if p.is_dir() and not p.name.startswith("._"))
    unreg = [d for d in dirs if d not in reg["revisions"]]
    cited = sorted({r for o in out for r in o["evidence_roots"].split(";") if r})
    snaps = [p for p in (V3 / "F00/integrity_snapshots_r1").rglob("*.json") if not p.name.startswith("._")] if (V3 / "F00/integrity_snapshots_r1").exists() else []
    summary = {"protocol_sha256": PROTOCOL_SHA, "clauses": len(out), "audit_clauses_parsed": len(audit), "status_counts": dict(Counter(o["status"] for o in out)),
               "rows_covered": len(cov), "rows_in_ledger": len(ledger), "cards_in_registry": len(card_reg), "cards_in_crosswalk": sum(r["in_crosswalk"] for r in card_rows),
               "pointer_failures": [o["clause_id"] for o in out if o["pointer_ok"] is False], "roots_cited": len(cited), "roots_cited_unregistered": [r for r in cited if not checks[r]["registered"]],
               "legacy_metadata_count": {"result_dirs_depth2": len(dirs), "unregistered_depth2": len(unreg), "unregistered_cited_by_current_claims": [r for r in cited if r in unreg],
                                         "integrity_snapshot_json_files": len(snaps)},
               "ledger_status_counts": dict(Counter(r["science_resolution"] for r in ledger)), "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (OUT / "unregistered_dirs.txt").write_text("\n".join(unreg) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "legacy_metadata_count"}, indent=1)); print(json.dumps(summary["legacy_metadata_count"]["unregistered_depth2"]))


if __name__ == "__main__":
    main()
