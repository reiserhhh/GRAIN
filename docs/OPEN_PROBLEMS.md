# GRAIN Open Problems Register v1

Created: 2026-07-05. Status legend: BLOCKING = must close before journal
submission of the methods paper; ENHANCE = strengthens but does not block.
Each entry: problem, why it matters, concrete resolution design, current state.

| ID | Problem | Class | Resolution design | State |
|----|---------|-------|-------------------|-------|
| OP-1 | Trait/state decomposition asserted, not measured: no quantitative trait-state spectrum per construct | BLOCKING | E5 spectrum (round-4 audited, MixedLM-corrected) + E7 horizontal-cut discovery (round-5 audited) | **CLOSED**: spectrum measured; state factor space != trait space (E7d); 2 decaying monthly state modes (f1/f3); states live below month scale -> session-scale sampling (new OP-16) |
| OP-2 | react/C3 signature evidence: stranger-null-corrected, thick-cell design never run at scale | BLOCKING for any signature claim | E6 on deep extraction with stranger null | **CLOSED (signature_revived, audit-adjudicated)**: fair matched estimator passes for first_person (+0.167) and directive (+0.179); novelty/adversity increments were estimand artifacts; builder script needs the estimand fixes folded in (OP-17) |
| OP-3 | P5 external-validity lockbox status | BLOCKING | Preserve the v0.1 preregistration bytes; report every opening without rewriting the seal | **OPENING #1 SPENT 2026-07-07**: overall rule failed 2/7 (required >=4/7); H2/H6 passed individually; one opening remains; Essays confirm-half unopened |
| OP-4 | Pre-commitment not provable: theory docs/scripts/results untracked in git | BLOCKING (process) | Commit the validation stack; adopt commit-before-run for all future confirmatory work | Waiting on user authorization |
| OP-5 | Construct inventory thin (3-4 usable style constructs; 1 is first-person rate) | BLOCKING (a scale needs breadth) | A-discover/B-confirm open-vocabulary discovery (word clusters + regions), then blind coding v2 | **CLOSED (discovery stage; round-6 audited, zero overturned claims)**: 15/15 confirmed on unseen users (r_B 0.38-0.65); inventory 4 -> 19 provisional. Remaining sub-gate: blind coding v2 + naming (wcl_60 = apostrophe-omission register; wcl_02 flagged near choice-duplicate) |
| OP-6 | Choice axes: single corpus (Reddit), class map drift unknown, X micro-choice n=31/17 symbols | BLOCKING for C1 generality | (a) cohort-split held-out confirmation; (b) larger X corpus; (c) choice-module pilot | **(a) CLOSED via OP-6a (referee sweep)**: class map + selection refit on cohort A; 5/5 selected axes confirmed on cohort B (r_B 0.48-0.68, mean shrinkage 0.027) — winner's curse resolved, C1 now T3. (b)/(c) remain open |
| OP-7 | Dialogue-register transfer unvalidated (clinical target is dialogue; all evidence is posts/essays) | BLOCKING for clinical claims; ENHANCE for methods paper | Pilot: N~30-60, 2 sessions each, fixed-prompt AI dialogue (Rulebook G); rerun PRED-1/B3-style checks on subject turns; measure assistant echo rate | **PRECURSOR CLOSED (OP-7a, 2026-07-07)**: on the Reddit-internal register boundary (top-level vs reply turns), 19/19 constructs register-robust (n=1,976, time-separated 2x2; ledger OP-7a row). AI-dialogue pilot itself still needs data collection — OP-7 remains open for clinical claims |
| OP-8 | English-only scorer; Japanese clinical deployment impossible | BLOCKING for JP deployment; ENHANCE for methods paper | Build JP lexicon set with equivalence protocol (translate -> independent back-check -> within-language re-validation per Rulebook D1) | **STAGE 1/3 CLOSED (2026-07-07)**: JP scorer v1 (fugashi/MeCab, frozen formulas, 7 lexicons) with P0-style self-test PASS (recovery 0.914-0.952 @120 sentences; docs/SUICA_JP_SCORER_PROTOCOL_V1.md). Stage 2 (independent back-check) and stage 3 (within-language re-validation, needs JP corpus) remain open — JP scores machinery-grade only |
| OP-9 | Embedding-based baseline missing: reviewers will ask "why lexicons, not embeddings?" | BLOCKING (reviewer-proofing) | Same disjoint-occasion battery on bge-large-en-v1.5 (CUDA-encoded on the 4070 box; --precomputed path added to the script) | **CLOSED**: n=3,183. M1 identification parity — embedding AUC 0.891 vs SUICA-19 AUC 0.887 (gap +0.004, bar 0.10). M2: embedding top-PC retest 0.66-0.72 (max 0.722) vs GRAIN best ~0.65 — comparable. M3 subsumption NOT material: median CV R^2 0.684, max 0.896, majority < 0.80; reverse R^2 0.57-0.83 (symmetric overlap). Position: complementary — GRAIN is an interpretable, frozen, model-free layer at identification parity; embeddings are slightly stronger at the top but opaque and version-dependent |
| OP-10 | Formal estimation: MoM regime-limited; MixedLM needs convergence management | BLOCKING | P0-B thin-cell harness (scripts/run_grain_p0b_thin_cell_regime_v3.py) | **CLOSED (round-6)**: P0B pass with multi-start REML (best-llf across bfgs/lbfgs/cg): recovers planted 0.05 -> 0.041, 0.10 -> 0.096, false positives 0.002. Builder's interim "MixedLM detection floor ~0.05" claim RETRACTED — it was a boundary-collapse optimizer artifact (round-6 audit; one falsely converged fit). Standing rule: thin-cell decompositions use multi-start MixedLM; MoM = screening only; estimator-free detectors as robustness layer. Residual sub-item: within-cell autocorrelation world (rho~0.1) still to add to the harness |
| OP-15 | Same-text adjacency inflates ALL parity (even/odd) split-halves (~2.5x on E5 state precision); several historical numbers use parity splits | BLOCKING (reporting hygiene) | Re-report state precision and any same-occasion reliability with CONTIGUOUS halves; tag every split-half as {parity | contiguous} in the Rulebook H rule | Opened by round-4 audit |
| OP-11 | Tension construct: state-track design not yet specified (per-occasion precision unknown) | BLOCKING for the state/emotion claim | E5 state split-half precision (this session) + per-occasion token requirement curve | E5 covers first half |
| OP-12 | Generalizability of the 400-comment extraction cap (sampling artifact risk) | ENHANCE | Rerun key numbers on the deep extraction | **CLOSED** (P1-deep: verdict cap-invariant; recorded mover: adversity 0.030 -> 0.129 on deep) |
| OP-13 | Norm drift / cohort stability over calendar time (platform language change) | ENHANCE | Era term in E5 | **CLOSED (first estimate)**: era share ~ 0.000-0.002 on all five constructs — no material drift at this window |
| OP-16 | State measurement design: states live below the month timescale (E7a fail; f1/f3 monthly persistence small, disattenuated f3 lag-1 ~ 0.64) | BLOCKING for the state/emotion track | Session/day-scale occasion sampling (clinical per-session transcripts align, Rulebook G3); event-anchored prompts; re-run E7 at week granularity on deep data as an interim probe | Opened by E5/E7 |
| OP-17 | Fold round-5 estimand fixes into E6/E7 scripts (matched class sets + equal gates + two-sided bootstrap in E6 :88/:96/:105/:119; multi-draw nulls and non-vacuous replication criterion in E7 :156-164/:206-214/:229-230) | BLOCKING (code hygiene before any rerun) | Apply audit's matched-estimator design as the script default; add null world to any congruence criterion | Opened by round-5 audit |
| OP-14 | Ethics/consent framework for clinical dialogue data | BLOCKING for clinical study, out of scope for methods paper | Standard IRB-style protocol; not an engineering task | Flagged |

## Referee-sweep additions (2026-07-05, from the 3-persona panel review)

| ID | Problem | Class | Resolution design | State |
|----|---------|-------|-------------------|-------|
| OP-18 | Tokenizer/lexicon hygiene for v4 scorer: U+2019 smart apostrophes split tokens (prevalence measured: 3.8% of slices; ">"-quote contamination 0.98%); shared words across lexicons manufacture covariance (know/must/could) | BLOCKING for the frozen release scorer | Normalize U+2019 -> U+0027 before tokenization; disjointify lexicons; rerun P1/E7 headline as robustness table; version as scorer v3.1 with changelog | **CLOSED (2026-07-07)**: scorer v4 built (suica_v4_lib.py) — disjoint lexicons (20 logged moves), battery BIT-IDENTICAL to frozen v3 (chain not restarted), normalization default recommended; ledger SCORER-V4. Residual: E7-family re-run under v4 rates (OP-23) |
| OP-19 | Empath is not LIWC; 24k-char input cap biases P3 toward non-redundancy | ENHANCE (LIMITATION if no license) | LIWC-22 licensed run, uncapped input, same CV protocol | Needs license decision by user |
| OP-20 | Per-analysis eligibility funnels absent (every analysis conditions on heavy posters) | BLOCKING (reporting) | Funnel table: raw users -> gap-eligible -> condition-eligible -> analyzed, per experiment, from manifests | Open (compile at freeze time) |
| OP-21 | No demographics; measurement invariance (gender/age/device DIF) untested — wcl_60 and directive plausibly device/age-coupled | NEEDS-DATA (partially checkable if PANDORA demographic subset usable label-free) | DIF analysis on available demographic splits; else own as limitation | Open |
| OP-22 | Blind coding v2 for the 15 OP-5 candidates (>= 3 model families + researcher's 120-item human subsample; distractors; chance level) | BLOCKING before construct-level claims | 240 T1 items (7-way, chance 1/7) + 90 T2 pairs; 4-family panel (deepseek/qwen3/llama3.1/base-mistral) + human 120 | **CLOSED (computed_pending_audit)**: 8/15 pass pre-committed rules; legibility classification added to registry (human+machine: epistemic; content-tier: 7/7 pass but coding not counted as validity; machine-mostly: 5; machine-only: apostrophe-omission — human 0/8, exactly as the researcher introspected). Panel agreement 0.58-0.78; parse rates 0.93-1.00. T2 inversions (wcl_15 0.25, wcl_03 0.46) recorded as discriminant diagnostics |
| OP-23 | E7 candidate state dimensions partially lexicon-algebraic (shared words); no external state criterion | BLOCKING for state-construct claims | Recompute E7 with disjoint lexicons (with OP-18); state criterion only via OP-7 pilot | Instrument fix DONE (scorer v4 disjoint lexicons); E7 recomputation under v4 rates + state criterion (OP-7 pilot) remain |
| OP-24 | Bootstrap-of-median fragility on coarse profiles (E6 CI lower bound 0.007) | ENHANCE | Report means alongside medians; larger replication cohort | Open |
| OP-25 | Threshold provenance: 0.10/0.30/0.45/0.60/0.65/0.75 are project conventions, not standards | ENHANCE (reporting) | Label as conventions; anchor to Nunnally/COTAN where possible | Open (wording task) |
| OP-26 | P0B misspecification worlds missing: within-cell autocorrelation (rho~0.1), zero-inflated/heavy-tail rates | ENHANCE | Add worlds to P0B harness | Open |

Referee TOP-5 status: (1) stale-report contradictions -> **CLOSED** (audit
banners on 7 reports); (2) OP-9 -> running; (3) choice-axis holdout ->
**CLOSED** (OP-6a pass); (4) lockbox eligibility+power -> computed this
session, folded into prereg; (5) scope/consistency pass -> **CLOSED**
(C3 harmonized, Rulebook A1 provisional, evidence-tier taxonomy in ledger,
construct registry with operational renames).

## Publication skeleton (methods paper)

1. Formal model (rind theory v3.1; assumptions A1-A3) + estimator validation
   on synthetic ground truth (P0).
2. Falsification series: centering falsified 3 ways (P2, E1 cross-fit) — the
   negative result IS a contribution (it contradicts the default
   "topic-as-nuisance" practice in text-based assessment).
3. Positive results: choice channel (E3), rind-fixation principle (PRED-1,
   3/4 span-matched), volume-reliability curves, trait-state spectrum (E5).
4. Governance: builder/auditor adversarial replication as method (3 rounds,
   each catching real artifacts) — reportable as a reproducibility protocol.
5. Limitations: single-platform trait evidence, English-only, dialogue
   untested, external validity pending P5.

Minimum closure set for submission: OP-1, OP-3, OP-4, OP-5, OP-9, OP-10, OP-11.

## No-lock falsification program additions (2026-07-07; FALSIFIER_MATRIX_V1)

| ID | Problem | Class | Resolution design | State |
|----|---------|-------|-------------------|-------|
| OP-28 | LLM-agent world (channel-separation constructive test: "tense person" vs "person who chooses tense venues" with planted profiles) designed but not executed | ENHANCE (construct logic; never human-validity evidence) | 2x2 {persona} x {free/assigned venues} via DeepSeek/local models; run full pipeline; attribution accuracy | Designed in FALSIFIER_MATRIX; not run |
| OP-29 | T1 additivity never attacked (interactive/multiplicative misspecification worlds) | ENHANCE | W-F worlds: y = flesh * rind coupling etc.; F6.3 share-agreement alarm as the detector | Open |
| OP-30 | Exact population-composition-bias term in the centering theorem (W-PHASE strong-coupling gap -1.404 vs -1.14) | ENHANCE (math) | Derive E[bhat_r] under flesh-coupled choice; closes the F4 second-order amendment | Open |
| OP-31 | Prospective registration harness ("time's lockbox"): seal structural predictions about not-yet-written text (X pipeline windows); label-free, infinitely re-armable | ENHANCE (process infrastructure) | Freeze prediction JSON + hash before window; score after | **SEAL #1 placed (2026-07-07)**: docs/SUICA_OP31_PROSPECTIVE_SEAL.md, 4 structural predictions, scored on next X window |
| OP-32 | Individuality-as-residual track (fingerprint thesis): distinctive-profile stability, distinctiveness magnitude, register-normed variants, cross-register distinctive transport; v4 reporting layer ("position + configuration + residual") | ENHANCE (theory-completing; paper differentiator) | Furr normative/distinctive decomposition on the battery profile with stranger nulls (machinery = E6/E9b lessons); pilot DONE (ledger OP-32-P: median 0.505 vs null -0.018, T1); next: audit, per-register norms, distinctive-accuracy analog with anchors (dev tier) | Pilot done 2026-07-07; audit pending |
| OP-33 | Co-selection construct family (stylome-indicated): the ~32% of identifiable person signal outside the battery (OP-9 M3) plausibly lives in n-gram/collocation co-selection (Wright 2017; van Halteren 2005), not single-word rates | ENHANCE (v4 growth direction) | Discover character/word n-gram profile constructs under the OP-5 A/B discipline + flesh-purity gate; target: shrink the M3 unsubsumed share | **21/48 co-selection axes confirmed (2026-07-07, transported fit)**; next: purity gate + M3 shrinkage |
| OP-34 | v4.1 gate refinement: SB-corrected class-disjoint retest for family assignment (current gate attenuation-uncorrected -> conservative; borderline wcl_03/36/11) | ENHANCE | SB-correct both arms from within-arm split-halves; re-gate | Open |
| OP-35 | Cross-regime comparability has no statistical shortcut (PRED-4 FAIL: FORM structure does NOT transport better; covariate adjustment barred by F4 when type is self-selected) | BLOCKING for any cross-document-type score comparison | Within-person multi-regime collection (the SUICA-native 2-arm design, with OP-7): same authors in free + fixed + domain-locked regimes; measurement-invariance testing (configural/metric/scalar) as the formal frame; regime-relative norms for levels meanwhile | Opened by PRED-4 (2026-07-07) |
| OP-36 | SUICA-native corpus (the instrument's own data): 2 regimes x 2 sessions x 2 languages, within-person, closes N1-N4 (state timescale, dialogue, cross-regime comparability, JP stage 3) at once | BLOCKING for clinical/cross-regime claims; the specified next study | docs/SUICA_NATIVE_CORPUS_DESIGN_V1.md: N=120/language, AI interviewer, anchor battery + optional informant (Furr distinctive accuracy), native preregistration, measurement-invariance frame; needs IRB + collection | Designed 2026-07-07 |

## E9 follow-on (2026-07-06)

| ID | Problem | Class | Resolution design | State |
|----|---------|-------|-------------------|-------|
| OP-27 | Deep-representation channel: activation-space probes / concept vectors on a local open model (beyond sentence embeddings); attention weights rejected as primary evidence (contested validity), occlusion attribution adopted for report-level explanations instead | ENHANCE (E10 candidate) | Mistral-7B hidden states on the 4070 via transformers; linear probes for the 19 constructs; G11 MTMM licensing; occlusion-based token attribution for interpreter reports | Designed, not started |
