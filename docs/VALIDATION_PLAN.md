# GRAIN Method Validation Plan v2

Created: 2026-07-04
Status: PROPOSED (from external review session 2026-07-04)

## 0. Purpose and framing

Goal of this plan: establish whether the GRAIN methodology is scientifically
sound, **before** any real scale development or field experiment.

"The methodology is correct" is not one provable statement. This plan
decomposes it into six falsifiable propositions (P0-P5). Each has a decisive
test, a pre-specified success criterion, and a defined interpretation on
failure. Passing P0-P4 is what "the method base is valid" means. P5 is
orientation, not the criterion.

Key framing decisions:

1. **Estimator correctness and signal existence are separated.** P0 proves the
   pipeline machinery on synthetic ground truth (no real data needed). P1-P4
   prove measurement properties on PANDORA. This is the honest meaning of
   "prove the methodology with only PANDORA available".
2. **Signal existence in text is not the bet.** Park et al. (2015), Eichstaedt
   et al., Pennebaker's function-word literature, and Schwartz open-vocabulary
   work already establish r ~ 0.2-0.4 text-personality links. The SUICA-specific
   bets are P2 (condition-fixed deviation is the right mechanism) and P4
   (base/react separation is real). Do not spend effort re-proving P1-level
   facts as if they were the contribution.
3. **All structure discovery and model selection in this plan is label-free.**
   No selection objective may contain Big5/MBTI terms (this repairs the 2026-07
   review finding F2: 35% anchor weight in `summarize_run`).

## 1. Falsifiable propositions

| ID | Claim (the u9061's thought, made testable) | Decisive test | Success criterion | On failure |
|----|---|---|---|---|
| P0 | The GRAIN estimator chain (slice -> rate -> condition-center -> base/react/residual -> variance shares) is computationally correct | Synthetic corpora with planted person/condition/interaction effects + null worlds | Recovered variance shares within +-20% relative of planted; planted-vs-estimated base rank r >= 0.9; false-positive rate <= 5% on null worlds | Fix estimator (or replace ad-hoc G approximation with mixed model) before touching real data |
| P1 | Author text carries stable person-level signal extractable by GRAIN scores | Disjoint-text, time-gapped test-retest of base scores on Tier-U users (early vs late comment halves, >= 3-month gap) | Spearman-Brown-corrected r >= 0.60 for >= 3 constructs, 95% CI reported; median across constructs >= 0.50 | Constructs are too noisy at current text volumes; report required-text-volume curve instead of claiming traits |
| P2a | Condition-fixed deviation (the MCD core idea) adds measurement value for **lexicon-rate constructs** | Paired comparison: base-score stability across disjoint condition sets, centered vs uncentered; pooled delta, cluster bootstrap | 95% CI of (centered - raw) stability delta > 0 | Centering is neutral for style-rate features (consistent with v2 G-theory 89/90 neutral cells); restrict the mechanism claim to P2b |
| P2b | Condition-fixed deviation adds value for **content-coupled features** (region-occupancy / MCD-style node scores) | Same paired design on node-occupancy scores | Same criterion | The core mechanism claim is not supported as operationalized; publish the negative result and revise theory |
| P3 | GRAIN scores are not redundant with existing lexica (LIWC/Empath category space) | Regress each frozen GRAIN base score on full Empath (+LIWC-like) category set, Tier-U | R^2 <= 0.75 for the majority of constructs | Reposition GRAIN as "condition-normalized administration layer over existing lexica" (still publishable, different claim) |
| P4 | base and react are separable: person x condition variance exists beyond sampling noise | Crossed random-effects decomposition (>= 5 real subreddit conditions per user); react-signature split-half across time halves within shared conditions | person x condition share CI excluding 0; react signature stability r >= 0.30 for >= 2 constructs | Drop the projective/CAPS branch from claims; GRAIN is base-only until richer designs exist |
| P5 | Frozen construct scores show preregistered directional external relations | <= 10 preregistered directional hypotheses, single lockbox opening, BH-FDR within the set | >= 50% of the set significant at q < 0.05, all significant effects in preregistered direction | External meaning unestablished; internal measurement claims (P0-P4) stand on their own |

## 2. Data tiers and contamination rules

| Tier | Users | Contents | Allowed use |
|------|-------|----------|-------------|
| U (development) | MBTI-axis users minus bridge minus Big5 (~8,650) | full comment streams, subreddit + timestamp metadata | unlimited exploration, structure discovery, tuning, P1/P2/P3/P4 |
| D (dev orientation) | same users' MBTI axis labels | EI/SN/TF/JP continuous | sanity orientation during development only; never a selection objective |
| L (lockbox) | Big5 1,401 + bridge 375/393; Essays corpus; any future corpus | labels + text | frozen-config confirmation only; **opening budget: 2 for the lifetime of this plan**; each opening requires a committed preregistration |
| X (auxiliary) | X-market authors (unlabeled) | posts | label-free replication of P1/P2 cross-platform, after fixing dedup |

Rules:

1. Freeze tier membership as CSVs under `data_sets/prepared/suica_tiers_v2/`
   in the first week; never edit afterward.
2. Big5/bridge users' **text** is excluded from all discovery and tuning.
3. Honest caveat to record in every report: Big5 users were touched by the
   historical 159-run anchor-weighted search, so lockbox confirmation is
   user-level-separated but not history-pure. Full purity requires a future
   third corpus; that remains the gold standard and is out of scope here.
4. Essays may be scored exactly once, at Phase 4, as an out-of-family
   transfer check. Essays has no conditions and no time axis; it can support
   only base-score transfer, not P2/P4.

## 3. Phases

### Phase 0 - Hygiene and governance (short)

- Apply the 2026-07-04 review fix list (Section 8).
- Create `docs/SUICA_CLAIMS_LEDGER.md`: one row per claim -> evidence file ->
  status (supported / directional / failed / retired). All future reports
  link into it. No number may appear in a ledger row that is not computed by
  a script from a CSV.
- Adopt builder/auditor separation (Section 5).
- Gate: fix list merged; ledger exists; tier CSVs frozen.

### Phase 1 - Estimator correctness on synthetic ground truth (P0)

Build `scripts/run_grain_synthetic_ground_truth_v2.py`:

1. Generate synthetic authors: word-category base tendencies b_u (planted
   person effects), condition effects c_c, interactions g_uc, slice noise.
   Text = bag-of-words draws from category-mixed vocabularies; pass through
   the **real** slicing/scoring/centering/aggregation code, not a mock.
2. Worlds: (a) full signal, (b) person-only, (c) condition-only, (d) pure
   noise, (e) label-shuffled versions of a-c.
3. Metrics: recovered vs planted variance shares; rank correlation of planted
   b_u vs estimated base; react recovery for planted g_uc; false-positive
   rates in null worlds.
4. Benchmark the current ordered-projection G approximation against a proper
   crossed random-effects fit (statsmodels MixedLM with vc_formula) inside the
   same harness. If the approximation is biased, switch estimators now.

Gate: P0 criteria met -> the pipeline is trustworthy machinery. This is also
the permanent regression harness against future vibe-coding drift: any later
code change must keep P0 green.

### Phase 2 - Label-free measurement structure on Tier U (P1, P2, P4)

Design upgrades applied here (all are review fixes):

- **Real conditions**: conditions = actual subreddits (top-k per user), keep
  users with >= 5 conditions x >= 3 slices per condition. The old n_conditions=2
  binarization is retired.
- **Leave-one-user-out population means** for condition centering.
- **Temporal gap**: early/late halves separated by >= 3 months of author
  silence buffer where possible.
- **React noise correction**: subtract expected sampling variance (bootstrap
  within condition) from observed across-condition variance before claiming
  react amplitude.
- Every stability number is tagged one of
  `{same-text-repartition | disjoint-text | cross-corpus}`; only the last two
  may be used in claims. Same-text-repartition numbers are pipeline
  determinism checks only.

Tasks:

- T2.1 (P1): disjoint-text, time-gapped base test-retest; report raw +
  Spearman-Brown-corrected r with CIs; also reliability-vs-slice-count curve
  (this doubles as the future administration manual parameter).
- T2.2 (P2a/P2b): condition-set split A/B per user (disjoint subreddit sets,
  matched sizes); base from A vs base from B; centered vs raw paired delta;
  run separately for lexicon-rate constructs and node-occupancy scores.
- T2.3 (P4): MixedLM crossed decomposition on slice scores; react signature
  stability (profile r over shared conditions across time halves) —
  Mischel-Shoda if-then signature test, properly.
- T2.4: multiverse slicing audit (96/128/semantic-shift): P1/P2 conclusions
  must be invariant to slicing policy, else slicing is a hidden researcher df.
- T2.5: X-market label-free replication of T2.1/T2.2 (after dedup fix).

Gate: P1 + at least one of P2a/P2b + P4 verdicts recorded in the ledger
(pass or fail — a clean fail also passes the gate; unresolved ambiguity does
not).

### Phase 3 - Construct set v2 and content validity (P3 + coding redesign)

- T3.1 Merge the tension family into one construct (review fix F3). Rebuild
  lexicons to be disjoint (no shared words across construct lexicons), or
  explicitly model overlap. Target inventory: ~5 constructs.
- T3.2 Re-derive/confirm constructs from Tier-U stable node structure with a
  **label-free** selection objective (reliability, cross-context stability,
  non-redundancy, method-factor penalty only).
- T3.3 Blind coding v2:
  - Items sampled across the score range (not only extremes) + topic-matched
    yoked controls.
  - Task = forced choice among ~5 construct labels + 2 distractor labels
    (defined chance level = 1/7) plus 0-3 salience rating; report accuracy vs
    chance, kappa, with CIs.
  - Coder panel: >= 3 model families (qwen3, mistral/persona-base, llama or
    gemma via Ollama; deepseek optional as 4th), **plus human coding by the
    researcher on a 120-item subsample** (human-LLM agreement reported).
  - Heldout items administered exactly once; failed constructs go back to
    development with a *new* item bank, never re-scored against existing
    ratings (retires the repair-rescoring pattern).
- T3.4 (P3): Empath/LIWC-style redundancy regression on frozen v2 scores.

Gate: construct set v2 frozen (formulas + lexicons + code hash committed);
coding verdicts in ledger.

### Phase 4 - Preregistered lockbox confirmation (P5)

- T4.1 Write `docs/SUICA_PREREG_LOCKBOX_OPENING_1.md`: exact frozen config
  hash, <= 10 directional hypotheses (e.g., tension-family -> Neuroticism +,
  Self-Positioning -> EI introversion +, Directive-Action -> TF thinking -,
  Novelty-Play -> Openness +), exact estimator, exact FDR procedure, success
  rule. Commit before any lockbox read.
- T4.2 Single run on lockbox (Big5 + bridge). Simultaneously: Essays one-shot
  base-score transfer; LIWC-baseline incremental ΔR^2 for each hypothesis.
- T4.3 Publish results to ledger regardless of outcome. Remaining lockbox
  budget: 1 opening, reserved for post-revision confirmation.

Expectation management: PANDORA official text baselines sit at mean r ~
0.25-0.32; single-construct directional r in the 0.10-0.25 range is the
realistic success zone. P5 failure does not invalidate P0-P4.

### Phase 5 - Method report and outward step

- `SUICA_METHODOLOGY_REPORT_V2.md`: claims ladder with pass/fail per
  proposition; explicitly lists retired claims (0.9x same-text stability
  headlines, tension-family independence, projective-hybrid wording until P4
  passes).
- If P0-P4 largely pass: this is a publishable methods contribution
  (Behavior Research Methods / Psychological Methods style), independent of
  P5. Preprint optional.
- Future work section: third corpus acquisition options; human-subject
  mini-study design (N~100-200, consented text + questionnaire) as the first
  "real experiment" the u9061 plans after method validation.

## 4. Formal measurement model (theory deliverable)

Write `docs/SUICA_MEASUREMENT_MODEL_V2.md` containing the explicit model:

```text
s_uci = mu + b_u + c_c + g_uc + e_uci
base(u)  = b_u          (estimated via condition-centered means, LOO population)
react(u) = structure of g_uc across c
choice(u)= P(c | u)     (informative, not missing-at-random — stated assumption)
residual = var(e) profile
```

with assumptions stated and numbered: A1 slices conditionally exchangeable
given (u, c); A2 population reference stable across the comparison window; A3
condition choice informative (therefore base is defined relative to the
observed condition support — this is a definition, not a bug). "The
methodology is theoretically correct" then reduces to: estimators for b_u,
g_uc, P(c|u) are consistent under A1-A3 (shown by P0 simulation) and the
components are non-degenerate in real text (shown by P1/P2/P4).

This document is what makes the 立体 (person x condition x slice) idea a
formal three-mode measurement claim — the legitimate statistical lineage is
Cattell's data box, G-theory, latent state-trait, and CAPS signatures, and the
doc should cite them as such.

## 5. Governance: AI-collaboration rules (anti-vibe-coding)

1. **Builder/auditor separation.** The agent (or session) that writes analysis
   code never writes its verdict into the claims ledger. A separate audit
   pass (fresh session, adversarial prompt, like the 2026-07-04 review) signs
   off each phase gate.
2. **Preregistration by commit.** Any confirmatory run: the plan + frozen
   config + success rule are committed first; the result references that
   commit hash.
3. **Mandatory negative controls.** Every stability/effect table ships with
   user-shuffle and condition-shuffle rows. A pipeline change that makes a
   negative control light up blocks the merge.
4. **No hardcoded numbers.** Report generators must compute every number from
   artifact CSVs (retires the "5/7 factors pass" literal string pattern).
5. **P0 as permanent CI.** The synthetic ground-truth harness runs after any
   scoring-code change; drift in recovered parameters fails the build.
6. **Lockbox budget is absolute** (2 openings). Log every read of Tier-L
   files.
7. **Language discipline in claims**: "projective", "validated", "invariant",
   "international" may appear in a claim only when the ledger row it links to
   is `supported`.

## 6. Stop rules

- If P0 fails and cannot be fixed in two iterations: halt all real-data work;
  the problem is machinery, not psychology.
- If P1 fails at all text volumes available: GRAIN on PANDORA is
  volume-limited; publish the reliability-vs-volume curve as the finding.
- If both P2a and P2b fail on the honest design: the condition-fixed-deviation
  mechanism claim is retired in its current form. This outcome is a real
  scientific result and gets written up, not retried until significant.
- Belief guard: the proposition list above is the falsification surface for
  "文書の中に人格・情緒のシグナルが必ずある". The belief is allowed to drive
  effort, never to veto a fail verdict.

## 7. Rough sequencing

- Phase 0: days.
- Phase 1: ~1 week (pure CPU, no labels).
- Phase 2: 2-3 weeks (heaviest; comment re-extraction with >= 5 conditions).
- Phase 3: 2-3 weeks (LLM coder panel via Ollama boxes; human subsample).
- Phase 4: 1 week.
- Phase 5: 1 week.

All CPU-feasible on the Mac node; LLM coding uses the existing Ollama
endpoints; no LoRA training is required anywhere in this plan.

## 8. Fix list from the 2026-07-04 review (checklist)

- [ ] Remove anchor terms from every selection objective (suica.py
      `summarize_run` anchor block never used for selection again).
- [ ] Retire same-text-repartition numbers from all claim surfaces; re-tag
      existing reports.
- [ ] Merge `suica_affective_tension_component` and
      `suica_tension_uncertainty_core`; report 5 constructs.
- [ ] Reconcile frozen config provenance (96/96/18 defaults vs 128/32/18
      search winner); document one canonical config.
- [ ] Add MBTI type codes + introvert/extrovert word forms to
      `PERSONALITY_LEAK_RE` in suica.py (parity with item-bank masking).
- [ ] Fix X scenario dedup inconsistency (dedup all three or none).
- [ ] Investigate "mean node split-half r: 0.0000" degenerate value in the
      true split-half report.
- [ ] Replace hardcoded summary strings with computed values.
- [ ] Rename construct labels to operational names until P5 passes
      (e.g., Self-Positioning -> First-Person Usage Tendency).
- [ ] Add condition-count >= 5 requirement to profile quality labels.
