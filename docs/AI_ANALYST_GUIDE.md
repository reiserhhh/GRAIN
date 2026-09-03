# GRAIN AI Analyst Guide v1 (AI活用分析手引き)

Created: 2026-07-06. Purpose: GRAIN is designed to be operated primarily BY
AI agents over document volumes no human can read. This guide fixes the role
architecture, the verbatim prompts, and the guardrails so that any capable AI
agent (or team of agents) can run a GRAIN analysis that inherits the
project's audited discipline. Six adversarial audit rounds produced this
design; every rule below traces to a caught failure.

## 1. Role architecture — no role may verify its own output

| Role | What it does | What it must NEVER do | Implementation |
|------|--------------|------------------------|----------------|
| R-SCORE | Compute all scores | Be an LLM. Scores come only from frozen deterministic code (suica package + v2 lib) | scripts CLI |
| R-BUILDER | Write/modify analysis code, propose designs | Sign verdicts on its own analyses; write ledger statuses beyond `computed_pending_audit` | agent session A |
| R-CODER | Blind item coding (T1/T2 prompts below) | See key files (`*_KEY_*`); know the study hypotheses; talk to other coders | >= 3 model families, temp 0 |
| R-AUDITOR | Adversarial replication: reproduce every headline number with independent code, attack estimands, run nulls | Reuse the builder's library for the recomputation under attack | fresh agent session B |
| R-INTERPRETER | Turn audited tables into human-readable reports | Introduce numbers not in artifacts; upgrade evidence-tier language | prompt §3.3 |
| R-HUMAN | Criterion anchor: subsample coding, clinical sign-off, lockbox opening approval | Be replaced. Human coding is the only non-LLM validity evidence | researcher |

Builder/auditor separation is not optional: in this project's history the
auditor overturned or corrected builder headlines in 5 of 6 rounds
(self-estimation leak, temporal confound, feature-set inflation, estimator
blindness, estimand asymmetry). Assume the builder's next mistake exists and
has not been found yet.

## 2. Standing guardrails (machine-checkable; enforce before believing any number)

- G1 **No condition-mean subtraction.** All centering variants are falsified
  (P2, E1 cross-fit). Rind variance is controlled by design (Rulebook A).
- G2 **Cross-fit every reference model**: any population/condition/norm model
  must be estimated on data disjoint from the units it scores.
- G3 **Nulls are empirical and >= 200 draws**; within-person profile claims
  use the STRANGER null, never within-person permutation alone.
- G4 **Every split-half is tagged {parity | contiguous}**; state-precision
  claims use contiguous only (parity inflates ~2.5x).
- G5 **Variance components**: multi-start REML (best restricted loglik across
  optimizers); never trust a single optimizer or its converged flag; MoM is a
  screen only.
- G6 **Any replication/congruence criterion must be shown able to fail**
  (include a shuffled null world before using the criterion).
- G7 **Selection objectives are label-free.** External anchors appear only in
  frozen, preregistered, budgeted confirmations (T4).
- G8 **Leakage hygiene**: personality-vocabulary masks (incl. MBTI type
  codes) at slice and item level; criterion-coupled communities excluded
  from choice features.
- G9 **Evidence-tier language** (T1 exploratory / T2 same-data pre-committed /
  T3 held-out / T4 lockbox): a report may not use a stronger word than the
  ledger row supports. "validated/confirmed" requires T3+.
- G11 **Direction scores need MTMM, not stability**: in a person-stable
  embedding space, ANY fixed direction yields person-stable scores (E9
  scrambled-anchor control: stability 0.49-0.55 for random-word anchors).
  An anchor/probe/concept-vector score may be adopted only with convergent >
  discriminant evidence against an independent measurement method.
- G12 **Every comparison names its license (L0-L4)**: score comparisons are
  valid only within their evidence-priced radius (COMPARISON_LICENSES.md). Cross-regime LEVEL comparisons are forbidden (F4 + PRED-4: no
  covariate shortcut, no form-coordinate shortcut); cross-regime claims are
  limited to direction and within-population rank order. C-family constructs
  are never compared as style across themes — they are read as choice.
  Forbidden without license: "higher/lower than" across regimes; permitted:
  "same direction as", "ranked similarly within their own populations".
- G10 **Machine-legible vs human-legible constructs**: constructs humans
  cannot recognize (e.g., apostrophe-omission register: human 0/8, machine
  r_B 0.649) are legitimate but must be labeled `machine_legible`; their
  content validity rests on mechanism description + stability, not coder
  recognition. Content/topic constructs pass coding trivially; coding is NOT
  counted as extra validity for them.

## 3. Fixed prompts (verbatim; do not paraphrase in production runs)

### 3.1 T1 construct-recognition coding (chance = 1/7)

System:
```text
You are coding text excerpts for a psycholinguistics study. Choose which ONE
of the seven descriptions (A-G) best characterizes what is DISTINCTIVE about
the excerpt, then rate how salient that feature is: 0=absent, 1=weak,
2=clear, 3=dominant. Reply with ONLY a JSON object like
{"choice":"C","salience":2} and nothing else.
```
User: `EXCERPT:\n"""\n{text}\n"""\n\nDESCRIPTIONS:\n{A-G lines}`
Constraints: temperature 0; options include 4 real foils + 2 distractors;
key held out; strip `<think>` blocks before parsing.

### 3.2 T2 intensity coding (chance = 1/2)

System:
```text
You are coding text excerpts. Decide which text, A or B, shows MORE of the
named feature. Reply with ONLY a JSON object like {"higher":"A"} and nothing
else.
```

### 3.3 Interpretation report (R-INTERPRETER)

```text
You are writing a measurement report from the attached score tables and the
claims ledger. Rules:
1. Every number you mention must appear in an attached artifact; do not
   compute new statistics.
2. Use evidence-tier language: T1 rows -> "exploratory observation";
   T2 -> "pre-committed result on development data"; T3 -> "held-out
   confirmed"; T4 -> "confirmatory". Never upgrade.
3. For each score, state: value, its reliability from the registry, and the
   population scope ("Reddit MBTI-disclosing users" unless the deployment
   has its own norms).
4. Forbidden words unless the ledger row licenses them: validated, proven,
   diagnostic, personality type, clinical significance.
5. End with the standing limitations block (population, English-only,
   state-track feasibility bar).
```

### 3.4 Adversarial audit commission (R-AUDITOR)

```text
You are the independent adversarial auditor. The builder's claims are listed
below with their artifacts. Your job is to OVERTURN them if possible.
Required: (a) reproduce each headline number with your OWN implementation
(do not import the builder's analysis functions for the quantity under
attack); (b) attack the estimand (same units? same estimator both sides of
every comparison? gates equal?); (c) rerun with the proper null (stranger
null for profiles; >= 200 draws; cross-fit for reference models);
(d) check criterion falsifiability (would a shuffled world pass?);
(e) verdict table: UPHELD / OVERTURNED / UPHELD-WITH-CAVEATS per claim, with
the single most important correction. Genuinely try to break the result —
in this project the builder was wrong in 5 of 6 audited rounds.
```

### 3.5 Construct-interpretation naming (used once per new candidate)

```text
Here are the top-loading terms and 6 high-scoring example excerpts for a
newly discovered text-behavior dimension. Propose: (1) an OPERATIONAL name
(what the text does, not what the person is); (2) a one-sentence scoring
definition; (3) whether the dimension is style-register, content-interest,
or machine-legible-only; (4) plausible confounds (device, register,
platform). Do NOT use trait or clinical vocabulary.
```

## 4. Pipeline call sequence for a new corpus (AI-runnable)

```bash
# 0. tiers/splits (label values never read)
python scripts/build_suica_tiers_v2.py            # adapt membership rules
# 1. prep: slice + score (R-SCORE)
python scripts/prepare_suica_v2_phase2_slices.py --input-parquet <corpus>
# 2. trait track: disjoint-occasion stability (T2 evidence)
python scripts/run_suica_p1_disjoint_retest_v2.py --tag <tag>
# 3. choice track: classes + axes + HELD-OUT confirmation (T3 evidence)
python scripts/run_suica_op6a_choice_axes_holdout_v3.py
# 4. inventory expansion if needed (A-discover/B-confirm, T3)
python scripts/run_suica_op5_construct_discovery_v3.py
# 5. coding validation for new candidates (R-CODER panel + human subsample)
python scripts/build_suica_op22_item_bank_v2.py
python scripts/run_suica_op22_llm_coding_v2.py --provider <p> --task t1|t2
python scripts/evaluate_suica_op22_coding_v2.py
# 6. estimator sanity on this corpus's cell layout
python scripts/run_grain_p0b_thin_cell_regime_v3.py
# 7. audit round (R-AUDITOR, fresh session) -> ledger statuses
# 8. only then: frozen prereg + external anchors (T4, budgeted)
```

AI may: run all of the above, propose designs, draft prereg text.
AI may not: open a lockbox, change frozen scorer formulas, promote its own
ledger statuses, or substitute for the human coding subsample.

## 5. Mass-document / clinical workflow sketch (the deployment target)

1. **Ingest** per session/document; deterministic scoring only (R-SCORE).
2. **Trait dashboard**: style-base + choice-profile scores with reliability
   bands from the registry; updated as text accumulates (volume curve gives
   the "enough text" indicator per construct).
3. **State track**: per-session first-person-deviation (current best state
   channel) reported with its honest precision (contiguous SB ~ 0.27 at ~1k
   tokens — below the 0.30 feasibility bar; the dashboard must show this
   uncertainty until session volumes raise it).
4. **R-INTERPRETER** writes the per-client narrative under prompt 3.3.
5. **Human sign-off**: the clinician confirms/edits; disagreements are logged
   as calibration data. No AI-only conclusions reach the client record.
6. Quarterly: R-AUDITOR replication on a sample of dashboards.

## 6. Known AI-analyst failure modes (all occurred in this project)

1. Self-estimation leak (reference fitted on the units it scores) — round 5's
   overturn of E1.
2. Wrong null (within-person permutation for profile claims) — E4 overturn.
3. Vacuous criterion (congruence bars that cannot fail) — E7b overturn.
4. Optimizer trust (converged=True on a boundary collapse) — P0B round 6.
5. Same-family coder panels agreeing with themselves — v1 coding retired.
6. Estimand asymmetry (same-user vs stranger computed on different unit
   sets) — E6 correction (both directions!).
7. Stale-report contradiction (artifacts not updated after audit) — referee
   TOP-1.
Treat this list as a pre-flight checklist: each item is one grep/one control
run away from being caught early.
