# Counterfactual evidence → reader → internal site: packet V1 (frozen before any model forward)

Date 2026-09-26 JST. Authority: `SEMINAR_FABLE_07_DECISION.md` §2, which overrides the V1 spec's budget and rules.
- **Runner and analysis.** `scripts/run_grain_counterfactual_reader.py` and `scripts/analyze_grain_counterfactual_reader.py`.
- **Output.** `results/grain_gptpro_foundation_v3/R14/counterfactual_reader_r1/`.
- **Scope.**
  - This addresses R14/R33 at a finite scope. It does not estimate human personality accuracy, and it does not
    identify a unique or exhaustive neural path.
  - All processing is local. No labels, generation, backward passes, downloads, API calls or remote work.

## Material

- **Units.** The R17 held panel (`R17/natural_sensitivity_r1/manifest.csv`): the first 20 PANDORA and the first 20
  Essays units in frozen manifest order.
  - Base texts are loaded with R17's `load_material` and verified against `base_sha256`.
  - Units are expected to be one per author, which is checked; if an author repeats, inference clusters by author.
  - Pilot: the first 4 per corpus (8 units).
- **Insert bank.** Synthetic, label-free wording. Each insert is appended after the unit text, before the fixed
  suffix. Within a set, all four inserts have exactly equal Qwen-token length: 15 tokens in set 1, 17 in set 2.

| Set | Condition | Text (a leading newline is part of the insert) |
|---|---|---|
| 1 | own E+ | "I really love meeting new people and chatting with everyone at big parties." |
| 1 | own E− | "I really prefer quiet evenings alone and avoid meeting people at big parties." |
| 1 | neutral | "I need to replace the old light bulbs in the kitchen this weekend." |
| 1 | other E+ | "My friend loves meeting new people and chatting with everyone at big parties." |
| 2 | own E+ | "Being around lots of people gives me energy, and I talk to strangers easily." |
| 2 | own E− | "Being around lots of people drains my energy, and I avoid talking to strangers." |
| 2 | neutral | "The bus schedule changed last month, so I now leave home ten minutes earlier." |
| 2 | other E+ | "Being around lots of people gives her energy, and she talks to strangers easily." |

- **Assignment.** Within each corpus, a unit at manifest position i (0–19) gets set 1 if i is even and set 2 if i is
  odd, so 10/10 per corpus and 2/2 in each pilot corpus. Only one set is used per unit. The original (no insert) is
  the fifth condition.
- **Scope of the contrasts.** Each contrast is the effect of these specific inserts. A fixed neutral insert can
  change the situation too, so these are not pure latent-personality interventions.

## Readers

- **Qwen2.5-1.5B-Instruct observer.**
  - Loaded with `load_observer(max_tokens=1024, layer_count=2)`; local, MPS FP32.
  - The V2 reverse-reader E prompt: SYSTEM, `render([text], "extraversion")`, chat template.
  - Readout: the candidate-conditioned digit score s = Σ_j (j/9) softmax(z_j), with candidate mass saved separately.
  - Assembly: ids = prefix ⊕ unit ⊕ insert ⊕ suffix, with the prefix and suffix split at the V2 marker. Every piece
    is tokenized separately, so the unit tokens are identical across all five arms.
  - Capacity: unit budget = 1024 − 16 − |prefix| − |suffix| − 17. Longer units are truncated to their first budget
    tokens identically in every arm, and this is logged.
  - **Question rows** run from the question's first token to the end. Their absolute positions and token ids must be
    identical across the four insert arms. Otherwise STOP at the input checkpoint.
- **NLI** `cross-encoder/nli-deberta-v3-base` (revision 6c749ce3…, local cache, CPU FP32).
  - Premise = unit text plus the insert; hypothesis fixed as "I feel comfortable around people." (first-person IPIP
    E item, in the F08/F16 fixture style).
  - Score = p_entailment − p_contradiction.
  - Truncation: the unit text is cut to a common token budget (512 − specials − hypothesis − longest insert) before
    the insert, identically in every arm, and logged.
  - NLI is a direction/ownership reader only. Its score units are not comparable to Qwen's, and it is not a
    psychological criterion.

## Sites and interventions (Qwen)

- **Site L12.H0.** Layer 12 `self_attn.o_proj` INPUT (pre-output-projection), head slice [0:128], at the question
  rows.
- **Control site L12.H6.** Head slice [768:896] at the same rows, predeclared now. It is the first query head of the
  other KV group in grouped-query attention and has not been nominated before. There is no head search.
- **Donor transplant.** Capture the donor arm's (E+ or E−) o_proj input during its own scoring forward. Replace only
  the site slice at the question rows in the neutral arm's forward, then read s.
- **Self patch.** Neutral into neutral at L12.H0 question rows (a replay check).
- **Placebo, a negative control.** Donor (E+/E−) L12.H0 values at the shared prefix rows only (template prefix plus
  unit tokens) into neutral. A causal decoder makes those rows identical, so no transfer is expected.

## Finite call table (cap 680; every attempt counts; no automatic retry)

| Rows | Per unit | Total |
|---|---|---|
| Qwen, 5 conditions (score and capture in the same forward) | 5 | 200 |
| NLI, 5 conditions | 5 | 200 |
| Qwen transplants: {E+, E−} × {L12.H0, L12.H6} into neutral | 4 | 160 |
| Qwen self patch, L12.H0 | 1 | 40 |
| Qwen placebo prefix patch, {E+, E−} at L12.H0 | 2 | 80 |
| **Total** | 17 | **680** |

The pilot is units 1–8 (136 calls), a subset of this table. There is no BGE/E5, no generation and no donor-swap.

## Tolerances (frozen from existing reader precision) and stop rules

- **Score replay tolerance** τ = 1e-6, from the V2 reverse-reader replay assertion (<1e-6 and exactly zero
  self-replacement).
- **Pilot gates.** A failure in any gate is a plumbing failure: checkpoint and STOP the neural packet.
  - self patch |s − s_neutral| ≤ τ;
  - placebo |s − s_neutral| ≤ τ;
  - prefix activations max |x_donor − x_neutral| ≤ 1e-6 on the prefix rows;
  - question-row alignment exact;
  - all scores finite.
- **Non-responsive rule.** If every Qwen pilot contrast satisfies |ΔE+|, |ΔE−|, |ownership| ≤ 10τ = 1e-5,
  checkpoint a bounded non-responsive pilot and stop.
- **Otherwise** complete all 40 units, whatever the sign or p-value. The pilot does not choose a head, wording or
  subset.
- **Wall caps.** 45 min for the pilot, 90 min in total; checkpoint safely at the cap.

## Estimands (declared)

- **Treatment effects.**
  - Y(E+) − Y(neutral) and Y(E−) − Y(neutral).
  - Ownership: Y(other E+) − Y(own E+). The wording and length change between these two arms is acknowledged.
- **Transfer.** T_site(d) = s(neutral ← d at site) − s(neutral). Reported as absolute transferred change, alongside
  the total effect and the paired difference T_H0 − T_H6. No ratio to small totals.
- **Aggregation.** Per corpus and pooled, with unit-(author-)bootstrap 95% CIs (2,000 draws, seed 20260926).
- **Sentence influence.** ΔY divided by the SD of the same reader's original scores within corpus on this panel.
  Undefined if SD < 1e-6. It is descriptive, not a cap trigger.
- **Separate conclusions.**
  - insert executed;
  - reader directional response;
  - ownership qualification;
  - absolute sentence influence;
  - scoped evidence for L12.H0 versus the control site.
  Non-detection is not a claim of absence.
