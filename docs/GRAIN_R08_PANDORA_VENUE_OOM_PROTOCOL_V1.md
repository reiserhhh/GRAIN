# R08 natural venue-sequence observable-operator model (OOM): protocol V1

Frozen 2026-09-26 JST, before any OOM fit or score.
- **Authority.** `NEXT_PACKET_16_CHANNEL_UTILIZATION_AND_NATURAL_OOM.md` Part C; `SEMINAR_FABLE_14_DECISION.md`.
- **Runner and output.** `scripts/run_grain_r08_pandora_venue_oom.py`, writing to
  `results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/`.
- **Budget.** CPU ≤ 10 min, 1 thread, 0 neural calls. No raw text, labels, questionnaires, MEPS or rescan.

## Question

Is there usable longer-history predictive information in this natural venue-choice process, beyond static personal
composition and the last venue? The process is observational. Later events are not do(C), and venue dynamics are not
read as psychological change or intervention.

## Inputs and roles (identical to packet 15 Part C)

- **Pinned inputs.** The same three pinned inputs as `run_grain_r08_pandora_history.py` (sha 128f5b64…): the event
  addresses and the reference units and addresses. The same row-identity checks apply.
- **Author roles.**
  - 403 source-train authors (source authors whose sorted index position is not ≡ 0 mod 5);
  - 101 source-validation authors;
  - 120 held authors.
- **Dictionary.** The top 20 conditions by source-train frequency, plus OTHER: 21 symbols.
- **Windows.** Events 0–24 initialize the history, and events 25–35 are scored (11 per author).
- **Baselines reused unchanged.** Global marginal; static personal composition with α = 3; order-one transition; the
  original blend .2·order-one + .8·static. α and w were chosen on source validation in packet 15 and are **not
  re-tuned**.
- **Exact reproduction gate.** The held static/order-one/combined/global log losses must reproduce
  `F14/pandora_history_comparison_r1/held_event_scores.csv` (sha 7e086a0e…) within 1e-12. Otherwise stop.

## OOM estimator (source-train sequences only)

- **Model.** A spectral HMM/OOM with single-symbol histories and tests (Hsu, Kakade & Zhang 2012; Jaeger 2000).
  - **Statistics**, pooled over all within-author window positions of source-train authors (stationarity
    assumption):
    - P1 = unigram marginal;
    - P21[j, i] = P(x_{t+1} = j, x_t = i);
    - P3x1[x][k, i] = P(x_{t+2} = k, x_{t+1} = x, x_t = i).
  - **Parameters.** U = the top-k left singular vectors of P21. b1 = Uᵀ P1. b∞ = (P21ᵀ U)⁺ P1.
    B_x = Uᵀ P3x1[x] (Uᵀ P21)⁺.
  - **Support.** Histories and tests are all 21 symbols, fixed; there is no further support selection.
- **Assumptions and capacity** (stated, not tested here).
  - The pooled population process is a stationary finite-rank OOM.
  - Personal differences enter only through the filtered state. The start state is b1, the stationary marginal.
  - Capacities are unequal and are declared:
    - static uses the author's full prefix counts;
    - order-one uses the last symbol;
    - the blend uses both;
    - the OOM filters the whole prefix through population operators with a k-dimensional state.
- **Prequential filtering** for each author:
  - Start at b1 and update through events 0…t−1.
  - For event t: raw q_x = b∞ᵀ B_x b_t for all x.
  - After scoring, update b_{t+1} = B_{x_t} b_t / q_{x_t}.
  - There is no population refit on validation or held authors.

## Probability, support and refusal rules (frozen)

1. **Raw diagnostics** per event: negative mass share Σ max(0, −q_x) / Σ |q_x|; normalization deviation |Σ q_x − 1|.
2. **Clip and renormalize.** q⁺ = max(q, 0). If Σ q⁺ ≤ 1e-12, or any value is non-finite, the event is **REFUSED**.
   Otherwise p = q⁺ / Σ q⁺.
3. **Floor.** p = max(p, 1e-4), renormalized. The number of events where the observed symbol was floored is recorded.
4. **Reset.** If raw q_{x_t} ≤ 1e-12 or the update is non-finite, the state resets to b1 and the reset is recorded.
5. **Numerical gate G1**, on source-validation scored events: mean negative-mass share ≤ .05, refusal rate ≤ .01 and
   reset rate ≤ .01.

## Rank rule (at most 3 candidates; frozen)

- **Candidates.** k ∈ {5, 10, 20}.
- **Eligible:** candidates passing G1.
- **Chosen:** the eligible rank with the lowest mean validation log loss over events 25–35, computed with the rules
  above.
- **Stop rule.** If no candidate is eligible, the OOM is a declared estimator/support failure. Outputs are preserved
  and held scoring of the OOM is **not** run.
- There is no adaptive expansion and no bin, window or rank search.

## Held scoring and reports

- **Paired comparisons** are on identical scored (non-refused) events: OOM − static, OOM − order-one, OOM − blend.
  Report log loss and Brier, with a paired author bootstrap of 2,000 draws (seed 20260926, authors resampled).
- **Coverage** is the share of non-refused events.
- **Full-denominator score** (labelled): refused events are scored with the source-train global marginal. This
  fallback is frozen here.
- **Decompositions.** Outcome in top-20 vs OTHER (event counts, mean losses, each subgroup's contribution to the
  pooled delta), and raw target novelty (new vs known raw venue for the author). Event-time gap terciles are
  descriptive.
- **Descriptive reconstruction.** TV between the model-implied and empirical bigram laws, on source-train (in sample)
  and source-validation.
- **Saved for replay.**
  - raw per-event predictions and losses for all models, with the full predictive distributions;
  - OOM diagnostics;
  - the frozen fit: U, b1, b∞, B_x, singular values, dictionary, author roles.
- **Interpretation limits.** OTHER holds about 72% of held events; a gain there is not prediction of rare venues. No
  do(C), personality dynamics or clinical reading.
