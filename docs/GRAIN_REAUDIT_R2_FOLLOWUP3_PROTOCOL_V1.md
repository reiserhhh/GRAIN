# GRAIN re-audit r2 follow-up packet 3 (FU3), protocol V1

Frozen 2026-09-26 JST before computation. Zero neural calls, no labels, no generation.
Seed 20260926. Held-split readout access (FU3-c) is logged before the run.

- **FU3-a (R21): intervention-signature preservation under a fitted map.**
  - Use the template-split OLS maps of F12/heldout_map_r1 (GRAIN to NLI, and NLI to
    GRAIN), fitted on one half of the templates.
  - On held-out templates, compute each intervention's effect in mapped space (mapped
    score after minus before). The interventions are the qualification flip and the pole
    flip.
  - Compare with the target reader's actual effect: per-pair correlation and ratio of
    mean effects, for the NLI qualification coordinate block.
- **FU3-b (R23): edit-signature preservation across encoders.**
  - Use the E5-to-BGE minimum-norm map of F12/encoder_gauge_r1, fitted on base and
    partner units of the other corpus.
  - Apply it to the R17 E5 edit deltas (contraction, reverse, punctuation) and compare
    with the actual BGE edit deltas: per-unit cosine and the ratio of norms.
  - Baseline: the same comparison with the partner (natural) delta.
- **FU3-c (R35): downstream analysis-choice multiverse for the R31 topic share.**
  - Choices: top-K subreddits K in {20, 30, 50}, principal components m in {3, 5, 10},
    and share vs log-share (log(share + 1/36)). That gives 18 universes, and all are
    reported.
  - Metric: the Extraversion LOO cross-validated R^2 of the historical semantic head's
    held-out predictions.
- **FU3-d (R33): comprehensibility proxy.** Mean words per sentence and mean syllables
  per word (vowel-group heuristic), with Flesch reading ease, for the F08 and F16 banks.
  The persona generations are also reported, as a reference only.
- **FU3-e (R04): extended combined world.**
  - Add opportunity O_t, where a condition is offered with probability .7 and missing
    otherwise.
  - Selection depends on theta, current state s_t and history H (the previous
    condition).
  - Add proficiency K_u scaling the reader noise, and query-specific readers (two
    readers with different sensor noise).
  - xi is estimated on 200 source persons by profile likelihood over a grid
    (-.1 to .35, step .05). E3 then uses the estimated xi.
  - Report RMSE for E1-E3 under nu1, and the state posterior SD at the final turn.
- **FU3-f (R05): P03 arms.**
  - Persons have a topic-selection preference tau, a response-style gain gamma, a
    semantic-interpretation bias delta (how they read the prompt) and a
    realization-verbosity rho.
  - Arms: A free choice (all four channels act), B forced topic (tau removed), C
    identical material (tau and delta removed; only style and realization act).
  - Estimate each channel's person variance as a difference of variances between arms,
    and compare with the truth. N = 500 per arm, disjoint persons.
