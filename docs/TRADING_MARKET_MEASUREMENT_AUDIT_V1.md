# GRAIN Trading-Text Measurement Audit v1

Date: 2026-07-10

## Ruling

The trading archive can be analyzed with GRAIN, but it must be treated as an
observational measurement study before it is treated as a prediction feature.
This audit found one reproducible author coordinate and several forward market
associations worth preregistered replication. It did not find a market signal
that survived the full exploratory multiplicity gate.

## Correct measurement design

1. Preserve post boundaries and pack slices only within
   `author x symbol x query-group collector context x ISO week`.
2. Treat `symbol` as the primary rind. Query groups are collector rules, not
   participant-selected psychological conditions; use exact query-group matches
   only as a stricter sensitivity design.
3. Do not estimate C1 choice from this archive. Symbol and query exposure were
   collector-mediated, so observed frequencies are not clean participant choice.
4. Estimate C2 with disjoint-week author retest, comparing mixed-rind,
   matched-symbol, and matched-collector-context designs.
5. Estimate C3 within author and symbol as the current score minus the mean of
   at least two strictly prior occasions. Never use future text in the baseline.
6. Aggregate authors with equal weight within symbol-week. Do not treat posts or
   author-week rows as independent inferential units.
7. Test market information as weekly cross-sectional Spearman IC. Separate past
   placebo, contemporaneous state sensitivity, future level, and future change.
8. Keep GRAIN discoveries, collection-volume controls, and market-persistence
   controls in separate multiple-testing families.

## Data and implementation audit

- Source database: 1,120,024 X posts.
- Eligible English snapshot: 560,321 rows; 7,099 exact within-author duplicates
  removed; 553,222 analyzed posts.
- 403,813 boundary-preserving slices from 181,699 authors and 78 symbols.
- 4,096 query-group strings remained after filtering; 2,354 were singletons and
  58,322 rows contained combined query rules. This is why query-group strings
  cannot be treated as psychological rinds.
- Collector regimes changed on 2026-05-18 and 2026-06-22.
- The fast scorer was bit-identical to the current GRAIN v4 scorer on its audit
  sample (`max_abs_diff=0`).
- Mixed ISO timestamp precision, strict C3 history masking, feature-specific
  market gates, and incomplete final-week targets are covered by tests.

## Measurement findings

Only one of 33 construct-by-design cells passed the pre-specified descriptive
gate (`SB >= .60`):

- `first_person_usage_v2`, matched exact collector context:
  Pearson `r=.447`, 95% CI `[.399,.493]`, Spearman-Brown `.618`, `n=1,103`.
- The primary matched-symbol sensitivity was just below the gate:
  `r=.425`, Spearman-Brown `.597`, `n=1,439`.
- All other style, novelty, tension, affect, uncertainty, and threat coordinates
  had insufficient retest reliability in the available stable collection phase.

The defensible measurement claim is therefore narrow: context fixation improves
the repeatability of first-person usage in trading text. The current archive does
not support treating the full GRAIN battery as stable trading-personality scores.

## Market-information findings

No GRAIN test passed family-wise BH control. The strongest descriptive candidates
were:

- Historical author-composition conflict/threat -> next-week volatility:
  mean weekly IC `-.248`, 13 weeks, unadjusted `p=.015`, SUICA-family `q=.894`.
  Direction was similar in the old 18-symbol and expanded regimes, but its
  association with volatility *change* weakened to IC `-.133` (`p=.098`).
- Historical author-composition uncertainty -> next-week volatility change:
  IC `-.183`, 13 weeks, unadjusted `p=.0205`, SUICA-family `q=.894`.
  It was negative in both the 18-symbol and expanded regimes, but only one
  high-volume-regime week was available and that week reversed direction.
- Within-author negative affect state -> next-week volatility:
  IC `-.222`, but the effect was concentrated in the 18-symbol regime and did
  not transport to the expanded regime.
- Text-level certainty -> next-week return:
  IC `.126`, but the effect was concentrated in the 18-symbol regime.

Market persistence was materially stronger than GRAIN: current weekly volatility
predicted next-week volatility at IC `.429`. Therefore the descriptive GRAIN
associations cannot yet be called incremental forecasts.

## What the other implementation must not do

- Do not call one post one validated GRAIN slice without preserving author,
  context, and occasion boundaries.
- Do not treat collector query-group combinations as latent psychological rinds.
- Do not infer participant choice from collector-created symbol/query coverage.
- Do not compute state deviations from one prior observation or from a baseline
  containing the current/future occasion.
- Do not count author-week rows as independent evidence; the present forward
  sample is only 13-15 independent weeks.
- Do not merge these exploratory features into NS-2 or runtime trading code.

## Next locked test

Keep the post-2026-06-22 collector regime and symbol universe stable, then accrue
at least eight new complete weeks. Before opening those weeks, freeze two primary
hypotheses:

1. `composition__uncertainty_rate` negatively predicts next-week volatility
   change after the current-volatility baseline.
2. `composition__conflict_threat_rate` negatively predicts next-week volatility
   level, with incremental value tested against current volatility.

Use the new weeks once, with week-level inference and no parameter tuning. A
promotion requires consistent sign, `|mean IC| >= .10`, one-sided preregistered
`p < .05`, and incremental improvement over current market persistence. Until
then, the result is a measurement feasibility finding plus two market hypotheses,
not a market predictor.

## Reproduction assets

- Audit runner: `scripts/run_suica_market_measurement_first_audit_v1.py`
- Tests: `tests/test_market_measurement_first.py`
- Aggregate report: `results/suica_market_measurement_first_v1_fixed/REPORT.md`
- Reliability table: `results/suica_market_measurement_first_v1_fixed/measurement_reliability.csv`
- Market tests: `results/suica_market_measurement_first_v1_fixed/market_information_tests.csv`
- Collection-regime sensitivity:
  `results/suica_market_measurement_first_v1_fixed/market_regime_robustness.csv`
- Trading snapshot exporter:
  `trading-agent-codex/scripts/export_suica_measurement_first_snapshot.py`
