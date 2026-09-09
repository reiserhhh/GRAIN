# Logic NEXT01/02: shared operators on actual material changes

Frozen before computing the new correlations, 2026-09-10. Discovery experiment,
not a preregistered psychological validation. No questionnaire outcomes are read.

## Question and finite ladder

The previous native patch found a largely common signed response. This does not
disqualify a shared measurement axis: different author positions and travels on
one axis can encode individual differences. Test the travels, rather than require
different authors to have different gradients.

1. Score the existing 12-author full/E1/E2 materials using the frozen 12 GRAIN
   factors (whole/native/fixed-128-cross, four each). No factor refitting.
2. Fit source-only Ridge maps from those factors to the frozen query function,
   separately for T/F (63 authors) and neuroticism (213 authors). Preserve three
   fixed estimands: author means, author-centered views, and adjacent-view
   differences centered by transition index. Alpha=100, training-only scale,
   no hyperparameter search or test-fitted calibration. Grouped five-fold source
   diagnostics; final maps use the entire declared source. Score held-out full
   positions and full-to-E1/E2 changes. Source views are across comments, not
   controlled experimental situations or consecutive conversation turns.
3. Reuse the existing scalar F02-to-state direction as a T/F change control,
   converting its normalized input back to raw factor-score units.
4. Evaluate E1 changes against both the frozen query function f and the cached
   native conditional response F. Decompose, without discarding cross terms:

   Delta F = beta' Delta G + (Delta f - beta' Delta G) + (Delta F - Delta f).

   These are represented, GRAIN-unrepresented query-readable, and fixed-reader
   mismatch channels. Exact summation is accounting, not causal identification.
   The second channel may motivate a new observation, not be deleted as noise.
5. Use the already cached signed E5 query-pole relevance as a different,
   question-aware coordinate. Measure full-to-E1/E2 changes, cross-query controls,
   and the exact removal-mixture identity. It is an encoder-based mathematical
   observation, not a claim to extract LLM states without a model.

## Material, endpoint and selection boundaries

- Source: `factor_path_routes_v1/source_r2/private_factor_views.csv` joined to
  `dynamic_path_channels_v1/private_path_views.csv`; 276 authors, three views.
- Evaluation: 12 disjoint authors in `complete_level_response_v1`, two queries.
  Corresponding view-0 text and E1/E2 indices in
  `multiquery_sensitivity_s1_r1/private_material.json`. These authors have been
  used in earlier research; this is source-disjoint, not a new blind replication.
- E1 deletes the union of the most pole-relevant excerpts across three questions;
  E2 is disjoint, equally many, approximately length-matched, lower-relevance.
  This is a material-availability intervention, not personality change or a
  freely chosen author action. Check actual length matching; retain every author.
- The semantic coordinate helped select E1; its E1 association is not independent
  saliency validation. Retain E2 and mismatched-query observations in the report.
- Full/E1 P/Q states are available. E2 has native responses and semantic scores,
  but no matching full multi-position P/Q cache. Do not fabricate Delta f for E2.
- Verify material/packet hashes, query-token identities, frozen model identity,
  exact full/E1 rendered input tokens and source/evaluation author disjointness.
- No new neural inference, fine-tuning, external API, MEPS text or lockbox use.

## Mathematics and primary literature connection

For n excerpts, k removed, and per-excerpt coordinate s:

    mean_kept(s) - mean_full(s)
      = (k/n) * [mean_kept(s) - mean_removed(s)].

This separates availability quantity and contrast in available evidence; neither
factor is automatically an author's preference. Query-dependent s adds a
candidate content/evidence channel to query-independent GRAIN factors.

[Mischel & Shoda (1995)](https://pubmed.ncbi.nlm.nih.gov/7740090/) distinguishes
average behavior and predictable situation-dependent patterns. It motivates
separate estimands, not treating a deletion as a real psychological situation.
[Szabo et al. (2015)](https://proceedings.mlr.press/v38/szabo15.html) provides a
distribution-regression precedent for bags of observations and two sampling
levels. Here factor aggregation and a query-pole mean are finite coordinates;
neither is assumed to be a sufficient or characteristic distribution embedding.

## Outputs and decisions

Private: `results/grain_vector_mechanisms/logic_natural_transport_v1/` with source
fits, held scores/predictions, additive channels, semantic mixture coordinates,
hash manifest and status. Public: aggregate report, code, tests and goal update.
Pearson r, author bootstrap 95% interval (1,999 draws), signed agreement and
MSE/zero-change for commensurate predictions; uncalibrated semantic coordinates
receive correlations only. Report all fixed alternatives, no winning refit.

- If fixed GRAIN maps transfer, develop shared-operator/different-input scoring.
- If f transfers while G does not, study a query-readable observation beyond
  the historical GRAIN factor bank (NEXT02), not more GRAIN hyperparameters.
- If semantic mixture tracks responses, develop availability/contrast as a
  mechanism candidate (NEXT03); if not, locate the missing query context.
- Native vs f disagreement identifies the reader approximation, not automatically
  nonlinear personality dynamics. Person-conditioned operators are a hypothesis
  requiring repeated designs, not an inference from these two endpoint pairs.

Stop this packet after one fixed run plus bug fixes and tests. Further hypotheses
receive new cards; do not keep resampling the twelve people until success.
