# Whole-system literature seminar: round 2

Date: 2026-09-12. This is a bounded exchange, not permission for experiments.

Read-only, no participant records, labels, keys, lockboxes, code changes, installs or jobs.
Use at most 6 targeted literature searches/fetches; no new broad review. Respond in
900-1400 words. Distinguish verified primary sources from leads.

## Codex independent findings

- Semantic Measures (Kjell et al., 2019; 10.1037/met0000191) directly constructs
  language-based psychological measures. Existing plans mentioned it but the central
  register missed it. The Text package paper is 10.1037/met0000542.
- ECD (Mislevy, Almond & Lukas, 2003; 10.1002/j.2333-8504.2003.tb01908.x)
  already connects construct claims, evidence rules, and task construction.
- AI-GENIE (Russell-Lasalandra, Christensen & Golino, 2026;
  https://link.springer.com/article/10.3758/s13428-026-03082-1) integrates LLM item
  generation, embedding-based redundancy/dimension analysis and human validation
  in 5 US samples, N=4964. It develops questionnaire items, not unconstrained
  author evidence. Its claim is structural, not all-purpose validity.
- JAM (Tan et al., 2026; https://arxiv.org/html/2607.08374v1) is unusually close:
  cross-theory pseudo-facets, graph/attention-pooled representations, LLM judgment,
  Essays and Kaggle MBTI, prototypes. Sections III-A/B explicitly use target labels
  and human-guided cross-theory links. 'Theory-agnostic' is portability of learning
  architecture, not evidence of discovering label-independent psychological constructs.
  Codex is still reading its harmonization and evaluation sections.
- ACE (Ghorbani et al., 2019; NeurIPS) already performs multi-resolution segmentation,
  representation clustering, outlier pruning and concept importance in vision.
  This is a close mathematical analogue to our slice-cluster-interpret sequence,
  not a text/author measurement theory. Rare-person cues make pruning nontrivial.
- sJIVE (Palzer et al., 2022; 10.1016/j.csda.2022.107547), sections 2.1-2.6:
  X_i = U_i S_J + W_i S_i + E_i; y = theta_J S_J + sum theta_i S_i + e.
  Shared versus view-specific structure plus criterion-guided factor extraction
  is an existing estimator, with explicit row-space identifiability conditions.
  JIVE's 'individual' means view-specific, NOT individual human.
- DSEM (Asparouhov et al., 2018; 10.1080/10705511.2017.1406803) already estimates
  individual random dynamic coefficients, measurement error, within/between levels.
  Oravecz et al. (2011; 10.1037/a0024375) supplies hierarchical latent SDEs for
  affect regulation. We should not reinvent these. Token index is not automatically
  psychological time, and free text is not random EMA observation.
- Structural Topic Models (Roberts et al., 2014) explicitly separate prevalence
  and content covariates; correlated/nested topic models also precede choice hierarchies.
- Wright et al., 'Assessing personality using zero-shot generative AI scoring of
  brief open-ended text' (PMC12974486), already assesses more than self-report
  convergence, including daily behaviors and mental health. Indexed original methods
  read; exact bibliographic details being checked.
- Concept Bottleneck Models (Koh et al., 2020), concept leakage (Mahinpei et al.,
  2021), CB-LLMs (Sun et al., ICLR 2025) already provide an interpretable/distilled
  pathway. A named bottleneck can carry unnamed task information. Our value cannot
  merely be teacher-to-small-model distillation.
- Mori-Zwanzig and computational mechanics are precedents for residual memory and
  predictive states; their identities do not establish a psychological process.

## Points to resolve together

1. Your round-1 suggestion to REPLACE learned contexts with DIAMONDS is too strong:
   user specifically wants potentially new vector-based structures. Use known
   taxonomies as interpretable anchors/comparators, retaining unclassified response
   variation. Contact/construal distinction is important, but text-inferred context
   can contain both and must not be called pure construal by definition.
2. One additive-dominant ESM study cannot be a universal null across all traits,
   contexts, nonlinearities, observation timescales. Make additive versus interaction
   a useful competition, not a prior rule that suppresses discovery.
3. Contextualized-questionnaire validity .11 vs .24 is not a numeric pass threshold
   for MCD. Different criterion/data/metric scope. It is directional precedent only.
4. Existing contextualized measures, ECD and optimal/adaptive design already go
   beyond 'no one predicts good tasks'. What precise unsolved problem remains for
   learned, self-selected text evidence? Avoid a novelty claim based on search absence.
5. Choose 3 most direct new predecessors you found and verify their primary source.
   SIMPA 2510.02811, longitudinal NLP 2601.07988, and contact/construal 2026 are
   high priority. Which implementation can genuinely be adopted rather than repeated?
6. Agree on a field-level RESEARCH TARGET, not publication guarantee: a constructive
   theory joining opportunity/selection, response evidence, reader interpretation,
   and criterion-linked measurement, which predicts a non-obvious design improvement
   or failure beyond established composition. Specify 3-4 deliverables, the closest
   opposing methods, and what would make it merely an engineering integration.

Please return agreements, substantive disagreements, corrected source information,
and a short final research-target statement. Do not initiate another broad search.
