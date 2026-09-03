# GRAIN Theory Manuscript: Evidence Provenance Map

**Status:** internal reproducibility companion to the trilingual Version 1
theory manuscript<br>
**Scope:** this file locates empirical statements; it does not upgrade an
exploratory result into construct validity or independent replication.

## Corpus and eligibility counts

| Manuscript statement | Primary repository source | Audited value | Scope note |
|---|---|---:|---|
| PANDORA analysis census | `reports/SUICA_M4_X4_THREE_LEVELS_REPORT.md`, “Census and the blocking anchors” | 17,640,062 parseable rows; 10,296 authors | Counts produced by the frozen repository parser and eligibility rules, not substitutes for the corpus paper's published summary counts. |
| Big Five and source-disjoint cohorts | `reports/SUICA_M4_X4_THREE_LEVELS_REPORT.md`, same table | 1,401 and 8,895 authors | Cohorts are defined by the repository protocol. The source-disjoint cohort supports label-free structural checks, not external replication. |

## Table 1 claims

| Question in manuscript | Primary source and analysis location | Exact result used in manuscript | Evidential boundary |
|---|---|---|---|
| Effect of unconditional condition centering | `reports/suica_p2_condition_centering_v2_s96.md`, “P2a lexicon constructs”; sensitivity in `reports/suica_p2_condition_centering_v2_s128.md` | Pooled change in disjoint-occasion stability: -.0936, 95% CI [-.1033, -.0839] at s96; -.0911, 95% CI [-.1017, -.0803] at s128. All five listed style measures changed negatively in both runs. | This is evidence against unconditional removal for these operators and measures. It is not proof that conditioning is always harmful. |
| First sealed validity test | `reports/suica_lockbox_opening_1.md`, confirmatory table and success rule | 2 of 7 counted directional hypotheses passed; registered rule required at least 4 and no significant wrong-direction result; `SUCCESS RULE MET: False`. | A failed omnibus criterion does not erase the two narrow associations. It blocks the broad claim. |
| Reproducible author structure versus Big Five | `reports/SUICA_M4_X3_TRAIT_JOIN_REPORT.md`, registered multivariate rows, realized bands, and scope statement; synthesis in `docs/SUICA_JPA_FEED.md` | Registered expression/dynamic distance relations were not detected outside realized null bands; the strongest reported marginal cell, adjusted level with Agreeableness, was r=.1413, p=.0042 and did not pass the registered Bonferroni threshold .00250. | Stable technical structure is not construct validity. Repeated analyses of PANDORA are analytical recurrence, not independent replication. |
| Between- versus within-person relation | `reports/SUICA_M4_X4_THREE_LEVELS_REPORT.md`, outcome paragraph and primary-arm table | Between-person slope .069062, 95% CI [.048843, .090518]; pooled within-person slope -.007316, 95% CI [-.011278, -.003048]; difference .076377, paired cluster-bootstrap 95% CI [.055630, .097254], B=1,000. | The signs differed for this registered relation. Equation (9) is a pooled fixed-effects slope, not an unweighted average of person-specific slopes. No whole-process ergodicity claim follows. |
| Person-by-context response component | `reports/SUICA_M4_X1C_VENUE_RESPONSE_REPORT.md`, outcome paragraph and primary-arm table | Degrees-of-freedom-corrected component .0190, 95% CI [.0171, .0193]; split-sample reproducibility r=.2788, cluster-bootstrap 95% CI [.2598, .2934]; permutation band [-.0161, .0152]. | This is a positive technical component estimate with modest reproducibility at the tested horizon. It is neither an individual score nor a named psychological trait. |
| Selection geometry and Big Five geometry | `reports/SUICA_M4_SR1_SELECTION_GEOMETRY_REPORT.md`, “Decomposition (b)” and primary table | Mantel r=.04899, one-sided permutation p=.001 over 852,165 pairs. | Statistically resolved but small and corpus-specific. The companion X3 report found several expression/dynamic coordinates within realized null bands. |

## Provenance rules for revision

1. Numerical edits to the manuscript must be checked against the primary report
   named above rather than copied from a later narrative summary.
2. Confidence intervals, permutation bands, eligibility counts, and registered
   decision rules travel with the point estimate.
3. A repository report is an internal analysis source. Public claims about a
   corpus, theory, or prior method require the cited scholarly source.
4. “Detected technical structure,” “measurement comparability,” “construct
   validity,” and “clinical utility” remain separate claim levels.
5. A new corpus or new outcome link should be added as a new provenance row;
   it must not silently replace the analysis population used here.
