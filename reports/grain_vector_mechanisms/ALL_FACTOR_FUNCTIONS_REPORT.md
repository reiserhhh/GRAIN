# Nine-factor functional construction (M28)

2026-09-09. **Completed:** the same reference-function construction now covers
all five Big5 domains and all four MBTI axes. This advances beyond the old
T/F, neuroticism and confirmation-seeking examples. No human label fitted the
functions and no model weight was changed.

## Complete result, not only the strongest targets

Two original M7 authors define each function. The old T/F and N matrices/states
are reused; seven new functions use the same symmetric integrated P paths and
full-origin Q derivative. A separate 12-author reference defines profile norms;
12 other S1 authors supply the results below, with two views and full/E1/E2.
Each query has 24 held full levels and 48 held natural edit responses. Intervals
resample authors, not individual edited observations.

| Query | Level r | Natural-response r | Response r 95% CI | Response MSE / zero |
|---|---:|---:|---|---:|
| Extraversion | .898 | .945 | [.913,.969] | .094 |
| Neuroticism | .821 | .678 | [.524,.783] | .640 |
| Agreeableness | .991 | .976 | [.958,.987] | .136 |
| Conscientiousness | .953 | .902 | [.850,.937] | .198 |
| Openness | .937 | .873 | [.821,.956] | .242 |
| EI | .838 | .550 | [.449,.825] | .724 |
| SN | .922 | .800 | [.697,.895] | .445 |
| TF | .871 | .796 | [.686,.875] | .381 |
| JP | .776 | .884 | [.836,.924] | .326 |

These r values measure how the mathematical function follows the **frozen
model's judgment and actual evidence response**, not human personality accuracy.
M27 separately reports the human Big5 r/MBTI AUC. In particular, a readable JP
function can faithfully describe a native JP judgment whose source AUC is near
chance. This gives a concrete route for separating a model's weak trait use from
failure to mathematically recover what the model does.

## What is common and what differs

All nine retain token configuration information. On natural responses,
configuration-only r ranges from .530 (EI) to .949 (A). The token mean alone is
especially weak for EI (.022), JP (.145) and TF (.212). Averaging the query matrix
can remove behaviorally relevant model information even when hidden-vector energy
remains. That result now spans more than the initial two trait examples.

The final-token contribution is not uniformly dispensable. For C, P-only r=.802
becomes P+Q r=.902; for JP, .764 becomes .884. In contrast TF and N do not gain
from adding Q. Thus the common object is an **additive query-conditioned function**,
not a universal dominant coordinate or identical recipe for every trait.

```
P = 1*m' + C
f_q(P,Q) = b_q + <G_q,P-P_R> + g_q'(Q-Q_R)
          = b_q + mean_q + configuration_q + final_q
```

Seven new G matrices, g vectors and reference coordinates are stored alongside
the two existing functions. Neither token rows across different questions nor
their absolute norm magnitudes are automatically comparable psychological axes.

## Sharing and the next useful question

The expanded response atlas has 36 query pairs x five strata, plus 2,880 signed
M/C/Q/remainder covariance terms. Maximum covariance accounting error is
6.51e-19. Strong native response links such as E/A (.638) and E/C (.696) are
retained by the functions (.626/.658). This gives executable shared-route
accounting without forcing one shared scalar onto all queries.

Conceptual similarity does not automatically make two native queries equivalent.
E and EI have response r=.047 in this panel. Across the 12 held authors' full
two-view means, their native r=.124 and functional r=.237. These are small-panel
model relations, **not** the human Big5/MBTI bridge matrix. They motivate asking
whether differences arise from evidence use, trait framing, or the output alphabet
(Big5 digits versus MBTI letters). A future bounded all-four-axis digit/letter
comparison can resolve a new mechanism question; it must not replace the frozen
benchmark post hoc. Query identity includes concept, referent, wording and output
coding, not just a human trait name.

The nine-level held same-author profile AUC is .580 under fixed 12-author norms.
Do not call it individual personality validity or compare it as a new independent
replication of the earlier three-coordinate AUC; these are the same small S1
authors. E1/E2 are legacy common edits selected for three old queries, not
target-optimized interventions for every new axis. New targets' response SDs are
retained; JP responses are small (SD=.0108) even though their directions are read.

## Execution and reproducibility

1,008 new full forwards and 98 suffix gradient calls: total1,106 calls,
1,354.61 seconds. The 20-minute first invocation paused after945 saved units;
the second completed the remaining states and reference integrals without
recomputing completed units. No API, training, new human outcome, bridge or
lockbox was used. Existing two-query states add288 to the analysis, total1,296.

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
"$PY" scripts/run_grain_all_factor_functions.py prepare --output-dir results/grain_vector_mechanisms/m28_reproduction
"$PY" scripts/run_grain_all_factor_functions.py collect --output-dir results/grain_vector_mechanisms/m28_reproduction
"$PY" scripts/analyze_grain_all_factor_functions.py --state-dir results/grain_vector_mechanisms/m28_reproduction --output-dir results/grain_vector_mechanisms/m28_analysis_reproduction
```

Current states: `results/grain_vector_mechanisms/all_factor_functions_m28_v1/`.
Completed analysis: `all_factor_function_analysis_m28_v2/`, with
`nine_query_functions.npz`, `native_function_metrics.csv`,
`frozen_component_ablations.csv`, `cross_query_relations.csv`,
`signed_path_covariance.csv`, fixed norms and private author positions.
Analysis v1 stopped before scoring because of a relative-path handling bug; v2
fixes path resolution and reuses the same neural data. No result-based rerun.

M29 source association is prepared for277 existing source authors (213 Big5,
64 MBTI) with all three views and no human fitting. M30 user-only counseling
queries are the independently running narrow-trait external screen. The overall
V5 goal remains active: do not label these pending links completed.
