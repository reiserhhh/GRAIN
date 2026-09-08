# All-factor inventory and source association atlas (M27)

2026-09-09. **Complete:** all nine PANDORA targets, 492 query/criterion cells,
27 paired author-bootstrap contrasts, and 22 recomputed Study 2 scale scores.
No new model fitting or inference in M27. These are new cross-target analyses
of frozen D1 predictions, not a second independent replication of D1.

## What the expanded matrix reveals

All five native Big5 queries have their largest *within-row* positive correlation
with the intended human domain on the same 213 source authors. For
conscientiousness the separation from extraversion is only .0017: this is an
observed pattern, not evidence that every contrast is significant. Native scores
are partly shared but not simply one interchangeable general impression.

| Native query | Human A | Human C | Human E | Human N | Human O |
|---|---:|---:|---:|---:|---:|
| A | .199 | .152 | .145 | -.075 | .092 |
| C | .102 | .168 | .166 | -.136 | .107 |
| E | .184 | .131 | .267 | -.117 | .177 |
| N | -.138 | -.087 | .019 | .224 | .072 |
| O | .064 | .069 | .097 | -.057 | .138 |

This supports expanding trait-specific operators rather than deriving every
query from the previously successful T/F or N function. Off-diagonal links are
retained for the shared/private path atlas; they are not erased as noise.

| Target | Native raw r / AUC | E5 supervised raw r / AUC | Routing question |
|---|---:|---:|---|
| Extraversion | .267 / - | .278 / - | Both routes have source information |
| Neuroticism | .224 / - | .042 / - | What does the query-conditioned observer use? |
| Agreeableness | .199 / - | .080 / - | Test its own configuration function |
| Conscientiousness | .168 / - | .043 / - | Separate specific use from general favorable impression |
| Openness | .138 / - | .244 / - | Does the current query underuse encoded information? |
| EI | .035 / .522 | .199 / .639 | Usable supervised encoder signal, weak native use |
| SN | .119 / .596 | .202 / .674 | Partial native use plus a larger encoder association |
| TF | .267 / .660 | .428 / .754 | Preserve as a useful mechanism anchor, not the whole theory |
| JP | -.003 / .500 | .188 / .608 | Informative availability/use contrast |

Big5 n=213; MBTI n=2,328. E5's reader uses source labels and the native observer
does not: the difference is a route comparison, not a controlled test of model
size or causation by reasoning. No Big5/MBTI population bridge is inferred from
these unmatched source groups. MBTI remains oriented E/N/F/J throughout.

Paired bootstrap (1,000 author draws): N native-minus-E5 delta r=.183,
95% CI [.014,.348]. Native-minus-E5 MBTI AUC differences are EI=-.117,
SN=-.078, TF=-.094, JP=-.107; all four intervals are below zero. This reinforces
the user's distinction between information being readable and a native query
actually using it. Whether *the same model state* contains the unused information
is answered by subsequent functional/source packets, not this encoder comparison.

Cue masking has a trait-specific pattern. E falls .056 versus raw and .079
versus placebo (latter CI [-.153,-.019]); N falls .048 versus raw but its
placebo contrast crosses zero. O does not decrease. Preserve both contrasts:
not all removed cue effects are target-specific. Full effects, p values and
BH q values are saved, including null results; the report is descriptive discovery.

## Counseling criteria are usable, with meaningful differences

The latest user request authorizes a separate local exploratory scale-linkage
branch. The 46 Study 2 text authors match all 46 pre/post scored participants.
All 22 scores have zero missing values and reproduce exactly from the actual
item/reversal definitions (maximum error 0). This does **not** mean 22 distinct
validated psychological factors: there are 17 logical target families across
22 occasions/composites, including exploratory splits and single indicators.

| Composite | Pre alpha | Post alpha | Use |
|---|---:|---:|---|
| Adapted AI mind attribution | .890 | .918 | General AI pre; this AI post |
| AI-directed attachment anxiety | .771 | .868 | Actual relationship pre; current wishes post |
| AI-directed attachment avoidance | .820 | .891 | Relational, not general clinical attachment |
| Positive AI trust | .690 | - | General pre-session attitude |
| AI distrust | .489 | - | Heterogeneous composite: keep all items/score, flag interpretation |
| Perceived AI empathy | - | .922 | Session appraisal |
| Session depth | - | .872 | Session appraisal |
| Session smoothness | - | .837 | Session appraisal |

Alpha describes within-sample item consistency, not validity. Agency/experience
splits, global ratings, privacy concern and the four ordered disclosure levels
are retained without pretending each is a validated interchangeable-item scale.
Pre/post mind attribution and attachment do not have identical referents/wording;
the query catalog must carry this change rather than treating differences as
pure within-person trait change. Native text/scale links have **not** run in M27.

## Theory-to-experiment link

[Funder's realistic accuracy model](https://pubmed.ncbi.nlm.nih.gov/7480467/)
separates relevant cue availability, detection and use. Our computational version
adds inspectable query-state functions between availability and judgment. The
empirical routing above instantiates that distinction; it does not make all
embedding directions personality dimensions.

[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html)
provides the finite-path calculus used for M28's G matrices. Averaging such
derivatives across fixed reference authors is our transferable-function
construction; its performance on new authors is an empirical question, not an
automatic corollary of path completeness. Additional psychometric/language and
mechanistic references are mapped to design choices in the V5 plan.

## Reproduction and next work

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
"$PY" scripts/analyze_grain_all_factor_inventory.py \
  --output-dir results/grain_vector_mechanisms/all_factor_inventory_m27_reproduction \
  --meps-root "/Volumes/mobile3/projects/project persona/data_sets/MEPS+AI_conv_experiment" \
  --allow-meps-scored-external --bootstrap 1000
```

Completed outputs: `results/grain_vector_mechanisms/all_factor_inventory_m27_v1/`.
`external_query_criterion_matrix.csv`, `predicted_query_matrix.csv`,
`true_criterion_matrix.csv`, `paired_author_contrasts.csv`, `meps_score_audit.csv`,
`meps_criterion_matrix.csv`, and `summary.json`. Private scores stay ignored.
The one-line executable shebang was added after this first run; the manifest
retains the actual pre-shebang code digest. Scoring/analysis behavior is unchanged.

M28 is the immediate neural construction: seven new query functions on the same
24-author panel, retaining the two old core functions. M30 separately freezes
phase/object-specific counseling prompts before joining text predictions to
these scales. Do not train on the 46 participants or rewrite queries against
their correlations. Full user-only coverage and condition-aware interpretation
are more valuable than another accuracy optimizer.
