# VM05: complete four-axis strong sparse reference

2026-09-06. All12 source-only axis/arm fits and all60 saved fold replays are
complete. **This closes the sparse reference packet, not VM05 or the bridge.**
The strong head's discussion/reference controls have just started. Full neural
representations and identical-eligibility comparisons are still pending.

## Complete comparison

Same6,393 MBTI-only authors,36 comments each, official axis-specific outer5
folds, inner3-fold selection, word(1,2)/char(2,5) TF-IDF plus LogisticRegression.
No PCA, rebalancing, threshold fitting, bridge training or LLM weight training.
This is a fixed-budget source subset, not the complete original PANDORA benchmark
or the old364-person bridge experiment. No direct comparison of their different
cohorts/metrics is valid.

| Axis | Raw AUC | Cue-mask AUC | Placebo AUC | Raw BA at .5 | Raw predicted positive | True positive |
|---|---:|---:|---:|---:|---:|---:|
| EI | .672221 | .672686 | .673967 | .503630 | .003285 | .207727 |
| SN | .669877 | .669909 | .668310 | .500000 | 1.000000 | .890192 |
| TF | .767212 | .765381 | .767227 | .657990 | .215705 | .352417 |
| JP | .610766 | .604841 | .609463 | .556540 | .190990 | .421555 |

These are **label-ranking AUCs**, not classification accuracies or Big5
correlations. At the frozen threshold EI/SN remain almost/all majority decisions.
A nontrivial ranking does not by itself yield balanced classification or establish
probability calibration. This does not mean a ranking signal is absent, nor does
class imbalance alone prove miscalibration. A threshold or calibration experiment
would require a new source-inner protocol, not tuning on these OOF/bridge labels.

| Axis | Cue minus raw AUC | Conditional95% interval | Cue minus placebo AUC | Conditional95% interval |
|---|---:|---|---:|---|
| EI | +.000465 | [-.001543,+.002563] | -.001281 | [-.003000,+.000385] |
| SN | +.000033 | [-.001567,+.001536] | +.001599 | [-.000151,+.003256] |
| TF | -.001831 | [-.002377,-.001294] | -.001846 | [-.002428,-.001248] |
| JP | -.005926 | [-.010618,-.001723] | -.004623 | [-.009100,-.000503] |

Intervals use1,000 paired-author draws, seed731, conditional on fitted OOF heads;
they do not refit models or adjust a broad multiple-comparison family. JP placebo
also reduces AUC by.001303 [-.002470,-.000256]. Thus generic masking sensitivity
exists, and the cue/placebo comparison is necessary. All full contrast rows are
retained; no arm was selected as a final winner.

## Mechanism implications

1. There is substantial axis-dependent readable association before LLM reasoning:
   a sparse statistical model is a necessary comparator, especially for TF.
   No improvement by a new neural observer should be claimed against only a weak
   lexical projection. This does not establish a universal personality classifier.
2. Cue removal has a small selective effect on TF and JP in this representation,
   unlike the earlier weaker head. "Cue-insensitive" is not a property that can
   be transferred between readout models. The remaining ranking is not explained
   away by this particular cue list, but other source/context shortcuts remain.
3. Separate representation readability, threshold decisions, semantic code
   fidelity and actual model use. These results address the first two only;
   they do not rescue the failed code-invariant LLM disposition probes.
4. The next causal alternative is shared discussion/community and reference
   exposure. The frozen84-condition strong-source suite checks this with retain,
   purge and five nested exact-count deletion controls. Weak-head controls/nulls
   cannot certify these stronger heads. Those checks are observational controls,
   not a randomized manipulation of a psychological trait.

## Reproducibility and acceptance

- EI raw/cue/placebo parents: `vm05_sparse_mbti_r2/EI_cont_*`.
- SN/TF/JP parents: `vm05_sparse_mbti_r3/{SN,TF,JP}_cont_*`.
- Complete aggregate: `results/grain_vector_mechanisms/vm05_sparse_all_axes_summary_r1/`;
  metrics.csv, paired_contrasts.csv, comparison_manifest.json, summary.json and
  source snapshot. Identity:
  `ce9738f884515cff51cabcaf3a992bcc44e009246e86734b0fecd3320f656748`.
- Run `scripts/summarize_grain_sparse_mbti.py --runs <all12 explicit parents>
  --output-dir <fresh private directory>`. It rejects missing/duplicate arms and
  different source cohorts. Failed historical SN output is not a parent.
- Twelve `vm05_sparse_{ei,sn,tf,jp}_{raw,cue,placebo}_replay_r1` packets verify all
  saved outer heads against their held-out inputs. Maximum error1.11e-16, zero
  refits and no new label values in the replay. JP placebo final fit took
  1,644.09sec, sampled peak RSS7.167GB, below the unchanged8GiB bound.
- New strong-control protocol/code frozen at493e3d9; explicit aggregate extension
  a10842a. Related suite115 tests passed; final11 focused tests passed. Two
  historical SWIG deprecation warnings, no test failure; not whole-repo testing.

The full-source observer queue remains independently frozen on Windows aa51b06.
No bridge outcome values or MEPS questionnaires were read. Released GRAIN scoring
is unchanged. Do not promote a sparse, neural or blended source winner before
the remaining full-source comparisons, same-ready analysis and controls close.
