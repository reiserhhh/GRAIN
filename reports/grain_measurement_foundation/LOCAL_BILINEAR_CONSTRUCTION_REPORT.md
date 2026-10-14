# Same-Unit Bilinear Construction: Two-Context Results

2026-09-13. C10/C14/C17 continuation. **Fixed comparison completed; retain the
simple reference, do not promote raw bilinear augmentation.** The all-plan goal
remains active.

## 1. Decision and contribution

The study now separates three different operations on the same authors and text:
adding two mean profiles, multiplying their mean profiles, and pairing the two
representations at each original text unit before aggregation. Three
within-author shuffles preserve all unit marginals but break pairing.

The main practical result is negative for the proposed upgrade. PANDORA's
same-unit arm has mean Big5 r=.210481 versus additive .323039; paired
Delta=-.112558, author-bootstrap 95% CI[-.169732,-.052454]. Essays gives
.207522 versus .221185, Delta=-.013663[-.047575,.028180]. All predefined arms
and all five targets are retained. This closes this specific construction test,
not the broader possibility of a useful GRAIN/evidence interface.

The important mathematical clarification is constructive: same-unit pairing
adds a within-author cross-covariance operator to an outer product of marginal
means. The 30 semantic means are themselves fixed projections of E5 means.
These distinctions tell future construction work which operation it is adding;
more coordinates, more information and a better estimated measure are different
claims.

## 2. Methods and identical available material

Protocol: [initial plan](../../docs/GRAIN_LOCAL_BILINEAR_CONSTRUCTION_PLAN_V1.md).
The no-PCA reference was added in a
[separate addendum](../../docs/GRAIN_LOCAL_BILINEAR_PROJECTION_ADDENDUM.md) during
Essays encoding, before either domain had target-head model artifacts.

| Local context | Source training | Source validation | Final source refit | Held | Units |
|---|---:|---:|---:|---:|---:|
| PANDORA Big5-only | 378 | 126 (fold1) | 504 | 120 (fold0) | 22,464 original comments |
| Essays | 1,929 | 284 (fold8) | 2,213 | 254 (fold9) | 13,790 consecutive chunks |

PANDORA uses exactly the existing 624-author,36-comment material, excluding the
old GRAIN reference authors. Essays uses all2,467 original unique authors.
Its chunks contain up to128 whitespace tokens and partition the entire original
essay without losing characters. Four Essays authors have one chunk; pairing
and marginal outer products are identical for them. Chunks are not time points,
independent occasions or verified psychological events.

Frozen GRAIN input is the old externally discovered word-TF-IDF/SVD24 runtime,
aggregated as mean/std48 or used at unit level. This is the **unit-coordinate
interface**, not the entire GRAIN system or the G1 support-gated geometry profile.
The latter retains its separate32-comment protocol and124 supported authors;
it is not forced onto the other500 authors to fill this table.

Pinned E5-large-v2 supplies normalized1024-dimensional unit vectors. PANDORA
reuses original CUDA comment caches. Essays uses the same revision locally on
MPS, masked mean/L2, query prefix, float32. All Essays units fit within512 model
tokens: maximum240, truncated0. The historical PANDORA reader has its recorded
512-token cap; this is not a claim that both readers consume identical internal
tokens. Every arm has the same available original material in its own domain.

Five questionnaire outcomes are continuous0-1 in PANDORA and binary0/1 in
Essays. Labels train only the final local statistical heads, never the encoder,
GRAIN runtime or semantic keys. PCA, TF-IDF, scaler and alpha selection are
source-training/validation only; refits use the combined source. Held labels do
not choose any hyperparameter, feature family, rotation, shuffle seed or sign.
Existing exploratory exposure remains declared; this is not a new lockbox.

Ridge alpha is selected from1/10/100/1000 by mean source-validation Pearson.
All heads remain continuous and unclipped. TF-IDF uses word1-3 and char2-5 grams,
10,000 features per block. E5 PCA uses64 source-fitted components. Combining
E5-64, GRAIN48 and semantic30 produces142 additive coordinates; the four
interaction types add720, giving862. No new neural model training was performed.

## 3. Inherited mathematics, now an executable comparison

Local outer-product pooling is inherited from
[Lin, RoyChowdhury & Maji (2015), section2](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf).
Their application uses image locations and CNN features; ours uses text units,
a frozen lexical representation and keyed semantic contrasts. We did not
reproduce its image experiments, signed-square-root/L2 descriptor normalization,
end-to-end training or classification objective. The present identity-preserving
raw pooling plus source scaling is one adaptation, not the whole bilinear family.

For author u with T units, let g_t be GRAIN24 and s_t be semantic30. Define:

```text
B_u = (1/T) sum_t g_t s_t'                    same-unit product
M_u = mean(g_u) mean(s_u)'                    marginal product
C_u = (1/T) sum_t (g_t-gbar_u)(s_t-sbar_u)'   covariance, divisor T
B_u = M_u + C_u
E_pi[(1/T) sum_t g_t s_pi(t)' | g,s] = M_u
```

The last equality follows because each semantic unit has probability1/T of
occupying each position under a uniform permutation. It also holds for T=1.
All permutations of a small numerical case and all real covariance identities
were checked. The shuffle is a counterfactual pairing control, not an estimate
of a psychological-null population or a causal intervention on an author.

If D contains keyed positive-minus-negative IPIP facet directions and h_t are
normalized E5 vectors, then:

```text
s_t = h_t D
sbar_u = hbar_u D
C_gs,u = C_gh,u D
```

Thus semantic30 does not add information beyond the full E5 mean. It can
supply directions missing from PCA64, and can alter finite-sample regularization.
The mean-projection identity replays within3.4e-9. Source PCA64 retains a median
.413187 of the facet-direction squared energy in PANDORA and .513162 in Essays.
These are geometric energy fractions, not percentages of personality information.

A same-unit reader adds the computable term <C_u,Theta> to marginal interaction.
Its exact fitted score can be decomposed into additive, marginal-product and
covariance contributions. None is automatically an independently identified
psychological construct.

## 4. Complete held comparison

Values are mean Pearson across allfive traits. These are the specified held
folds, **not full5-fold/10-fold performance** and not comparable with historical
pooled r~.196 as evidence of a new improvement.

| Frozen feature family + source-selected Ridge | PANDORA n=120 | Essays n=254 |
|---|---:|---:|
| Word+char TF-IDF | .274526 | .231108 |
| E5 unit mean / PCA64 | .318143 | .233362 |
| E5 unit mean, no PCA (addendum) | .313561 | .182236 |
| GRAIN unit mean/std48 | .154239 | .103040 |
| Keyed semantic means30 | .261687 | .183025 |
| Additive E5-64 + GRAIN48 + semantic30 | .323039 | .221185 |
| Additive + marginal outer product | .138304 | .207596 |
| Additive + same-unit product | .210481 | .207522 |
| Additive + shuffled product,924 | .094614 | .205225 |
| Additive + shuffled product,925 | .190444 | .187808 |
| Additive + shuffled product,926 | .192958 | .191310 |

The paired arm exceeds each shuffle in point estimates, but the only interval
excluding zero is PANDORA versus seed924: +.115868[.019395,.200744]. The other
two PANDORA shuffle intervals and all Essays shuffle intervals cross zero.
That one comparison does not establish a general same-unit advantage; seed924
also selected a different alpha on source-validation. There is no held seed
selection or multiplicity-adjusted claim of superiority.

| Comparison, mean-r difference | PANDORA 95% CI | Essays 95% CI |
|---|---|---|
| Paired minus additive | -.112558[-.169732,-.052454] | -.013663[-.047575,.028180] |
| Paired minus marginal outer | +.072177[-.015366,.152087] | -.000074[-.031295,.030279] |
| Additive minus E5/PCA64 | +.004896[-.035930,.046612] | -.012177[-.035517,.008794] |
| Paired minus full E5 | -.103080[-.176795,-.034468] | +.025286[-.032874,.079241] |
| E5/PCA64 minus full E5 | +.004582[-.043549,.058684] | +.051126[.006011,.092534] |

All intervals use1,000 paired author resamples conditional on fitted scores;
they exclude refit uncertainty and are not multiplicity-adjusted acceptance
gates. The inherited JSON scope string contains 'oof'; here it means predictions
outside fitting authors in one held fold, not pooled fullCV.

Essays illustrates why keeping all dimensions is not automatically better for
a finite estimator: PCA64 outperforms the full1024 reference in this comparison,
despite discarding some semantic-direction energy. It does not show those
discarded directions contain no signal. The full reference chose alpha1000,
the edge of the fixed source grid; we did not expand it after viewing held data.

## 5. What the score decomposition reveals

After undoing source scaling, the paired model satisfies exactly:

```text
prediction = additive_component
           + vec(M_u) beta_interaction
           + vec(C_u) beta_interaction
```

The covariance contribution's held SD is .051-.083 in PANDORA, while its
criterion correlations range from-.105 to+.153 acrossfive traits. In Essays
its SD is .063-.078 and criterion correlations range from-.089 to+.095.
Thus this fitted model assigns appreciable score variation to covariance,
but most of that isolated variation is weakly aligned with the questionnaire
outcomes. These are post-hoc components of one fitted head, not separately fitted
predictors or proof that all covariance is noise. PANDORA Neuroticism's+.153 is
retained in the full table, not promoted as a new factor.

Every unit's bilinear contribution can be summed back to the product portion
of the author score. Exact numerical addressability is delivered; source
ownership, factuality and psychological interpretation are separate from this
algebraic attribution. No raw patient or author quotation is published.

## 6. Consequences for the production method

1. Retain compact additive and semantic references; do not replace them with
   this raw862-column interaction model. The decision is supported in both
   source-validation rankings and the held results, not rescued by a new grid.
2. Keep mean profiles, within-panel covariance and paired evidence as distinct
   mathematical objects. Covariance is retained for research, not deleted or
   renamed pure noise.
3. An address join is necessary for some explanations but is not sufficient
   to turn a semantic affinity into evidence of an author's trait. Subsequent
   construction work should address valid source/event/condition definitions,
   rather than add more raw cross-products to the same reader.
4. A local method may change regularization or use a target-directed projection
   without creating information absent from its input. Link that claim to the
   correct comparison, including the full-embedding control.

Remaining Study B work includes native LLM/frozen-readout arms on matched
material, whole/chunk semantic comparison, target-defined condition/path objects,
the four-axis MBTI map under its own folds, and independent psychological
interpretation. This tranche does not make them complete. Original G1 support,
MEPS validation-only and lockbox restrictions remain unchanged.

## 7. Reproduction and checks

Main source fitting took32.43s for PANDORA and40.34s for Essays. New E5 encoding
took about661.7s between cache contract and completion timestamps. Existing
22,464 PANDORA comment vectors were reused. Final outputs comprise20 main heads
plus2 no-PCA heads; alpha selection fitted80 main and8 companion validation heads.
These are statistical fits, not neural finetuning.

The first attempt stopped before encoding/fitting because300 item vectors
spanned more than one cache batch. The loader was corrected to gather verified
contiguous batches; the completed run is `local_bilinear_r2`. R1 is preserved.
No active process was restarted due to an observation timeout.

Independent arithmetic replays all36,254 unit addresses/coordinates and every
author's paired, marginal and shuffled matrix. Maximum pooling difference
5.21e-18; covariance-identity difference5.83e-18. All20 head predictions and100
trait correlations agree; maximum cold-head error2.89e-15. The companion's
full-vector heads replay within4.56e-15, with complete per-trait metrics added.
Source/held separation, source-only scaler means and original unit coverage
were checked. 57 selected tests pass, including19 new bilinear/source tests;
Ruff and diff checks pass.

Artifacts are local and gitignored:
`results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2/`.
Each domain stores units/addresses, feature matrices, all source tuning, model
hashes, frozen scores, per-trait/arm metrics and comparison summary. Top-level
`audit.json`, `paired_score_accounting.csv`, `full_semantic_metrics.csv` and
`projection_control/` preserve the independent calculations. Individual texts,
IDs, vectors, fitted models and attribution arrays must not be publicly committed.

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
STUDY='results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_next'
"$PY" scripts/run_grain_local_bilinear.py --output "$STUDY"
"$PY" scripts/run_grain_bilinear_projection_control.py --study "$STUDY"
"$PY" scripts/audit_grain_local_bilinear.py --output "$STUDY"
"$PY" -m pytest -q tests/test_grain_local_bilinear.py tests/test_grain_local_measure.py tests/test_grain_common_factors.py tests/test_grain_predecessors.py tests/test_grain_semantic_link.py
```

Run from the independent GRAIN repo with the existing data, cached E5 weights and
old frozen GRAIN artifacts. The main command resumes only a hash-matching partial
run and skips verified complete fits; the companion requires a fresh
`projection_control` directory. No new package download, native LLM generation,
external text upload, commit, push or lockbox opening is part of this run.

