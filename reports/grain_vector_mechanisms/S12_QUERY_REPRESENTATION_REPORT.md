# S12: a joint key-value vector recovers part of the useful source contrast

2026-09-08. COMPLETE.213 N source authors, three cue-mask views each.
639 unchanged MPS/FP32 forwards in953.63seconds,0 backwards.90 teacher-only
affine readers;3,834 OOF predictions frozen before joining exposed source labels.
CPU fitting plus outcome analysis took3.56seconds.

## Main finding

**Within-comment KV pairing makes the useful configuration contrast recoverable
by a small linear reader where ordinary body/E5 means did not.** This advances
the mainline from an intervention effect to a concrete vector representation.

For the primary target C=S_new-M, every fixed arm is shown:

| Representation | Coordinates | Target r | Target MSE / train-mean MSE | Source N r |
|---|---|---:|---:|---:|
| Body-state mean | PCA64 | .0074 | 1.2403 | .1161 |
| Body-state mean | PLS2 | .0588 | 1.2441 | .1285 |
| Query state | PCA64 | .2401 | 1.0343 | .1216 |
| Query state | PLS2 | .1645 | 1.0950 | .1695 |
| KV pairing | PCA64 | **.3203** | **.9063** | .1429 |
| KV pairing | PLS2 | .2362 | 1.0021 | .1116 |

S11's matched E5-mean target r was -.0187/-.0183 for PCA64/PLS2. The underlying
native C retains its S9 source association r=.2116. No representation, coefficient
sign, or alpha was selected from questionnaire outcomes.

Paired target-recovery results, conditional on the saved OOF fits:
- KV/PCA versus body/PCA: delta r=.3128, bootstrap95%CI[.1177,.4878].
- KV/PCA versus E5/PCA: delta r=.3390, CI[.0919,.5246].
- KV/PLS versus body/PLS: delta r=.1774, CI[.0079,.3075].
- KV/PCA target r bootstrapCI[.1604,.4399]; FisherCI[.1942,.4359].
- KV/PCA reduces MSE relative to body/PCA by .3340 units of the train-mean
  control's MSE, CI[.2055,.6623]. Its own MSE is9.37% below that null control.
  The paired gain over a poor body reader is not a9.37% psychological gain.

This selects KV/PCA as the next **mechanism shortcut candidate** using the
declared target-fidelity criterion, rather than selecting query/PLS because
its external r happens to be larger.

## Source outcome linkage and the two notions of success

KV/PCA C readout has N r=.1429, Pearson p=.0372 and FisherCI[.0086,.2721].
Its paired-author bootstrap correlation CI is[-.0055,.2702]; the small positive
association has interval-method sensitivity. Its external advantage over body
means is only delta r=.0268, CI[-.1099,.1672], not an established prediction gain.
Query/PLS C has N r=.1695, FisherCI[.0359,.2972], bootstrapCI[.0170,.3061], but
its target MSE remains worse than the train-mean null. Both results are retained.

Therefore:
1. Representation-level recovery is the strong result: the joint KV reader
   decodes the intervention contrast better than the tested means.
2. A positive human-label association survives in compact readers on these
   already exposed source authors. It is not a new high-accuracy personality
   benchmark or evidence of significant superiority over the native teacher.
3. Source associations are exploratory/pointwise across the18 displayed arms.
   The current goal is a usable mechanism chain, not a calibrated questionnaire.

Covariance accounting for KV/PCA:
Cov(Y,C)=.0010462 = Cov(Y,C_hat) .0003223 + Cov(Y,C-C_hat) .0007238.
That is approximately31% of this signed source covariance at the point estimate,
not31% of personality variance and not a causal mediation percentage.

## Complete-teacher/background controls

The contextualized query state most strongly predicts the projected background:
PCA query target r=.7454 for M and .6765 for S_new, versus body .6753/.5881
and KV .5407/.5431. That high complete-teacher/background fidelity is a different
property from recovering C. It explains why following only teacher-score r can
miss the relational contribution of interest.

All18 arms, external Fisher intervals, target MSE, paired external comparisons,
selected alphas and additive covariance identities are in the ignored output CSVs.
T/F's full2328-author student AUC=.7173 remains unchanged; the64-author S9
diagnostic subset was not retrained here.

## Actual mathematical readout

For author u, view v, head h and comment token group G, use actual keys, values,
and effective contextualized query from layer17:

\[
z_u={1\over3}\sum_v \operatorname{concat}_h\left[
 \sum_G\sum_{i\in G}
 \left(p^h_{uvi}-{P^h_{uvG}\over |G|}\right)V^h_{uvi}
\right],\qquad
\widehat C_u=b+\beta^\top z_u .
\]

P_G is the native attention mass of that comment. This contrasts weighted
within-comment values with a same-mass uniform value mean. It does not erase
comment choice or claim that attention mass is psychological importance.
Fold-specific standardization, PCA and Ridge compose into a saved affine map
in the original1536 coordinates; every held-out prediction replays from that map.

No verbal chain-of-thought, generated explanation, or human label enters that
reader. The contextualized vectors still come from a pretrained decoder prefix.
S12 collected them during full native forwards for an exact comparison; it did
not yet benchmark an early-exit prefix as a faster standalone execution path.

The S10 identity dA/dq=Cov_q(V,K) explains a local operation underneath z.
The new Fisher geometry note derives the value variance reachable by constrained
query changes; that variance is over tokens, not over human traits. The source
experiment links a computational route to author measurements without conflating
the two covariance populations.

## Verification and next step

Origin replay max error0 across all639 views. Native attention head reconstruction
max error4.42e-6. Same fixed model, tokens, numerical execution and S4 author folds;
all90 affine models replay. The collector and analysis are complete; no neural
process remains running.

Next: implement a frozen prefix-only KV extractor plus the existing fold-specific
affine readers, and demonstrate numerical replay and saved computation on a small
fixed source subset. Preserve the query-state/PLS comparison as a second mechanism
route, not as an outcome-selected production winner. After prefix replay, use
existing caches for a bounded joint/query interaction analysis or lower-rank
compression; do not reopen E5-mean hyperparameter grids or broad generation queues.

## Reproduction and artifacts

Plan: docs/GRAIN_QUERY_REPRESENTATION_S12_PLAN.md.
Theory: docs/GRAIN_QUERY_INFORMATION_GEOMETRY_V1.md.
Collection: scripts/run_grain_query_representation.py.
Fit: scripts/analyze_grain_query_representation.py.
Paired target analysis: scripts/analyze_grain_query_recovery.py.

Private ignored root:
results/grain_vector_mechanisms/query_representation_s12_r1/
- manifest.json, run_status.json, private_packets/author_*.json and .npz
- analysis/private_models/, private_oof_before_outcomes.csv, oof_frozen.json
- analysis/metrics.csv, paired_intervals.csv, covariance_split.csv, selected_models.csv
- analysis/summary.json, target_recovery/paired_target_intervals.csv, summary.json.

Re-run into a new output directory, not over this frozen packet:

~~~bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
"$PY" scripts/run_grain_query_representation.py prepare \
  --output-dir results/grain_vector_mechanisms/query_representation_s12_r2
"$PY" scripts/run_grain_query_representation.py run --max-authors 213 \
  --output-dir results/grain_vector_mechanisms/query_representation_s12_r2
"$PY" scripts/analyze_grain_query_representation.py \
  --collection results/grain_vector_mechanisms/query_representation_s12_r2 \
  --output-dir results/grain_vector_mechanisms/query_representation_s12_r2/analysis
"$PY" scripts/analyze_grain_query_recovery.py \
  --analysis results/grain_vector_mechanisms/query_representation_s12_r2/analysis
~~~

No protected questionnaires, bridge sets, lockboxes, LoRA, API, or remote push.
