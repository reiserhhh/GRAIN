# NEXT76: a numerical content channel, with a different observable error geometry

2026-09-11. Finite pass65 of the ACTIVE GRAIN logic goal. Primary `7b91260`
frozen before source fitting and application. No new neural/API/human-label/MEPS
calls; existing12-author cached materials only.

## Findings and decision

**The local program can now compute both query selection and content values from
H16 without reading current native K/Q/V or their captured derivatives.** Its
source value correction improves new8 raw V mean squared error by about11.1%
and dV error by4.5%. This is a concrete removal of one runtime dependency.
The fixed readout g and neural H16 remain explicit inputs.

**A better value-vector approximation is not automatically a better measurement.**
With the response-coordinate selector, replacing identity V with learned V raises
the native-query K+V contrast error from .783140 to .862421. The raw vector
improvement and the action deterioration coexist in the same fixed experiment.
Do not promote the learned value arm as a complete response predictor.

Both response channels are now computed from one scalar: selection covariance
plus the selected change in content. The source-query native comparator shows
partial cancellation, whereas the learned functions amplify their two channels.
The next step is to reuse source-only endpoint readers and assemble an actual
query-to-endpoint component, not improve global V reconstruction indefinitely.

## Exact construction

- Same source4LOO/new8all4, both neuroticism/confirmation queries, all four frozen
  GRAIN directions and both existing own/pooled readout modes. New8 is disjoint
  from source fitting but repeatedly exposed in prior discovery cards.
- Reuse source NEXT75 rank32 basis/67-column causal-context features. Identity
  V0 is H16 through fixed attention22 RMSNorm/V matrices. Fit Vhat=V0+Phi C and
  D Vhat from one function using source V/dV, source-normalized joint losses,
  alpha=.01, unpenalized explicit constant. Ten solves, no new8 selection.
- Two frozen selectors x two values = four pure numerical functions. Native-V
  versions of both selectors plus original source-Q/native-KV are comparators.
  No new fitted basis, dimension, alpha, rescaling or source intercept deletion.
- Application uses H16/layout/fixed matrices/source maps/g-only packets. Stage
  guards deny native V/dV/K/Q, native captures and old prediction targets.
- Every function exports scalar, key partial, value partial and their sum. The
  two target queries hold Q fixed; DQ and residual/MLP/later-suffix channels are
  not included in K+V totals. g is also fixed in this local derivative.

## Value and tangent approximation

Equal-author means; raw, not centered, value/dvalue errors:

| cohort | query | identity V MSE | learned V MSE | identity dV MSE | learned dV MSE |
|---|---|---:|---:|---:|---:|
| source4 LOO | N |.537377|.481319|.323027|.310084|
| source4 LOO | Confirmation |.538380|.482061|.323117|.310153|
| new8 | N |.520716|.463122|.318528|.304202|
| new8 | Confirmation |.521340|.463561|.318586|.304219|

All4-source normalized residual losses: V .875581/.875126, dV .899480/.899452.
The new8 dV SSE/zero remains .995892/.995761; residual teaching yields a limited
but measurable vector improvement, not near-complete tangent reconstruction.

Explicit bias-column norms are2.875896/2.882069. Its projected scalar RMS on
new8 is .051242/.049033 averaged over both modes. Its K/V derivative is zero
within4.063e-17, since it adds the same fixed vector to every token. Other source
centering constants are not included in this particular bias-column account.

## End-to-end local response comparison

Own mode, new8 N-minus-confirmation,32 factor actions from8 authors. These are
model-response correlations, not human personality correlations.

| selector | values | native-Q K+V r | native-Q K+V SSE/zero | predicted RMS |
|---|---|---:|---:|---:|
| identity | identity |.866692|1.354273|.120349|
| identity | source learned |.865849|1.475068|.123747|
| response coordinates | identity |.864612|.783140|.100452|
| response coordinates | source learned |.862394|.862421|.103416|
| identity | native comparator |.876527|.806845|.102763|
| response coordinates | native comparator |.857904|.403092|.078045|
| source-Q/native-K | native comparator |.845321|.613897|.016861|

Native-Q target RMS=.062505. The direct source-Q target is .016861. Against the
latter, response-selector identity/learned V SSE/zero is26.932384/28.822514.
Different target functions must not be exchanged to claim intermediate recovery.

Per-query native-Q total errors for the response selector: identity values
1.751740/6.608140; learned values1.879690/6.736578 (N/confirmation). All factors,
queries, components and modes, including these unfavorable results, are exported.
Local scalar contrast r changes .434018->.486225, but SSE/zero1.827096->1.852583;
sign agreement remains0/8. Again, ordering alone is insufficient for a substitute.

## What the two channels reveal

For the response selector with source-learned values, new8 contrast key/value
RMS=.096775/.006665, sum RMS=.103416. Its energy exceeds the sum of marginal
energies by13.66% through a positive cross term. With the same selector and native
V/dV the excess is only1.20%. The original source-Q/native-KV comparator instead
has total/sum energy=.765203: cancellation. These are query/component-specific
computational relationships, not personality variance percentages.

With selection held fixed, the observable value error is a *weighted projection*
of the vector error, not its ordinary Frobenius norm. An update can reduce many
irrelevant-to-this-readout coordinates while increasing the signed error that
the particular readout sees. The algebra is exact; the proposed specific remedy
(source endpoint assembly) is the next experiment, not an already demonstrated cure.

## Verification and outputs

- Audit PASS;10fits,24primary query packets and exact numeric replay, six exact
  analysis CSV replays. Row inventory:8/336/1344/168/168/84. Public tables:7.
-64 full same-scalar central differences max8.882586e-11. Original native-Q
  key/value channels and earlier comparator replay max9.992007e-16.
-41 focused tests pass; Ruff and whitespace checks pass. Successful numeric
  stages including replay23.629569CPU seconds, below180s. Imports/provenance
  preflight outside numeric-stage timing. No runtime/model restart required.
- H16 and material-specific fixed g remain neural-derived; current native V
  is removed only from the candidate application, not from teacher evaluation.
  Old score/lockbox/release artifacts are unchanged, no raw data exported.

Private: `results/grain_vector_mechanisms/logic_source_value_channel_v1/`.
Public: [aggregate tables/provenance](source_value_channel_v1/).

Runner `scripts/run_grain_source_value_channel.py`: separate stages `prepare`,
`fit`, `predict`, `evaluate`, `predict --folder replay`, `evaluate --folder replay`,
`audit`. Publisher `scripts/report_grain_source_value_channel.py`. Use the parent
venv and a fresh output directory for repetition; never overwrite this frozen run.

[Theory](../../docs/GRAIN_SOURCE_VALUE_CHANNEL_THEORY_V1.md),
[NEXT77 source endpoint assembly](../../docs/GRAIN_SOURCE_ENDPOINT_ASSEMBLY_PLAN_V1.md).
