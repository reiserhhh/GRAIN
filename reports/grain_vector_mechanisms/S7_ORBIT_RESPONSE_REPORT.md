# S7: shared nonlinear distribution response, not shared ordering response

2026-09-08. Complete. Same eight S6 authors, three queries, two selected-comment
groups, three seeded within-comment permutations.168 forwards,0 backwards,
239.05seconds after model loading. No questionnaire labels or new training.

## Finding

The strong N/confirmation linkage from S6 localizes to the **nonlinear
distribution-response component**: r=.9730, author-bootstrap95% CI
[.8575,.9928]. Its ordering-sensitive counterpart has r=-.0776,
CI[-.6582,.7789]. Thus the shared mean-null response is more specifically
associated with how these queries read dispersed token states than with a
common sensitivity to placing the same states at different token positions.

This is a result about this decoder's responses to declared internal-state
interventions. It is not an emotional diagnosis, an external-scale correlation,
or a claim that normal sentence order is unimportant.

## What is held constant

Each permutation moves complete block16 state vectors only within a selected
comment. It preserves their complete unordered multiset, all moments, total
norm, count and unselected/query states. The original token positions remain
in the decoder; assigning different states to them can affect subsequent
positional attention. These are not claimed to be meaning-preserving edits.

Three uniform permutations use seed731 + panel_index*100 + edit_index*10 +
repeat. The same local permutations apply across queries. All24 unaltered
origins agreed with S6 scores within1e-5. S6 supplies the mean-collapsed endpoint.

## Algebra and finite estimation

Let Pi H be the comment-mean projection, and Rf(H)=mean_g f(gH) over the full
within-comment permutation group. Then

\[
f(H)-f(\Pi H)=[f(H)-Rf(H)]+[Rf(H)-f(\Pi H)].
\]

The first difference is arrangement-sensitive; the second compares averaging
judgments to judging an averaged representation. S7 uses the average of three
random group elements as an estimate of Rf. The empirical identity has zero
numerical error, but three draws do not equal full group enumeration. Trial
score dispersion is retained as Monte Carlo information, not suppressed.

## Quantitative comparison

| Query | Original-to-shuffled RMS | Arrangement component RMS | Distribution component RMS | Total collapse RMS | Mean within-path MC score SD |
|---|---:|---:|---:|---:|---:|
| T/F | .00348 | .00305 | .01569 | .01355 | .00180 |
| N | .00716 | .00692 | .01880 | .02471 | .00128 |
| Confirmation | .00265 | .00228 | .01002 | .00900 | .00107 |

These are response magnitudes, not independent variance fractions. Components
can cancel: arrangement/distribution correlations are T/F=-.744, N=.820 and
confirmation=-.541. Do not divide the RMS columns to report percentages of
personality information. All individual permutations and empirical component
scores are retained in private local artifacts.

| Matched N/confirmation component | r | Author-bootstrap95% CI |
|---|---:|---|
| Arrangement | -.0776 | [-.6582,.7789] |
| Nonlinear distribution | .9730 | [.8575,.9928] |

The intervals resample authors while retaining their paired edit conditions.
They describe this eight-author panel with its three fixed random draws, not
a broad population or uncertainty from arbitrarily many new permutations.

## The next mathematical object

For centered orbit displacement r_g=vec(gH-Pi H), E_g r_g=0. A local expansion
suggests a testable approximation:

\[
Rf(H)-f(\Pi H)\approx
\tfrac12\operatorname{tr}(\nabla^2f(\Pi H)\,\operatorname{Cov}_g(r_g)).
\]

This is **query-conditioned curvature contracted with token dispersion**, not
ordinary variance selection. Next test the approximation and its higher-order
remainder before implementing a lightweight operator. S5's comment-level E5
kernel mean is a different representation, so its weak N result does not refute
this internal-token mechanism. Connecting that mechanism to the missing D1
teacher/label covariance remains a separate step.

The finite-group construction and scope are developed in
[theory V3](../../docs/GRAIN_AGGREGATE_CONFIGURATION_THEORY_V3.md), drawing on
Etingof et al.'s representation theory and Zaheer et al.'s invariant-function
framework. The upcoming curvature test is motivated by the Integrated Hessians
literature, not by assuming that all decoder functions are quadratic.

## Reproduction

Runner: `scripts/run_grain_orbit_response.py`.
Outputs: `results/grain_vector_mechanisms/orbit_response_s7_r1/`, including source
hashes,168-call cost ledger, private permutation/component rows, aggregate tables
and completion summary. Results remain gitignored; tracked reports contain
aggregates only. The focused suite now passes189 tests; Ruff checks pass.

```bash
python scripts/run_grain_orbit_response.py \
  --panel results/grain_vector_mechanisms/multiquery_sensitivity_s1_r1 \
  --mean-null results/grain_vector_mechanisms/mean_null_path_s6_r1 \
  --output-dir results/grain_vector_mechanisms/orbit_response_s7_replay
```
