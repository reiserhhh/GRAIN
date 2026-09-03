# GRAIN L4-to-L5 Reference-Frame Protocol

Date: 2026-07-30  
Opening: `F05-L45-REFERENCE-FRAME`  
Current ruling: `L45_OPENED_DISCOVERY_CANDIDATE`  
Maximum licensed layer: `L4_technical_object`

## 1. Question and boundary

This protocol asks when an anonymous L4 technical object may become a
comparable reference-relative measurement candidate. It does not ask whether
the object is personality, emotion, state, intelligence, or any other
psychological construct.

The minimum reference object is

\[
R=(\nu,\lambda,\omega,\chi,\mathcal U),
\]

where:

- \(\nu\) is a versioned reference-population measure;
- \(\lambda\) is the target facet/opportunity measure;
- \(\omega\) is the representation, slicing, and aggregation operator;
- \(\chi\) is the common reference chart;
- \(\mathcal U\) is the declared generalization universe.

Changing any member of \(R\) creates a different score version. A batch mean
or cohort-relative distance is not a reusable score.

## 2. Reference-relative estimand

For author \(u\), facet \(f\), and occasion \(o\), let
\(\bar h_{uof}\in\mathbb R^d\) be the observable L4 vector summary. The frozen
facet-standardized occasion vector is

\[
x^\lambda_{uo}
=
\int \bar h_{uof}\,d\lambda(f).
\]

When event composition differs from \(\lambda\), the implementation uses only
observed facet provenance and refuses when

\[
\lambda \not\ll Q_u
\quad\text{or}\quad
\lambda \not\ll Q_\nu .
\]

The reference center and covariance chart are

\[
\mu_R
=
\mathbb E_{V\sim\nu}[x^\lambda_V],
\qquad
\Sigma_R
=
\operatorname{Cov}_{V\sim\nu}(x^\lambda_V),
\]

with registered target-population weights. The common coordinates are

\[
z_{uo}
=
(x^\lambda_{uo}-\mu_R)
(\Sigma_R+\epsilon I)^{-1/2}.
\]

A calibration-only stability operator \(D_R\) is selected from independent
occasions. The candidate score is

\[
\widehat M_u^R
=
\frac{1}{O}
\sum_{o=1}^{O}D_R(z_{uo}).
\]

The complete output is not just a point:

\[
\widehat{\mathfrak M}_u^R
=
\left(
[\widehat M_u^R]_{\mathcal G_R},
\widehat\Sigma_u^R,
\mathcal S_u^R,
n_{\mathrm{eff},u},
\delta_u^R,
R,
\mathrm{provenance}
\right).
\]

Here \(\mathcal S_u^R\) is support coverage, \(n_{\mathrm{eff},u}\) is
effective information, and \(\delta_u^R\) is a minimum detectable technical
difference. If coordinates are not anchored, the output is restricted to a
relation, distance, or gauge-equivalence class.

## 3. Necessary conditions

An E45 instance must satisfy all of the following:

1. L4 objects share a common chart or pass a registered invertible-transport
   audit.
2. \(\nu,\lambda,\omega,\chi,\mathcal U\) are frozen and versioned.
3. Reference, calibration, and scored authors are disjoint or cross-fitted.
4. Target facet support and selection are identified; otherwise the score is
   refused.
5. Event count and event composition remain separate.
6. At least two valid occasions exist, with dependence declared.
7. Reference, calibration, occasion, event, and operator uncertainty are
   resampled or otherwise propagated.
8. A/Q and other observational aliases forbid hidden-component attribution.
9. Reference drift creates a new version and recalibration, never silent
   recentering.

These requirements are enforced by
`schemas/suica_foundation_study.schema.json` and
`suica_core/foundation_contracts.py`.

## 4. Error object

The observable region is fitted without generator truth. Each draw resamples:

1. reference authors within declared target strata;
2. calibration authors;
3. within-author occasions, centered around the observed author mean and
   corrected for the finite number of occasions;
4. event means from observed within-event variance.

Generator truth is accessed only after the region is frozen to estimate
coverage. This operationalizes the generalizability-theory principle that
prompt/facet, occasion, method, and reference variability are distinct error
sources. It remains a synthetic error model, not a completed human-text
G-study.

## 5. Counterworlds and refusals

| World | What it attacks | Required action |
| --- | --- | --- |
| `composition_shift` | count/composition collapse | standardize to \(\lambda\), report ESS |
| `reference_mixture` | wrong reference mixture | target-weight the reference chart |
| `support_hole` | positivity failure | `REFUSE_NONOVERLAP` |
| `aq_gauge_alias` | absolute component attribution | score total relation only |
| `choice_response_alias` | unknown facet provenance | refuse |
| `person_occasion_alias` | one occasion posing as stability | refuse |
| `correlated_replicate_shock` | pseudo-replication | refuse |
| `operator_kernel` | noninvertible chart transport | refuse transport |
| `informative_precision` | uncertainty related to author magnitude | retain conditional uncertainty audit |
| heavy-tailed \(t_5\) | Gaussian-only robustness | preserve gates under the second noise law |

## 6. Opened discovery history

### V1: informative failure

V1 passed recovery, correction, alias, transport, and MDD gates but failed
observable uncertainty:

\[
\text{coverage}=0.7141.
\]

The bootstrap resampled reference authors, calibration authors, and event
means but conditioned on the four realized occasions. It therefore omitted
person-by-occasion variation and produced intervals that were too narrow.

### V2: occasion channel restored

V2 added observable within-author occasion resampling plus finite-occasion
correction. It restored mean coverage near the nominal region, but its
decision report pooled conditions under a field named “minimum coverage.”
The numerical result was retained, but the gate definition was corrected.

### V3: current opened candidate

V3 used fresh development seeds and condition-specific confidence bounds:

| Gate statistic | Result |
| --- | ---: |
| Clean score-correlation lower bound | 0.9876 |
| Clean normalized-error upper bound | 0.1592 |
| Composition-correction gain lower bound | 0.2247 |
| Reference-mixture correction gain lower bound | 0.6449 |
| Minimum uncertainty-coverage lower bound | 0.9250 |
| Maximum condition mean coverage | 0.9875 |
| MDD false-positive upper bound | 0.1125 |
| Two-MDD power lower bound | 1.0000 |
| Five mandatory refusal rates | 1.0000 each |
| Truth-isolation error | 0.0 |

The result is `L45_OPENED_DISCOVERY_CANDIDATE`, not L5 confirmation. Full
artifacts are under
`results/suica_l45_reference_frame/l45_reference_frame_discovery_v3/`.

## 7. Psychometric alignment

The protocol aligns with established work in three limited ways:

- generalizability theory treats multiple facets and conditional standard
  errors as distinct measurement-error channels;
- multivariate G theory warns that a reliable total or correlation does not
  automatically make an individual profile interpretable;
- reference-sample weighting research shows that alignment to the target
  population matters for comparable scores.

GRAIN's original contribution here is the combined interface over
facet-indexed high-dimensional technical vectors, explicit alias worlds,
operator-gauge refusal, and observable reference/occasion/event uncertainty.
This is an original composition of supported principles, not proof that every
component is itself novel.

Primary methodological anchors:

- Brennan (1998), conditional SEM in generalizability theory:
  <https://doi.org/10.1177/014662169802200401>
- Huebner and Skar (2021), conditional SEM for writing assessment:
  <https://doi.org/10.7275/vzmm-0z68>
- Jiang and Raymond (2018), multivariate G theory for score profiles:
  <https://doi.org/10.1177/0146621618758698>
- Lu and Kim (2021), target-population weighting in equating:
  <https://doi.org/10.1002/ets2.12313>

## 8. What remains open

1. A separately sealed, fresh-randomness synthetic confirmation with no
   retuning.
2. A real study-specific G-study/D-study over the facets actually sampled.
3. Conditional uncertainty under sparse, MNAR, and facet-specific occasion
   interactions.
4. Reference and operator transport across independently collected panels.
5. Only after E45 closure, a separate E56 study may test external
   psychological or behavioral interpretation.

Until then, reusable human scores, person change, batch comparison, and all
psychological naming remain refused.
