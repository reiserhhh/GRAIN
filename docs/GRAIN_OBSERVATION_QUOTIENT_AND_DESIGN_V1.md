# Observation Quotients and Design-Dependent Routes

2026-10-08. HM18 executed and numerically verified. Companion to
OBSERVABLE_ACTION_GEOMETRY_V1; earlier frozen plans remain historical snapshots.

## 1. What changed

The useful context law remains

\[
L_q(x,c)=b_q^\top x+x^\top A_qc,\qquad A_q=S_q+H_q.
\]

HM18 distinguishes three objects: a matrix in source coordinates, its scalar
action on a declared intervention design, and the reader's native response.
They have different geometries. This is a construction rule for the
text-evidence-to-target route, not an additional named personality factor.

On full-property application interventions, S/H action correlations are
.99904/.99974/.99969. On receiver token-state reversal they are
.03805/-.07207/.33930. Thus near-equivalence is design-dependent, not a
universal equivalence of symmetric and skew actions.

## 2. A quotient that preserves what is measured

For observations \(D_i=x_ic_i^\top\) and a declared positive measure \(w_i\),
define

\[
\mathcal E_\nu(B)_i=\langle B,D_i\rangle_F=x_i^\top Bc_i,
\qquad \|B\|_\nu^2=\sum_i w_i\mathcal E_\nu(B)_i^2.
\]

The space \(\mathcal N_\nu=\ker\mathcal E_\nu\) consists of operator
differences invisible to this design. The measured object is the equivalence
class \([B]\in\mathcal B/\mathcal N_\nu\), equipped with the induced inner
product. It is not the full ambient matrix. Numerical ranks use the fixed
100*eps*max(1,largest_eigenvalue)*n rule, not a predictive rank selection.

Under orthogonal gauge Q, carry both vectors and operator:
\(x'=Q^\top x,c'=Q^\top c,B'=Q^\top BQ\). Actions and quotient geometry
remain unchanged. A general nonorthogonal coordinate change requires the
dual/metric transformation too; arbitrary whitening is not an isometry.

There are two different rank questions:

- The full source observation map on all operator coefficients has rank182
  for receiver dyads, inside a numerical187-coordinate support.
- The restriction to the two particular fitted directions S_q/H_q has
  rank2 on nondegenerate evaluated arms. Poor conditioning, not exact
  identity, explains their full-property near-collinearity.

The second statement does not identify every symmetric or skew matrix.
The first is not a count of psychological dimensions. With 34,969 possible
coefficients and rank182, the numerical source-null dimension is34,787.
Minimal-norm/source-span compilation chooses an extension, not a claim that
all other psychologically meaningful extensions are impossible.

## 3. Design changes separability

For the fitted two-channel action matrix \(V_i=(s_i,h_i)\),

\[
G_\nu=\sum_iw_i V_iV_i^\top.
\]

Its eigenvalues determine numerical separability of those fixed directions;
centering before covariance measures a different object. Report both.
For application full-property, condition numbers are1815/6170/7590;
receiver reversal gives3.397/2.170/2.128, centered reversal1.677/1.961/1.985.
Reversal preserves the token-state multiset, not natural sentence semantics.
It reveals positional response, not a confirmed psychological rotation.

Validation NA reversal is still highly correlated (.9857/.9907); the
separating effect is not universal across every lexical family. All arms,
families and amplitudes remain in the result rather than selecting a
favorable subset as a new scoring rule.

## 4. Midpoints add a distinct quadratic observation

Because \(m_i=c_i+\kappa x_i/2\),

\[
\operatorname{sym}(x_im_i^\top)
=\operatorname{sym}(x_ic_i^\top)+\frac\kappa2x_ix_i^\top,
\quad
\operatorname{skew}(x_im_i^\top)=\operatorname{skew}(x_ic_i^\top).
\]

The symmetric source observation rank increases182 to264; the skew rank
stays182. The gain82 is a measurement-direction result. It neither means
82 new traits nor proves more human personality information. HM19 will
separate this self-quadratic increment from the original observation span
and compare its frozen native action, without a new accuracy grid.

## 5. Shared routes need their observation measure

Define target sharing at the action level:

\[
K_{pq}^{\nu}=\sum_iw_iL_p(x_i,c_i)L_q(x_i,c_i),
\qquad C_{pq}^{\nu}=K_{pq}^{\nu}-\bar L_p\bar L_q.
\]

Keep target-specific units; correlation discards scale. For application
full-property symmetric actions, source Frobenius cosine versus action
cosine is +.30463 versus-.14426 for NA/low, and-.01676 versus+.40178 for
NA/trust. An operator angle therefore cannot define a shared trait route
independently of the observation law.

The compiled native-action covariance also differs from the measured reader
covariance: for NA/low the latter correlation is+.04242 while receiver
compilation gives-.29595. Good marginal response reproduction does not
automatically reproduce multi-target dependencies. This points to joint
observation/action structure, not another per-target alpha search.

## 6. Information geometry connection and limits

In an explicitly specified homoscedastic Gaussian observation model,
\(y_i=\mathcal E(B)_i+\epsilon_i\), the vectorized coefficient information
matrix is proportional to \(D^\top WD\). The dual weighted dyad Gram has
the same nonzero eigenvalues. Without that error model, HM18's Gram is an
observation seminorm, not an estimated psychometric Fisher matrix or a
confidence interval for a person's score.

Casey et al. distinguish uncertainty in model parameters from uncertainty
in useful predictions and construct experiments from response sensitivities.
This motivates targeting the missing action direction rather than trying
to recover every matrix coefficient. Their biological model, priors,
noise assumptions and optimality claims are not transplanted here.

OpenAI/math's metric/connection draft uses source evaluation and sufficiently
rich frames to identify maps, then extends them under PDE/geometric
hypotheses. We borrow the order of reasoning: characterize which evaluations
distinguish objects before claiming global uniqueness. Text interventions
do not satisfy its unique-continuation theorem merely by being vectors.

## 7. Executed evidence and reproducibility

Canonical results: results/grain_math_response_v1/action_observability_r1;
alternate arithmetic: action_observability_verification_r1. All2172 query
addresses/724 state pairs/7arms/source-validation-application splits are
retained. No new fits, model calls, API, private text or neural training.
The statistics are result-guided diagnostics of old controlled reader
interventions, not new human data or completion of the original18/100/44.

Run sequentially with the existing project environment:

```bash
python scripts/run_grain_math_action_observability.py prepare --out <fresh-output>
python scripts/run_grain_math_action_observability.py source --out <fresh-output>
python scripts/run_grain_math_action_observability.py evaluate --out <fresh-output>
python scripts/verify_grain_math_action_observability.py --packet <fresh-output> --out <fresh-verification>
```

Preserve canonical frozen outputs. Source diagnostics are hashed before
held values are loaded; no held rotation, reweighting or model selection.

## Sources Actually Used

- [Casey et al., Optimal experimental design in an EGFR signaling and
  down-regulation model](https://arxiv.org/html/q-bio/0610024v1), Methods
  II.1-II.2 and results on prediction/parameter uncertainties read this
  packet. Equations3-5 and sequential design motivate a response-sensitive
  observation map. No reproduction of their biochemical fit.
- [Sethna, Experimental Design in Sloppy Systems](https://sethna.lassp.cornell.edu/Sloppy/ExperimentalDesign.html),
  author explanation read as context, not a new mathematical theorem.
- [OpenAI/math](https://github.com/openai/math), pinnedcommit
  adc7f1241b42e322a6451854ab7e4b4c146bf78a, metric/connection draft,
  build/sections/07-global.tex lines1-245 read this packet. Not the entire
  manuscript or an independent verification of its proposed proof.
