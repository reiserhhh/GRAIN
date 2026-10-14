# F16 T2/T3 controlled-text observation protocol

Frozen before application evaluation, 2026-09-23. This is a new observation
extension, not a replacement for the original F05/F13/F16 numeric runs.

## Target and model

Reuse the three existing core source ensembles (200 fits/seed). Preserve
`s_t=rho*s_(t-1)+w_t`, `Y_t=a+b*z_t+s_t+e_t`, stationary initialization,
`rho=.99*tanh(eta)`, and Gaussian source prior on `(a,b,eta)`.
T2 is state at the requested time 47; T3 is realized response averaged over
times 48..51 with the frozen condition plan `(+1,-1,+1,-1)`.
Keep the full covariance of `(s_47,Y_48,Y_49,Y_50,Y_51)`.

Render first 48 archived events using the existing F08 public synthetic bank:
eligibility Q independently Bernoulli(.5); eligible pole B=1[Y>=0], ineligible
pole independent fair coin. Source/application wording pools remain separate.
Scoring receives only event/text address, author, condition, time and history,
not Y, B, Q or latent parameters. Reuse pinned actual NLI/GRAIN reader outputs;
32 validation texts calibrate the joint categorical channel, disjoint from
32 old GRAIN training texts and 32 application frames. This finite reused bank
is not independent natural-language generalization.

## Calculation and approximation

For prior state `s~N(m,v)`, offset `d=a+b*z`, observation noise R and category c:
`A=.5*E[0,1,c]+.25*(E[0,0,c]+E[1,0,c])`,
`B=.5*(E[1,1,c]-E[0,1,c])`, `L(s)=A+B*Phi((d+s)/sqrt(R))`.
Let `k=(d+m)/sqrt(R+v)`, `Z=A+B*Phi(k)`,
`Z'=B*phi(k)/sqrt(R+v)`, `Z''=-B*k*phi(k)/(R+v)`.
Exact one-step posterior moments under a Gaussian prior are
`m_new=m+v*Z'/Z`, `v_new=v+v^2*(Z''/Z-(Z'/Z)^2)`.
The next step projects that distribution to a Gaussian. This is assumed-density
filtering (ADF), not an exact non-Gaussian state posterior. Negative/nonfinite
variances fail, not eigenvalue flooring. Compare fixed-parameter filtering with
a non-Gaussian state-grid calculation; do not hide approximation error.

Source prior integration uses 256 scrambled Sobol points in three parameter
coordinates per source fit, weighted by the accumulated text likelihood.
Average 200 source/reader components equally (cut/bootstrap convention).
Report conditional-state/future, personal-parameter, and combined source/reader
covariance separately. Future innovations affect multiple future steps; their
loadings are summed before squaring for the mean target. No future observations
are inputs. Norms are frozen from all 300 source authors before application.

## Fixed comparisons and numerical checks

Three seeds 20260915/16/17; 300 source, 200 application each, no author selection.
Arms: hybrid joint (NLI pole/GRAIN qualification); NLI joint; same hybrid
factorized channel; known pole/eligibility oracle; naive Gaussian signed reader
with fractional qualification precision; eligible continuous-Y reference.
The last two use exact scalar Gaussian filtering conditional on the same
parameter nodes; thus parameter integration is matched, not a new optimizer.
All six arms predict both original targets, with MSE/bias/r/coverage/width and
paired author-bootstrap MSE differences (2000 fixed replicates). Failed
calibration and approximation cells remain; no threshold or posterior tuning.

Two independent QMC scrambles on the first 12 application IDs/seed assess score
and SD discrepancy; >.02 triggers a separately saved 1024-node diagnostic,
not replacement of the primary result. First two application IDs/seed, source
components 0/100/199 at their prior means: exact state-grid 301/601 nodes over
12 stationary SD, compare log evidence/state moments against ADF. State-grid
resolution differences and finite support are reported separately. This panel
is an approximation diagnostic, not a full alternate integrated-posterior audit.

Cold process: bundle+input only, exact object agreement within 1e-10.
Eight scored text evidence cards per target: two full histories, two missing
conditions, positive-report selection, condition-dependent quotation,
all quotation, all hypothetical. Additional missing-tail, empty, revision,
retraction and alias controls retain requested time and future plan.
Actual controls are not interpreted as changes in an author's psychology.

## Theory sources and boundaries

Gaussian moment matching/assumed-density filtering is standard; see Särkkä,
*Bayesian Filtering and Smoothing* (2013), chapter 6:
https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
The closed one-observation derivatives above are calculated for our affine
probit channel, not copied from a Kalman linear-measurement update.

Preserve numeric-unit anchors, observed/borrowed support and all old failures.
No new neural/API calls, source refits, human text/questionnaires, MEPS training,
remote work, lockbox, commit or push. T3 here is an integration extension, not
another independent-builder claim. Original 44+6 scope remains open.
