# GRAIN relation geometry and transported observation rulers

2026-10-10. HM126 finite mechanism-development result, not psychological validity.
Canonical output: `results/grain_math_response_v1/relational_isometry_r1`.
All prior versions, failures and original18/100/44 scope remain intact and OPEN.

## 1. Mechanism decision

HM125 showed that a source-only fixed contextual ruler can distinguish virtual
authors whose twelve-dimensional lexical-GRAIN laws coincide. Its source span,
however, captures only15.36% of new-instance actor-action energy. HM126 tests
whether that loss is mainly coordinate orientation, rather than missing or
changed relation geometry. This is a relation/measurement-law test, not another
personality predictor or teacher-fidelity optimization.

Public instrument anchors partly help: their supported orthogonal transport,
fitted on wording0, is closer to the frozen source relation on wording1 than
no rotation in all five nontrivial cases, and closer than a fixed wrong-family
anchor correspondence. No scale, rank, layer, regularization or target-answer
fit was selected. Anchors are paid observations, not free source-only transfer.

But pure isometry is insufficient. Even the best held-data rotation leaves
relative vector residuals .242--.392. The full relation Gram has high overall
alignment (.850--.958), yet different-instance correct-family ranks135/720 and
187/720. High global shape similarity does not uniquely locate named relations.
The held-data optimum is a diagnostic bound, never the deployed transport.

## 2. Supported polar transport and its observation meaning

The original view0 wording0 controls define X in R^(2304 x k), k=12 or6:

    X_j=mean_(b,o) b o h_source,wording0,j,b,o; B=X^+.

Known public wording0 controls in a target rendering define Y. Thin QR gives
X=Qx Rx, Y=Qy Ry. The small SVD is Ry Rx^T=U Sigma V^T. Put

    L=Qy U, R=Qx V, T=L R^T.

L and R have orthonormal columns. T maps the calibrated source support to the
target support and is a partial isometry, not a uniquely identified full-space
rotation. An orthogonal completion O=T+O_perp is arbitrary outside that support.
Since B annihilates the source orthogonal complement,

    B O^T h = B T^T h

for every such completion. Thus the supported observation is uniquely specified
although the entire hidden vector transport is not. This is established polar/
Procrustes algebra, not an original mathematical theorem.

The deployed numeric readout has no design-label/question/answer argument:

    z_c(h)=B T_c^T(h-mu_c),
    origin_c=B(mu_c-mu_0),
    r_c(h)=(I-L_c L_c^T)(h-mu_c).

Store all three, plus the raw source-centered score and raw vectors. Origins
may reflect context, choice and state; the complement may contain uncalibrated
relations. Neither is automatically noise. The fixed no-rotation arm uses the
same calibration origin, so paired-author comparisons isolate orientation,
not a favorable recentering of the assessed sample.

The source and context anchors have the same registered family/condition design.
The numerical scorer itself receives none of those labels. A registered PersonA
carrier and pretrained contextual upstream encoding are still required. It is
not raw-text inference without an encoder, nor automatic natural role binding.

## 3. Actual held-out transport

Wording0 supplies calibration; wording1 never enters frame/transport fitting.
The two wordings are opportunity formulations, not independent populations.
Expression families are known controlled designs, not human-confirmed equal
behavior magnitudes. Source wording1 is the frozen reference, not an identity
matrix by assumption. Table entries are matrix discrepancy norms relative to
that reference, not prediction errors or psychological r-values.

| Held-out view | Cells | No rotation | Matched public anchors | Wrong anchors | Gram alignment | Best rotation residual | Correct-family rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same-event rewording |12|1.0274|.7092|1.6202|.9529|.2464|1/720|
| Different behavior instance |12|1.0808|.7262|1.5833|.8509|.3198|135/720|
| Same-event rewording |6|.8426|.3995|1.7657|.9580|.2419|1/720|
| Different behavior instance |6|.8814|.5472|1.6820|.8503|.3134|187/720|
| Fresh HM124 instance |6|.9543|.7261|1.4276|.9224|.3922|5/720|

The two source controls recover their own finite readout up to floating-point
error; that is a construction check, not a scientific replication. The fresh
bank lacks condition+1. Six-cell results do not infer missing conditions or
replace the required twelve-cell tests where those conditions are observed.

65 preserved virtual author laws change actor/behavior association with all
margins fixed. Each source/context bank integrates the same signed profiles.
For cells j, p=(1+b o theta_j)/(4 n_cells) over wording1. Thus

    n_cells (S_theta-S_minus_theta)/2=C_c theta,
    C_c=B T_c^T D_context,wording1.

Fresh-bank contrast discrepancy is .9422 without rotation, .7545 with matched
anchors and1.4422 with wrong anchors. These virtual distributions are not new
people, and theta is not a Big5/MBTI or clinical construct.

## 4. From Euclidean axes to a conditional relation metric

An explicitly result-after supplement asks whether a common design-family mode
merely inflated Gram similarity. With u=ones(k)/sqrt(k) and P=I-u u^T, preserve

    G=(u u^T)G(u u^T)+PGP+(u u^T)GP+PG(u u^T).

The common trace share is .219/.252 in source, and .229--.325 in changed views.
It is not the majority. Looking only at the specific block does NOT recover
the different-instance identity: ranks worsen to195/720 and194/720; fresh is
14/720. The simple common-background explanation is not supported. All blocks
remain in the saved account; this is not a centering/denoising algorithm.

The direction-sensitive account uses the source metric G0=X^T X and a target
metric Gc=Y^T Y. For a nonzero declared relation coefficient theta,

    ||Y theta||^2 / ||X theta||^2
      =q^T L_c q / q^T q,
    q=G0^(1/2)theta,
    L_c=G0^(-1/2) Gc G0^(-1/2).

The generalized eigenvalues of L_c bound the ratio exactly. Pure rotation
would make all gains1; a uniform unit change would make them one common scalar.
Observed held-out ranges are:

| View | Cells | Minimum directional energy gain | Maximum |
| --- | ---: | ---: | ---: |
| Same-event rewording |12|.506|2.808|
| Different instance |12|.440|4.269|
| Same-event rewording |6|.861|2.533|
| Different instance |6|.550|3.655|
| Fresh instance |6|1.017|5.573|

These are squared vector-displacement gains, not percentages of personality
variance. They show direction-dependent deformation, not just a common change
of orientation or magnitude. Differences may reflect renderer behavior and
actual semantic strength of the controlled behavior alternatives; this packet
does not uniquely attribute them to a single internal neural cause.

The next measurement object should retain a contextual relation metric and
qualified observation operator, not assume a global Euclidean ruler. A generic
form is

    h_c=mu_c+D_c theta+r_c,
    distance_c(theta_u,theta_v)^2
      =(theta_u-theta_v)^T G_c (theta_u-theta_v).

This is a finite observation model, not a proof that human psychological theta
exists or that all natural texts share these coordinates. A GRAIN construction
must specify reference condition, calibration relation, units, observation law
and uncalibrated channels before interpreting author positions psychologically.

## 5. Next mainline question

Do qualified response operators preserve relation meaning more strongly than
raw vector distances when both undergo context-dependent deformation? Integrate
HM119 actor-by-focus kernels and S21/S22 function-space observation geometry
before proposing another run. This connects to source action/target readout,
not another better-performing regression or arbitrary whitening.

Keep source/reference, independent public anchors, assessed author evidence,
and external psychological interpretation distinct. No extra experiment was
started at closeout. No automatic new factor naming or universal invariance.

## 6. Prior art and reproduction

[Schonemann1966](https://engineering.purdue.edu/~bethel/schonemann_orth_proc.pdf)
already solved orthogonal Procrustes and discusses null-space nonuniqueness.
[Kornblith et al2019](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf)
study representation-similarity invariances; our uncentered Gram comparison is
not their CKA. [Cohen/Welling2016](https://proceedings.mlr.press/v48/cohenc16.pdf)
formalize equivariance; no transformation group is assumed true for text.
The pinned [OpenAI matrix-jets manuscript](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex),
lines71--108, motivates separating stable observed support from an unidentified
completion. Its harmonic-PDE hypotheses are not established for these vectors.

Methods are mature; the system contribution is a scoped executable transport
construction plus evidence separating orientation, anisotropic deformation,
and unobserved relation channels. It is not a new mathematical theorem.

```sh
python scripts/run_grain_relational_isometry_hm126.py prepare --out NEW_OUTPUT
python scripts/run_grain_relational_isometry_hm126.py execute --out NEW_OUTPUT
python scripts/score_grain_relational_isometry_cold.py \
  --bundle NEW_OUTPUT/00/common_condition_minus__fresh_instance_bundle.npz \
  --input NEW_OUTPUT/00/common_condition_minus__fresh_instance_held.npz \
  --out NEW_OUTPUT/cold_scores.npz
python scripts/readback_grain_relational_isometry_hm126.py --out NEW_OUTPUT
python scripts/analyze_grain_relational_geometry_metric_hm126.py --out NEW_OUTPUT
```

9new tests,15combined focused tests,8new Python Ruff/format checks pass.
Independent-formula score/Gram audit24372entries max1.99e-13; supplemental
block/eigen-ratio audit max1.63e-14. These are same-agent alternate arithmetic,
not blind independent review. Separate cold CLI scored24vectors, .08wall/.07CPU
seconds at displayed timing precision. Runner phases plus supplement2.156CPU
seconds remain within the original600second cap. All source/code/input/result
hashes verify;0new neural calls, API, private data, training, Claude or Git.
