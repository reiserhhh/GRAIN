# GRAIN: Evidence Composition and the Author-Measurement Bridge

2026-10-09. HM61 finite empirical update, not a new mathematical theorem or a
completed personality instrument. Parent goal active; original scope open.

## The Object That Changed

HM60 isolated six behavior contents and six target queries. HM61 crosses the
six low/high behavior alternatives in full mixed accounts, holding author,
event order and queries fixed. Two lexical families provide 128 accounts and
768 passive local readings. Source family coefficients were sealed before
application acquisition. Clause families appeared in previous isolated probes;
the compositions, not all semantic materials, are new.

The observation operator is not the author construct:

    R_q(E,C) = target-conditioned reading of evidence E in narrative context C.

The six bits specify concrete text alternatives, not independent human traits.
Their uniform design permits exact finite decomposition without asserting
independence in a natural author population.

## Composition Before and After the Answer Link

For each centered ten-candidate logit vector, write the complete cube expansion:

    z_q(x) = zbar_q + sum_j A_qj*x_j
             + sum_(|S|>=2) K_qS*product_(j in S) x_j
    R_q(x) = ell(z_q(x)).

Here zbar is the mean over the 64 designed accounts, not an absent-evidence
score. Coefficients are uniform Fourier/Mobius contrasts. Exact reconstruction
and Parseval are standard mathematical identities, not empirical discoveries.

The same link ell converts native logits and constant-plus-first-order logits
to scores and high/low log-odds. Thus an additive pre-answer operation can
already produce higher-order answer effects. The control separates that known
possibility from actual nonadditive pre-answer operations.

HM61 higher-order nonconstant energy:

| Stage | Family0 | Family1 |
| --- | ---: | ---: |
| Body endpoint at block16 | 5.49% | 4.42% |
| Query endpoint at block16 | 9.36% | 6.57% |
| Centered candidate logits | 6.35% | 3.12% |
| Native score | 6.95% | 4.45% |
| Same score link applied to additive logits | 0.83% | 0.88% |

These are finite-design squared-response shares, not psychological information
or causal mediation fractions. Different stages have different ambient metrics;
subtracting their fractions is not a layer-by-layer information partition.
Lexical/length changes remain part of the alternative texts.

Nonadditivity already exists in the query-independent body state and in centered
logits. This rejects a pure final-link explanation in these materials. It does
not identify a unique attention circuit or an interaction between human traits.

## Conditional Evidence Effects

For a scalar response with coefficients a_qS:

    Delta_j R_q(x_-j)
      = 2*sum_(S containing j) a_qS*product_(k in S excluding j) x_k.

The one-bit effect depends on co-occurring evidence whenever relevant higher
terms are present. A fixed isolated-event weight need not describe every
account, but the dominant additive skeleton can still be useful.

Mixed first-order operators resemble their isolated operators: score cosine
.9094/.8957 and log-odds .9357/.9178 across the two families. Corresponding
target effects are positive in 190/192 and 192/192 conditional edges. The two
negative edges concern anthropomorphism in family0. Shared modes and selective
components remain; neither is automatically noise or a psychological factor.

## Transfer Is a Different Question from Exact Within-Table Reconstruction

Source first-order coefficients and application counterparts have cosine
.8981 (score) and .9088 (log-odds); second-order cosines are .6582/.6607.
All prespecified orders1/2/6 predict application conditional edges without
using application intercepts or coefficients:

| Source order | Score pooled edge RMS error | Log-odds pooled edge RMS error |
| --- | ---: | ---: |
| 1 | .039778 | .355640 |
| 2 | .040702 | .381966 |
| 6 | .041002 | .385125 |

The higher-order description is exact within the source table, but more detail
does not improve aggregate transport here. This is a mechanism decision, not a
chosen final predictor: preserve the common skeleton and context-specific
interactions instead of increasing order to maximize teacher agreement.
All six targets and all declared orders remain in source_transfer.csv.

## Geometry: Declared Interpolant, Not a Latent Personality Space

The finite cube table defines a multiaffine interpolant on [-1,1]^6. Its
Jacobian is a convex combination of corner Jacobians. Positive minimum
eigenvalues of all symmetric corner Jacobians therefore suffice for strong
monotonicity. In a pre-fixed nonsingular source frame A0, kappa<1 with
kappa=max_corner||A0^-1 J-I|| gives a sufficient lower sensitivity bound.

Neither certificate passes in any of the four family/unit cases. This does
not prove noninjectivity. Corner P-matrix tests alone would not establish the
general Gale-Nikaido condition. No frame, rank or regularization sweep is
launched to rescue a geometric certificate. The polynomial extension is not a
measured continuous text manifold or human personality mapping.

## The Next Mainline Object

An observation-law result must connect to, not replace, a measurement layer:

    eta_ue = a_u + B_u*Z_ue + s_u(t_e) + epsilon_ue
    r_ueq  = T_q(eta_ue, C_ue) + delta_obs_ueq.

This is a research specification, not an HM61 estimate. a_u is an author
component, B_u a personal conditional response, and s_u a time-specific state;
the admissible event, conditions, units and reference population must be fixed.
GRAIN choice and opportunity distributions must also be retained rather than
centered away. Model-response derivatives are not automatically the author's B_u.

The next finite task must answer which target-independent vector observation
units and selective action laws can support this author layer, using scoped
existing public/licensed assets. It must test a specific aggregation or
identification alternative, not add another reader-fidelity or prompt grid.
No new private-human experiment is authorized by this document.

## Sources, Provenance and Scope

[Ferrere et al. 2026](https://arxiv.org/html/2603.02673v2), Sections1--4 and
Remark3.6, supply the decomposition context. This experiment uses the classical
independent balanced Boolean special case, not their general dependent-input
construction. [Gale/Nikaido 1965](https://fig.if.usp.br/~marchett/fluidos/jacobian-matrix-global-univalence-mapping_gale-nikaido.pdf),
introduction/Sections2--4, provides the global-injectivity comparison; the fixed
sufficient bounds above have elementary direct proofs.

OpenAI/math inverse-boundary manuscript introduction, fixed commit
adc7f1241b42e322a6451854ab7e4b4c146bf78a, was re-read in full this packet. Its
multi-input response-operator organization is an analogy, not an application
of its elliptic/manifold/uniqueness theorem to an LLM. Advanced terminology is
not evidence of mathematical novelty.

Finite numerical verification: 768 actual readouts, alternate maximum error
3.638e-12, exact body-prefix agreement, source barrier verified;79 focused
tests/Ruff pass. Same-agent arithmetic verification is not independent human
replication. No new training, generation, patch, API, private text or publication.
Artifacts: results/grain_math_response_v1/heterogeneous_evidence_cube_r1.
HM45 remains deferred;HM59 template rescue stays closed. Author psychological
meaning, B_u and independent LLM-free new-text inference remain separate open
mainline tasks, not inherited from a finite factorial observation table.
