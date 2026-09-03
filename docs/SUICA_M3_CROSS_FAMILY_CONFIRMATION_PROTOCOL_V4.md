# GRAIN M3-V4 Cross-Family Confirmation Protocol

Status: **M3_CROSS_FAMILY_V4_READY_FOR_CLEAN_SEAL; formal confirmation not
sealed**

## 1. Question

M3-V4 tests whether author-dependent text-path structure can exist outside
ordinary means, low-order moments, and lag-0:2 covariance, and whether a fixed
family-blind measurement battery can recover that structure on an independent
replicate.

This is a mathematical existence test. It does not establish personality,
construct validity, human-text generalization, or clinical utility.

## 2. Cross-family planted objects

### CF-D: distribution geometry beyond degree four

On a common bounded support, author \(u\) has

\[
f_u(x)=2^{-d}\{1+\eta q_u(x)\}, \qquad \eta=0.92,
\]

where \(q_u\) is built from Legendre products orthogonal to every monomial of
total degree at most four. A shared unknown orthogonal rotation is applied
after sampling. Tail, skew, multimodal, and copula worlds use different
unseen basis families, while the estimator always uses the same random ECF.

### CF-O: nonlinear response operators beyond joint poly3

For actor \(u\), partner \(v\), occasion \(o\), and opportunity \(t\),

\[
Y_{uvot}
=B_3(Z_{ot},P_{ot})^\top\beta
+a_u+b_v+d_{uv}+e_o
+R_u(Z_{ot})+C_u(P_{ot})+\epsilon_{uvot}.
\]

\(R_u\) and \(C_u\) are spline, Voronoi, or neural surfaces projected
orthogonally to the joint polynomial space through degree three. Train and
test use the same actor and partner populations but disjoint actor-partner
dyads. Actor, partner, dyad, and occasion nuisance effects are stable across
replicates and sum-to-zero constrained.

### CF-KP-H: equal low-order covariance, different dwell hazard

A hidden alternating renewal state \(S_{u,t}\in\{-1,+1\}\) has

\[
p_u(D=1,2,3,4)
=(0.20,\ 0.25+\theta_u,\ 0.30-2\theta_u,\ 0.25+\theta_u).
\]

Thus \(E[D_u]=2.6\) and \(P(D_u=1)=0.2\) for every author, while the complete
hazard vector varies. Residual-life initialization makes the sampled path
stationary. Observation is a shared unknown rotation of the latent path plus
isotropic Gaussian noise.

### CF-KP-R: equal scalar autocovariance, different probability current

A three-state cycle has

\[
P_d(i,i)=s,\quad
P_d(i,i+1)=a+d\delta,\quad
P_d(i,i-1)=a-d\delta,
\]

where \(d\in\{-1,+1\}\). Since \(P_{-d}=P_d^\top\), every real scalar emission
has the same autocovariance under direction reversal, while its signed
forward-reverse probability current changes. The hidden state is mapped
through an asymmetric scalar emission, Gaussian observation noise, and a
shared unknown rotation.

### CF-KP-N: equal linear covariance, different nonlinear dynamics

Each latent channel follows stationary ARCH(1):

\[
Z_{u,t}=\sigma_{u,t}\epsilon_t,\qquad
\sigma_{u,t}^2=(1-\alpha_u)+\alpha_u Z_{u,t-1}^2,
\]

with \(0.05<\alpha_u<0.45\). Therefore

\[
E[Z_{u,t}^2]=1,\qquad
E[Z_{u,t}Z_{u,t-\ell}]=0\quad(\ell\ge1),
\]

although the conditional-variance operator varies by author.

## 3. Fixed estimator battery

Every observation task runs all objects. No estimator receives the generator
world, generator seed, hidden state, oracle parameter, or truth path.

1. **CF-D expected object:** one isotropic four-scale random ECF. No quantile,
   copula-harmonic, anchor-kernel, or train-stability feature selection is
   allowed.
2. **CF-O expected object:** train-frozen Laplace Nyström/RKHS response
   surfaces, residualized against joint poly3. Test responses are never used
   to estimate nuisance effects.
3. **KP-H expected object:** a fixed nonparametric dwell-hazard profile and a
   held-out Jeffreys-smoothed survival log score.
4. **KP-R expected object:** a signed forward-minus-reverse characteristic
   kernel mean on delay windows.
5. **KP-N expected object:** a fixed RFF delay-VAMP/Koopman operator. Author
   deviation from the leave-author-out pooled operator is shrunk by one
   train-only cross-occasion coefficient before held-out prediction.
6. **Cheap controls:** degree-four moments, joint poly3 response slopes, and
   lag-0:2 covariance.

Expected geometry is computed from independent test features, not train
features. Held-out increments compare an author-specific train object with a
leave-author-out pooled train object.

## 4. Independent mathematical validity

Truth-open code independently recomputes:

- density normalization, minimum density, and every rotated moment tensor
  entry through total degree four;
- joint-poly3 projection norm;
- nuisance sum-to-zero constraints;
- train/test dyad disjointness, complete actor/partner coverage, graph
  connectivity, and incidence rank \(A+P-1\);
- renewal \(E[D]\), singleton probability, cycle stationarity and transpose
  relation, and theoretical \(\Gamma(0{:}2)\);
- exact alias support and hidden-emission-kernel controls.

Empirical covariance is not required to agree at \(10^{-8}\). Only the
population generating law must match exactly; empirical lag covariance is a
finite-sample diagnostic.

## 5. Confirmation isolation

Formal execution is four separate commands/processes:

1. `confirmation_v4`: clean committed code and one exact config are sealed
   before randomness exists.
2. `generate_v4`: post-seal external randomness creates observations and an
   AES-GCM encrypted truth lockbox.
3. `fit_v4`: reads observations and independent estimator seeds only. The
   source contains no generator or truth loader.
4. `open_v4`: verifies every task/hash, opens truth once, performs independent
   validity checks and seed-then-author hierarchical bootstrap, and atomically
   publishes results through a hash-chained open ledger.

Blind/open commands cannot accept a replacement config. Formal seal refuses a
dirty worktree and any config containing a predeclared seed.

## 6. Current development evidence

The alias construction is now a genuine support-identifiability counterexample:
its author-specific surface is

\[
r_{u,j}(x_j)=\theta_{u,j}\operatorname{sign}(x_j)
             (|x_j|-1)_+^4.
\]

The author operator therefore exists and varies outside the registered
support, while it is exactly zero for every observed opportunity
\(x_j\in[-1,1]\). It is not implemented by disabling the signal.

The post-fix V4 smoke run completed all mathematical validity, refusal, alias,
and two-sided null-calibration checks. At deliberately small event budgets:

- all three operator worlds recovered condition and partner geometry strongly;
- renewal hazard and cycle direction recovered on independent replicates;
- hidden-emission and outside-support aliases remained near chance;
- tail distribution shape was underpowered;
- ARCH structure and its small proper held-out predictive increment were
  recovered.

The final post-fix event-power frontier selected 1,024 events for tail,
multimodal, and ARCH worlds. At that budget, six-seed development power was
1.00, 1.00, and 0.833 respectively. These are development-seed findings only.
The complete 12-seed V4 preflight passed every registered development gate:

| Object family | Expected AUC | Oracle geometry | Proper held-out gain |
| --- | ---: | ---: | ---: |
| CF-D, four worlds | 0.636-0.680 | 0.331-0.468 | 0.0126-0.0432 |
| CF-O, six target cells | 0.99991-0.99997 | 0.977-0.988 | 0.335-0.490 |
| CF-KP, three worlds | 0.758-0.809 | 0.829-0.876 | 0.00000227-0.0290 |

All validity fractions were 1.0, all refusal maxima were zero, and all
knockout geometry magnitudes were at most 0.0419. The two aliases produced
AUCs 0.4984-0.5008 with absolute geometry 0.0145-0.0523. Across six null
families, the maximum mean AUC deviation from 0.5 was 0.0119.

The decision artifact is
`results/m3_cross_family_preflight_v4/decision.json`; its run manifest and
artifact inventory independently verify. This licenses preparation of a
clean pre-randomness seal only. It does not license the V4 confirmation claim:
the 32-repeat external-randomness run in
`configs/m3_cross_family_confirmation_v4.json` remains unopened. In
particular, the ARCH proper gain is positive but extremely small and remains
a legitimate formal-confirmation failure point.

## 7. Mathematical lineage and contribution boundary

The component mathematics are established rather than invented here:

- characteristic kernel mean embeddings justify representing a distribution
  by an injective RKHS object
  ([Simon-Gabriel and Schölkopf, 2018](https://jmlr.org/papers/v19/16-291.html));
- conditional mean embeddings admit a regression interpretation, while their
  operator form requires explicit assumptions
  ([Park and Muandet, 2020](https://proceedings.neurips.cc/paper/2020/hash/f340f1b1f65b6df5b5e3f94d95b11daf-Abstract.html));
- probability currents formalize time-reversal asymmetry in irreversible
  Markov systems
  ([Kaiser et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6566214/));
- VAMP/Koopman methods estimate nonlinear dynamical operators from observables
  and support held-out variational comparison
  ([Mardt et al., 2018](https://www.nature.com/articles/s41467-017-02388-1)).

M3-V4's candidate contribution is not any one of these theories. It is the
registered synthesis: construct mutually low-order-indistinguishable author
worlds from several mathematical families, recover each with one
generator-blind battery, require independent replicate geometry and proper
held-out gain, and refuse exact observational aliases. A passed synthetic
confirmation would establish this existence-and-measurement result only.
