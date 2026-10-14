# Source observation channels and author evidence integration

2026-10-13 JST. HM234, canonical author_observation_channel_r1.
This is a finite measurement-construction account, not a new operator theorem
or a psychological accuracy result. Original18/100/44 requirements remain OPEN.

## 1. The measurement object includes the integration law

HM233 supplied one-event source meanings on current author events. HM174 had
already constructed three integration laws. HM234 joins them on the SAME24
authors/288 comments/32 comparison edges/five target actions. The main object is

    score_q = F_q(history; observation_family, source_reference, integration_law)

The integration law cannot be omitted from the measurement specification.
Changing it with the vector observations fixed can change cross-target
relationships and the direction of meaning/action contacts. These are not
necessarily a common gain or an improved approximation of an LLM answer.

## 2. Exact one-event round trip

Choose a SOURCE author U uniformly among n386, then observe cell A with law
P_u(a). The SOURCE target function is g_q(u)=Y_uq-pi_q, where Y is the already
serialized binary SOURCE outcome. SOURCE-only calibration and panel exclusion
are inherited; no new raw questionnaire or application outcome is used.

    R(a)=n^(-1) sum_u P_u(a)
    (Tg)(a)=sum_u P_u(a)g(u)/(n R(a))
    (T*f)(u)=sum_a P_u(a)f(a)
    B=T*T=P diag(R)^(-1) P'/n

The adjoints use the uniform-author and R-weighted event inner products.
For the old field W_q=psi_q-pi_q, the source return score is exactly

    theta_q=P W_q=B g_q.

For any author function g, Jensen's inequality gives

    0 <= <g,Bg> = ||Tg||_R^2 <= ||g||_U^2
    ||Bg||_U^2 <= <g,Bg>.

B is symmetric, nonnegative and row-stochastic. Its centered nontrivial
eigenvalues are principal inertia components. There is no PSD clipping,
inverse, rank selection or fitted gain. Tiny floating negative eigenvalues
are retained as arithmetic values, not named negative information.

This corrects a tempting but invalid use of the tower property:
E[E[Y|A]|U] does not in general equal Y(U), because these sigma-fields are
not nested. Under the declared empirical iid resampling law, infinitely many
events make their averaged one-event score converge to Bg(U), not necessarily
g(U). Sampling variation can vanish while the round-trip difference remains.
This is not a theorem that all natural-text personality inference is limited
by the current eight-cell channel. The channel uses just coordinate0/5 cuts
and a binary condition, not all GRAIN15 or the full embedding.

## 3. Observation design versus numerical precision

The two historical windows have17 eligible observations each. Define
Q_u(a,b)=P0_u(a)P1_u(b), corresponding to independently selecting one record
address from each fixed, disjoint catalogue. Returning the first or second
cell with probability1/2 is an author-independent Markov kernel G, with

    QG=P, where P=(P0+P1)/2.

Hence U -> pair -> single forms a Markov chain under THIS design. Conditional
Jensen gives B_pair-B_single positive semidefinite: for every g, the pair
posterior has at least the single posterior's squared norm. It does NOT give
a Loewner inequality for B_pair^2-B_single^2, or guarantee better application
scores, target semantics, human stability or population validity.
Selector independence does not establish independence of natural occasions.
The same construction can also retain more accidental permuted-label
alignment. Richer information alone does not assign psychological meaning.

## 4. Three previously constructed, now connected author laws

All three share exactly the same one-event probabilities psi_q(a):

    M_mean,q(n) = sum_a n_a psi_q(a)/N
    M_iid,q(n) = pi_q product_a rho_q1(a)^n_a /
                [pi_q product_a rho_q1(a)^n_a
                 +(1-pi_q) product_a rho_q0(a)^n_a]
    M_shared,q(n) = sum_u Y_uq product_a P_u(a)^n_a /
                   sum_u product_a P_u(a)^n_a.

In the middle rule, conditional independence is imposed at the binary target;
in the last, one common SOURCE author index links the events and five labels.
The empirical author catalogue is not a fitted number of psychological types.
No rule is selected using native answers; source support failures would remain
missing. Equal-cell distinct events count, repeated event IDs are rejected.

Choice is retained. Factoring P_u(a)=P_u(c)P_u(a|c) yields

    L_u(n)=L_choice,u(n_c) L_within,u(n).

Even when two removal arms have identical condition counts, L_choice,u is a
common factor across ARMS, not necessarily across SOURCE authors. It reweights
the reference author distribution before normalizing the within-condition
posterior, and cannot generally be cancelled out of the author mixture.
In contrast, HM233's additive choice-score difference is exactly zero for
those matched arms. This is an algebraic consequence of the chosen laws, not
a separate causal estimate of the psychological role of topic/situation.

## 5. Actual SAME-event meaning connections

For each integration rule m, form edge difference X_et^(m) from full histories
and contract it with the fixed model action T_eqr:

    K_tqr^(m) = mean_edge X_et^(m) T_eqr.

Keep three independent indices: SOURCE meaning t, reader target q and encoding
target r. Full, source-discrimination and reader/source-interaction arrays
are retained for both actual and the ONE old permuted source law:2250 entries.
Native/full-body-increment/deletion each retain complete5x5 matrices.
Units are probability contrast times model nats, not Pearson correlation.

Actual E/N author-score cosine is -.525668/- .541533/- .020236 for mean/iid/
shared. For native same-name N contact, the signs are negative/negative/
positive (-.000488/- .005632/+ .013038). But shared-author removal E and N
contacts turn negative (-.003175/- .005423) from positive mean contacts.
Thus the common-positive-gain explanation fails on this fixed material;
the shared-author law changes the relation between levels and interventions.
It is not a universal rescue: A/C/O native contacts remain negative, and the
permuted shared-author law also has positive E/N contacts. Interpretation must
remain tied to SOURCE law and actual observation operation.

## 6. Source channel accounts

The largest centered component is .389389(single) versus .604747(pair).
All64 pair cells are supported; garbling replay error1.665e-16; refinement
operator minimum eigenvalue-5.481e-17 retained. This is one fixed design
comparison, not an optimization of those numbers.

SOURCE posterior-field energy divided by centered label energy:

| Target | Single event | Two-window pair |
| --- | ---: | ---: |
| E | .0091994 | .0205394 |
| N | .0019273 | .0135645 |
| A | .0077760 | .0206028 |
| C | .0037144 | .0134162 |
| O | .0105149 | .0208741 |

These are empirical SOURCE calibration quadratic forms, not fractions of true
personality information or held-out explained variance. The permuted rows
also increase; actual single N is smaller than its permuted counterpart.
The ratios locate a coarse observation-family bottleneck without declaring
natural-text information absent or improving prediction accuracy.

## 7. Execution and mathematical provenance

SOURCE bundle and all input/code/result seals pass. Cold scoring precedes
action loading, all240 histories are product-replayed, all2250 tensor entries
loop-replayed, inherited HM233 means match within8.015e-16. Alternate maximum
2.331e-15;42 focused tests/3Python lint-format pass. Same agent, not a blind
third-party replication. No new model/API/training/private/application-label
call; historical anonymous geometry still includes17 panel authors. The source
raw/application cue-mask difference is preserved, not solved by this packet.

Calmon etal2017 Section1.3 Eq4 and Section2 supply the adjoint/spectral theory:
https://arxiv.org/html/1704.00820 . Raginsky2011 SectionIII-A supplies explicit
Markov-kernel comparison language, following Blackwell1951/1953:
https://maxim.ece.illinois.edu/pubs/raginsky_ISIT11.pdf . This turn read those
primary sections; the Blackwell1953 original site exposed only an iframe, so
do not claim its complete proof was newly read.

OpenAI/math adc7f124 Gaussian-fiber sections/06-fiber.tex1-150 supplied the
question of information relative to an observation family; its Gaussian/
sphere/memory hypotheses are not transferred. HM174's empirical multi-event
laws remain prior project work, not rediscovered inventions.

## 8. Next scientific decision

Mean-score attenuation and integration dependence are now distinguishable
without improving neural readout precision. The next missing construction is
a SOURCE-supported joint observation law on actual condition/occasion/event
relationships, versus the catalogue-iid surrogate, on qualified author units.
Inventory HM170/174/177/188/214 sequence/occasion interfaces BEFORE designing
another finite experiment. Add a genuinely different observation prediction,
not a longer history, stronger posterior, spectral inverse or accuracy grid.
Preserve choice/state/dynamics and previous successes/failures. This packet
does not close the full psychological measurement objective.
