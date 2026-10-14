# Observed path dependence and author mixture: HM235

2026-10-13 JST. Registered secondary complete and verified; the primary remains
uncomputed after a fixed-budget entropy-reference failure. Goal active/nojob.
Original18 protocols/100cards/44questions remain OPEN. All previous evidence
and the primary failure are retained. No predictive fitting or precision grid.

## Measurement object and the new interface

HM174/234 used independent selectors from author event catalogues. HM172 had
already connected adjacent chronological states to source targets. Here the
observation unit is an actually realized consecutive triple (A,B,C) from the
fixed HM172 paths, with identities and strict timestamps inherited from HM170.
The eight states retain the recorded venue bit and two frozen anonymous GRAIN
coordinate bits. Conditions are not removed or treated as nuisance-only.

There are386 SOURCE authors, split203/183 before outcomes, with two windows
and16 overlapping triples per window. The12,352 triples are not12,352 subjects.
The120 disjoint historical held authors supply240 cold windows/3,840 triples.
Only previously sealed SOURCE binary targets are read. No held outcomes, raw
questionnaires, model answers or private text are used. Representation/cohort
reuse is historical; the windows span sampled publications, not a conversation.

For each author u, p_u(a,b,c) is the normalized observed triple table. Let g
denote the full measured five-bit profile, not an inferred psychological type:

\[
F(g,a,b,c)=\frac1n\sum_u1\{Y_u=g\}p_u(a,b,c).
\]

The registered laws are:

\[
I(g,a,b,c)=\frac{F(g,a,b,+)F(g,+,b,c)}{F(g,+,b,+)},
\]

\[
J(g,a,b,c)=\frac1n\sum_{u:Y_u=g}
\frac{p_u(a,b,+)p_u(+,b,c)}{p_u(+,b,+)}.
\]

An absent middle-state mass contributes zero to these mass laws. It does not
license a zero or neutral posterior on an unsupported observed cell. I imposes
adjacent conditional independence after pooling; J imposes it before pooling.
Neither is an identified stationary human transition law or proof that an
individual has no memory. J is an empirical finite-author comparison law.

H, the fourth registered common-path reference, would preserve all adjacent
profile margins and the pooled triple table under maximum entropy. The primary
F-H question specifically concerns profile assignment to that common path.
Its IPF run exhausted10,000 sweeps at error2.0291653554812238e-8 against the
fixed1e-10 tolerance. H and the primary are NOT_COMPUTED; no tolerance/iteration
change or rerun is performed. This numerical failure is not evidence of absent
path or psychological mechanisms. The separate secondary uses only I,J,F.

## Why pooled path dependence can contain author differences

Define alpha_u(a|b)=p_u(a,b,+)/p_u(+,b,+),
gamma_u(c|b)=p_u(+,b,c)/p_u(+,b,+), and

\[
\omega_u^{g,b}=\frac{1\{Y_u=g\}p_u(+,b,+)}
{\sum_{v:Y_v=g}p_v(+,b,+)}.
\]

Conditional author weights depend on b; equal weighting inside each b is
generally wrong even though authors were equally weighted before conditioning.
Direct substitution yields the standard mixture identity:

\[
J-I=F(g,+,b,+)\operatorname{Cov}_{\omega^{g,b}}
\bigl[\alpha_u(a|b),\gamma_u(c|b)\bigr].
\]

Thus the same observed path dependence can include covariance across authors,
not only an extra temporal mechanism inside an author. This is not a new
covariance theorem. The post-outcome descriptor verifies it against all eight
registered source tables, retaining the old permuted profile assignment.
Maximum mass error is2.6021e-18; this is an arithmetic identity, not psychological
validation or independent replication.

## Keep interactions instead of treating a remainder as noise

The laws do not have an automatically additive KL decomposition. The exact
account retains a signed term:

\[
D(F\Vert I)=D(F\Vert J)+D(J\Vert I)
+\sum_{g,a,b,c}(F-J)\log(J/I).
\]

Terms with zero J are omitted only where F is also zero; illegal support fails.
SOURCE means over the four bank/window tables, in nats:

| Assignment | Total F/I | Remaining F/J | Mixture J/I | Signed cross |
| --- | ---: | ---: | ---: | ---: |
| Actual | .608223 | .328565 | .304987 | -.025329 |
| Old permutation | .622333 | .333051 | .312700 | -.023418 |

None of these quantities is a fraction of personality, noise or prediction
error. They compare the fixed empirical tables; reducing a divergence is not
a promotion criterion. The similar permuted account also prevents attaching
target meaning to generic pooled dependence.

For target q and each law L, define psi_Lq(a,b,c)=P_L(Y_q=1|a,b,c). All5 targets
are contracted from the full profiles. Let w=psi_F-psi_J and m=psi_J-psi_I.
On the same observed support, total field t=w+m. For disjoint author banks and
different windows:

\[
\langle t_0,t_1\rangle_R=
\langle w_0,w_1\rangle_R+\langle m_0,m_1\rangle_R+
\langle w_0,m_1\rangle_R+\langle m_0,w_1\rangle_R.
\]

The fixed R is the pooled SOURCE triple mass; coverage is reported, not silently
renormalized. A missing joint support stays unknown, not zero recurrence.
These are signed probability-field products, not Pearson correlations or p-values.

| Target | Total | Remaining | Mixture | w0 x m1 | m0 x w1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E | .000974 | .000239 | .000112 | .000847 | -.000224 |
| N | .000557 | .000318 | .000355 | .000263 | -.000378 |
| A | .000073 | .000149 | -.000289 | .000176 | .000037 |
| C | .000710 | -.000643 | .000493 | .000446 | .000414 |
| O | -.000129 | -.000456 | -.000039 | .000380 | -.000014 |

C's sign change cannot be summarized as removing an independent positive
mixture factor: the cross terms matter. The remaining field is not automatically
a pure dynamic personality signal. The old permutation has larger E/N total
and remaining recurrence; it remains in every output, with no permutation search,
p-value or target selection. F-H assignment has not been substituted by F-J.

## Construction and verification

The cold secondary reader sees only the source field array and opaque held
author keys, unique event IDs, states and timestamps. It averages16 source
posterior readings per window; it does not multiply overlapping triples as
independent observations or compute a full latent-author posterior.
All240 held windows are supported for all three laws/five targets/assignments.
Support is a finite operational result, not external personality accuracy.

Primary prepare succeeded once; compute failed once. A separately sealed
secondary ran once, then cold/verify succeeded once; seal checks were repeated
without recomputing science. Direct recounting
of12,352 triples is exact; direct Markov-table error8.6736e-19; cold replay0.
The separate post-outcome descriptor changes no scientific observations or
source laws. KL/field identities agree within6.592e-17. Audits are same-agent,
not blind or third-party.35 focused tests pass. Six new Python files pass Ruff;
the sealed secondary companion retains one UP034 parenthesis-style warning.
All seven files pass format checking. No scientific code is edited after sealing.

## Literature and the next mechanism, not a precision branch

[Shalizi and Crutchfield2001](https://csc.ucdavis.edu/~cmg/papers/cmppss.pdf)
define history equivalence through future predictions. A three-event table is
not an epsilon-machine or all-future sufficient state. [Perrone et al.2024](https://arxiv.org/html/2404.04343)
motivates declaring multivariate dependence constraints; no uniform-margin
population or full copula theory is transplanted here.

[Miles and Webber2026](https://link.springer.com/article/10.1007/s11222-026-10842-8),
abstract,1.1 and2.2-2.4, explicitly relax the one-chain assumption with Markov
mixtures. Their variational EM, component-count search and classification-error
theorem are not implemented or claimed as our innovation. Our empirical J is
a finite neighbor-product reference, not their fitted stationary mixture model.

The pinned OpenAI/math adc7f124 Gaussian-fiber section1-135 is reread for
information relative to a specified observation family. Its spherical signal,
independent Gaussian rows and bounded-memory hypotheses do not hold here;
the theorem is not transferred. Covariance/KL/bilinear identities above use
standard probability and algebra, not a new high-math theorem.

The useful mainline increment is an actual ordered-event SOURCE/cold interface
and a distinction between pooled author heterogeneity, remaining path structure,
and their target-field interactions. Before a next finite plan, inventory
HM138/143/159/177/191 observation constraints and target-function interfaces:
which target-relevant history function is identifiable from the observed source,
without demanding a reconstructed full sparse profile law? Competing predictions
and a new usable reading must be declared first. No IPF precision rescue,
longer-history/rank/component/classifier grid or new model job is queued.

Canonical outputs: `results/grain_math_response_v1/observed_path_memory_r1/`;
secondary `partial_markov_accounts_r1/`; descriptor `mixture_decomposition_r1/`.
