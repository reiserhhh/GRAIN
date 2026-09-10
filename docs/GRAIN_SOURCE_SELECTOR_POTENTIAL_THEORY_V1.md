# Query-conditioned selection and GRAIN response are two properties of one function

2026-09-11. NEXT74, finite pass63. Standard differentiable regression and softmax
geometry applied to the same-material interface; no claim of a new generic theorem.

## 1. The actual function

For a fixed source reference S and query q, let Qbar_Sq be its source-only anchor.
Let H denote H16 token states, M the fixed token mask, and P_M remove an allowed-
token constant. The scalar logits for every head are

    ell_hat(H)=P_M ell_identity(H;Qbar_Sq)+P_M Phi_S(H) B_Sq.
    d ell_hat[D_f]=P_M J_ell_identity(H)D_f+P_M D Phi_S(H)[D_f] B_Sq.
    a_hat=softmax(ell_hat+M).

Phi has the old source rank32 state projection, its causal prefix mean and fixed
positions. Its scalar and derivative share B. P_M is applied consistently to both;
this changes no attention or covariance response. It is gauge fixing, not removal
of author choice or psychological content. The constant coefficient is pinned0.

With native projected value v_m held fixed in this *key-channel* calculation:

    S_qm(H)=sum_h E_(a_hat_h(H))[v_mh].
    D S_qm(H)[D_f]=sum_h Cov_(a_hat_h)(v_mh,d ell_hat_h[D_f]).

The actual finite differences verify this coupling. The whole neural score also
has query/value/downstream dependencies; those are not removed by this equation.

## 2. What it resolves and what it separates

NEXT73 located the importance of a query's selection measure. NEXT74 now learns
that measure from source materials, rather than only measuring its effect:
new8 KL improves25.62%/33.56%. This is a usable local computational replacement
for part of selection, retaining source queries plus H16/native V.

But proximity of function values on observed inputs does not determine derivative
proximity. Even in one dimension, epsilon*sin(x/epsilon^2) converges uniformly to
zero while its derivative can grow like1/epsilon. In this experiment primal
improvement and oversized GRAIN derivatives coexist. The example explains the
logical possibility; it is not an identification of our model's actual error.

Two fixed objectives were tested:

    E_value(B)=E_S ||P_M(ell_hat-ell_teacher)||^2 / source_value_energy.
    L_value(B)=E_value(B)+alpha ||B||^2.
    L_joint(B)=E_value(B)
             +E_(S,f) ||P_M(dell_hat_f-dell_teacher_f)||^2 / source_direction_energy
             +alpha ||B||^2.

Both fits include the same ridge penalty; the value-only arm omits only the
direction term. Current teacher keys never enter new8 application. This is a
directional instance of value-plus-derivative teaching, related to
[Sobolev training, Czarnecki et al.,2017](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html).
That work motivates derivative supervision, not a guarantee that any fixed
low-rank feature map can satisfy it.

## 3. The next mathematical distinction

For source derivative design A, residual targets Y and shared coefficient B,
the best unconstrained linear derivative approximation has error

    min_B ||A B-Y||_F^2 = ||(I-AA^+)Y||_F^2.

This is a source-space capacity statement. Compare it to the observed regularized
joint loss before blaming data quantity or selecting another alpha. The scalar
and derivative targets may also prefer different B inside the same function
family. NEXT74 did not run this projection; it must not be described as a measured
proof of missing span yet.

GRAIN supplies meaningful *directions to test*, while the selection function
supplies a query-specific measure. Neither replaces the other. Their connection
is stronger than two author scores added together: it is the response of one
query-conditioned function along an address-matched GRAIN direction. A response-
informed source coordinate basis is the next concrete candidate, not an unlimited
accuracy or factor search.

## 4. Rejoin the user's two ideas

The [same-material theory](GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md) already
separates joint local distribution, signed covariance, native token responses and
interpretation episodes. NEXT74 advances the numerical selector inside that path.
Uniform weighting, positive relevance weighting and signed response aggregation
are distinct estimands; the first two cannot automatically stand in for the third.

Psychodynamic/TA concepts can be added as independently defined *queries or coding
targets* at the interpretation interface. A high response to such a query is first
a property of the computation on that text, not a clinical fact about its author.
The [research addendum](GRAIN_RELATIONAL_INTERPRETATION_RESEARCH_ADDENDUM_V1.md)
connects this to current primary studies without starting new clinical data use.
