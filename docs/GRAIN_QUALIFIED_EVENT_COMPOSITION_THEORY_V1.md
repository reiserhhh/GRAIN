# Qualified event composition: encoder, observation and author law

2026-10-10. HM110 finite construction. This extends the observation interface,
not the psychological names/validity of six declared model queries. The original
18 protocols/100 cards/44 computational questions remain open.

## 1. The object to construct

For a fixed context c, define an encoder E and declared observation F_q:

    X0 = E(mid,mid;c)
    Xi = E(x,mid;c); Xj = E(mid,y;c)
    Xij = E(x,y;c)
    R_ij = Xij - Xi - Xj + X0.

In HM110 X is current-token K/V after blocks0--16 with common tokens/positions.
The context includes actor qualification, other four mid behaviors, material
bank and the current carrier. Mid means a particular described behavior, not
absence or a psychological zero. All term definitions depend on that anchor.

Three constructions must not be conflated:

    A_q = F_q(Xi) + F_q(Xj) - F_q(X0)
    B_q = F_q(Xi + Xj - X0)
    Y_q = F_q(Xij).

A is the independent-read composer; B is the single-state composer followed
by nonlinear attention; Y is an actual jointly encoded record. Both A and B
are source-constructible without reading source joint records. B's argument
is a numerical counterfactual, not necessarily a realizable natural sentence.

## 2. Exact composition account

For every corner and target,

    Y_q-A_q = (B_q-A_q) + (Y_q-B_q).

The first term is the declared reader's interaction on constructed states.
The second is its response to the encoding remainder R_ij. Smooth F permits
the exact formulas, with Delta_i=Xi-X0 and Delta_j=Xj-X0:

    B_q-A_q = integral_0^1 integral_0^1
      Delta_i^T Hess F_q(X0+s Delta_i+t Delta_j) Delta_j ds dt,

    Y_q-B_q = integral_0^1
      grad F_q(Xi+Xj-X0+t R_ij)^T R_ij dt.

These are standard multivariable calculus, not newly discovered theorems.
They describe the declared function, not unique causal paths inside the LLM.
Anchored decomposition has explicit prior theory in Kuo et al. (2010),
Theorem2.1/Example2.3:
https://web.maths.unsw.edu.au/~fkuo/pubs/preprint/ksww09-decomp.pdf .
Only pp0--5 and the start of p6 were read in this packet. No general dependent
ANOVA, RKHS theorem or continuous human-trait law is inferred here.

The numerical experiment checks the signed identity directly, not by tuning
integral grids. RMS terms do not sum. Their inner products may be negative;
signed alignment is not a fraction of psychological information.

## 3. The author consequence is a joint law, not another factor name

At fixed ownerA, the four corners are x,y in {-1,+1}. Every conditional
observation on this support has an exact representation

    Y_q(x,y;A,c) = b0_q + bi_q x + bj_q y + bij_q xy.

Hence an author's evidence expectation contains

    S_uq(c) = b0_q + bi_q E_u[x|c] + bj_q E_u[y|c]
                       + bij_q E_u[xy|c].

Two author evidence laws can have the same individual marginals and different
joint moment E_u[xy|c]. HM110's P+/-=(1+/-.4xy)/4 are positive and give

    S_+q-S_-q = .8 bij_q.

For A, bij_q=0 identically. For Y, it is observed rather than stipulated.
Thus improving marginal averages cannot recover this contrast. The missing
quantity is evidence co-occurrence within a qualified unit, not a new trait
axis and not a reason to erase topic/condition means. Across conditions use
the explicit author selection measure P_u(c), not attention partition mass.

With owners crossed, bij_q(A)=a_110+a_111; bij_q(B)=a_110-a_111.
The 111 term tracks ownership-conditioned composition in this design. B rows
are not evidence about A. Actor-name/position/semantic effects still coexist.

## 4. Executed results and construction decision

Canonical `results/grain_math_response_v1/qualified_composition_r1`:
292 actual prefixes,41.617 modelseconds,26 source anchor captures frozen before
application forwards. No source pair outputs enter the single-state composer.
Two cold numeric subprocesses score60 designed20-event laws (not60 humans).

| Declared readout units | Source workshop | Application material |
| --- | ---: | ---: |
| Total Y-A RMS | .075650090 | .086558050 |
| Reading B-A RMS | .019186533 | .016866862 |
| Encoding Y-B RMS | .073655588 | .084389035 |
| Actual author joint-law contrast RMS | .016745795 | .018998168 |
| Independent-read author contrast RMS | ~1e-16 | ~1e-16 |
| Composed-state author contrast RMS | .002695390 | .003888832 |
| Author encoding-remainder contrast RMS | .016565026 | .019020468 |

The encoding remainder has signed alignment .941821/.956270 with Y-A. This is
an anchored counterfactual account, not causal mediation or personality variance.
Even application-local single observations do not recover the actual joint
author contrast. Better single-event estimation alone is therefore not the
identified remedy on this support.

All15 pairs/six queries remain. A result-after companion shows actual author
contrast pattern cosine .076036 across banks,52/90 same-sign cells. Both
scenario and wording change. Existence of a composition defect transports;
its specific coefficients do not transport by the coarse behavior-slot names.
This is a material-specific result, not rejection of all shared response laws.

The interface must retain contextual joint observations and declared author
co-occurrence, alongside marginal levels. It must not silently merge B, count
independent reports as joint events, or claim joint records are human trait
values. HM61 supplies prior native-score interaction evidence; HM110 adds the
actual single-state construction boundary and qualified author-law consequence.

## 5. Relation to higher-math inspiration and next unresolved construction

OpenAI/math commit adc7f1241b42e322a6451854ab7e4b4c146bf78a,
metric/unitary-connection manuscript06-matrix.tex lines1--120: one fixed family
of complementary first/second observations motivates this design. Its harmonic
jets, constrained spans and PDE inverse theorem are not transferred to text.

The next construction question is not whether interactions exist (already
answered), or whether more pair tables fit a teacher. It is which actual
relational state of two qualified events determines their contextual response,
so a source-built operation can act on new evidence without copying a table.
Separate a common operation from material-specific coefficients; inventory
existing typed-relation/source-operator tests before a new bounded experiment.
Natural authors, psychological linkage and a standalone new-text encoder-free
system remain distinct requirements of the full mainline.

34 focused tests and primitive/corner/event-sum checks pass; maximum arithmetic
discrepancy4.44e-15. No model/predictor fit, API/private read, training, publication
or HM107 rescue. These checks verify this finite packet, not whole-project closure.
