# GRAIN: observation equivalence under compound role edits

2026-10-10. This document derives the HM79 design before interpreting its new
readings. The algebra is established finite-group/factorial mathematics;
the empirical question concerns its consequences for GRAIN evidence units.

## 1. What is and is not being measured

The present observable is a frozen reader's response to an account. Its input
contains focal behaviour x, companion behaviour y, companion actor g and an
unrelated notebook actor h. This is not a human trait variable or a new scale.
An author measurement later aggregates such qualified observations; it must
not silently equate a compound role edit with an isolated psychological cause.

HM68 observes only the subset S={(x,y,g,h):h=-g}. It can establish effects of
an actual balanced actor exchange. It cannot identify separate g and h effects
without an extra assumption or extra observations. Numerical exactness on S
does not resolve this underdetermination.

## 2. Finite character algebra and the observation kernel

Let G=(C2)^4, with coordinates x,y,g,h in {-1,+1}. The algebra A of real
functions on G has dimension16 and orthogonal characters

    chi_abcd=x^a y^b g^c h^d,  a,b,c,d in {0,1}.

Equivalently A=R[x,y,g,h]/(x^2-1,y^2-1,g^2-1,h^2-1).
The observation map R restricts a function to S. It has rank8 and an
eight-dimensional kernel, generated as an ideal by (1+gh):

    ker(R) = (1+gh) A.

Indeed e+=(1+gh)/2 is the indicator of the unobserved half h=g; a function
vanishes on S exactly when it equals e+ times itself. This proves the kernel
claim without assuming the true reader response is low-degree or linear.
The complementary idempotent e-=(1-gh)/2 selects the historical half.

Therefore old observations specify an equivalence class [F] in A/ker(R),
not a unique function F on the full role design. Better arithmetic, a higher
correlation to the same scores or zero source residual cannot choose a
representative of this class. The foldover reads the complementary idempotent
sector, supplying new distinctions instead of improving the old readings.

This is an exact statement about sixteen designed inputs. It is not a theorem
that arbitrary natural text admits four sufficient coordinates.

## 3. Alias map and the meaning of an ownership gate

Because h=-g, restriction gives

    R(chi_abcd)=(-1)^d x^a y^b g^(c+d mod2).
    beta_abc=alpha_abc0-alpha_ab(c xor1)1.

In particular

    beta_xyg=alpha_xyg-alpha_xyh,
    beta_xy=alpha_xy-alpha_xygh.

Thus a three-way old gate mixes two independent assignments. A fourth-order
full interaction can also appear as an ordinary second-order old interaction.
It is inaccurate to say that the old experiment observed every component of
the full relation merely because its eight-cell expansion was exact.

At fixed assignments, the x-by-y difference is

    M(g,h)=4(alpha_xy+g alpha_xyg+h alpha_xyh+gh alpha_xygh).

The companion and anchor role edits are independent finite-difference
operators on the complete design. Their context dependence is measured by
alpha_xygh. This does not require one physical vector axis per coefficient:
coefficients can describe the observation function rather than a stored state.

The old balanced exchange preserves actor counts, while single-role edits
alter name exposure. Consequently nonzero alpha_xyh may reflect attention,
recency, name exposure or entity context. The experiment observes and separates
assignments; it does not automatically identify those submechanisms.

## 4. Connection to relative author measurement

For a declared evidence composition P_u on G, the target readout is

    Theta_u,q = E_(P_u)[F_q]
              = sum_J E_(P_u)[chi_J] alpha_J,q.

It has two distinct ingredients: the author's evidence composition, and the
common reader's observation law. The reader's alpha is not the author's own
psychological response parameter. An observation unit must retain the roles
and relations required by the chosen law before author-level aggregation.

The predefined P+/-=(1+/-epsilon xygh)/16 pair preserves every marginal of
order at most three, but

    Theta_plus,q-Theta_minus,q=2 epsilon alpha_xygh,q.

If actual reader observations give a nonzero term, the pair demonstrates a
finite relational consequence omitted by lower-order author summaries. It is
not a pair of measured people and is not a new human-personality correlation.
Signed cross terms in the old alias account must remain; there is no unique
nonnegative percentage of psychological information attributable to each role.

## 5. Development decision and lineage

Independent assignment is a prerequisite to deciding what GRAIN should store
as evidence identity, contextual input and their relation. It is mechanism
development, not a demand that an observer reach perfect accuracy.

NIST's [aliasing](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3343.htm)
and [foldover](https://www.itl.nist.gov/div898/handbook/pri/section3/pri338.htm)
sections are the direct design precedent. Group characters, restriction maps,
quotient algebras and idempotents are standard mathematics, not original GRAIN
theorems. [Causal abstraction](https://jmlr.org/papers/volume26/23-0058/23-0058.pdf)
provides the related requirement to declare what high-level intervention a
low-level edit is intended to represent. No DAS fitting is performed here.

OpenAI/math's metric/connection inverse-problem introduction provides a
research-design analogy: first characterize observation ambiguities and then
add distinguishing observations. Its elliptic uniqueness result is not a
text theorem. Complete acquisition and alternate calculation are now verified:
2880 fresh forwards, 2880 exact reuses, 708.462 model seconds, no fitting.
All-query score RMS is .098332/.087543 for 8alpha_xyg, .030570/.023407 for
8alpha_xyh and .028326/.017660 for 8alpha_xygh across the two families.
Corresponding centered-logit joint RMS .087904/.070753 shows that the joint
relation is already present before probability conversion. These are designed
reader-response units, not trait correlations or information percentages.
The prescribed P+/- differences have score RMS .003541/.002207 with unchanged
lower-order marginals. Keep contextual actor assignment in the evidence unit;
the name-exposure and role-structure submechanisms still require a new design.
The full result report is in the parent project, execution/
ROLE_ASSIGNMENT_DEALIASING_REPORT_20261010_ZH.md. Frozen primary files are intact.

The original 18/100/44 scope, natural-text observation-unit transport, author
psychological interpretation, personal response operators and dynamic paths
remain separate work. This finite algebra does not close them by definition.
