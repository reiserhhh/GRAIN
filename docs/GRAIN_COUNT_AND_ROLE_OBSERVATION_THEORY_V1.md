# GRAIN: count and role relations in an observation law

2026-10-10 JST. HM80 mathematical companion, written during frozen acquisition
and completed after actual capture/alternate arithmetic verification.
No primary protocol, material, query, collector or verifier is changed here.

## 1. The mainline question is information, not accuracy

An observation law F_q maps a qualified account to a fixed target readout.
It is not the author's psychological response law. Improving a numerical
approximation to F_q does not by itself discover a personality mechanism.
The development question is which distinctions an evidence unit must retain
before it can support author comparison.

HM79 showed that independently assigning the companion and notebook actors
changes mixed-content observations. However, with two binary actor slots,
g,h in {-1,+1}, the count of Person A events is N_A=2+(g+h)/2 and

    gh = (N_A-2)^2-1.

Therefore a joint gh coefficient need not distinguish relation structure from
count curvature. More precise observations on that support cannot remove
this alias. HM80 adds an independently assigned third actor i, rather than
fitting the old response more closely.

## 2. The finite observation space

Let D={-1,+1}^5 with coordinates (x,y,g,h,i). The first two coordinates
select two fixed behaviours. The remaining coordinates assign the actors of
three fixed event slots. The focal event always belongs to Person A, so

    N_A = (5+g+h+i)/2.

The observation space V=R^D has dimension32. Under the uniform designed
measure, the signed characters chi_J=product_(j in J) j are orthonormal.
Every function, linear or nonlinear, has the exact finite expansion

    F_q = sum_(J subset {x,y,g,h,i}) alpha_J,q chi_J.

This is established finite Fourier/factorial mathematics. It is an expansion
on the declared inputs, not a claim that five variables suffice for language.
Vector observations obey the same expansion coordinate by coordinate.

## 3. Count-compatible and role-indexed components

S3 acts by permuting the designed role addresses g,h,i while holding x,y
fixed. No equivariance of the neural model is assumed. Define

    P_count F = (1/6) sum_(sigma in S3) rho(sigma) F,
    F_count = P_count F,
    F_role = F-F_count.

The orbits are exactly the classes (x,y,N_A). Thus P_count is also uniform
conditional averaging given (x,y,N_A). It is self-adjoint and idempotent.
Its range has dimension16 and its orthogonal complement has dimension16.

    E[F_role | x,y,N_A] = 0,
    <F_count,F_role> = 0,
    Var(F) = Var(F_count) + E[F_role^2].

The last equality uses E[F_role]=0. It is a decomposition of observed
variation under a designed measure, not psychological information fractions.
Both components remain part of the observation law.

## 4. An exact limitation of this binary role design

S3 has trivial, standard and sign irreducible types. Its central projectors
on this particular V are

    P_trivial = (1/6) sum_sigma rho(sigma),
    P_standard = (2/6) sum_sigma (fixed_points(sigma)-1) rho(sigma),
    P_sign = (1/6) sum_sigma sign(sigma) rho(sigma).

Here rank(P_trivial)=16, rank(P_standard)=16, and P_sign=0.
For the last statement, any binary triple contains two equal entries. The
transposition of those entries fixes the input. A fully alternating function
must then be its own negative at every input, and must vanish.

Consequently F_role lies entirely in standard isotypic copies. This does
not mean it is one two-dimensional psychological factor: there are multiple
copies, content conditions and output channels. Nor does it imply that an
LLM internally implements these representations. It characterizes the input
design's observation space. Seeking an alternating component here would be
an impossible design task, not an accuracy problem.

## 5. Pair relations versus nonlinear count curvature

At content mask xy, keep all three pair coefficients

    a = (alpha_xygh, alpha_xygi, alpha_xyhi),
    a_count = mean(a) (1,1,1),
    a_role = a-a_count.

Their common component is count-compatible because

    gh+gi+hi = ((2N_A-5)^2-3)/2.

Any observation depending only on (x,y,N_A), including an arbitrary nonlinear
one, has a_role=0. Unequal pair coefficients therefore reject total-count-only
explanation for that component on D. This is stronger than finding arbitrary
role variation: main role effects may exist while this joint component is
count-compatible.

Ordered event positions remain different in count-matched inputs. A nonzero
a_role can include position-sensitive attention and presentation effects.
The immediate conclusion is a required distinction in observed accounts,
not an isolated semantic cause or a measured human trait.

## 6. Consequence for author-relative measurement

For an author's declared evidence composition p_u on D,

    Theta_u,q = sum_d p_u(d) F_q(d)
              = sum_(x,y,n) p_u(x,y,n) F_count,q(x,y,n)
                + sum_d p_u(d) F_role,q(d).

The count-compatible component retains composition/choice information.
The role-indexed component retains within-count arrangement information.
Neither is automatically nuisance. Orthogonality at the reference design
does not make either component psychologically valid or irrelevant.

For two distinct role edges e1,e2 in {gh,gi,hi}, the frozen HM80 hypothetical
compositions are

    delta(d) = xy(e1-e2)/2,
    p_plus/minus(d) = (1 +/- epsilon delta(d))/32, epsilon=0.5.

They preserve the whole distribution of (x,y,N_A) and every marginal of
degree at most three. Yet their observation contrast is

    Theta_plus,q-Theta_minus,q
        = epsilon (alpha_xye1,q-alpha_xye2,q).

This is a finite consequence for author-summary design. These compositions
are not people. The three edge-pair contrasts are linearly dependent, not
three independently discovered traits. Real author compositions and their
psychological interpretation are separate empirical inputs.

## 7. Nonuniform populations and natural text

Uniform group averaging is orthogonal only under an invariant reference
measure. Do not apply this uniform identity to a nonuniform natural corpus
and silently call its remainder noise. For a frozen reference measure mu,
the appropriate orthogonal count projection is instead

    P_mu F(d) = E_mu[F | x,y,N_A](d).

It is defined on classes with positive reference mass. Empty classes remain
unobserved, rather than receiving fabricated observations. It need not equal
the Reynolds average when mu is not invariant. The scoring reference,
evidence support and transport assumptions must be declared separately.
Changing the author under assessment must not refit this reference.

The present complete design identifies its observation law without missing
cells. Natural text does not provide counterfactual readings for every role
assignment. A source-built joint evidence map and cold scoring are therefore
the next development interface, not a free result of this exact expansion.

## 8. Mathematical precedent and what the experiment can add

Characters, conditional expectation, Reynolds projection and S3 isotypic
decomposition are standard tools. They are not renamed as original GRAIN
theorems. The novel empirical candidate is whether this frozen observer has
role-indexed joint action, and whether retaining it changes useful author
comparisons beyond ordinary content/count summaries.

OpenAI/math's [isotypic and coefficient-integral discussion](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Signed-tensor-densities-and-diagram-budgets-for-coordinate-sweeps-September-26-2026/build/sections/03b-isotypic.tex)
was reread at commit adc7f1241b42e322a6451854ab7e4b4c146bf78a, especially
lines72-140 and206-244. It motivates retaining separate sectors and declaring
projection domains. Its signed-tensor density bounds, overlap inequalities
and moment recurrences are not transferred to text or personality.

HM80 follows one frozen finite battery. It does not require a near-perfect
reader, a parameter grid, another peripheral actor, or a hidden-state factor
name before developing the next evidence-unit interface. Original18/100/44,
natural author transport, personal response operators, psychological meaning
and independent LLM-free scoring remain open at their actual scope.

## 9. Actual finite result and development decision

All5760 new forwards completed in1481.770 cumulative model seconds, with
zero retries/fits. The independent count-class and S3 calculations agree;
all2880 same-count input-token bags and8640 one-token role edits were checked.
An alternate explicit-sum verifier checked63,882,066 scalar comparisons,
with maximum error8.882e-15. This is same-agent alternate arithmetic, not
third-party replication. The focused41 tests and Ruff/format checks pass.

Across all three fixed query roles, scaled pair-coefficient RMS (8alpha) is:

| Stage | Family0 count | Family0 role | Family1 count | Family1 role |
| --- | ---: | ---: | ---: | ---: |
| Native score | .00904859 | .00917505 | .00760940 | .00756222 |
| Centered logits | .04830380 | .04519133 | .03642359 | .03719620 |
| Query endpoint | .00268113 | .00240986 | .00225545 | .00232716 |
| Body endpoint | .00785011 | .01188657 | .00698932 | .01137022 |

Stage units are different and must not be added or compared as information
percentages. Role-indexed pair action is present before probability conversion
and is comparable to count curvature in this pair sector. It is not all of
the observation: the whole score role component accounts for4.09%/5.19% of
designed variation, while body role variation is49.96%/50.90%. Those are not
personality signal fractions.

The540 prescribed profile contrasts are all nonzero above1e-6 in score units.
Their family-specific RMS is .000993228/.000818635, with identical full
content-by-count distributions and all degree<=3 marginals. This is a small
finite scoring consequence, not a practical prediction improvement. The
contrast identity's maximum error is1.776e-15. Counts and relations both stay
in the evidence unit; lower-order summaries cannot determine this contrast.

Do not append another count-control or source-linear fidelity grid. Develop
the inherited source-built joint evidence map and cold author observation
interface. The current full table does not itself provide that interface.
Position-sensitive attention, new-material transport and psychological meaning
remain distinct questions, not reasons to discard the measured component.
The parent report is execution/COUNT_VERSUS_ROLE_INTERACTION_REPORT_20261010_ZH.md.
