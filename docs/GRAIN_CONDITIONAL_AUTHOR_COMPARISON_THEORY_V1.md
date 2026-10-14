# GRAIN: conditional author comparison from vector-valued alternatives

2026-10-11 JST, HM140. Executed finite constructor and counterexamples.
This is an application of established choice/conditional-likelihood theory,
not a newly discovered statistical theorem or a completed personality scale.

## 1. A different target from imitating an observer

HM139 calibrated mathematical functions to native LLM actions. An exact source
fit did not determine a transportable function. Here the response law is instead
DECLARED before simulation. No native LLM target is read, fitted or imitated.
The question is whether this law produces an executable relative-author score
from the frozen GRAIN dictionary, and exactly what that score identifies.

The text dictionary supplies coordinates of possible expressions. It does not
by itself supply their psychological utility or prove why a human chooses them.
The new piece is a tested vector-to-relative-choice measurement interface, not
an empirical discovery that people follow its law.

## 2. Vector menus and the conditional response law

Let g(j,0),g(j,1) be two frozen15-dimensional expression coordinates. The menu
contrast is d_j=[g(j,1)-g(j,0)]/s, where s is a source-only scalar norm. The
experimental menus are hypothetical pairs assembled from distinct public
fictional records, NOT documented opportunities available to real authors.

For known free opportunities and independent binary choices, assume

    logit P(Y_uj=1) = kappa_j + d_j^T theta_u.

kappa_j is arbitrary but shared across the authors being compared. The utility
scale is common and fixed. This is a model assumption, not a semantic property
of an embedding or an established human-personality law.

For independent counts X~Bin(n,p_u),R~Bin(r,p_ref), condition on K=X+R.
With delta=theta_u-theta_ref,

    P(X=x | K,delta)
      = C(n,x) C(r,K-x) exp(x d_j^T delta) / Z_j(K,delta),
    Z_j = sum_h C(n,h) C(r,K-h) exp(h d_j^T delta).

Derivation: the joint mass is proportional to
C(n,x)C(r,K-x) exp[x(kappa+d theta_u)+(K-x)(kappa+d theta_ref)].
Terms exp[K(kappa+d theta_ref)] and the unconditional binomial normalizers are
constant across feasible x and cancel. The author contrast remains.

    grad_delta log P = [x-E(X|K,delta)] d_j,
    -Hessian_delta log P = Var(X|K,delta) d_j d_j^T.

This cancels a SHARED response intercept in a conditional likelihood. It does
not delete topic/situation coordinates or population means from raw embeddings.
Exposure counts and author context selection are retained as separate evidence.
Different kappa values alter how much information is observed through K, even
though the law conditional on the same K is independent of kappa. Scores from
different random data sets are NOT claimed identical.

## 3. What can be measured and which coordinates remain arbitrary

All source probabilities stay unchanged under a common shift

    theta'_u=theta_u+h, kappa'_j=kappa_j-d_j^T h.

Absolute origin is therefore not identified; relative theta_u-theta_ref is.
Individual shifts in Null(D) also leave all source choice probabilities unchanged.
The measurable parameter is a relative equivalence class modulo Null(D), not
automatically a named trait or a complete author's identity.

On the actual15-coordinate dictionary,12 source menus have rank12/null3. A
unit coefficient witness changes source log odds by only9.72e-17 but changes a
new menu log-odds contrast by.361732. Source plus the12 new menus has rank15;
this is potential coverage under the SAME declared law, not proof of natural
expression transport or discovery of three new psychological factors.

The new-menu unseen energy3.1563% is a dictionary geometry fraction, not a
personality-information percentage. As in HM139, choosing zero for unobserved
directions would be a convention, not evidence that their author value is zero.

## 4. Executable scoring and its finite scope

The constructor fixes the top2 SOURCE singular directions, anonymously U01/U02,
solely for a finite2D inference demonstration. Identification uses ALL15; the
other13 are not fitted or silently assigned zero. A fixed121-atom uniform prior
supports exact finite posterior integration. It is not a hyperparameter search.

Cold input contains only menu address, known free/forced qualification, author
counts and frozen reference counts. Truth, ID, text, scales and teacher readings
are rejected. All324 declared-world configurations receive nonconstant numerical
profiles, covariance and conditional information; no winning axis/world chosen.
The reference is a simulated fixed anchor theta_ref=0, NOT a heterogeneous pooled
reference population. Future population norms require their own identified model.
Shared reference counts make author comparisons dependent; no independent-sample
population CI is claimed. These are finite-model posterior quantities.

Forced reports supply no choice trials in this model. They can still contain
style/state/other evidence in other GRAIN channels. Unknown opportunity or event
qualification is rejected, not rewritten into a free choice. If no common free
support exists, the returned posterior is the prior and the status is explicitly
NO_IDENTIFYING_CHOICE; its zero-centered prior mean is not an observed zero trait.

## 5. Exact counterexample: relative position is not its psychological source

Consider two declared worlds with the same common kappa and vectors:

    stable world: theta_u=h_u, eta_uj=0;
    personal-context world: theta_u=0, eta_uj=d_j^T h_u.
    logit p_uj = kappa_j + d_j^T theta_u + eta_uj.

Both have exactly the same probabilities. In paired simulation they also have
the same observations and every posterior atom, error0. This is not failure to
estimate a weak signal: it is structural nonidentification of its asserted cause.
The rule measures the conditional expressive contrast but cannot decide whether
to call it stable preference or author-specific condition response from these
records alone. More precise teacher imitation cannot resolve this equivalence.

Similarly exp[tau(kappa+Dtheta)] is unchanged if tau doubles and kappa/theta are
halved. Effective log-odds parameters are measurable only in a declared scale;
underlying utility magnitude and choice variability cannot be separated for free.
Coordinate rotations applied to vectors AND prior atoms preserve posterior
weights to1.07e-15. An axis name is not what gives the comparison its meaning.

## 6. Relation to existing GRAIN and next construction decision

M03/F03 already preserve opportunity/selection and supply numerical author
models; HM135 already implements choice likelihood examples. They are prior
internal results, not rediscovered here. HM140 adds the frozen GRAIN menu
dictionary, conditional relative-author scorer, full-space observability and
paired personal-context alias to a single runnable interface.

Retain this as ONE possible conditional-expression channel. Do not replace
choice composition, state, trajectory, interaction or shared/target-specific
vector mechanisms with it. Next scientific work must specify and test the
event/offer-to-vector observation law and distinguish joint response mechanisms,
not optimize this toy world's scoring accuracy or pretend native scores are
psychological truth. Original18/100/44 remain open, no parent-card promotion.

## 7. Ancestry and actual verification

Train chapter3.1--3.2: logit response and utility-scale normalization;
https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf .
Baker/Saxe/Tenenbaum2009 sections2--3/Eq1--2: conditional action-to-goal inverse
planning; https://web.mit.edu/9.s915/www/classes/cognition2009.pdf . Their goals,
MDPs and human experiments are not our expression coordinates.
Andersen1973 conditional estimation is historical ancestry; publisher abstract
only accessed, not an alleged read of its inaccessible full derivation:
https://doi.org/10.1111/j.2044-8317.1973.tb00504.x .
OpenAI/math pinned adc7f124, conditional-sweeps01-setup1--105 reread; actual
conditioning support inspires the design. No fair-coin/mixing theorem transferred.

Primary source files were frozen after60 focused tests/3 Python lint+format
passed.324 profiles/108 joint-binomial enumeration cases replay; profile error0,
conditional formula error2.00e-15, gauge error2.22e-16. Same-agent different-form
arithmetic, not independent blind reconstruction. CPU.620381s,0 new neural/API/
private/training/Git/automation/lockbox. Preserve the scientific code seals.

