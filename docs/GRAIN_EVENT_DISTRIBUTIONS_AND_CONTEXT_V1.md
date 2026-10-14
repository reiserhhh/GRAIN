# Event Distributions and Context: The Observation Interface

2026-10-09. HM57 extends the HM55-HM56 interface, not their frozen results.
The object is the composition of textual evidence under a fixed local reader.
No human personality accuracy or universal psychometric theorem is claimed.

## Known geometry versus the missing observation object

For isolated event logits l_e on one common answer alphabet, define
p_e=softmax(l_e). Two standard composition laws are:

    m_w = sum_e w_e*p_e
    g_w = normalize(product_e p_e**w_e) = softmax(sum_e w_e*l_e).

The second law can move a scalar answer expectation outside the isolated
expectation interval. This ordinary categorical nonlinearity must not be
renamed as a new semantic mechanism. It has a precise local description:

    p_t=softmax(t*l_x+(1-t)*l_y), d=l_x-l_y
    d E_p[v]/dt = Cov_p(v,d)
    d^2 E_p[v]/dt^2 = E_p[(v-E_p[v])*(d-E_p[d])^2].

The two KL objectives have different minimizers. Writing m=sum w*p_e and
log g=sum w*log p_e-log Z, expansion gives:

    sum w*KL(p_e||r) - sum w*KL(p_e||m) = KL(m||r)
    sum w*KL(r||p_e) - sum w*KL(g||p_e) = KL(r||g).

This is finite categorical geometry, not a claim that the model answer law is a
calibrated Bayesian posterior about a person. These are declared event/mention
weights, not independent-likelihood products or estimated precision weights.
References: [Nielsen 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7650632/) and
[Bandyopadhyay & Chung, III-B](https://arxiv.org/html/1403.3117v3).

## Sufficiency for reproducing a reader is a fiber condition

Let S(H) contain chosen unordered isolated-event/mention distributions, and
F(H) contain whole-account target readouts. A deterministic T with F=T o S
can exist on a finite material set only if:

    S(H1)=S(H2) implies F(H1)=F(H2).

Necessity follows by applying T. Sufficiency on that finite set follows by
assigning one value to each image of S, if every fiber has one F value. This
is a standard factorization statement; it is not a new GRAIN theorem.

If two accounts in a fiber have different readouts a,b, every common readout
c obeys:

    (||a-c||^2+||b-c||^2)/2
      = ||a-b||^2/4 + ||c-(a+b)/2||^2.

Thus the best paired profile RMS is ||a-b||/2. More flexible functions of the
same S cannot evade it. This is an observation-information bound, not a
psychological error floor. Two fixed pools failing alone would not exclude
other nonlinear pools; an identical full summary with different outputs does.

## What HM57 actually adds

All 1296 HM56 raw readouts were replayed without another forward or fitting.
Entire ten-answer laws, not only six means, were compared. Qualified own event
distributions stay identical for same-event repetition, other-person additions
and order reversal. Own mention distributions stay identical between repeated
and distinct occasions, other-person additions and order reversal.

The corresponding native profiles change in every paired case. Both event
and mention summaries stay identical for other-person additions and reversal.
Keeping uncertainty shape alone therefore does not recover the event identity,
ownership/context or order that the declared summary has forgotten. This
statement is about these specified isolated-event distributions; it does not
exclude contextual embeddings or richer relational GRAIN representations.

The RMS distances of fixed pools sometimes improve and sometimes worsen.
They are not the promotion criterion. N-B=(G-B)+(N-G) has a signed cross term
in squared loss; no fraction is labeled an explained personality percentage.

## Consequence for a text-measurement base

Retain three objects rather than treating every mismatch as noise:

    E(H): qualified current events with actor, occasion and claim relations
    M(H): own descriptive mentions, emphasis and report history
    C(H): order, other-person context, choice and condition relations
    F_q(H)=T_q(E(H),M(H),C(H)).

T_q remains unidentified. E and M are not interchangeable. C can influence a
reader without another person's action becoming the focal person's action.
Context-sensitive author expression may eventually be measured separately;
it is not automatically an individual trait or nuisance to remove.

The next mechanism test should determine whether the event-identity contrast
survives consistent occasion-label renaming. A difference based on equality
relations should survive their preservation; a specific digit/code association
may not. A finite group action and its invariant projection can separate these
objects without teaching a predictor to imitate the reader. This is a plan,
not an executed group-invariance result. Original full research scope is open.
