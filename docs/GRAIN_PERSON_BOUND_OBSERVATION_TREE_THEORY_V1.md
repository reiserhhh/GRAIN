# Person-bound observation trees and qualified criterion contacts

2026-10-14 JST. Mathematical account of the frozen HM288 construction, written
while its sole collection is running. No outcome selected these coordinates.
The categorical factorization is standard probability/response-tree mathematics.
The potential empirical contribution is the connection of both observation
processes to the same real author evidence and fixed SOURCE criterion.

## 1. Evidence absence and behavioral contradiction are different observations

HM287 finds an ownership/query interaction in common item responses,and an
exploratory SOURCE A contact. That does not determine whether the reader is
endorsing behavior or finding one author's text easier to assess. A forced no
can represent a negative conclusion,an unsupported conclusion,or a response habit.
Counting it as a low behavior score mixes distinct measurement operations.

Quote-source NLI and unresolved-condition assignment already exist in GRAIN.
They address source/condition qualification,not this same-author behavioral
support/claimed-assessability observation. Ordinary NLI or abstention is therefore
not new here. Their relevance is the need for an explicit observed unavailable
state,without deleting the underlying evidence or imputing a neutral trait.

The new fixed observer law has0=contradicts,1=supports,2=insufficient evidence.
All24 historical authors,96 comments,48related dyads,50 items and eight role/order
corners stay. SOURCE F/J/D levels and all criterion components are unchanged.
The model's claimed availability is not calibrated human answerability or truth.

## 2. Exact three-category probability coordinates

For a single native endpoint,let z0,z1,z2 be selected logits. Define

\[
 d=z_1-z_0,\qquad a=\log(e^{z_0}+e^{z_1})-z_2,\qquad
 s=\sigma(a),\quad b=\sigma(d).
\]

The within-alphabet probabilities are exactly

\[
 \pi_1=sb,\qquad\pi_0=s(1-b),\qquad\pi_2=1-s.
\]

On the interior simplex this chart is invertible:

\[
 d=\log(\pi_1/\pi_0),\qquad
 a=\log((\pi_0+\pi_1)/\pi_2).
\]

Adding a common logit offset changes neither coordinate. Implementations use
logsumexp/logaddexp,do not clip rare branches,and retain the full-vocabulary
mass of the alphabet separately. These are not fractions of personality.
Formation of a is nonlinear:compute it at each native corner before role or
item averaging;the logsumexp of averaged logits is another observation.

A designed binary recording that maps insufficient into no observes only sb.
For every s in[p_yes,1],b=p_yes/s produces the same yes probability. The worlds
(s,b)=(.5,.8) and(.8,.5) both give.4 yes,but their three-category laws differ.
This is a basic identification counterexample,not evidence that actual people
or the old model used those worlds. It explains why better precision alone
cannot identify both processes from that specified binary recording.

## 3. Conditional direction can be visible only on a rare branch

For one categorical observation with coordinates(a,d),the ordinary Fisher
information matrix is

\[
 I(a,d)=\begin{pmatrix}s(1-s)&0\\0&sb(1-b)\end{pmatrix}.
\]

To see this,the a score is1-s for an answered outcome and-s for outcome2. The
d score is1-b for outcome1,-b for outcome0,and0 for outcome2. The cross moment
vanishes because b(1-b)+(1-b)(-b)=0 within the answered branch.

As s approaches0,a large conditional d remains numerically defined but carries
little information in actual non-abstaining outcomes. This is an observation
support issue,not proof d contains no author-related structure. Conversely,
local Fisher orthogonality does not imply that authors' a/d fields are independent,
uncorrelated,or separate personality factors. Both may depend on the same evidence.
This analytic illustration is not a new fitted metric or primary test in HM288.

## 4. Preserve person incidence and display coupling

For each item j and dyad u,ownership o,queried actor t and display d0 take values
in{-1,+1}. For either response field f in{direction,assessability},keep all
eight finite characters

\[
 \widehat f_{uj,S}=\frac18\sum_{o,t,d_0}
      \left(\prod_{r\in S}r\right)f_{uj}(o,t,d_0).
\]

The intended display-even person-bound field has S={o,t};the display-dependent
person field has S={o,t,d0}. Common,query,ownership and other mixed fields are
also retained. Position dependence is an observation condition,not automatically
discarded text noise. Ordinary character algebra and role-binding nulls were
already established;source qualification is the separate scientific question.

Old high/low public item keys produce both keyed direction and common responses.
For each target bank q,

\[
 K^f_{uq,S}=(\overline{\widehat f}^{+}-\overline{\widehat f}^{-})/2,
 \quad C^f_{uq,S}=(\overline{\widehat f}^{+}+\overline{\widehat f}^{-})/2,
 \quad\gamma^f_u=\frac15\sum_q C^f_{uq,\{o,t\}}.
\]

The global common field is fixed,not selected to correlate with A. Claimed
assessability can itself be author-related information;it is not removed as a
nuisance. No unavailable categorical observation is converted into a neutral
human trait score. Continuous logits and raw category probabilities both stay.

## 5. The same SOURCE law qualifies both fields

Reuse HM287's complete fixed source weights. In particular,

\[
 e_{iq}=Y_{iq}-E_{\rm SOURCE}[Y_q\mid Y_{-q},bank],\quad
 W^{F,rel}_{ui}=W^F_i(x_u)-W^F_i(x_{v(u)}),\quad
 a^{F,source}_{uq}=\sum_i W^{F,rel}_{ui}e_{iq}.
\]

Here the notation a_source is a SOURCE author reading,not the new response-tree
coordinate a. Implementations explicitly name the latter assessability to avoid
type confusion. F and J are reference-model levels;F-J is a model contrast,not
an absolute human trait or an add-comment causal effect.

For both new fields independently compute

\[
 T_{q,f}=\frac1{48}\sum_u a^{F,source}_{uq}\gamma^f_u
         =(\alpha^f)^Te_q,\qquad
 \alpha^f_i=\frac1{48}\sum_u W^{F,rel}_{ui}\gamma^f_u.
\]

All five SOURCE targets and both fields form10 predefined primary contacts.
They share the original199 conditional assignments/target and use Holm10;
the two channels are not independent experiments. Exact orbit mean0 and
variance(alpha^f)^T Vq alpha^f are retained,not interpreted as Gaussian z tests.
These signed contacts are not r/AUC or calibrated individual psychology scores.
All F/J/D,criterion composition/within/total,banks,spaces and cross-query contacts
remain. Cyclic dyads share writers;there are24 people,not48 independent cases.

## 6. Changing the question is not automatically preserving the old law

New instructions explicitly distinguish contradiction from insufficiency. This
can alter support direction itself. Therefore save

\[
 \Delta d_{uj}(o,t,d_0)=d^{new}_{uj}(o,t,d_0)-d^{old}_{uj}(o,t,d_0).
\]

No theorem identifies old binary logits with new conditional logits. Old token2
mass was not an instructed semantic uncertainty outcome and is not relabeled
after the fact. The finite new-category-to-binary merging is well-defined on
the NEW law;it does not make the OLD prompt a demonstrated Blackwell coarsening.
This boundary prevents a format/task change from being called purification.

## 7. Scientific use and stop conditions

Direction qualification can support an author-bound behavioral-reading candidate
under the new law. Assessability qualification can support an author-bound
judgment-opportunity candidate. Both together indicate linked response processes,
not independent human factors. If neither qualifies,this specific interface has
not supplied a new criterion-qualified bridge;old A/O/C findings remain at their
own operations.

Always retain contract change and display coupling. More token2,the number of
tests,or alphabet compliance is not success. There is no human answerability
gold standard here. After the fixed capture and one audit,stop this ternary/
prompt/yes/abstention/head/metric/rank/precision branch. A further package needs
a different mechanism prediction or newly usable measuring construction.

## 8. Literature and borrowing boundary

[De Boeck & Partchev2012](https://www.jstatsoft.org/article/view/v048c01) provides
response trees including missingness and agreement/certainty. Only its primary
abstract was accessible;PDF fetch failed. We do not claim a full-paper review,
fit its GLMM or infer latent human response-process parameters from model logits.
[Braun2025 Sections1/3.2](https://arxiv.org/html/2509.08480v1) studies response-
format/no bias in selected models and legal-domain tasks;those results motivate
a competing explanation,not a conclusion about our observer's cause.
[Madhusudhan2024 Sections3/4](https://arxiv.org/html/2407.16221v2) keeps answerability
separate from answer/abstention;we do not possess its QA truth labels here.

OpenAI/math adc7f124 reconstruction02-foundations236-251 supplies conditional
Jensen/degradation reasoning. Its independent tree branches and reconstruction
threshold hypotheses are not transferred. The probability chart,conditional
moments and finite role projection are mature mathematics. The empirical
contribution must be the qualified same-author evidence/response construction,
not a newly named theorem or a claim that any three-category model measures people.

Native collection remains neural. The cold numeric interface is model-free only
after these actual observations have been obtained. Original18/100/44 OPEN;
no API/private/MEPS/new training/Git/Claude/automation/lockbox/oldqueue permission.

## 9. Finite actual result and use

The sole384family/19200question capture completed1876.022882modelseconds. Native
three-category probabilities are not constant(.402491/.246748/.350761 means).
Person-bound/display-coupled RMS are.380287/.546718 for direction and
.159749/.722908 for assessability. All fields and condition coupling remain.

Primary SOURCE A direction contact+.001057533 has oldconditional rank.010 and
Holm10.100;assessability A-.000010960 has rank.985/Holm1. This retains a direction
candidate under the new law,not a confirmed trait or a formal test of channel
differences. Both SOURCE banks and all5common query descriptions are positive
for A direction. The same historical evidence is reused,not independent studies.

Contract change has person-bound RMS.200646. New A minus old A contact+.000148072
is a task-function difference,not accuracy improvement or purified uncertainty.
The next distinction is conditional behavioral meaning versus broad statement
endorsement/halo. High/low keys are not exact logical complements;inventory
existing polarity/scope/SOURCE interfaces before a new finite operation test.
No ternary/prompt/head/rank/metric/precision rescue of this packet.

Original analysis/verify failed because the inherited collector exported two
zero bias entries for a bias-free three-row head. Primary data/code stay sealed.
A separate read-only companion certifies Qwen2bias=False and weight metadata
absence,and represents the SAME absent bias by three zeros. No new model or
forward;equivalent analysis/cold/2000reference accounts pass. Canonical results
are person_bound_observation_tree_r1/bias_export_recovery_r1.30tests/4formats pass;
primary testI001 style warning and both original failures retained. Same-agent
arithmetic,not blind replication. All handles terminal/nojob;full goal remains
active and original18/100/44 OPEN.
