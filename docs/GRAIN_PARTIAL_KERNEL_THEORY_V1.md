# Partial observation with a shared unknown response law

HM290,2026-10-14 JST. One completed approximate numerical construction and
finite natural-history test;original18/100/44OPEN. No new SMC/personality theorem.

## 1. The newly executable object

HM281 already integrates the unknown SOURCE transition law on a complete path.
HM282 marginalizes hidden bits only after fixing each SOURCE law at its mean.
HM290 adds a usable approximation to the genuinely different object

    L_i(O)=E_(mu,K|SOURCE_i) sum_(x compatible with O)
              mu(x0) product_t K(x_(t-1),x_t).

One kernel is shared inside a path and across the declared episodes. Replacing
it by its mean first generally changes L. For a uniform Beta row,the complete
path000 with a uniform two-state start has probability1/6,not1/8. This known
Dirichlet-moment example is a software/model distinction,not a new theorem.

The independent observed outcome is the next inherited vector-state of an actual
public comment,not an LLM's opinion about an author. The states are coarse venue
and anonymous GRAIN-coordinate codes;they are not independently named actions,
emotions or true personality types. Upstream encoding remains neural.

## 2. The sufficient memory is a distribution of counts

Given a SOURCE component i and hidden path statistic (a,n0,N),the next complete
state b has collapsed probability

    q_i(b|a,N)=(alpha_i,ab+N_ab)/(alpha_i,a+ + N_a+).

For an observed allowed set M,the probability is m_i=sum_(b in M)q_i(b).
Resample predecessor statistics by m_i,then draw b from q_i restricted to M,
then increment its count. Unknown kernels are analytically marginalized through
conjugacy;the particles represent unknown hidden paths/counts,not new kernels
drawn independently at every event. SOURCE identity stays attached to each row.

The fixed-budget estimator is

    Lhat_i(O_1:T)=product_t mean_r m_i,r,t.

Likelihood estimation and normalized posterior approximation are different:
even when a standard likelihood estimator has the usual unbiasedness property,
log likelihoods and normalized SOURCE beliefs need not be unbiased. No such
finite-particle unbiased posterior claim is made here. Completely observed paths
have deterministic counts and reproduce the existing exact beta-ratio law.

An ordinary HMM current-state marginal is insufficient when the same unknown
K is reused: future probabilities also depend on the hidden count distribution.
This is a required measurement-state variable,not a new intrinsic author factor.

## 3. Cold SOURCE-linked measuring rule

Keep both old SOURCE assignments and all32 joint profiles. At each observation,

    W_g=sum_(i:code_i=g) w_i,
    theta_q=sum_i w_i Y_iq,
    departure_q=theta_q-SOURCEmean(Y_q).

SOURCE coordinates stay fixed while belief moves;this is epistemic updating,
not automatic human trait or emotional-state change. Scores reference a compiled
catalogue of expression modes,not a literal identification of the new author
with an old SOURCE participant. All condition choice and expression remain.

Input:identified/timestamped frozen vector-state episodes plus an immutable
SOURCE bundle. Output:all joint profiles,five marginals and next-state laws.
No neural inference,LLM judge,current-author labels or batch standardization is
needed for the scoring/update layer. The encoder/cell construction is a separate
dependency,so this is not wholly non-neural raw-text processing.

## 4. Natural conditioning versus SOURCE-conditional design

Six old contracts remain: none,F,C,S,SgivenC,CgivenS. F/C/S use the integrated
evidence (C/S approximated). SOURCE-conditional design uses L_i(F)/L_i(C),or
L_i(F)/L_i(S),with the original source reference rather than naturally reweighted
reference counts. Within i,the fully observed F updates the kernel counts.
These are specified conditional experiments,not identified human do(C) and not
the average of individual-kernel likelihood ratios.

Finite estimates can violate true coarse>=full monotonicity. ALL flags remain
exported;no clipping,epsilon or neutral-score imputation conceals them. Hence
the conditional readouts are approximations,not certified exact conditional
probabilities. Source posterior and predictive outputs themselves normalize.

## 5. Actual construction and scientific result

Same386 SOURCE/120 held authors,two18-event episodes,three references,two old
anchor assignments and first-state-matched donor map.64particles,two fixed
realizations;realization0primary,realization1a numerical check,not a new human
replication or a model-selection opportunity. No SOURCE parameters were refitted.

Pooled primary late transition log-score differences,nats per transition:

|Early contract|own-fresh|own-stranger|new minus old fixed same contract|
|---|---:|---:|---:|
|none|0|0|-.030121|
|full|-.080175|+.138442|-.051427|
|condition|-.001559|+.043187|-.025968|
|expression|-.009334|+.080392|-.023940|
|expression_given_condition|-.072124|+.118513|-.045746|
|condition_given_expression|-.050007|+.096385|-.037932|

All self-stranger means remain positive in both banks and both realizations.
Primary C-only own-fresh range[-.016849,+.012439],S-only[-.030245,+.010468]:
these small negative means are not exact zeros or confirmed memory gains.
Realization1pooled C/S own-fresh=-.001701/-.012121,own-stranger=+.038372/+.078553.
Full replay is EXACTLY the old known integrated result,not a new discovery.

Integration did not beat the old fixed-mean predictive levels. In particular,
the fresh integrated comparator itself differs from old fresh by-.030121.
Comparing the two own-fresh differences alone would conceal that baseline
shift. No best predictor,noise-free filter or accuracy gain is promoted.

The actual addition is a missing-observation measuring capability and an
extension of author relevance to shared-parameter uncertainty. Improved kernel
faithfulness does not automatically make a SOURCE law fit natural expression
better. Law mismatch,observation scope and author relevance remain separate.

## 6. Numerical boundary,not another precision campaign

Pooled mean absolute individual log-score differences across realizations:
C=.016427,S=.015607 nats/transition. Early profile total-variation means:
.031608/.061977;conditional designs.052855/.086206. Thus group directions
survive this check,while fine individual readings can be numerically sensitive.
No particle budget is increased to turn them into precision achievements.

Exact coarse/full monotonicity violation flags,bank0/bank1/pooled:
realization0=550/627/980,realization1=567/491/933. These are prefix-SOURCE cells,
not people or psychometric error percentages. The theoretical integral remains
defined;exact long masked integration is stillOPEN. The adapter is approximate.

## 7. Execution,ancestry and next decision

One112.430600-second compute under360cap,1440partial filters/8640 late programs
of120people. Streaming donor inversion reuses each early filter once for its
owner and old recipient,avoiding a multi-GB particle archive.37focused tests
(15new),fourPythonlintformat and sealsPASS. Full beta likelihood replay0,old
carry3.16e-13,fresh8.53e-14,normalization8.09e-14,cold0. Sameagent,notblind.
No scientific failure,retry,post-seal edit,new neural/API/private/train/Git/
Claude/automation/lockbox/old queue. PID14210gone,session71818exit0.

[Carvalho etal2010](https://arxiv.org/pdf/1011.1098),sections2-3,provides mature
conditional-statistic/resample-propagate particle learning. Our discrete
Dirichlet-count implementation is not its Gaussian example or a new theorem.
OpenAI/math adc7f124 Posterior-replicas.../08-streaming1-105 motivates conditioning
on a sufficient current memory before predicting. Its Gaussian angular/sample
complexity bounds and fresh-side independence do not apply to these texts.
No recent-math theorem is transferred by merely citing that file.

Stop particle/seed/prior/decay/state/weight/forecast-accuracy refinement. Next
inventory SAME-event SOURCE criterion/response-function meaning:what source-
linked numerical object can be interpreted,not which generic forecast wins.
Keep the existing positive HM171 source-E score and all earlier nulls/fields.
Original18/100/44 and full personality-interpreter theory remainOPEN.

Canonical:results/grain_math_response_v1/partial_kernel_r1.
Cold CLI:scripts/score_grain_partial_kernel_cold.py --bundle <source.npz>
--input <record.json>. No new scoring batch affects the fixed reference.
