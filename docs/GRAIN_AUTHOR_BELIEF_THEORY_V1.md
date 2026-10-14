# SOURCE-author belief and measurement dynamics

HM281,2026-10-14 JST. A whole-path constructor and finite natural-history
mechanism experiment, not a new probability theorem or a trait accuracy result.
Canonical outputs: author_belief_r1/runtime_io_recovery_r1, with a separate
readonly_verification_r1. Both preserved engineering failures remain visible.

## 1. The new object is a belief law, not another LLM judgment

GRAIN's existing source-relative fields and local triple functionals did not
provide a flexible complete-path generative observation. HM264's fixed source
inventories lost support for all complete held paths. This constructor changes
the observation model: a reference author's unknown transition rows are drawn
from an explicit finite Dirichlet law. It does not repair the old inventory,
claim its source windows were future budgets, or make a saturated binary
observer into a valid psychometric instrument.

Input is a qualified sequence of the inherited8 states, recording venue and
two GRAIN coordinate bits. Its underlying text encoder remains neural. The
path/scoring/update computation itself needs no neural inference or LLM answer.
All386 SOURCE authors/two18-event windows and120 disjoint historical held
authors/two windows are retained. There are no new raw/private/API/train calls.
Source labels are old five-bit criteria, not new latent personality types.

## 2. Exact conditional model after source compilation

Let c0_i and c_iab be source starts and within-episode transitions. There is no
cross-episode edge. Let r0 and r_ab be source-pooled start and row frequencies.
All global source cells must be observed. One fixed prior row mass gives

    alpha0_i=c0_i+r0, alpha_iab=c_iab+r_ab.

The global base is empirical SOURCE information; this is a compiled empirical-
Bayes conditional model, not a fully specified hierarchical prior before source
data. Per-author unobserved transitions receive explicit prior mass, not observed
counts. Positive support alone is NOT a validation criterion.

Draw a reference index I uniformly, then independent Dirichlet start/transition
rows with these parameters. After observed episode-start counts n0 and transition
counts n, the component likelihood is

    L_i= B(alpha0_i+n0)/B(alpha0_i)
         * product_a B(alpha_ia+n_a)/B(alpha_ia).

B is the ordinary multivariate beta function. For the fixed-row alternative,
replace each row by its source posterior mean before observing new events.
No mixture-component number, bandwidth, prior strength or state count is fitted.

The full shared-author update is

    w_i(history) proportional to prior_i*L_i(history),
    q_i(b|a,history)=(alpha_iab+n_ab)/(sum_b alpha_iab+n_a),
    predictive(b)=sum_i w_i*q_i(b|a,history).

Two episodes share I and row parameters, but each starts anew. The next state
is predicted BEFORE it is incorporated. Beta ratios and this sequential rule
give the same likelihood; correlated overlapping triples are not multiplied.
The reference index represents a catalogue mode for a NEW author, not a claim
that the new person literally is one of the SOURCE participants.

## 3. The author score and its movement are different quantities

For each retained actual or old-permuted SOURCE assignment,

    profile_g(t)=sum_(i:code_i=g) w_i(t),
    theta_q(t)=sum_i w_i(t)*Y_iq,
    departure_q(t)=theta_q(t)-mean_SOURCE Y_q.

All32 profiles, five marginals and the fixed reference remain. No product of
marginals or batch-relative renormalization is substituted. These are exact
SOURCE-anchor posteriors of the compiled model, not human trait posteriors.

Source Y values and the latent component/kernel can remain fixed while
theta(t) moves. Such movement is epistemic updating about the author model.
It does not by itself measure an actual person's emotional state or personality
change. Conversely, this does not assume real people cannot change; a changing
human response law is a separate mechanism to model and compare.

## 4. Two memory operations, not two independent pieces of evidence

Cross author-weight updates with kernel-row updates:

| Program | Update author weights | Update kernel uncertainty |
| --- | --- | --- |
| fixed_reset | no after first state | no |
| fixed_shared | yes | no |
| kernel_reset | no after first state | yes |
| kernel_shared | yes | yes |

All initial and first-transition predictions are identical. Reset programs
hold weights at their first-state posterior and are explicit predictive ablations,
not coherent whole-author posteriors. The two shared programs have the exact
mixture meanings above. One event updating both latent index and rows is NOT
double counting: its beta-ratio likelihood confirms the joint update.

This model is Markov exchangeable: two paths with the same starts and transition
counts have the same component likelihoods and final shared posteriors. Therefore
it cannot explain every previously observed within-count order contrast. It
adds a complete-path observation law, not a universal model of language motion.

## 5. Natural-history findings, no selected best predictor

Three SOURCE references and the SAME120 held authors. Pooled transition log-score
changes in nats/transition, with exploratory whole-author percentile intervals:

| Contrast | Mean | 95% interval |
| --- | ---: | --- |
| author update, fixed rows | +.034420 | [.006501,.063187] |
| kernel update, fixed author weights | +.020135 | [-.002509,.042730] |
| kernel update, shared author | -.031332 | [-.038245,-.024137] |
| two-operation interaction | -.051467 | [-.069024,-.034583] |
| own early carry versus fresh late, fixed | -.058870 | [-.088376,-.032344] |
| own early carry versus fresh late, kernel | -.080175 | [-.114036,-.047632] |
| own versus stranger carry, fixed | +.117598 | [.081278,.151504] |
| own versus stranger carry, kernel | +.138442 | [.093386,.180650] |

Strangers are a single cyclic reassignment within identical early first-state
groups, covering120/120, not full topic/time/demographic matching. Stranger
initialization is a counterfactual observer operation, not a same-author record.
Own-minus-stranger signs remain positive in both source banks; own-minus-fresh
signs remain negative. Same-author relevance and safe reuse are different claims.
Two banks are not independent human replications. Intervals condition on the
SOURCE estimate and model and are exploratory, not family-adjusted confirmation.
For stranger contrasts the resampling unit is the recipient-author DELTA with
the donor map fixed. Those deltas share donor/recipient people, so these
percentile ranges do not establish independent-population95% coverage. Their
signed finite means remain the observed result; no bootstrap refinement follows.

The combined program is NOT promoted: two individually nonnegative mean changes
do not add. A priori fixed priors/kernels can be mismatched to natural data;
these results do not uniquely attribute negative carry to human change, context,
sparse source estimation or posterior overcommitment. No parameter rescue follows.

Pooled final kernel-shared effective source count averages5.3819 of386. Large
SOURCE target departures appear for both actual and old-permuted anchors
(roughly.26-.30 RMS), so sharp belief/score movement alone is not psychological
meaning. Label assignments never enter path prediction. Probability numerator
contributions average .69270 SOURCE/.18176 prior/.12554 application learning,
not percentages of personality information or measurement error.

## 6. What this enables and what the next mechanism must distinguish

New reusable chain:

    text-vector states + qualified episodes
      -> frozen SOURCE kernel-law bundle
      -> full reference-author belief and next-state predictions
      -> SOURCE-profile/target functionals at every prefix.

This does not use an LLM's unsupported yes/no judgment to define target meaning.
It joins WHO-relative reference laws and WHEN-history updating while retaining
WHERE condition choice and WHAT frozen expression coordinates. It is not the
same evidence as HM280's48 binary neural pairs and does not close that bridge.

Next ask WHICH conditional components can be transported: condition selection,
within-condition response, author mixture or kernel uncertainty. Keep all parts
and source references. A source-observed conditional operator is needed before
another finite experiment; not prior/weight/state/rank/length/accuracy tuning.
Changing the memory rule must be a new mechanism hypothesis, not optimizing away
negative carry. No new job is queued; original18/100/44 remains OPEN.

## 7. Mathematical ancestry and actual verification

[Fortini/Petrone2015, Sections2.1/2.3/3](https://arxiv.org/html/1406.5421v3)
already establishes predictive/kernel-mixture and exchangeability relationships.
The finite Dirichlet integral and Bayes update are standard. No infinite
recurrence, uniqueness or human-personality theorem is transferred.
[Pinned OpenAI/math foundations98-114/166-232](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
motivates keeping latent law, observation and posterior distinct; it supplies
design inspiration only. The local delta is the implemented complete-path
constructor and the tested distinction between relevance and transport.

Primary compute failed before science on exclusive runtime creation. One scoped
status-only companion used the unchanged scientific code and a reduced179s cap.
Compute6.132597s; primary process charge.261422s. Full4320 predictive program
paths are descriptions of120 people, not4320 subjects. The frozen verifier then
completed rising-product checks/cold export but failed on NumPy-bool JSON output;
partial verification.json is preserved. A separate read-only gamma/beta audit
checks2880 shared fresh/carry/stranger endpoints, all source counts/codes and
cold trajectories. Maximum likelihood3.907985e-14/profile1.121325e-14,
prediction normalization9.992007e-16/share1.443290e-15/cold0.
32 focused tests,10 new; two existing SWIG warnings. Primary/recovery six Python
lint/format checks pass; the post-outcome auditor keeps a sealed RUF007 stylistic
violation (Ruff exit1, zip versus pairwise), rather than changing its recorded code hash.
Same-agent different arithmetic, not blind replication. No scientific rerun.
