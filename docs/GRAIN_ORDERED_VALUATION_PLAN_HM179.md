# HM179: universal ordered valuation, not another best-readout search

2026-10-11. Proposed AFTER HM178; reuse of its outcomes is explicit.
No untouched preregistration, psychological validity or fresh cohort claim.

## Unknown and competing mechanisms

HM178 SOURCE N-specific context contacts have negative descriptive alignment
with the native mean score, but positive alignment with every digit-vs-zero
log-odds action. Do not infer from this that changing the ordinal spacing will
connect the target. Nor assume that an average captures the entire response.

- A: the negative scalar direction is specific to equal digit spacing;
  another increasing valuation can reverse the signed pooled correspondence.
- B: every increasing valuation has the same opposing direction; spacing
  cannot repair this connection and the target interpretation law must change.
- C: both occur, conditional on author or view. Keep individual and view
  structure instead of claiming one global personality direction.

## Inventory, units and boundaries

The same 24-author, two-view, five-subset native cache has ONE N query only.
Other five-target fields are SOURCE covectors compared with this N response,
not five native target queries. Historical generic multi-query experiments
are not these authors or these evidence events and will not be joined here.
The ten logits define a distribution CONDITIONAL ON the ten digit candidates,
not the complete-vocabulary LLM response. The upstream encoder remains neural.

Reuse HM178's fixed source-only 6+6 relation fields and all components, two
arms and five traits; no new labels, native fit, raw/private text or reruns.
This adds a comparison interface, not a new source observation law.

## Exact construction

Let p_V be the old digit distribution for each natural subset. Define

    m = p_(R+S) - p_(R+C) - p_S + p_C.

It has zero total mass. It is twice the difference between the valid
distributions (p_(R+S)+p_C)/2 and (p_(R+C)+p_S)/2. This does not make it a
personality posterior or a new neural intervention.

For any increasing valuation v with v_0=0,v_9=1, set

    w_k=v_k-v_(k-1)>=0, sum_k w_k=1,
    t_k=sum_(j>=k) m_j, k=1,...,9.

Then m.v=sum_k w_k*t_k. The exact range over ALL weakly increasing normalized
valuations is [min(t),max(t)]. For strictly increasing valuations these are
infimum/supremum, generally not attained. Never replace the strict class with
a selected step scorer in deployment. Equal spacing is the mean of all tails.

For SOURCE increments x across authors, form

    b=sum_u (x_u/||x||_2)*m_u.

Apply the same bounds to b; these are unscaled signed alignment units, not
Pearson r, cosine optimization, accuracy, p-values or a calibrated score.
For individuals retain sign(x_u)*m_u and mark unresolved zero x separately.
Primary scope is the two-view author mean (24 authors, not 48 people); both
views are retained independently. Do not mix views into independent subjects.

## Finite scope and decision

150 pooled certificates: 2 arms x 5 components x 5 target fields x 3 scopes.
All nine tails and all per-author certificates retained. No utility fitting,
threshold/digit selection, monotonicity relaxation, gamma/OT tolerance grid,
rank/head/layer/precision refinement or deletion of failures.

- Primary N-specific author-mean all-negative bounds: stop monotone-spacing
  rescue; inventory source-supported target/readout interpretation interfaces.
- Mixed bounds: scalar correspondence is underidentified by ordinal order;
  do NOT pick the favorable weighting. Specify an independently supported
  target valuation law before another linkage experiment.
- All-positive bounds: preserve evidence of ordinal agreement at this scope,
  not human personality accuracy. Mixed individual/view findings remain.
- Source zero: unresolved, never a successful null or rejected author.

## Tests, audit and prior art

Test positive, negative, scale-dependent and zero worlds; reject non-zero-mass
signed measures, invalid distributions and arbitrary nonmonotone valuations.
Test translation invariance, strict valuations, view aggregation, source-zero
handling and direct enumeration independent of the reverse cumulative sum.
Verify all parent seals, actual cached candidates, old mean/log-odds accounts,
all 150 certificates and alternate positive-minus-negative distribution laws.
This is same-agent independent arithmetic, not a blind audit.

Classical first-order stochastic order and finite cone duality are prior art:
[Sarver lecture notes, section4.1](https://sites.duke.edu/toddsarver/files/2021/07/Micro-Lecture-Notes.pdf).
[Mueller and Wiesel2026, section2.2/3](https://arxiv.org/html/2607.28215)
clarify comparison over function classes and order/transport duality. We do
NOT implement their new almost-order coefficient or claim a new theorem.
The pinned OpenAI three-state manuscript foundations175-232 distinguishes an
observation law from an arbitrary readout functional; design analogy only.
Its tree assumptions, threshold theorem and conditional independence are NOT
imported into author text. Original18/100/44 remain OPEN, no new queue/Git/API/
training/private reading/Claude/automation/lockbox.
