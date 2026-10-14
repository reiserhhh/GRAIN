# HM183: author sampling versus conditional evidence in a source-law update

2026-10-12. Seal before outcomes; no precision or prediction search.

Inventory: HM171 declares author-balanced target observations; HM175 separates
profile pooling from same-author event dependence. HM181 lifts a source action
to actual author/triple atoms; HM182 supplies marginal and matched target
contrasts. None of these decomposes the SAME HM181 lift into its actual author
marginal and its conditional evidence law. HM121 token attention masses concern
another sampling level, not these natural author/triple observations.

Question: does the profile-preserving source operation preserve equal author
sampling, or does its apparent target response mix source-author selection with
within-author evidence selection? A second question is whether instability of
an action-derived vector means instability of the updated natural target reader.

Competing mechanisms: (1) only conditional evidence changes; (2) author sampling
also changes and contributes to target-reader updates; (3) both contributions
and their interaction matter. Report all results, not a winning predictor.

Use the unchanged SOURCE386/32 binary profiles/two4+4+4 views/49408 actual
triples, HM181 weights, and HM182 actual4096 features and fields. All source
labels inherited; no application labels, raw/private texts, new neural/API,
training, Git, Claude, automation or lockbox. The encoder remains neural.
Six actions: each frozen component and their frozen sum. No dose/rank/grid.

For original p(u,t)=a(u)b(t|u)=1/(386*64), set h=p-d_q (or p-sum_q d_q).
Require actual nonnegative h and positive author marginal a'=sum_t h;
otherwise stop and preserve failure, never clip/retry/change the operation.
Let b'=h/a'. Retain four laws p,a'b,ab',a'b'. Exactly:

    h-p=(a'-a)b+a(b'-b)+(a'-a)(b'-b).

All four profile marginals must match the original; individual author masses
need not. A factorized law uses only existing atoms, not newly generated texts
or a causal intervention on people. Do not erase topic/choice/state.

For each law, actual weighted profile mean is sum_(u,t in profile)law*Phi/pi_g.
This compiles all five marginal/matched/reference/composition fields using the
same positive source reference. Preserve all target/action pairs. Retain the
three signed reader changes and squared-norm cross terms, not information
fractions. Compare total reader state before/after separately from its change.

Exact marginal-reader change also obeys, for pi_r unchanged,

    D_original-D_joint = [W_qr+(1-2pi_r)W_q0]/[2pi_r(1-pi_r)].

This is a conditional-mean identity, not a novel derivative theorem. HM181 W
is an action footprint, not the original target field. Both must remain.

Compile all2x6x4x5x4 fields, then cold-score the same240 frames independently
with only numeric event coordinates/slots/IDs. Missing triple opportunity stays
missing. No comparison against LLM answers or psychological accuracy in this
packet. The new ability is a source-law update compiler retaining sampling
levels; it is not a new trait intervention or unlabeled denoiser.

Focused tests: pure-author/pure-evidence/mixed/null laws, invalid support,
exact target-change identity with unequal class sizes, all32 profile reference
preservation, and cold input label rejection. Alternate direct signed author
weights and Kronecker cold features; fixed tolerance1e-12, no refinement.

Prior art: finite disintegration, conditional means and product differences are
mature. Szabo2016 JMLR17(152) separates subject and within-subject sampling.
Kennedy2023 arXiv2203.06469 Sections2/3 distinguishes a functional from its
distributional derivative; no influence-function inference guarantee applied.
OpenAI/math pinned three-state common-input observation construction motivates
checking which shared variable is sampled. No tree threshold theorem transfer.
Original18/100/44 remain OPEN. Stop this branch after its mechanism decision;
do not optimize a small update until it resembles an LLM endpoint.
