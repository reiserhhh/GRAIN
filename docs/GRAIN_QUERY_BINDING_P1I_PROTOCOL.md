# P1I: queried entity, judgment criterion and the remaining suffix

2026-09-07. Freeze after complete/audited P1H, before the new patch outcomes.

P1H's query-clause effect follows its semantic role under clause-order reversal,
not simply later position. Test whether it is concentrated at the queried entity,
the judgment criterion, their combination, or other contextualized positions.

Reuse all48 P1H prompts with unchanged model, evidence, questions and scoring.
For each question explicitly mark the target name (Alex/Blair, or room for venue)
and criterion (initiate the conversation rather than wait; outgoing rather than
reserved; open). These intervals are declared from the authored grammar, never
selected by patch scores. Include each preceding separator space to retain whole
leading BPE tokens; all omitted/boundary tokens stay in the remainder region.

At block16 form a complete N/K/R binary patch factorial covering all Q. N=entity,
K=criterion, R=other Q positions. Mask000 reuses native,111 reuses audited P1H Q;
collect the other6 masks. Add one fixed-seed, count-matched random Q subset outside
N and one outside K, per case. Match each role-swap pair's random positions. These
controls match size, not all syntax/distance properties; report that limitation.

48x(6+2)+4 native/self checks =388 new forwards, cap500/600s/9GiB. The source
snapshot and every parent activation/output hash must match before collection.
Full111 must equal the actual parent Q effect, NOT necessarily the donor endpoint:
E and D remain recipient. No extra model fitting, layer search or private data.

Outputs: all effects, per-task/order restoration-error energy, matched random
contrasts, all seven inclusion-exclusion terms (no unique-causal attribution),
token intervals and independent scalar/layout audit. Negative venue and both
direction rows stay visible. Unit of authored diversity remains four settings.

Continue by what distinguishes explanations. Criterion dominance would localize
a useful candidate, not prove a personality-specific feature. Entity dominance
would prioritize reference binding. R dominance would reject a simple named-word
path; no deleting R as noise. Nonadditivity calls for a distributed conditional
operator, not arbitrarily assigned causal shares. No all-token or all-head scan.
