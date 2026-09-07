# P1D: partition the late query pathway and test a compact response-shape model

2026-09-07, follow-up after complete P1C. No new human/material validity claim.

P1C actual1362 forwards/306.2s: canonical trait donor residual-error energy falls
to.000789 when E is copied atblock04, but stays.962627 for E atblock16. Q alone
atblock16 reaches.008099; D alone remains.889314. Event-target has a comparable
E-to-Q redistribution. All E/Q/D together restore the donor with zero error.
This motivates localization inside Q, not arbitrary additional layer search.

Use existing36 cases: event-target, canonical trait-code1, venue, each12 paired
texts. Atblock16 divide ordered Q indices into four contiguous near-equal token
count bins, before interpreting the words. Preserve D and E. Run all16 masks;
0000 uses the already verified baseline,15 nonempty masks each require1 forward.
Full1111 must reproduce P1C Q-only, not necessarily the full donor.3 unpatched
baselines are recaptured; four-bin self-copy controls for the same3 cases. Expected
546 actual forwards, hard maximum800 /600seconds, same MPS FP32 frozen observer.

Save token-bin memberships and every patch score. Compute all main and higher
interaction coefficients with exact16-cell factorial inversion. Equal interaction
allocation (Shapley) is optional descriptive accounting, not unique causal attribution.

Compact operator test: on the alphabetically first3 scenario IDs, fit shared
response-shape coefficients separately for event and trait; evaluate on the other3.
Feature matrix = four bin-presence indicators, then indicators plus six pairwise
products. Output target = effect divided by the already measured full-donor score
gap. At evaluation, use the supplied donor gap to convert the predicted normalized
effect back to margin units. Report all-bin-copy comparator and no-change comparator.
This tests cross-setting transport of the shape of an intervention effect;
it DOES NOT predict a personality score without LLMs, and its known donor gap is
not a label-free discovery. Venue gaps can be near zero, so its original effects
are fully reported but it is not fitted as a normalized personality pathway.

Material is already exposed by P1B/P1C. The16-mask outcomes and fitting rule are
fixed before P1D. No choosing bins/layers/rank after results. If additive/pairwise
fails, retain this as evidence of interactions or variation rather than widen the
search silently. End this packet and reassess information value before any further
neural run. No old queue resume and no Claude call.
