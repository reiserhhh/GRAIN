# GRAIN R15 Rational Speech Acts listener vs the NLI reader stack, protocol V1

Frozen 2026-09-26 JST before the comparison. The pole-readout profile by status was
inspected once to choose the stack definition below: pole alone is quotation-blind by
design. Cached F16 scores, no neural calls. Standard RSA (Frank and Goodman), no novelty.

- Speaker states w: supports, rejects, no position. Uniform prior.
- Utterances: actual(p), negated-correction(p), quoted(p), hypothetical(p) for p in
  {support, reject}, as rendered in the F16 bank.
- Literal truth:
  - actual(p) and negated(p) are true iff w = p;
  - quoted(p) is true in every w (it reports another person);
  - hypothetical(p) is true iff w = none ("I have not made that decision").
- Costs are proportional to words: actual 0, negated and quoted 1, hypothetical 1.
- Listeners:
  - L0 literal;
  - L1 pragmatic, with S1 proportional to exp(alpha (log L0 - cost)), for
    alpha in {1, 4} (declared; not fitted);
  - SURFACE, which reads the embedded clause as the speaker's stance whatever the
    status.
- Reader stack, from cached scores per item: with pole readout pr and NLI qualification
  q, P(supports) = pr q, P(rejects) = (1 - pr) q, P(none) = 1 - q. Cell means are taken
  over status x pole.
- Metric: mean total-variation distance between each listener and the reader stack over
  the 8 cells, with the per-status breakdown.
