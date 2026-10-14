# GRAIN-C1-01 protocol V1.2 supplement (r3): pilot design contrast

Frozen 2026-10-02 ~19:57 JST after r2 and before any r3 output. V1 and V1.1 apply unchanged
(worlds, sample sizes other than the pilot, information boundary, kernel, tolerances, acceptance
rule, np arms, declared lexicon signs). r1 and r2 verdicts stand as recorded.

## Why (post-r2, disclosed)

In r1 and r2 every anchored arm passed W2 (Delta) in 1-2 of 3 seeds, failing only the style check.
The style-by-context nuisance is anchored in the pilot only through 9 training writers, i.e. 9
independent style values, so chance pilot covariance decides the leak. The r2 nuisance projection
failed in one seed because a mixed style/target component cleared the binary anchor test.
Question: is the conditional target constructible with an adequately designed pilot, or is
there a structural limit for anchored readouts?

## Single change

Pilot 36 writers x 16 texts (27 train, 9 validation) instead of 12 x 16 (9 / 3). Same per-writer
budget, same label law, same CV rules. Fresh seeds 20261008, 20261009, 20261010. One contrast, not
a grid; no further pilot sizes are run in this packet.

## Decision use

Pass in W2 3/3 by a non-oracle arm: conditional-response construction is feasible in this declared
world when the pilot has enough independent writers, and C2 must specify a pilot writer budget and
context variation per writer. Failure: anchored readouts need a direct register/style anchor (or
declared context-varying register labels) for conditional targets.
