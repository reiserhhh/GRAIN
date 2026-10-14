# R08 text/behaviour quotient: protocol V1 (frozen before any computation)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_13_PROVENANCE_AND_EQUIVALENCE.md` Part B. Sources: P02 pp.17–19
(PSR; psychological bisimulation over A_C with nuisance g ∈ G; construct as S/∼), P03 pp.22–23 (predictive-state
equivalence under controlled contexts).
- **Prior evidence.** `F15/bisimulation_metric_r1`; FU1-f, which the r3 audit called true by construction (an iid
  nuisance left out of the reward); `PACKET12_SOURCE_CLAUSE_CROSSWALK.md`.
- **Runner and output.** `scripts/run_grain_r08_text_behaviour_quotient.py`, writing to
  `results/grain_gptpro_foundation_v3/F15/text_behaviour_quotient_r1/`.
- **Scope.** Synthetic mathematical observations only, not validated language or behaviour. CPU, 2 workers × 1 BLAS
  thread, 0 neural calls, cap 15 min. At most 64 states per world and 6 worlds. Exact computation throughout.
- **Standard theory used, no novelty claimed, locators not asserted:**
  - lumpability (Kemeny & Snell);
  - probabilistic bisimulation (Larsen & Skou; Givan, Dean & Greig);
  - symmetry-induced reductions (MDP homomorphisms, Ravindran & Barto);
  - predictive state representations (Littman, Sutton & Singh);
  - the bisimulation metric (Ferns, Panangaden & Precup, as in F15).
  The finite theorem below is proved in full in the decision; nothing is extended beyond it.

## Model

- **System.** A finite controlled system with states S, probe actions A_C = {a0 neutral, a1 evaluative probe}, and a
  joint kernel K_a(o, s′ | s).
- **Observation.** One observation per step, o = (Y, B).
  - Y = (c, f) is coded text: content c ∈ {0, 1}, surface form f ∈ F.
  - B ∈ {0, 1} is coded behaviour.
- **Views** are garblings of o: raw (Y, B); Y only; B only; construct (c, B), which drops the form; content only (c).
- **Test probability.** For a word w = (a1, o1, …, ak, ok), p(w | s) = e_sᵀ M_{a1,o1} ⋯ M_{ak,ok} 1, where
  M_{a,o}[s, s′] = K_a(o, s′ | s).

## Worlds (s = (x, h, r): static trait x ∈ {0, 1}, hidden state h ∈ {0, 1}, register r)

Emissions are independent given (s, a), and the next state is independent of o given (s, a).
- **Trait:** x′ = x.
- **Hidden state:** P(h′ = 1) = p_h.
- **Register:** r′ = r with probability .9, otherwise uniform over the other register values.
- **Form:** f = r with probability .9, otherwise the other form. (W5 uses .8 and uniform over the other two.)
- **Laws:** content c ~ Bern(p_c) and behaviour B ~ Bern(p_b).

| World | States | p_h | p_c | p_b | Declared role |
|---|---|---|---|---|---|
| **W1 legit_nuisance** | 8 (r ∈ {0, 1}) | .3 + .4h + .15x + .1a | .2 + .5h + .2·a·x | .2 + .3x + .3h | Register is pure nuisance: G = form/register flip is an automorphism |
| **W2 conflict_behaviour** | 8 | as W1 | as W1 | W1 − .2r | Register changes behaviour: the orbit collapse must lose B information |
| **W3 conflict_future** | 8 | W1 + .2r | as W1 | as W1 | Register changes future hidden state only: one-step invariance holds, futures differ |
| **W4 unknown_transport** | 4 base states (x, h), no register | .5 + .3(2h − 1) + .1(2x − 1) | .8 if h = 1 else .2 (a0); a1 adds .1(2x − 1) | .3 + .4x | New channel c′ = K(c), K ∈ {identity K1, swap K2} unknown, modelled as a static latent over 8 augmented states |
| **W5 graded_near_nuisance** | 12 (r ∈ {0, 1, 2}) | as W1 | as W1 | W1 − .02r | Graded small register effect: approximate metric and ε non-transitivity |
| **W6 passive_unresolvable** | 8 | .3 + .4h + .1a (no x) | as W1 | as W1 | The trait never affects passive text: only B or the probe can reveal it |

G in W1–W3 is g: r ↦ 1 − r on states and f ↦ 1 − f on observations: an invertible relabeling that changes raw text
symbols. Dropping f (the construct view) is a non-invertible garbling, not a group element.

## Computations (all exact)

1. **Observable equivalence per view and action set.**
   - States s ∼ t iff p(w | s) = p(w | t) for all words.
   - Computed as equality of rows of the future-span basis. The basis is Krylov closure of {1} under M_{a,o} for the
     view's observations, which needs at most |S| dimensions.
   - Primary computation in float64 with tolerance 1e-12. An independent computation uses exact Fraction arithmetic
     and exact rank.
   - Views compared:
     - mean summary: E[c | s, a0], a fixed feature;
     - one-step raw-Y law under a0 (short view);
     - Y only, passive {a0} (long passive view);
     - Y only, {a0, a1} (long probed view);
     - B only;
     - raw (Y, B), {a0, a1} (unquotiented model);
     - construct (c, B), {a0, a1}.
2. **Coarsest probabilistic bisimulation** for the raw and construct views, by partition refinement on (block,
   per-action per-observation block mass).
3. **G checks (W1–W3).**
   - Raw-law invariance: K_a(o, g s′ | g s) = K_a(o, s′ | s).
   - Equivariance: K_a(σo, g s′ | g s) = K_a(o, s′ | s).
   - Construct-view invariance of the one-step law.
   - Lumpability of the orbit partition for the construct view, with the largest violation named.
   - The quotient kernel when the orbit partition is lumpable.
4. **Future-law total variation** by enumeration, for declared pairs and horizons k = 1..4.
   - Action sequences: passive (all a0) and probe-first (a1 then a0).
   - Views: Y only, B only, construct.
5. **Prediction loss of a merged view.** Expected Brier increase for B at t+1 and t+3 under passive dynamics, weighted
   by π = uniform initial state propagated 500 steps.
   - Formula: Σ_s π(s)(p_view − p_s)², where p_view is the π-weighted block mean.
   - Also the loss for content c at t+1.
6. **Bisimulation metric (construct view).** c_R = .1, c_T = .9, reward term = total variation of the one-step
   construct-view law, W1 via LP (HiGHS), stopping at max change < 1e-9.
   - Worlds with at most 12 states.
   - Cross-check: the archived A01 `c2_xor` metric (max d .130) must be reproduced within 1e-6.
7. **W4 transport.**
   - Observable equivalence on the augmented space {K1, K2} × S under c′ only and under (c′, B).
   - Invariant scoring: a statistic invariant under the relabeling, P(c′_{t+1} = c′_t).
   - Calibrated transport: the expected log-likelihood ratio for K per anchor observation from a known state.
8. **W5 threshold relation.** Compute d between registers 0/1, 1/2 and 0/2 at fixed (x, h).
   - The relation d < ε is non-transitive for some ε iff max(d01, d12) < d02, and the interval of such ε is reported.
     ε is never chosen from results.
   - Single-linkage closure versus complete-linkage (diameter) blocks is reported.

## Pre-declared contrasts and falsifiers

- **F1 (implementation check).** In W1, the orbit partition must be lumpable for the construct view, and
  construct-view futures of s and g s must be identical at k = 1..4. Otherwise STOP: the implementation is wrong.
- **F2.** In W2 and W3, if the orbit collapse shows zero construct-relevant loss, the conflict design failed. That
  is reported, not redesigned. Expected: W2 loses B information at k = 1; W3 has one-step invariance but future
  total variation > 0 from k = 2, and a lumpability violation.
- **F3.** In W6, if passive Y (any length) separates x, the "passive text cannot reveal it" claim is false. Expected:
  B and probe a1 resolve it.
- **F4.** In W4, if the c′-only laws of (K2, s) and (K1, πs) differ (total variation > 1e-12), the non-identifiability
  claim is false; π is (x, h) ↦ (1 − x, 1 − h). If the (c′, B) laws coincide, B does not resolve.
- **F5.** In W5, if max(d01, d12) ≥ d02, no non-transitive threshold exists. That is reported as found.
- **F6.** The float and exact Fraction partitions must agree, and enumeration must agree with the Krylov verdicts on
  declared pairs. Otherwise STOP.
- **No search.** One parameter set per world, fixed above. No seeds, no grid, no redesign after results.
