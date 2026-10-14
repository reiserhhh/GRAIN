# R08 text/behaviour quotient protocol V1, amendment A1 (before any world result)

Date 2026-09-26 JST. V1 (sha 64a9ed3d…) is unchanged.
- **Defect.** The frozen W3 law p_h = (W1 p_h) + .2r gives p_h = 1.05 under a0 and 1.15 under a1 at state x = 1,
  h = 1, r = 1. That is not a probability. Attempt 1 failed in the W3 transport LP (negative transition mass). It is
  kept in `F15/text_behaviour_quotient_r1_failed_attempt1/`, and no world result was computed.
- **Change.** W3 uses p_h = (W1 p_h) − .2r: the register lowers the hidden-state switching probability. Its range is
  [.1, .95].
  - The declared role is unchanged: the register changes only the future hidden state, so one-step construct laws of
    s and g s coincide while their futures differ.
  - The sign is not chosen from any result.
- **Gate.** The runner now checks that every declared probability lies in [0, 1] and that every kernel row sums to 1
  exactly (Fractions) before any computation, and fails closed otherwise.
- **Nothing else changes.** Every other world, view, tolerance and falsifier stays as frozen.
