# NEXT80: teach source reader motion, not only source reader positions

2026-09-11. PLANNED, not executed. Following NEXT79's source-range obstruction,
develop one finite source directional-reader field and test it on the unchanged
four endpoint functions. No bandwidth, factor, alpha, rank or query search.

## 1. Fixed source-only intervention

For each of the same five source references S and each member i in S:

    c_i = (H16_iN,last + H16_iC,last)/2
    v_iS = (mean_{j in S} c_j-c_i) / ||mean_{j in S} c_j-c_i||
    epsilon_S = .01 sqrt(h_S^2)

h_S^2 is the same source median squared distance as NEXT79. A zero direction or
zero h_S skips its motion column and records degeneracy; do not choose a fallback
direction from recipient outcomes. Membership is applied before forming v or h.
There are 16 source/reference pairs:4*3 plus4, not16 independent authors.

Change only the last H16 row of *both* source query states by +/-epsilon_S*v_iS.
BODY, other rows, query difference, layout, weights and candidate alphabet stay
fixed. This moves the paired context while retaining query-specific offsets.
Run the actual frozen suffix17..end; hook the post-attention input at block22,
then capture r_iq^+/- = gradient of original fixed L_q with respect to that
post-attention last vector. It is **not** the H16 gradient or block17 reader.
Reproduce the eight unperturbed source NEXT51 readers before using new columns.

    j_iS = [(r_iN^+-r_iC^+) - (r_iN^--r_iC^-)] / (2 epsilon_S)
    J_iS = j_iS v_iS^T.

J_iS is a rank-one directional secant model. It is not a recovered full Hessian
or proof that the full reader depends only on c. Measure source motion outside
the original d_i affine span, and its projected action on native/source evidence.

## 2. One numerical development

Retain the NEXT79 constant and position-kernel comparators. Add exactly:

    d_hat_motion(c) = sum_i pi_i(c)[d_i+j_iS v_iS^T(c-c_i)]
    rhat_N/C(c) = m_S +/- d_hat_motion(c)/2
    Fhat_q(H) = rhat_q(c)^T phat_q(H_q).

The existing four phat functions and source h/pi/m remain unchanged. No recipient
reader/target, native Q/K/V/g, human Y or MEPS enters application. Derivative
teaching is allowed **only on source states**. Source4LOO packages cannot use the
excluded author's context to construct a teaching direction, even indirectly.

Differentiate this scalar. For context changes, retain both Dpi times the local
reader and pi*J_iS*Dc. For the existing BODY directions Dc=0. Verify synthetic
context derivatives and all four actual BODY function derivatives. The unbounded
local extrapolation is an explicit limitation: record displacement along v in
units of epsilon and h, but do not add a tuned clip/gate after seeing outcomes.

## 3. Budget and decisive readout

- One local frozen observer load, no API/Claude/new model download or old queue.
- At most76 native reverse calls:8 source parity,64 perturbations,4 half-step
  checks for the first all4 source/two queries. Stop on parity failure or a
 120-second native compute cap; preserve partial outputs, do not restart blindly.
- One fixed half-step check estimates finite-difference sensitivity, not an
 epsilon search. If its secant relative discrepancy exceeds .20, close this
 numerical derivative estimate and report the issue rather than tuning on new8.
- Native reader source parity: relative L2 <=1e-3 and absolute max <=1e-4.
  These are numeric path-consistency checks, not scientific effect thresholds.
- At most90 CPU seconds for construction/application/evaluation/replay,
 zero regression coefficient solves. Record loading, failed and successful
 native costs separately. Do not regenerate current target neural outputs.

Report every old score/action/term/factor panel. Primary mechanism questions:
does motion expand source output directions, and does that expansion improve
new8 differential K+V projected error and move its opposing modulation toward
the native sign without worsening useful scalar variation? Keep source4 results.
Mixed results remain a located mechanism, not a full-predictor promotion.

If motion is locally present but fails to transport, next distinguish finite
context extrapolation from omitted BODY-conditioned reading. Do not conclude
that context is unimportant or return to a kernel sweep. H16 and full-L gaps stay
explicit. This card is a source-teaching intervention, not natural text editing
or a new human psychological experiment.

## 4. Files and precedent

Suggested `source_reader_motion.py`, `scripts/run_grain_source_reader_motion.py`,
focused tests, private `logic_source_reader_motion_v1/` and cohort-only report.
Pin source membership/material, code, scalar definition, capture boundary,
derivative columns and predictions in that order; one full cached replay.

[Solak et al. (NIPS 2002)](https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html)
motivates combining observed values and local derivative information; our field
is not their GP posterior or uncertainty guarantee. Its compatibility comes
from differentiating the stated Fhat, not borrowing integrability from a citation.
[NEXT79 theory](GRAIN_CONTEXT_DIFFERENTIAL_READER_THEORY_V1.md) derives the source
range obstruction that this finite experiment specifically addresses. Goal ACTIVE.
