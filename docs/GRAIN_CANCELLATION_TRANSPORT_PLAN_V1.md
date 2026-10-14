# HM40: Source-Hidden Compensation and Transport

Status: planned after verified HM39, not executed. Preserve all HM35-HM39
code, data, unsuccessful verifications and source checkpoints. GoalACTIVE.

## Question

HM39 finds substantial link/decoder components that compensate outside the
source-total active space. Does the source compensation space explain the
total action that is uncovered in application? This changes whether to keep
a joint observation family before compiling another response law. It is not
an accuracy/rank/regularization search or a new human validity experiment.

## Mathematical design

Use fixed per-query U from verified HM36. At each HM39 midpoint, let

```text
r_L=(I-UU^T)v_link,  r_D=(I-UU^T)v_decoder
c=(r_L-r_D)/2,  e=(r_L+r_D)/2
r_L=c+e, r_D=-c+e, r_L+r_D=2e.
```

Retain e, numerical remainder and all signed crosses; do not presume e=0.
The exact latent pair-to-total map T(v_L,v_D)=v_L+v_D kills(u,-u).
Synthetic witnesses must show that separate transports can break this
cancellation, and that a common transport preserves it. A total-only law
cannot distinguish the paired witnesses before such unequal transport.
This factor-through property is standard mathematics, not a novel theorem.

Construct W only from source c columns,32perquery(16nodes x2arms), not from
application totals. Project/orthogonalize against the frozen U, then use stable
thinQR+smallSVD with the original HM36 rank_multiplier100 machine threshold.
Do not tune a rank or add numerical near-zero modes to force improvement.
Report the complete spectra, orthogonality and source numerical floor. Also
report sensitivity to the measured roundoff bound without selecting a winner.
If the retained directions are not resolved over that floor, label the affected
dimensions numerically unresolved rather than a psychological discovery.

The existing roundoff is the parent-total minus the actual saved component sum.
Its Frobenius norm and its column-normalized analogue are resolution proxies,
not certified bounds on either component's separate neural error. Record both
spectra and the number above these proxies and the unchanged1e-8physical
comparison tolerance; none of these sensitivity labels changes the selected
machine rank. No application spectrum selects a source direction.

One same-rank random W control perquery is projected against U and frozen,
with seed4007+query_index. Compare old U, U+W and U+W_random on the same
application observations. A higher-dimensional source union is not compared
only with a smaller baseline: the same-rank control is mandatory. No joint
cross-query concatenation or application-dependent coordinate selection.

## Fixed scope and budget

All96HM39midpoint paths,3queries/2roles/2wordings/8H-B-Q cells/2physical arms.
No full-path integral claim or new-text replication.0new model/API/training
calls. At most3source compensation bases and3same-rank random controls,
300CPU wallseconds including preparation, failed attempts and verification.
Before numerical work freeze code/tests/spec/design and all ancestors.
Source bases/hash barrier must be committed to disk before application arrays
are loaded for evaluation. Retain the inherited complete scientific scope.

## Outputs and checks

- Source and application c/e norms, component cosine, physical scalar actions,
  source complement total and signed component cross moments, every address.
- Complete source W geometry/spectrum, numerical floor and matched-rank control.
- Cold total covector coverage and, separately, DeltaP input coverage.
- Cold action remaining outside U/W and captured in W, with all signed terms;
  zero-ish denominators stay explicit, not enormous percentage claims.
- Pair/wording contrast and condition tables byquery/role/arm; no selected cells.
- Alternate direct projections/QR geometry and exact quotient/transport toys,
  physical tolerance1e-8 plus declared machine-bound ratio audits.

Tests: equal/unequal transports, invisible source pair, known nonlinear
fiber map example, zero compensation, independent component units, gauge,
orthogonal versus nonorthogonal basis, random-control dimensions/seed,
source-first role and call/budget enforcement. A learned W is an observation
family, not a new人格因子 or an LLM-free predictor.

## Decision

If W carries cold total action beyond matched random, preserve this joint
family and next study source-only component reading transport. If it does not,
the source-hidden compensation explanation is insufficient; inspect the
state-conditioned output-link/Jacobian composition or downstream decoder
modules with a separately bounded design. If compensation is target-specific,
keep separate target laws; do not impose one universal noise deletion.

The actual application derivative is an evaluation reference only. Original
18/100/44/natural authors/B_u/psychological interpretation remain open.
No privateMEPS/API/Claude/automation/newtraining/commit/push/tag/lockbox/oldqueue.

## Literature and reading boundary

MIT18.952 multilinear algebra notes, section1.2 proposition1.2.1 and related
quotient/dual exercises (PDFpages7-11), provide the standard factor-through
criterion. For our T, a block transport descends exactly when it preserves
ker(T); in the separate-map example this means G_L u=G_D u on the kernel.
This is not a new algebraic theorem or a proved universal model transport.
https://math.mit.edu/classes/18.952/spring2011/chapter1.pdf

OpenAI/math fixedcommit adc7f1241b42e322a6451854ab7e4b4c146bf78a,
expansion.tex was read in full in this packet. Its joint rectangular tests and
integral-reduction warning motivate retaining an observation family rather
than concluding that cancellation of one output means absence of structure.
Its formal-series/subspace-expansion proof does not apply to floating neural
states or prove our compensation-space transfer.
https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026/build/sections/expansion.tex
