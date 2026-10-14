# GRAIN observation marks and conditional response actions

2026-10-10. HM122 primary comparison failed; retain HM121 observation results.
This document interprets frozen results, not a replacement preregistration.
Canonical results: `results/grain_math_response_v1/grain_mass_action_r1`.

## Measurement and control are different maps

An anonymous GRAIN coordinate describes an evidence unit. A high correlation
with a reader response does not make that coordinate a target-selective control
direction. Keep three maps separate:

1. Evidence -> observation marks, including role, condition and support.
2. Marks -> a specified intervention on the reading instrument.
3. Intervention -> the full conditional vector of reader responses.

HM121 tests a supported-path observation association. HM122 tests a finite
intervention constructed from the same frozen marks. They are different claims.
No human trait is manipulated, and model-response cosine is not psychological r.

## Finite action and its standard mathematical basis

For source covariance Sigma and source high-minus-low mark contrast delta_q,
use the normalized covector

    b_q = Sigma^+ delta_q / sqrt(delta_q^T Sigma^+ delta_q).

Within a text body, center the marks and clip once at [-3,3]. For baseline
probability p_i=m_i/M_body, the finite preference projection is

    p_q,+/- = argmax_p { +/- epsilon sum_i p_i s_qi - KL(p || p_base) }
    p_q,+/-i = p_base,i exp(+/- epsilon s_qi) / Z_q,+/-.

The maximization is over the baseline support with probabilities summing to one.
Fixed epsilon=.5; exterior mass, K/V, positions and current text remain unchanged.
This is classical exponential tilting, not a new theorem. Wrong-position arms
preserve the same score and mass multisets in this equal-token-mass design.

## A conditional response matrix, not a one-factor/one-trait dictionary

For text x, focus f, condition c, input cue q and output reader r, define

    A[x,f,c,q,r] = (F_r(x; m_q,+,f,c) - F_r(x; m_q,-,f,c)) / 2.

Retain high-low behavior, focus and assignment contrasts as separate linear
operations on A. Also retain the assignment-even component; it is not deleted
as noise. Diagonal target action and all off-diagonal actions are observed.

In HM122, the diagonal absolute action minus mean off-diagonal absolute action
is negative and worse than positional misalignment on both new-instance wordings:
correct-minus-wrong=-.072822/- .081231. Baseline numeric agreement passes while
target selectivity fails. A more accurate diagonal readout cannot by itself
justify discarding off-diagonal actions or naming each input cue as a trait axis.

## Exact dependency and empirical coupling

If two clipped score functions satisfy s_r=-s_q, then their plus/minus tilted
measures swap exactly. For ANY fixed reader, their finite response changes are
negatives. If s_r=s_q, their measures and finite changes coincide. This identity
requires the same support, baseline measure, clipping and dose; it is not a
claim about semantically named traits. Positive proportionality without clipping
can instead be represented by a dose change, not necessarily the same action.

The actual cues are only approximately related. Result-after diagnosis found
new-instance input/output E-A cosines -.982824/-.982762 and O-anthro
+.962688/+.979093. They describe instrument-channel overlap, not human trait
correlations or a causal isolation of the failure mechanism.

Four linear mark coordinates bound the tangent parameter dimension by four,
before the chosen nonlinear operations. This DOES NOT bound the finite response
configuration rank by four: nonlinear curves can span more output directions.
Nor does it prevent a four-dimensional representation from classifying six labels.

## Literature and design inspiration

- Wainwright and Jordan (2008), exponential families, maximal entropy and
  overcomplete parameterizations, Sections3.1--3.2 read in this packet:
  https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf
- Jain and Wallace (2019), attention weights alone versus explanations:
  https://aclanthology.org/N19-1357/
- Wiegreffe and Pinter (2019), controlled explanation diagnostics:
  https://aclanthology.org/D19-1002/
- OpenAI/math adc7f1241b42e322a6451854ab7e4b4c146bf78a, matrix-jets
  section71--108: full coverage and right-inverse conditioning inspire the
  question of complementary observation/action channels. Its PDE assumptions
  and theorem do not establish a language or personality result.

The next decision is which conditional response configurations add independent
information about qualified author evidence, not which dose best reproduces a
teacher. Inventory prior HM105--119 constructions first; no next experiment is
authorized by this interpretation document. Preserve source/application roles,
choice, conditions and state; do not convert this finite failure into rejection
of all GRAIN or a numerical-precision rescue.
