# GRAIN: contextual dual frames and author comparison

## Development result

HM125 constructs a fixed numeric scorer from96 original source vectors. It
accepts an application vector, not a trait question, model answer, or application
family/actor/polarity label. A separate cold CLI scored48 cached application
vectors without importing a neural runtime. This repairs a concrete interface
gap after HM124: frozen lexical GRAIN observations collapse the actor exchange,
whereas contextual coordinates can distinguish the associated author laws.

The scorer is not a raw-text mathematical replacement for an LLM. Its input is
a contextual V state from the frozen pretrained encoder, at a registered PersonA
carrier. It does not solve unknown-person parsing, evidence eligibility, or
human psychological validity. Calibration uses source factorial design labels.

## Construction and exact account

Let h_e in R^2304 be the nine-token/two-head current V observation. The source
provides six behavior families and two conditions, each with two wordings and
four polarity/actor corners. Define

    d_fc = (1/8) sum_(w,b,o) b o h_wfcbo,
    D = [d_01,...,d_12], mu = E_source h,
    B = D^+, z(h) = B(h-mu), r(h) = h-mu-D z(h).

The twelve columns have full rank, condition number10.0058699, inverse operator
norm3.0063675. B D=I is a calibration identity, not an empirical discovery.
RD01--RD12 are anonymous calibrated coordinates, not discovered personality
factors. The residual r, source origin mu, choices and conditions are retained.

For a declared author event law P_u, use the explicit expectation

    S_u = E_(P_u) z(h), R_u = E_(P_u) r(h), H_u = E_(P_u) h.

This is not neural partition-weighted attention or an encoding of a concatenated
conversation. Positive laws are generated for a finite mechanism control:

    p_u(w,f,c,b,o) = (1+b o theta_u,fc)/(4 n_wordings n_cells).

The65 profiles are neutral,24 signed axes and40 signed dense profiles. They are
virtual distributions over a known text bank, not65 people. They change joint
actor/behavior association but leave all individual margins unchanged. Theta is
material-generation truth, not a personality score. The entire12-dimensional
slice-level lexical-GRAIN
coordinate distribution remains identical (maximum change6.94e-18), while the
contextual author coordinates differ. This is a stronger control than checking
only lexical means, but the factorial/fiber mechanism itself is established.

For an application view v, compute D_v for evaluation only and define

    M_v = B D_v,
    n_cells (S_theta - S_minus_theta)/2 = M_v theta_visible.

No application inverse, recentering, rotation or model fitting is used to rescue
scores. The identity is verified on all32 paired profiles per view. A second
formula uses direct probability sums and independent least-squares solves.

## Observed transport, not prediction accuracy

| View | Rows | Action energy captured by source span | Positive corresponding directions | Off-diagonal / diagonal norm | Origin drift norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source calibration |96|1.0000|12/12|~0|0|
| Same-event rewording |96|.3637|11/12|1.8570|.7064|
| Different instance |96|.2235|9/12|2.0865|.8966|
| Fresh HM124 instance |48|.1536|2/6|3.4707|7.2175|

Energy is the squared Euclidean norm of these factorial action columns, not a
percentage of psychological information. Direction signs are descriptive, not
significance tests. The fresh view has only condition-1; six condition+1
parameters yield exactly identical observed laws and scores, and remain
unobserved. Cross-coordinate spillover does not recover those conditions.

Future-question pre-edit V vectors agree exactly across all12 saved question
suffixes. Thus this particular causal cut does not use the future question.
The source frame nevertheless transports poorly: in the fresh view84.64% of
actor-action energy remains outside its span. Coordinates can show substantial
author differences while failing to preserve what the same coordinate means.
There is no contradiction between existence of relational information and poor
transport of one fixed coordinate chart.

## Consequence for the mainline

For a contextual local representation h(c,theta)=mu_c+D_c theta+remainder,
the source ruler reads

    z_0(h) = B_0(mu_c-mu_0) + B_0 D_c theta + B_0 remainder.

This separates three problems: reference-origin movement, coordinate mixing,
and information outside the source action span. More accurate computation of
the same z_0 does not fix any of these by itself. Statistical accuracy can help
only after choosing observations that identify the desired distinction.

The next mechanism-development question is whether a source-constructed
contextual relation/transport law can preserve the declared author contrast on
a held-out expression, without application labels or answer fitting. Existing
HM114/HM119 conditional-response constructions must be compared, not repeated.
Calibration anchors must be distinct from the author being assessed. Context
origins can carry author choice and state information; they are not automatically
noise to subtract. Residuals likewise remain an observation channel.

No next experiment has been started by this packet. Original18 protocols,
100 cards and44 computational questions remain OPEN.

## Mathematical ancestry and contribution boundary

- [Park, Choe and Veitch,2024](https://proceedings.mlr.press/v235/park24c.html)
  distinguish representation, probing and steering, and motivate a language-aware
  inner product. We use Euclidean geometry here, not their causal inner product.
- [Park and Muandet,2020](https://papers.neurips.cc/paper_files/paper/2020/file/f340f1b1f65b6df5b5e3f94d95b11daf-Paper.pdf)
  develop conditional vector-valued expectations. These finite declared laws
  do not inherit their statistical consistency results.
- The pinned [OpenAI matrix-jets manuscript](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex),
  lines71--108, constructs a bounded right inverse from a uniformly spanning
  family. This inspires rank/inverse-sensitivity checks, not a transfer of its
  PDE assumptions or uniqueness theorem to language.

Dual frames, factorial contrasts and expectations are mature mathematics.
HM125's contribution is system development and a finite transport diagnosis:
an executable query-score-free author-coordinate interface plus the measured
failure of a source-only global ruler. No novel mathematical theorem, new human
trait, or prediction improvement is claimed.

## Reproduction and verification

Canonical output: `results/grain_math_response_v1/relational_dual_frame_r1`.
Prepared inputs and code are sealed; do not overwrite or edit sealed code.
Run a new explicitly named output for a reproduction:

```sh
python scripts/run_grain_relational_dual_frame_hm125.py prepare --out NEW_OUTPUT
python scripts/run_grain_relational_dual_frame_hm125.py application --out NEW_OUTPUT
python scripts/score_grain_relational_dual_frame_cold.py \
  --frame NEW_OUTPUT/00/frame.npz --input NEW_OUTPUT/01/fresh_instance.npz \
  --out NEW_OUTPUT/cold_scores.npz
python scripts/readback_grain_relational_dual_frame_hm125.py --out NEW_OUTPUT
```

6new tests and15combined focused regression tests pass. Five new Python files
pass Ruff check and format. Alternate arithmetic readback covers781800 numeric
entries with maximum error5.75e-14; it is same-agent second-formula verification,
not independent blind replication. Recorded runner-phase CPU time1.112seconds
excludes the separate short cold-CLI overhead; no model calls or failures.
Source/app materials were already known: result-after construction, not a new
preregistered replication. Old HM124 lint failures remain sealed and unaltered.
