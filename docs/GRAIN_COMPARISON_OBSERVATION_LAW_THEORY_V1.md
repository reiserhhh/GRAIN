# Comparison observation laws and author measurement

2026-10-12 UTC. HM228 completes the original HM226 finite graph in a separate
recovery package. This is an empirical measurement-interface result, not a new
Hodge theorem, a psychological factor discovery, or prediction optimization.

## Main finding

The same 24 author histories support different target organizations under
standalone scoring and pairwise comparison. E/N centered target cosine changes
from -.349749 to +.749677; the raw exchange-odd edges already have cosine
+.693144. Thus this change is not introduced solely by graph projection.
All ten comparison-position target cosines are positive (.523492 to .913765).
The first singular direction accounts for .790534 of raw squared singular
values, or .797664 after equal column scaling. These are descriptive matrix
quantities, not proportions of personality information or a new human factor.

The comparison task changes pair context, wording, response alphabet, and
readout together. The observation-law change is observed; counterpart alone
is not isolated as its cause. Familiar target names do not, by themselves,
guarantee preservation of what is being measured.

## Exact finite accounts

Let X_u be a fixed author's 12-event history, q a target, and pi the complete
presentation/response contract. The frozen observer supplies a final state:

    H_uvq = E(X_u, X_v; q, pi)
    z_uvq = log p(A | H_uvq) - log p(B | H_uvq)
           = (W_A - W_B) H_uvq

The same output bias, if present, must also be included; this frozen head has
none. No personality label is supplied to the observer. Both author orders
are observed, not synthesized:

    d_uvq = (z_uvq - z_vuq) / 2
    b_uvq = (z_uvq + z_vuq) / 2

For the fixed connected 32-edge graph, let B have +1 at the first author and
-1 at the second. Use the sum-zero reference gauge:

    theta_q = argmin_{sum theta = 0} ||d_q - B theta||^2
    c_q = d_q - B theta_q
    B.T c_q = 0
    z_uvq = theta_uq - theta_vq + c_uvq + b_uvq

theta is a graph-referenced position, c a signed non-additive relation account,
and b an exchange-even presentation account. This is an exact decomposition
of this observer's data, not a causal decomposition into personality/noise.
Keep all three accounts; do not erase c or b and announce a stable trait.

## Cardinal and ordinal structure are different

All five targets have nonzero c. Its squared magnitude relative to d is
.063345 E, .022437 N, .049177 A, .036266 C, and .028665 O. Most observed edge
magnitude is representable by additive positions, but exact cardinal
additivity fails. These ratios are not psychological variance fractions.

Nonzero cardinal circulation does not exclude a nonlinear monotone scalar
order. Therefore a separate global directed-graph check uses signs of the
observed d values, at margins 0 and 1e-4. E/N/C/O are acyclic on all 32 observed
edges at both margins. A has the concrete strict cycle:

    15 -> 16 -> 17 -> 15
    edge magnitudes: .041737556, .009439468, .020342827 nats

No strict scalar ordering realizes these three A comparisons simultaneously.
This statement concerns the fixed observer on these edges, not human A or
unobserved author pairs. SCC detection and an independent Kahn/DFS witness
agree on all ten target/margin cases. Same-agent verification is not a blind
third-party replication.

## Presentation and external anchors remain separate

All 15 self-comparisons favor the second slot, with z from -.111866 to
-2.626394 nats. Exchange-even RMS exceeds odd RMS for every target; the even
component is not negative on every ordinary pair. A self-control anomaly is
not evidence that all pair relations are caused solely by slot bias.

Human questionnaire E/N has r=-.529160 on this panel. Comparison-position
same-target contacts are E+.077177, N-.420922, A-.157398, C+.145312, O+.169934.
No label flip, fitted calibration, or prediction promotion is performed.
The older SOURCE-A-position/human-N contact remains +.695631; its permutation
family tail is .015 in this 100-contact analysis and .031 in the different
HM225 250-contact family. Historical data reuse is not a new holdout test.

## Instrument contract and provenance

The original HM226 resource failure is preserved. HM227 failed a hidden-state
absolute tolerance after two checks; max state drift .000267029. The existing
log-odds drift was .0000228882, selected-logit .0000534058, and normalizer
.0000477151, all within the original 1e-4 output scale. HM228 explicitly
narrowed the instrument contract after these outputs; it did not rename the
failed gate PASS or retrospectively call this revision prospective.

Full-key row-blocked attention keeps the same model, inputs, causal/padding
mask, dtype, and row size. Original 27 batches/135 endpoints are inherited;
40 batches/200 endpoints were captured once. The interrupted original batch
027 was explicitly re-observed, not described as never attempted. HM228 model
time is 713.558914 seconds; including failed HM227, recovery time is
764.255671 seconds within the original 1200-second recovery cap. No backend
grid, training, API, private data, or output-fitting is used.

The two compatibility checks do not establish a uniform error bound for every
unobserved input, hidden coordinate, KV value, or future generation. Original
HM226 analysis/verifier code remains unchanged (its payload label is HM226);
HM228 manifests and receipts identify the separate recovery execution.
Target Gram/SVD description is explicitly post-outcome exploratory. All old
failed seals and all signed observations remain intact.

## Prior art and the next construction decision

- [Jiang et al.: statistical ranking and combinatorial Hodge theory](https://arxiv.org/abs/0811.1067) supplies the position/cycle machinery and the cardinal/ordinal distinction.
- [Xu et al.: non-transitivity in LLM-as-a-Judge](https://arxiv.org/abs/2502.14074) already studies LLM comparison cycles and baseline dependence.
- [Todd et al.: Function Vectors](https://arxiv.org/abs/2310.15213) already studies extraction and reuse of neural task actions; ordinary vector transport is not our new theorem.
- [OpenAI/math](https://github.com/openai/math), pinned adc7f124, free-group cocycle construction, motivates prefix-qualified actions only. Its representation assumptions are not established for natural text or psychometrics.

The present contribution is a SAME-author/SAME-event observation interface
that retains position, relationship, and presentation, and an empirical
decision that nominal targets change organization across tasks. The next
inventory is source-qualified target meaning and observation-law transport
(HM171/HM173/HM221 plus GRAIN's anonymous author/choice/state accounts).
Before a new finite run, specify the competing mechanisms and a new usable
construction: shared writer configuration versus target-specific valuation
versus task-induced target aliasing. More ranking edges, Bradley-Terry fits,
label flips, head/layer grids, or floating-point precision are not that
construction. Neural encoding/reading still remain; original18/100/44OPEN.
