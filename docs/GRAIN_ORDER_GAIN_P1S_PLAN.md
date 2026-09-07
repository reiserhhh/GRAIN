# P1S: paired encoding amplitude versus finite conditional gain

2026-09-07. Freeze before calculation. Cache-only follow-up to completed P1R;
zero additional neural calls, CPU cap60seconds, no new fitting or private data.
Question: which part of the fixed-direction response changes under record order?

Use ALL64 matched person cases in P1P(original)/P1R(reversed), including both
criteria, roles, room facts and targets. The direction d is unchanged. For each
order o, compute signed projection amplitude a_o=d.T(n_donor-n_recipient), and
projected logit shift y_o=f_o(n_o+a_o*d)-f_o(n_o). Then g_o=y_o/a_o is a finite
directional gain, not a derivative. If any |a_o|<1e-6, retain and flag the row,
stop gain promotion; do not drop it or pick another direction.

Exact symmetric paired accounting:

    y_1-y_0 = mean(g)*(a_1-a_0) + mean(a)*(g_1-g_0)
              amplitude term          gain term.

Both components can correlate/cancel. Report squared terms AND signed cross term
per criterion and setting. This is descriptive accounting of a causal order
manipulation, not two independently isolated causal mechanisms. Gain includes
baseline anchor, all other recipient states and the subsequent network. No term
is a personality score, probability or semantic-information percentage.

Also make one frozen predictive check: y_pred=g_original*a_reversed. Compare
its SSE with unchanged y_original and zero against y_reversed. It uses no reversed
output during prediction and fits no parameter, but requires the original case's
LLM response and is not an LLM-free or new-person prediction model. Exact symmetric
accounting uses both outputs and must NOT be advertised as prediction.

Recompute remaining-complement and readout-interaction energies at both orders:
I=f_natural-f_projected-f_complement+f_base. Retain their cross term and native
answer errors. Always report all4 settings rather than choosing a favorable one.
Save64 rows,2 criterion summaries,8 setting summaries and16 order/criterion/setting
remainder summaries plus4 pooled order/criterion rows. Independent reconstruction
uses stored arrays and raw logits; synthetic tests cover identity/cancellation,
constant-gain transfer and near-zero rejection. Separate source snapshot/audit.

Continuation depends on the result: amplitude-only success supports a narrow
conditional transport approximation; changing gain requires context-sensitive
functional geometry, not just vector-length normalization. Neither result removes
native answer bias or proves psychological validity. Do not launch a larger model,
rank search or old source queue as an automatic follow-up.
