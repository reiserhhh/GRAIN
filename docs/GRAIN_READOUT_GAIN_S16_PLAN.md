# S16: separate readout gain, comment allocation and conditional response

2026-09-08. Next bounded mainline experiment after full S15. Not yet collected.

## Question

Does S15's mass advantage come from total body exposure B or from selecting
different comments? Can a second conditional cumulant account for its remaining
within-comment response error? Neither question needs a new questionnaire fit.

For the first24 fixed S15 source authors, same3views/common query and frozen
reader, compute B=sum_G P_G, r_G=P_G/B and pi_G=|G|/sum_H|H|.
For centered values and centered L=q^T K within G, define
D_G=exact tilted-mean difference,
L_G=Cov_G(V,L), Q_G=E_G[(V-E_G V)(L-E_G L)^2].

Five fixed readouts, summed over heads and passed to the existing reader:

1. B sum_G r_G D_G: exact source replay;
2. B sum_G pi_G D_G: retain gain, remove comment-selective mass;
3. B sum_G r_G L_G: S15 exact-mass covariance replay;
4. B sum_G pi_G L_G: gain-only covariance;
5. B sum_G r_G (L_G+Q_G/2): fixed second-order conditional readout.

S15's uniform-mass covariance is already available as a secondary replay
comparison. Do not refit, rescale, flip signs, choose moments on Y, or attach
psychological names to heads. Outside-body tokens remain in B's normalizer.

## Measurements and stop

Record internal fidelity r, MSE normalized by exact author-score variance,
bias/variance of error, all12 heads' body/outside/first-token attention masses,
within-body comment entropy and readout contributions. Describe mass separately
from signed functional contribution; large attention does not imply large beta
contribution. No human outcome join on this24-author mechanism pilot.

72 prefix calls,0 suffix/backward calls, ten-minute cap. All three replays must
match S14/S15 within1e-8 in scalar scores. Complete this fixed pilot, then choose
the next mathematical question from measured gain/allocation/moment behavior.
Do not automatically expand to213 or start a new neural queue merely to fill it.

The underlying equations and verified primary papers are in
`GRAIN_ATTENTION_MASS_MECHANISM_NOTE.md` and theory V4. The attention-sink papers
motivate gain inspection but their necessity results are not assumed for Qwen.

Output: `results/grain_vector_mechanisms/readout_gain_s16_r1/` (ignored).
Report: `reports/grain_vector_mechanisms/S16_READOUT_GAIN_REPORT.md`.
New runner should reuse the proven prefix extractor and S15 source validation,
leaving all frozen S12-S15 implementations and artifacts unchanged.
