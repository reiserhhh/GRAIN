# Condition-indexed writer response observations

2026-10-11. HM165 is PARTIAL_BUDGET, not a completed vector-response bridge.
The original18/100/44 scope remains OPEN. No accuracy promotion or new theorem.

## 1. The new observation, and its scope

An old controlled article is followed by a new self-choice task. The system
policy and article body are independently crossed at high, neutral and low
states. This is a re-rendered local-model continuation, not a human longitudinal
event or an uninterrupted original generation session. Generation seeds are
replicates, not authors. Counterfactual policy/body pairings are controls.

The neutral-policy contrast

\[
\Delta_X(a,c,o)=q(0,x_+,a,c,o)-q(0,x_-,a,c,o)
\]

asks whether the actual article bodies act on a subsequent response when the
current explicit personality prompt is absent. It differs from predicting the
persona label or asking another reader whether an author is extraverted.
The response is the local model's constrained A/B law, not human behavior.

On the completed prefix, this contrast averages +.039841 for free social
participation and +.035245 for required participation. The direct current-policy
contrasts average +.173479 and +.098395, respectively. These are signed effects
under two different interventions, not separable percentages of personality.
The presence of the historical-text channel is supported here; universal
orientation or population psychological meaning is not inferred from it.

## 2. Define an observation law before interpreting its interactions

Let ell(p,h,a,c,o) be semantically aligned A/B log odds. For this defined binary
protocol q=sigma(ell). The unrestricted option mass is separately retained.
At a neutral-policy/empty-history reference define

\[
\ell_{sep}(p,h)=\ell(p,0)+\ell(0,h)-\ell(0,0),\quad
J_\ell=\ell-\ell_{sep}.
\]

Then the probability interaction has the exact signed account

\[
J_q=
\underbrace{\sigma(\ell_{sep})-q(p,0)-q(0,h)+q(0,0)}_{C_{link}}
+\underbrace{q(p,h)-\sigma(\ell_{sep})}_{R_{response}}.
\]

No interaction is deleted. C_link is present even in a model whose log odds
are additive. R_response is the probability consequence of a nonadditive
log-odds table; it does not uniquely localize a neural interaction.
Neither is a psychological information fraction or an independently identified
natural indirect effect. This is standard finite factorial/GLM algebra.

For the matched high-high minus low-low diagonal, the free-social mean
interaction -.073866 splits into -.048323 link curvature and -.025542 response
remainder. In required-social choice, -.013035 splits into -.015280 and +.002245.
Thus a negative probability interaction need not mean the two evidence channels
inhibit each other internally. The remainder has a positive mean in the second
condition. All orders, positive/negative cells and cross-domain choices remain.

## 3. Protocol dependence is part of the measured response

After semantically aligning action labels, define

\[
\ell_{even}=(\ell_0+\ell_1)/2,\quad
\ell_{odd}=(\ell_0-\ell_1)/2,
\quad\ell_o=\ell_{even}+(-1)^o\ell_{odd}.
\]

Apply the same account independently to q. Averaging probabilities is NOT
sigmoid of averaged log odds. The even field is the balanced-order protocol
functional; it is not proved to be a position-free psychological trait.
The odd field is order susceptibility, not automatically noise to discard.
Different action orders are different model inputs, not an established gauge.

History-effect signs reverse across the two orders in 5/21 free-social pairs,
7/21 required-social, 10/21 known-planning and 12/21 unknown-planning. The
planning history means are near zero (+.001291/+.006350), but their order-even
RMS is .035392/.037995 and order-odd RMS .045613/.075726. A small pooled mean
does not imply absence of all text-conditioned response information.

## 4. Three-state contrasts distinguish polarity from nonneutral presence

Use the orthonormal columns

\[
b_0=(1,1,1)/\sqrt3,\quad b_1=(-1,0,1)/\sqrt2,
\quad b_2=(1,-2,1)/\sqrt6.
\]

For each full table Y, A=B^T Y B and Y=B A B^T. Retain all nine coordinates.
b1 is the high-minus-low contrast; b2 compares the two nonneutral states with
the middle control. For history, b2 also includes text-presence, length and
content differences; for policy it compares explicit prompts with the neutral
prompt. It is not a continuous quadratic personality response.

The two-diagonal natural contrast is blind to the polarity-by-polarity term:
that term is unchanged under simultaneous sign reversal. The diagonal can
still contain polarity-by-presence and presence-by-polarity interactions.
Consequently, one high/low score cannot stand in for the entire response table.
These are measurement contrasts, not nine newly discovered personality factors.

## 5. Consequence for the GRAIN mainline

Keep generation policy, article evidence, current response condition, observation
channel and diagnostic judgment distinct. An operational target may be

\[
T_{a,c,o,p}(x)=\ell(p,x,a,c,o),\qquad
\Theta_u(T)-\Theta_v(T)=\int T\,d(P_u-P_v).
\]

The second equation is an author functional only when P and the event/condition
support are specified; synthetic replicates here do not supply a human P.
The next constructive interface is to freeze source-observed response functions
and apply them to new article VECTORS before their response outcomes are opened.
Source covariance witnesses are an available construction, not calibrated
conditional expectations or a theorem of psychological identification.

HM165 planned this interface but did not finish its source capture. No source
program was compiled, no cold application was scored, and no application choices
were observed. Do not rerun the exhausted primary collection or pretend its
partial source became complete. A new finite construction must explicitly keep
the missing neuroticism/source/application cells pending, not change the goal.

## 6. Prior work and mathematical provenance

[Gilg et al. 2026](https://arxiv.org/html/2605.13339v3), Sections2.1/2.3,
AppendixA.3 biography injection and AppendixL.2 already study evaluative vectors,
fine-grained context changes, positional versus task-identity effects. These
ideas are not our field-wide discoveries. Their learned probe is not copied
here; the response table and prior GRAIN vectors remain separate objects.

[Han et al. 2025](https://arxiv.org/html/2509.03730v2), discussion on alignment
and prompting, motivates separating linguistic trait expression from behavior.
[Zhou et al. 2026](https://arxiv.org/html/2609.32758v1), abstract/introduction,
motivates separating source reuse from target recalibration. No target refit,
temperature tuning, persona accuracy grid or new neural training occurs here.

OpenAI/math pinned adc7f124, Thorp-sweep section05 inverse-color/trace arguments,
was reread. Its separation of conditional laws is a design inspiration only;
permutation independence, density bounds and spectral conclusions have not been
established for the text model. The actual calculations use standard factorial
contrasts, binary links and finite linear transformations, not a transferred
advanced theorem. The contribution is a concrete observation interface and its
conditional limitations on the SAME historical article IDs.

## 7. Reproducible finite account

Primary: results/grain_math_response_v1/writer_history_bridge_r1.
37 real32-input batches,1184/5904 unique histories,493.199781 model seconds.
The480-second cap is checked between atomic batches; the final batch overshoot
is recorded, not hidden or renewed.168/864 full tables and21/108 complete pairs.
Only extraversion/source diary-email-opinion prefix; not a balanced sample.

Separate post-outcome companions preserve partial and observation-law descriptions.
All3024 contrast coordinates and1512 link cells are retained. Alternate raw-logit
replay uses tanh and explicit finite sums;1184 draws agree, max numerical error
1.78e-15. Same-agent arithmetic audit, not independent behavioral replication.
39 focused tests and9 Python lint/format checks pass; those are engineering
checks, not a claim that the original18/100/44 research scope is complete.
