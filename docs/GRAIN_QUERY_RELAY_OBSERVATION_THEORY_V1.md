# Content-Independent Query Relays and Target Observation

HM201 construction and empirical update, 2026-10-12 JST. Sections1-4 were
written before outcome analysis; Section5 records the completed fixed run.
Scientific inputs and outcomes remain sealed, no rescoring or precision search.

## A finite observation problem, not a score optimization

Let B contain all excerpt tokens, R all post-body rows before the final row f,
and C the common pre-body prefix. HM200 changes only the final row's conditional
body selector. It leaves R's already body-conditioned representations intact.
Thus preserved SOURCE contact can originate in direct B->f reading, B->R->f
relay, or their interactions. Neither raw attention nor derivative allocation
alone distinguishes these possibilities.

HM201 changes the observation operator, not the writer or personality label.
The all-layer mask removes each B->R edge while retaining B->f edges. This is
graph surgery, not natural text denoising or input-independent degradation.

## Architectural content blindness

For fixed token addresses, pre-body prefix and query words, a causal layer is
row-local except for its attention-weighted value combination. Under the cut,

    h_R^(l+1) = T_R,l(h_C^l, h_R^l; mask),
    h_B^(l+1) = T_B,l(h_C^l, h_B^l; native causal mask).

R cannot access f by causality. At l=0 its token/position embeddings are fixed.
Induction implies h_R^l is independent of body token content for every l.
Changing body length changes positions and support: length must remain a
declared condition, not be silently removed as noise. The tiny-network test
changes body IDs at equal length and checks every relay row in every layer;
the final endpoint still changes, demonstrating the retained direct route.

This suggests a conditional query template bank:

    K_R,l(q; L,C), V_R,l(q; L,C),
    h_f^(l+1) = T_f,l(h_f^l, K_B,l(x),V_B,l(x),
                     K_R,l(q;L,C),V_R,l(q;L,C), K_C,l,V_C,l).

The bank can in principle be compiled without the new author's body content.
Final recursion is still nonlinear and neural; this does NOT establish an
LLM-free raw-text scorer. Surviving SOURCE contacts motivate a smaller target
observation interface. Relay-dependent contacts require a source-supported
summary instead. No compiler fidelity grid before this empirical decision.

## Two intervention coordinates and signed interaction

Let D_ab be the deletion response, a=relay cut, b=foreign final selector.
Keep the same intact donor for b=1 even when a=1. C is unexchanged. Then

    Delta_R = D10-D00,
    Delta_S = D01-D00,
    Delta_RS = D11-D10-D01+D00,
    D11-D00 = Delta_R+Delta_S+Delta_RS.

This finite factorial identity is standard, not additive causal mediation.
The cut changes softmax denominators and downstream nonlinear states. For
fixed SOURCE action X covariance distributes over signed effects, correlations
and squared norms do not. Do not omit cross terms or call effects personality
information fractions.

## Prior work and prospective empirical contribution

HM42/43 already allocate derivatives across body/query/remainder at one cut.
HM86/87 examine selection/contextual values; HM155/156 qualified downstream
carriers. Here the new interface is an actual all-layer forward relay cut on
SAME natural evidence, frozen five-target SOURCE fields and intact selectors.
It is not the first finding that query/context routes exist.

[Path-patching guidance](https://arxiv.org/html/2404.15255v1) supplies mature
mediation/amplification and redundant-route distinctions. [Attention
identifiability](https://arxiv.org/html/1908.04211v2) already challenges unique
attention explanations. [OpenAI/math fixed three-state foundations](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
inspires explicit observation operators; its input-independent degradation
and conditionally independent branches are NOT assumed for this model.

The pre-outcome question was which target-supported action remains and what
it enables us to construct. Original18/100/44 stay OPEN.

## Empirical update: route-conditioned meanings, not a universal body-only map

The fixed194 batches/970 inquiries completed in1085.797725 model-seconds.
Every body-to-relay weight is exactly0;body K/V across queries and preserved
outside weights remain unchanged. All five targets and four corners were
analyzed once, then independently recomputed arithmetically by the same agent.

A SOURCE/full contact changes from .699/.532 to .338/.392 under relay cut;
with the frozen foreign selector it is .346/.428. The first SOURCE bank's
geometric covariance change is -6.092985e-5 with conditional95%CI
[-9.644096e-5,-2.346093e-5];the second is -2.922393e-5 with CI crossing0.
The contact is attenuated,not abolished or converted into a psychological
accuracy score. Both application views retain positive A full contacts.

More discriminating is the unchanged GRAIN complement: A .479/.488 becomes
-.107/-.148. Both banks' covariance-change CIs are negative:
[-1.111099e-4,-2.202018e-5] and[-7.913016e-5,-1.398824e-5]. The retained
GRAIN15 part changes .236/.124 -> .589/.557, but its positive covariance
changes have CIs crossing0. Correlation increases are not evidence for a
better measurement or greater information: variances and signed components
also change. The full retained+complement covariance identity still holds.

N full contacts remain positive at .500/.314;cut+foreign .584/.462. Its
two relay covariance-change CIs cross0. Historical held-author N bank1
remains negative -.233/-.087;do not use pooled positivity to erase that.
E/C/O all remain recorded,not optimized. Most interaction covariance CIs
cross0;the tiny C interaction is a self-control scale,not a new mechanism.

This gives an operational distinction beyond HM200: precise final body
selection was not necessary for A contact, whereas an evidence-conditioned
pre-answer relay materially changes how the complement contacts the target.
It does NOT prove a unique semantic mediator: the cut also renormalizes
attention and changes downstream states, and generic context/coherence
remains a competing explanation. This is a controlled local model mechanism,
not a human neural route or universal psychological information fraction.

For a frozen SOURCE coordinate X_t and each declared route configuration a,b,

    C_t,ab = Cov(X_t,D_ab),
    C_t,10 = C_t,00 + Cov(X_t,Delta_R).

The SOURCE coordinate alone does not specify the observation meaning. A
target-measurement compiler must declare evidence encoding, relay construction
and endpoint valuation, while preserving choice/state/qualification/reference.
The body-only observation is usable for some contacts,but is not the complete
replacement for the original reading law. Do not tune the cut to restore r.

Next inventory SAME-event relay encoding/carrier and endpoint-reading
interfaces;distinguish SOURCE-linked summary from generic query normalization
before designing a conditional summary compiler. Any new intervention must
add construction/identification ability,not another fidelity grid. Current
E5/Qwen dependencies remain. This is new system-specific empirical evidence,
not a new chain-rule or tree-reconstruction theorem.
