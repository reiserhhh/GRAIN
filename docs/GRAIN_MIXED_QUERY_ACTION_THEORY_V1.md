# Query-Conditioned Evidence Action: An Addressed Differential Law

## Object and Domain

Fix a token layout, receiver body/system states B, mapped property states H,
and final query state Q at the layer16 hook. Earlier query states P vary in
their own ordered physical coordinates. The frozen decoder defines a scalar
digit-alphabet readout F(H,B,P,Q). This is a model observation function, not
the author's personality or a learned author response operator.

For the underlying real-valued smooth computation, alpha(P)=d_H F is an H
covector. The conditional mixed map K(P)=d_P alpha maps a P tangent to an H
covector. Local changes of evidence action can therefore be expressed without
semantic factor names. H and P remain separately addressed; their coordinates
must not be pooled into one undifferentiated embedding.

## Exact Differential Identity and Its Numerical Approximation

For gamma(t)=P1+t(P0-P1), DeltaP=P0-P1, and a fixed H tangent deltaH:

    <alpha(P0)-alpha(P1), deltaH>
      = integral_0^1 <K(gamma(t))[DeltaP], deltaH> dt.

Proof: differentiate the scalar a(t)=<alpha(gamma(t)),deltaH> by the chain
rule, then integrate a'(t). This is the ordinary fundamental theorem of
calculus on a fixed finite-dimensional coordinate domain, not a new theorem.
Changing B, Q or token layout while differentiating would require additional
terms or a new chart; the HM35 experiment does neither.

HM35 differentiates the same live leaf graph twice. It obtains a P covector
v_t=d_P<alpha,deltaH>, which can be contracted with any predeclared aligned P
direction. No large dense Hessian is formed. Fixed t=0,.5,1 gives trapezoid
and Simpson approximations; both are retained. The floating implementation
and quadrature errors are distinct from this real-valued identity.

The original physical H step is a rounded float32 perturbation. Its recorded
half-displacement is used, not a symbolically equivalent reordering that
could change the executed endpoints. The integrated contrast concerns the
gradient action. Comparing it with native finite-H actions introduces a
separate finite-H approximation error.

## Source Support Is Not Target Activity

Let D be the frozen HM34 source-displacement matrix and PiD its orthogonal
projector in the original physical metric. Decompose DeltaP into

    s=PiD DeltaP; u=(I-PiD) DeltaP.
    <v_t,DeltaP>=<v_t,s>+<v_t,u>.

Both terms are evaluated on the SAME full gamma path, and both are signed.
They may cancel. Squared total action contains the cross term2*s_action*u_action.
Neither ||u||^2/||DeltaP||^2 nor a squared-action fraction is a fraction of
psychological information. An unsupported direction can be inactive, active,
or counteract a supported direction.

The two original H directions define only a partial target-action observation
family. Its local quadratic form sum_arm(v_arm v_arm^T) is positive semidefinite
but has rank at most2 at each node. It is not a complete Hessian, a complete
personality metric, or evidence that all remaining directions are noise.
It suggests a next mechanism experiment: source-only observation directions
chosen by target activity, rather than arbitrary vector variance. Actual HM35
results must determine whether that is the next useful step.

## Coordinate Discipline

Under a change P'=T P, tangents transform by T and covectors by T^{-T}; their
pairing is unchanged. An orthogonal projector in physical Euclidean
coordinates generally becomes a metric projector in arbitrary new
coordinates. Recomputing a Euclidean projector after a nonorthogonal transform
would change the scientific question. The current protocol keeps the original
metric and ordered token coordinates instead of making an invariance claim.

## Mainline Interpretation

This law separates three previously entangled objects: available source
directions, query-state-dependent reading, and finite evidence displacement.
It can explain why the same text evidence affects different target queries
differently. It does not by itself identify which response is psychologically
valid. The encoder up to the zero-based layer-index16 output hook also remains;
local mathematical analysis is not yet an entirely LLM-free measurement tool.

The OpenAI math expansion argument inspired retaining a common observation
domain and testing a family of directional views. Its Laurent-field rank
theorems are not used as proofs about this floating decoder. The actual
justification here is the explicitly stated differential identity and the
addressed intervention data.
