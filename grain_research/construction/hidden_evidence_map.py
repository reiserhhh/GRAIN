"""GRAIN-C1-01: hidden evidence map, target response versus author fingerprint.

A numeric high-dimensional observation layer with an unknown linear mixing map.
Each event vector x mixes a target expression e = a_u + b_u z + noise with a stable
author fingerprint f_u, a target-irrelevant author-by-context style response r_u z,
topic material, shared context content and wording noise.

Estimators receive only an ``EstimatorView``: source and application observation
vectors, recorded conditions, opaque author indices, a pilot bank with independently
declared labels (when the world declares them) and a public lexicon. Hidden maps,
latents and true expressions live in ``EvaluatorTruth`` and are read only by the
evaluator and by the explicitly privileged oracle arms.

Standard pieces reused: classical calibration (projection regressed on an exact
declared label), ridge / principal-component regression, Fisher discriminants,
the F04 noise-corrected random-coefficient moments and its Gaussian BLUP.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields

import numpy as np
from scipy.linalg import eigh

CONDITIONS = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
D = 64
KF = 4
KT = 12
SOURCE_TOPICS = tuple(range(8))
HELD_TOPICS = tuple(range(8, 12))
LEXICON_SIZE = 8
G_E = 1.5
FP_SD = 2.0
STYLE_SD = 1.0
TOPIC_GAIN = 1.5
CTX_GAIN = 1.0
WORD_SD = 1.0
EVENT_SD = 0.5
PILOT_LABEL_SD = 1.25
N_SOURCE, N_APP, EVENTS = 300, 200, 32
PILOT_WRITERS, PILOT_PER_WRITER, PILOT_TRAIN_WRITERS = 12, 16, 9
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
PCR_GRID = (2, 4, 8, 16, 32)

WORLDS = {
    "W1_shared_level": {"slope": False, "pilot": "expression", "targets": ("level",)},
    "W2_conditional": {"slope": True, "pilot": "expression", "targets": ("delta", "level")},
    "W3_missing_anchor": {"slope": True, "pilot": "topic_only", "targets": ("delta", "level")},
    "W4_confounded_anchor": {"slope": True, "pilot": "confounded", "targets": ("delta", "level")},
}
TARGET_VECTORS = {
    "level": np.array([1.0, 0.0]),
    "delta": np.array([0.0, 2.0]),
    "held_m_1/3": np.array([1.0, 1.0 / 3.0]),
}


def _unit(v):
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class EstimatorView:
    """Everything a non-privileged estimator may read."""

    source_x: np.ndarray
    source_z: np.ndarray
    source_author: np.ndarray
    app_x: np.ndarray
    app_z: np.ndarray
    app_author: np.ndarray
    pilot_x: np.ndarray
    pilot_writer: np.ndarray
    pilot_label: np.ndarray | None
    pilot_label_kind: str
    lexicon_index: np.ndarray
    lexicon_sign: np.ndarray


@dataclass(frozen=True)
class EvaluatorTruth:
    """Hidden generator state. Never passed to a non-privileged estimator."""

    maps: dict
    source_latent: dict
    app_latent: dict
    pilot_latent: dict


ESTIMATOR_FIELDS = tuple(f.name for f in fields(EstimatorView))
FORBIDDEN_IN_VIEW = ("maps", "theta", "fingerprint", "style", "expression", "topic", "a", "b")


def world_maps(seed):
    """Seed-specific unknown mixing map shared by all worlds of that seed."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1601, 0]))
    sign = rng.choice([-1.0, 1.0], LEXICON_SIZE)
    lex = np.zeros(D)
    lex[:LEXICON_SIZE] = sign
    rest = np.zeros(D)
    rest[LEXICON_SIZE:] = rng.normal(size=D - LEXICON_SIZE)
    u_e = _unit(_unit(lex) + _unit(rest))
    return {
        "u_e": u_e,
        "U_f": rng.normal(size=(D, KF)) / np.sqrt(D),
        "u_s": _unit(rng.normal(size=D)),
        "U_t": rng.normal(size=(D, KT)) / np.sqrt(D) * TOPIC_GAIN,
        "u_c": _unit(rng.normal(size=D)) * CTX_GAIN,
        "lexicon_index": np.arange(LEXICON_SIZE),
        "lexicon_sign": sign,
    }


def emit(maps, e, f, s, topic, z, noise):
    """x = G_E u_e e + U_f f + u_s s z + U_t[:, topic] + u_c z + WORD_SD noise."""
    return (
        G_E * np.outer(e, maps["u_e"])
        + f @ maps["U_f"].T
        + np.outer(s * z, maps["u_s"])
        + maps["U_t"][:, topic].T
        + np.outer(z, maps["u_c"])
        + WORD_SD * noise
    )


def author_latents(seed, world, role, n, slope):
    """Author parameters and event draws with role-separated random streams."""
    w = list(WORLDS).index(world)
    r = ("source", "application").index(role)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1602, r, 0]))
    a = rng.normal(size=n)
    b = rng.normal(0.5, 0.5, size=n) if slope else np.zeros(n)
    f = rng.normal(0.0, FP_SD, size=(n, KF))
    s = rng.normal(0.0, STYLE_SD, size=n)
    ev = np.random.default_rng(np.random.SeedSequence([seed, 1602, r, 1]))
    z = np.tile(np.array(CONDITIONS), EVENTS // len(CONDITIONS))
    topic = ev.choice(SOURCE_TOPICS, size=(n, EVENTS))
    eps = ev.normal(0.0, EVENT_SD, size=(n, EVENTS))
    noise = ev.normal(size=(n, EVENTS, D))
    # world index only selects slope on/off; latents are common-random across worlds
    del w
    return {"a": a, "b": b, "f": f, "s": s, "z": z, "topic": topic, "eps": eps, "noise": noise}


def render_authors(maps, lat):
    """Emit all events of all authors from latents (used for CRN interventions)."""
    n = len(lat["a"])
    z = np.broadcast_to(lat["z"], (n, EVENTS))
    e = lat["a"][:, None] + lat["b"][:, None] * z + lat["eps"]
    x = np.empty((n, EVENTS, D))
    for u in range(n):
        x[u] = emit(
            maps, e[u], np.broadcast_to(lat["f"][u], (EVENTS, KF)),
            np.full(EVENTS, lat["s"][u]), lat["topic"][u], z[u], lat["noise"][u],
        )
    return x, e


def pilot_bank(seed, maps, kind):
    """Authored pilot texts by separate writers with declared labels."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1603, 0]))
    n = PILOT_WRITERS * PILOT_PER_WRITER
    writer = np.repeat(np.arange(PILOT_WRITERS), PILOT_PER_WRITER)
    fw = rng.normal(0.0, FP_SD, size=(PILOT_WRITERS, KF))
    sw = rng.normal(0.0, STYLE_SD, size=PILOT_WRITERS)
    z = rng.choice(CONDITIONS, size=n)
    topic = rng.choice(SOURCE_TOPICS, size=n)
    xi = rng.normal(size=n)
    if kind == "confounded":
        v = _unit(rng.normal(size=KF))
        c = (fw @ v) / FP_SD
        e = PILOT_LABEL_SD * (0.9 * c[writer] + np.sqrt(1 - 0.81) * xi)
    else:
        e = PILOT_LABEL_SD * xi
    noise = rng.normal(size=(n, D))
    x = emit(maps, e, fw[writer], sw[writer], topic, z, noise)
    label = None if kind == "topic_only" else e.copy()
    return x, writer, label, {"e": e, "topic": topic, "z": z, "fw": fw, "sw": sw}


def build_world(seed, world):
    """Return (EstimatorView, EvaluatorTruth) for one seed and world."""
    cfg = WORLDS[world]
    maps = world_maps(seed)
    src = author_latents(seed, world, "source", N_SOURCE, cfg["slope"])
    app = author_latents(seed, world, "application", N_APP, cfg["slope"])
    xs, es = render_authors(maps, src)
    xa, ea = render_authors(maps, app)
    px, pw, plab, plat = pilot_bank(seed, maps, cfg["pilot"])
    src["e"], app["e"] = es, ea
    view = EstimatorView(
        source_x=xs.reshape(-1, D),
        source_z=np.tile(src["z"], N_SOURCE),
        source_author=np.repeat(np.arange(N_SOURCE), EVENTS),
        app_x=xa.reshape(-1, D),
        app_z=np.tile(app["z"], N_APP),
        app_author=np.repeat(np.arange(N_APP), EVENTS),
        pilot_x=px,
        pilot_writer=pw,
        pilot_label=plab,
        pilot_label_kind={"expression": "declared_expression", "confounded": "declared_expression",
                          "topic_only": "declared_topic_only"}[cfg["pilot"]],
        lexicon_index=maps["lexicon_index"].copy(),
        lexicon_sign=maps["lexicon_sign"].copy(),
    )
    truth = EvaluatorTruth(maps=maps, source_latent=src, app_latent=app, pilot_latent=plat)
    return view, truth


def view_digest(view):
    """Hash of every estimator-visible array (isolation evidence)."""
    h = hashlib.sha256()
    for name in ESTIMATOR_FIELDS:
        value = getattr(view, name)
        h.update(name.encode())
        if value is None:
            h.update(b"None")
        elif isinstance(value, str):
            h.update(value.encode())
        else:
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- readout arms
@dataclass
class Readout:
    arm: str
    direction: np.ndarray | None
    offset: float = 0.0
    unit: float = 1.0
    status: str = "OK"
    info: dict | None = None

    def phi(self, x):
        return (x @ self.direction - self.offset) / self.unit


ANCHOR_T_MIN = 4.0


def calibrate(projection, label, with_t=False):
    """Classical calibration: projection = alpha + gamma * declared label."""
    xm = np.column_stack([np.ones(len(label)), label])
    (alpha, gamma), *_ = np.linalg.lstsq(xm, projection, rcond=None)
    if not with_t:
        return float(alpha), float(gamma)
    resid = projection - xm @ np.array([alpha, gamma])
    s2 = float(resid @ resid) / max(len(label) - 2, 1)
    se = np.sqrt(s2 * np.linalg.inv(xm.T @ xm)[1, 1])
    return float(alpha), float(gamma), float(gamma / se) if se > 0 else np.inf


def _ridge_fit(x, y, lam):
    xm, ym = x.mean(0), y.mean()
    xc = x - xm
    w = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ (y - ym))
    return w


def _lowo_select(x, y, writer, grid, fit):
    """Leave-one-writer-out choice over a fixed grid, pilot train writers only."""
    scores = []
    for g in grid:
        err = 0.0
        for w in np.unique(writer):
            tr, te = writer != w, writer == w
            d = fit(x[tr], y[tr], g)
            p = x @ d
            a, c = calibrate(p[tr], y[tr])
            pred = (p[te] - a) / c
            err += float(np.sum((pred - y[te]) ** 2))
        scores.append(err / len(y))
    return grid[int(np.argmin(scores))], dict(zip([str(g) for g in grid], scores))


def source_pcs(view):
    xc = view.source_x - view.source_x.mean(0)
    _, sv, vt = np.linalg.svd(xc, full_matrices=False)
    return vt, sv


def direction_author_stability(view, ridge=1e-3):
    """Top Fisher author discriminant on source data (label-free)."""
    x, a = view.source_x, view.source_author
    n = a.max() + 1
    means = np.stack([x[a == u].mean(0) for u in range(n)])
    within = x - means[a]
    sw = within.T @ within / (len(x) - n)
    sb = np.cov(means, rowvar=False) - sw / EVENTS
    vals, vecs = eigh(sb, sw + ridge * np.trace(sw) / D * np.eye(D))
    return _unit(vecs[:, -1]), float(vals[-1])


def direction_response_stability(view, ridge=1e-3):
    """Top random-slope component: between-author slope covariance vs sampling."""
    x, z, a = view.source_x, view.source_z, view.source_author
    n = a.max() + 1
    zz = z[a == 0]
    xm = np.column_stack([np.ones(len(zz)), zz])
    proj = np.linalg.solve(xm.T @ xm, xm.T)
    xs = x.reshape(n, len(zz), D)
    coef = np.einsum("ke,ued->ukd", proj, xs)
    resid = xs - np.einsum("ek,ukd->ued", xm, coef)
    s2 = np.einsum("ued,uef->df", resid, resid) / (n * (len(zz) - 2))
    sampling = s2 * np.linalg.inv(xm.T @ xm)[1, 1]
    between = np.cov(coef[:, 1, :], rowvar=False) - sampling
    vals, vecs = eigh(between, sampling + ridge * np.trace(sampling) / D * np.eye(D))
    return _unit(vecs[:, -1]), float(vals[-1])


def _orient_label_free(d):
    """Deterministic sign rule when no anchor exists: largest |loading| positive."""
    return d * np.sign(d[np.argmax(np.abs(d))])


NUISANCE_AUTHOR_COMPONENTS = 4
NUISANCE_RESPONSE_COMPONENTS = 2


def _author_moments(view):
    x, a = view.source_x, view.source_author
    n = a.max() + 1
    means = np.stack([x[a == u].mean(0) for u in range(n)])
    within = x - means[a]
    sw = within.T @ within / (len(x) - n)
    sb = np.cov(means, rowvar=False) - sw / EVENTS
    return sb, sw


def _slope_moments(view):
    x, z, a = view.source_x, view.source_z, view.source_author
    n = a.max() + 1
    zz = z[a == 0]
    xm = np.column_stack([np.ones(len(zz)), zz])
    proj = np.linalg.solve(xm.T @ xm, xm.T)
    xs = x.reshape(n, len(zz), D)
    coef = np.einsum("ke,ued->ukd", proj, xs)
    resid = xs - np.einsum("ek,ukd->ued", xm, coef)
    s2 = np.einsum("ued,uef->df", resid, resid) / (n * (len(zz) - 2))
    sampling = s2 * np.linalg.inv(xm.T @ xm)[1, 1]
    between = np.cov(coef[:, 1, :], rowvar=False) - sampling
    return between, sampling


def nuisance_patterns(view, px, py, ridge=1e-3):
    """C1 r2 supplement: stable source components the pilot cannot anchor.

    Filters are top generalized eigenvectors of author-level and author-by-context
    (slope) between-author covariance. A filter is anchored when its pilot
    calibration |t| >= ANCHOR_T_MIN. Unanchored filters contribute their source
    activation pattern (between covariance times filter) to the nuisance set.
    """
    patterns, log = [], []
    for name, (between, noise), m in (
        ("author", _author_moments(view), NUISANCE_AUTHOR_COMPONENTS),
        ("response", _slope_moments(view), NUISANCE_RESPONSE_COMPONENTS),
    ):
        vals, vecs = eigh(between, noise + ridge * np.trace(noise) / D * np.eye(D))
        for k in range(1, m + 1):
            filt = vecs[:, -k]
            _, _, t = calibrate(px @ filt, py, with_t=True)
            anchored = abs(t) >= ANCHOR_T_MIN
            log.append({"family": name, "rank": k, "eigenvalue": float(vals[-k]), "anchor_t": float(t),
                        "anchored": bool(anchored)})
            if not anchored:
                patterns.append(_unit(between @ filt))
    return (np.column_stack(patterns) if patterns else np.zeros((D, 0))), log


def fit_arm(arm, view, train_mask=None, fixed=None, keyed_declared_sign=False):
    """Build one readout from the estimator view only (plus pilot train rows)."""
    has_label = view.pilot_label is not None
    if train_mask is None:
        train_mask = view.pilot_writer < PILOT_TRAIN_WRITERS
    px, pw = view.pilot_x[train_mask], view.pilot_writer[train_mask]
    py = view.pilot_label[train_mask] if has_label else None
    fixed = fixed or {}
    info = {"uses": [], "free_parameters": 0}
    if arm in ("np_ridge", "np_pcr"):
        if not has_label:
            return Readout(arm, None, status="MISSING_ANCHOR",
                           info={"reason": "pilot declares no label for the target expression"})
        base = fit_arm("pilot_ridge" if arm == "np_ridge" else "pilot_pcr", view, train_mask,
                       {k: v for k, v in fixed.items() if k != "nuisance"} if fixed else None)
        nmat = fixed.get("nuisance") if fixed else None
        nlog = None
        if nmat is None:
            nmat, nlog = nuisance_patterns(view, px, py)
        d = base.direction
        if nmat.shape[1]:
            d = d - nmat @ np.linalg.lstsq(nmat, d, rcond=None)[0]
        info = {**base.info, "uses": base.info["uses"] + ["source_x(nuisance patterns)", "source_z"],
                "free_parameters": base.info["free_parameters"], "nuisance": nmat, "nuisance_log": nlog,
                "nuisance_removed": int(nmat.shape[1])}
        alpha, gamma, t = calibrate(px @ d, py, with_t=True)
        info["anchor_t"] = t
        status = "OK" if abs(t) >= ANCHOR_T_MIN else "ANCHOR_UNSUPPORTED_DIRECTION"
        return Readout(arm, d, alpha, gamma, status, info)
    if arm in ("pilot_ridge", "pilot_pcr") and not has_label:
        return Readout(arm, None, status="MISSING_ANCHOR",
                       info={"reason": "pilot declares no label for the target expression"})
    if arm == "pilot_ridge":
        lam = fixed.get("lam")
        cv = None
        if lam is None:
            lam, cv = _lowo_select(px, py, pw, RIDGE_GRID, _ridge_fit)
        d = _ridge_fit(px, py, lam)
        info = {"uses": ["pilot_x", "pilot_label", "pilot_writer(CV)"], "free_parameters": D + 2,
                "lambda": lam, "cv": cv}
    elif arm == "pilot_pcr":
        vt = fixed.get("vt")
        if vt is None:
            vt, _ = source_pcs(view)
        k = fixed.get("k")
        cv = None

        def pcr(x, y, kk):
            return vt[:kk].T @ _ridge_fit(x @ vt[:kk].T, y, 1e-8)

        if k is None:
            k, cv = _lowo_select(px, py, pw, PCR_GRID, pcr)
        d = pcr(px, py, k)
        info = {"uses": ["source_x(SVD)", "pilot_x", "pilot_label", "pilot_writer(CV)"],
                "free_parameters": int(k) + 2, "k": int(k), "cv": cv, "vt": vt}
    elif arm == "keyed":
        d = np.zeros(D)
        d[view.lexicon_index] = view.lexicon_sign / LEXICON_SIZE
        info = {"uses": ["public_lexicon"] + (["pilot_label(unit)"] if has_label else []),
                "free_parameters": 2 if has_label else 0}
    elif arm == "svd1":
        vt = fixed.get("vt")
        if vt is None:
            vt, _ = source_pcs(view)
        d = vt[0].copy()
        info = {"uses": ["source_x(SVD)"] + (["pilot_label(unit,sign)"] if has_label else []),
                "free_parameters": 2 if has_label else 0, "vt": vt}
    elif arm == "stability_author":
        d = fixed.get("d")
        if d is None:
            d, _ = direction_author_stability(view)
        info = {"uses": ["source_x", "source_author"] + (["pilot_label(unit,sign)"] if has_label else []),
                "free_parameters": 2 if has_label else 0, "d": d}
    elif arm == "stability_response":
        d = fixed.get("d")
        if d is None:
            d, _ = direction_response_stability(view)
        info = {"uses": ["source_x", "source_z", "source_author"] + (["pilot_label(unit,sign)"] if has_label else []),
                "free_parameters": 2 if has_label else 0, "d": d}
    else:
        raise ValueError(arm)
    if has_label:
        alpha, gamma, t = calibrate(px @ d, py, with_t=True)
        info["anchor_t"] = t
        status = "OK" if abs(t) >= ANCHOR_T_MIN else "ANCHOR_UNSUPPORTED_DIRECTION"
        return Readout(arm, d, alpha, gamma, status, info)
    if not (arm == "keyed" and keyed_declared_sign):
        d = _orient_label_free(d)
    return Readout(arm, d, 0.0, 1.0, "UNIT_AND_ORIENTATION_UNIDENTIFIED", info)


def oracle_readouts(view, truth):
    """Privileged references: true event expression, and a least-squares readout of it."""
    e = truth.source_latent["e"].reshape(-1)
    x = view.source_x
    xm = x.mean(0)
    w = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x - xm]), e, rcond=None)[0]
    d = w[1:]
    alpha, gamma = calibrate(x @ d, e)
    return Readout("oracle_map", d, alpha, gamma, "PRIVILEGED",
                   {"uses": ["TRUE source expression", "source_x"], "free_parameters": D + 1})


# ---------------------------------------------------------------- response kernel
def design(z):
    return np.column_stack([np.ones(len(z)), z])


def fit_moments(phi, z, author):
    """Vectorized F04 fit_source_moments (linear mode), equal-design authors."""
    n = author.max() + 1
    y = phi.reshape(n, -1)
    xm = design(z[author == 0])
    xtx_inv = np.linalg.inv(xm.T @ xm)
    beta = y @ (xm @ xtx_inv)
    resid = y - beta @ xm.T
    df = n * (xm.shape[0] - 2)
    sigma2 = float(np.sum(resid ** 2) / df)
    sampling = sigma2 * xtx_inv
    raw = np.cov(beta, rowvar=False) - sampling
    vals, vecs = np.linalg.eigh((raw + raw.T) / 2)
    cov = (vecs * np.maximum(vals, 1e-6)) @ vecs.T
    return {"mu": beta.mean(0), "Sigma": cov, "sigma2": sigma2, "raw_eigenvalues": vals,
            "source_authors": int(n), "noise_df": int(df)}


def blup(fit, xm, y):
    """Gaussian BLUP for authors sharing one design; returns OLS too."""
    s, mu = fit["Sigma"], fit["mu"]
    v = xm @ s @ xm.T + fit["sigma2"] * np.eye(len(xm))
    w = np.linalg.solve(v, xm @ s).T
    beta = mu + (y - xm @ mu) @ w.T
    post = s - w @ xm @ s
    ols = y @ (xm @ np.linalg.inv(xm.T @ xm))
    return beta, post, ols


def score_targets(readout, fit, x, z, author, mask=None):
    """Score every author for every declared target from the frozen readout+kernel."""
    n = author.max() + 1
    phi = readout.phi(x).reshape(n, -1)
    zz = z[author == 0]
    keep = np.ones(len(zz), bool) if mask is None else mask
    beta, post, ols = blup(fit, design(zz[keep]), phi[:, keep])
    out = {}
    for name, t in TARGET_VECTORS.items():
        out[name] = {"score": beta @ t, "var": float(t @ post @ t), "ols": ols @ t}
    return out


def readout_mechanism(readout, maps):
    """Evaluator-only: response of the readout to each hidden mechanism (per unit)."""
    d, g = readout.direction, readout.unit
    return {
        "gain_expression": float(G_E * d @ maps["u_e"] / g),
        "fingerprint_per_sd": float(FP_SD * np.linalg.norm(maps["U_f"].T @ d) / abs(g)),
        "style_response_per_sd": float(STYLE_SD * d @ maps["u_s"] / g),
        "topic_sd": float(np.std(maps["U_t"][:, list(SOURCE_TOPICS)].T @ d) / abs(g)),
        "context_content": float(d @ maps["u_c"] / g),
        "wording_sd": float(WORD_SD * np.linalg.norm(d) / abs(g)),
    }
