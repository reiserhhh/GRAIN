#!/usr/bin/env python3
"""G-5: Pragmatic/Epistemic act separation & cross-fitted register-channel Kronecker probe.

Operates deterministically (zero test-time LLM calls) to separate register/genre channel
shifts (length x reply/quote x script) from shared latent content traits (G11/G12).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

# Bilingual EN + JA/CJK deterministic epistemic/pragmatic lexicons (domain-neutral)
_OWNERSHIP_RE = re.compile(
    r"(?:\b(?:i|i'm|i've|my|mine|myself|personally|in my view|for me)\b|私|わたし|僕|俺|自分|個人的)"
)
_CERTAINTY_RE = re.compile(
    r"(?:\b(?:definitely|certainly|clearly|undoubtedly|must be|always|never|guaranteed)\b|確実|間違いなく|絶対|必ず|明らか)"
)
_HEDGE_RE = re.compile(
    r"(?:\b(?:maybe|perhaps|might|could|seems?|probably|somewhat|if|unless|assuming|depending)\b|かも|たぶん|おそらく|思われ|もし|場合によって)"
)
_PROMO_RE = re.compile(
    r"(?:\b(?:must|should|need to|everyone|huge|massive|obviously|wake up|stop ignoring|look at)\b|[!！]{1,}|べきだ|みんな|必見|絶対)"
)
_RETRO_RE = re.compile(
    r"(?:\b(?:was|were|had|did|happened|yesterday|last year|previously|in hindsight|used to|back then)\b|だった|でした|たしか|以前|当時|振り返)"
)
_FORWARD_RE = re.compile(
    r"(?:\b(?:will|going to|plan to|expect|next|soon|eventually|forecast|prepare|tomorrow)\b|予定|つもり|これから|今後|期待|予測)"
)
_QUOTED_RE = re.compile(
    r"(?:^>|\b(?:according to|they said|people say|reportedly|someone posted|heard that)\b|「[^」]+」|らしい|とのこと|と言われ)"
)
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def extract_pragmatic_acts(texts: Sequence[str]) -> np.ndarray:
    """Deterministic (N, 5) pragmatic-act profile: [ownership, cert_vs_hedge, promo, retro_vs_fwd, quoted]."""
    out = np.zeros((len(texts), 5), dtype=np.float64)
    for i, raw in enumerate(texts):
        txt = str(raw)
        low = txt.lower()
        n_units = float(max(1, len(low.split()) + len(_CJK_CHAR_RE.findall(txt))))
        norm = math.sqrt(n_units)
        own = len(_OWNERSHIP_RE.findall(low)) / norm
        cert = len(_CERTAINTY_RE.findall(low)) / norm
        hedge = len(_HEDGE_RE.findall(low)) / norm
        promo = len(_PROMO_RE.findall(low)) / norm
        retro = len(_RETRO_RE.findall(low)) / norm
        fwd = len(_FORWARD_RE.findall(low)) / norm
        quoted = len(_QUOTED_RE.findall(low)) / norm
        out[i] = np.tanh([own, cert - hedge, promo, retro - fwd, quoted])
    return out


def register_channel_probs(
    texts: Sequence[str],
    *,
    is_reply: Sequence[bool] | np.ndarray | None = None,
    short_chars: int = 60,
    long_chars: int = 220,
) -> np.ndarray:
    """Deterministic 6-channel register simplex: (short/med/long) x (standalone vs reply/quote)."""
    n = len(texts)
    probs = np.zeros((n, 6), dtype=np.float64)
    span = float(max(1, long_chars - short_chars))
    for i, raw in enumerate(texts):
        txt = str(raw).strip()
        length = float(len(txt))
        w_short = max(0.0, min(1.0, (float(short_chars) + 20.0 - length) / 40.0))
        w_long = max(0.0, min(1.0, (length - float(long_chars) + 40.0) / span))
        w_med = max(0.05, 1.0 - max(w_short, w_long))
        len_p = np.array([w_short + 0.02, w_med, w_long + 0.02], dtype=np.float64)
        len_p /= float(np.sum(len_p))

        if is_reply is not None:
            rep = 1.0 if bool(is_reply[i]) else 0.0
        else:
            rep = (
                1.0
                if (txt.startswith(">") or "@" in txt[:24] or "Re:" in txt[:16])
                else 0.0
            )
        mode_p = np.array(
            [1.0 - 0.85 * rep - 0.075, 0.85 * rep + 0.075], dtype=np.float64
        )
        probs[i] = np.outer(len_p, mode_p).ravel()
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


def kronecker_register_features(
    embeddings: np.ndarray, reg_probs: np.ndarray
) -> np.ndarray:
    """Explicit observable surface-channel expansion [emb, emb (x) p_reg]."""
    emb = np.asarray(embeddings, dtype=np.float64)
    rp = np.asarray(reg_probs, dtype=np.float64)
    if emb.ndim != 2 or rp.ndim != 2 or len(emb) != len(rp):
        raise ValueError("embeddings (N, D) and reg_probs (N, R) must align")
    inter = (emb[:, :, None] * rp[:, None, :]).reshape(
        len(emb), emb.shape[1] * rp.shape[1]
    )
    return np.hstack([emb, inter])


class CrossFittedRegisterCalibrator:
    """Author-disjoint (G2) L2 ridge calibrator over Kronecker register-interaction features."""

    def __init__(
        self,
        *,
        alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
        n_folds: int = 5,
        seed: int = 517,
    ) -> None:
        self.alphas = tuple(float(a) for a in alphas)
        if (
            not self.alphas
            or not np.isfinite(self.alphas).all()
            or min(self.alphas) <= 0
            or n_folds < 2
        ):
            raise ValueError("positive penalties and at least two folds required")
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.best_alpha_: float | None = None

    @staticmethod
    def _fit_ridge(
        X: np.ndarray, y: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, float]:
        mu_x = np.mean(X, axis=0)
        mu_y = float(np.mean(y))
        Xc = X - mu_x
        yc = y - mu_y
        d = Xc.shape[1]
        # Primal/dual automatic switch for speed and conditioning
        if len(Xc) >= d:
            gram = Xc.T @ Xc + alpha * np.eye(d)
            w = np.linalg.solve(gram, Xc.T @ yc)
        else:
            gram = Xc @ Xc.T + alpha * np.eye(len(Xc))
            dual = np.linalg.solve(gram, yc)
            w = Xc.T @ dual
        return w, mu_y - float(mu_x @ w)

    def _author_folds(self, author_ids: Sequence[Any]) -> np.ndarray:
        unique_au = sorted({str(a) for a in author_ids})
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(unique_au))
        k_folds = max(2, min(self.n_folds, len(unique_au)))
        au_to_fold = {
            unique_au[idx]: int(pos % k_folds) for pos, idx in enumerate(perm)
        }
        return np.array([au_to_fold[str(a)] for a in author_ids], dtype=int)

    def fit_predict(
        self,
        embeddings: np.ndarray,
        reg_probs: np.ndarray,
        targets: np.ndarray,
        author_ids: Sequence[Any],
    ) -> dict[str, Any]:
        """Cross-fit predictions on held-out author folds (G2)."""
        Phi = kronecker_register_features(embeddings, reg_probs)
        y = np.asarray(targets, dtype=np.float64)
        folds = self._author_folds(author_ids)
        if (
            y.shape != (len(Phi),)
            or len(folds) != len(Phi)
            or not np.isfinite(y).all()
            or len(set(folds)) < 2
        ):
            raise ValueError(">=2 authors and aligned finite targets required")
        unique_folds = sorted(set(folds.tolist()))

        best_mse = float("inf")
        best_alpha = self.alphas[0]
        for alpha in self.alphas:
            oof = np.zeros_like(y)
            for f_id in unique_folds:
                tr = folds != f_id
                va = folds == f_id
                w, b = self._fit_ridge(Phi[tr], y[tr], alpha)
                oof[va] = Phi[va] @ w + b
            mse = float(np.mean((y - oof) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        # The global penalty is for future deployment only. Every reported
        # held-author prediction selects its penalty without that author's labels.
        best_oof = np.empty_like(y)
        ids = np.asarray(author_ids)
        outer_penalties = []
        for outer in unique_folds:
            tr = folds != outer
            train_x, train_y = Phi[tr], y[tr]
            inner = self._author_folds(ids[tr])

            def loss(alpha, inner=inner, train_x=train_x, train_y=train_y):
                s = 0.0
                for k in np.unique(inner):
                    keep = inner != k
                    w, b = self._fit_ridge(train_x[keep], train_y[keep], alpha)
                    s += np.sum((train_y[~keep] - train_x[~keep] @ w - b) ** 2)
                return float(s)

            al = (
                min(self.alphas, key=loss)
                if len(np.unique(inner)) > 1 and len(self.alphas) > 1
                else self.alphas[0]
            )
            w, b = self._fit_ridge(train_x, train_y, al)
            best_oof[~tr] = Phi[~tr] @ w + b
            outer_penalties.append(al)

        w_full, b_full = self._fit_ridge(Phi, y, best_alpha)
        self.coef_ = w_full
        self.intercept_ = b_full
        self.best_alpha_ = best_alpha
        return {
            "oof_predictions": best_oof,
            "best_alpha": float(best_alpha),
            "oof_mse": float(np.mean((best_oof - y) ** 2)),
            "outer_penalties": outer_penalties,
            "n_folds": len(unique_folds),
        }

    def predict(self, embeddings: np.ndarray, reg_probs: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError(
                "CrossFittedRegisterCalibrator must be fit before predict"
            )
        Phi = kronecker_register_features(embeddings, reg_probs)
        return Phi @ self.coef_ + self.intercept_


def mtmm_validity_audit(
    trait_a_reg1: np.ndarray,
    trait_a_reg2: np.ndarray,
    trait_b_reg1: np.ndarray,
    trait_b_reg2: np.ndarray,
) -> dict[str, float]:
    """G11 Multitrait-Multiregister check: convergent (same-trait cross-reg) vs discriminant."""

    def _corr(u: np.ndarray, v: np.ndarray) -> float:
        ua = np.asarray(u, dtype=np.float64) - np.mean(u)
        va = np.asarray(v, dtype=np.float64) - np.mean(v)
        den = float(np.linalg.norm(ua) * np.linalg.norm(va))
        return float(np.dot(ua, va) / den) if den > 1e-12 else 0.0

    conv_a = _corr(trait_a_reg1, trait_a_reg2)
    conv_b = _corr(trait_b_reg1, trait_b_reg2)
    convergent = 0.5 * (conv_a + conv_b)
    disc_monomethod = 0.5 * (
        abs(_corr(trait_a_reg1, trait_b_reg1)) + abs(_corr(trait_a_reg2, trait_b_reg2))
    )
    disc_heteromethod = 0.5 * (
        abs(_corr(trait_a_reg1, trait_b_reg2)) + abs(_corr(trait_a_reg2, trait_b_reg1))
    )
    return {
        "convergent_mean": float(convergent),
        "discriminant_monomethod_abs_mean": float(disc_monomethod),
        "discriminant_heteromethod_abs_mean": float(disc_heteromethod),
        "mtmm_margin": float(convergent - max(disc_monomethod, disc_heteromethod)),
    }


__all__ = [
    "CrossFittedRegisterCalibrator",
    "extract_pragmatic_acts",
    "kronecker_register_features",
    "mtmm_validity_audit",
    "register_channel_probs",
]
