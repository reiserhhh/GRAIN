"""Legacy implementation module for the GRAIN analysis method.

Core slicing/scoring utilities for turning free-form author text into
comparable text-behavior coordinates (sliced utterances measured in their
naturally occurring contexts).

Historical notes (kept for provenance; see docs/CLAIMS_LEDGER.md):
- The original expansion "shared user-invariant comparative architecture"
  encoded the condition-centering idea that was later FALSIFIED three
  independent ways (P2 2026-07-04; E1 cross-fit audit 2026-07-05). Do not
  center scores on condition means; control context by design (RULEBOOK).
- `summarize_run` below contains the historical anchor-weighted search
  objective (35% Big5/MBTI). It is DEPRECATED for selection: all current
  selection objectives are label-free (guide rule G7).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


BIG5_TRAITS = ("Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness")
MBTI_AXES = ("EI_cont", "SN_cont", "TF_cont", "JP_cont")
TOKEN_RE = re.compile(r"\w+(?:['’-]\w+)?|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
PERSONALITY_LEAK_RE = re.compile(
    r"\b(?:big\s*five|big5|mbti|personality|trait|percentile|score|"
    r"openness|conscientiousness|extraversion|extroversion|agreeableness|"
    r"neuroticism|introversion|enthusiasm|withdrawal|industriousness|orderliness|"
    r"intj|intp|entj|entp|infj|infp|enfj|enfp|istj|isfj|estj|esfj|istp|isfp|estp|esfp|"
    r"introverts?|introverted|extroverts?|extroverted|extraverts?|extraverted|"
    r"myers[- ]?briggs|enneagram)\b",
    re.IGNORECASE,
)
METHOD_RE = re.compile(
    r"\b(?:http|https|www|reddit|subreddit|imgur|youtube|youtu|wikipedia|"
    r"link|links|url|com|org|net|io|co|deleted|removed|edit|upvote|downvote|karma|"
    r"amp|jpg|jpeg|png|gif|html|wiki|pcpartpicker|product|ref|utf|search|gallery|"
    r"watch|comments|thread|post|posts)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"\b(?:https?://\S+|www\.\S+|\S+\.(?:com|org|net|io|co)\S*)", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class SuicaSliceConfig:
    """Slicing parameters shared by SUICA strategies."""

    strategy: str
    slice_tokens: int = 96
    slice_stride: int = 96
    min_slice_tokens: int = 24
    max_slices_per_user: int = 28
    semantic_max_tokens: int = 160
    semantic_shift_quantile: float = 0.85
    random_seed: int = 42
    prior_run_dir: str = "results/token_slice_hmcd_optimization_v1/s96_d2_root24_base"


@dataclass(frozen=True)
class SuicaFactorConfig:
    """Vectorization and factorization parameters."""

    method: str
    tfidf_max_features: int = 22000
    tfidf_min_df: int = 3
    svd_dims: int = 96
    root_k: int = 24
    child_k: int = 5
    max_depth: int = 2
    min_node_slices: int = 80
    flat_k: int = 24
    gmm_k: int = 18
    factor_count: int = 10
    factor_min_reliability: float = 0.10
    factor_min_features: int = 12
    ridge_alpha: float = 10.0
    text_control: str = "raw"
    random_seed: int = 42


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving punctuation as local sequence tokens."""
    return TOKEN_RE.findall(str(text or ""))


def sentence_split(text: str) -> list[str]:
    """Conservative sentence-ish split for Reddit text."""
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(str(text or "")) if part.strip()]
    if not parts and str(text or "").strip():
        return [str(text).strip()]
    return parts


def _slice_tokens(tokens: list[str], *, start: int, end: int) -> dict[str, Any]:
    return {
        "token_start": int(start),
        "token_end": int(end),
        "token_count": int(end - start),
        "token_mid_frac": float((start + end) / 2.0 / max(1, len(tokens))),
        "slice_text": " ".join(tokens[start:end]),
    }


def fixed_token_slices_for_text(
    text: str,
    *,
    slice_tokens: int,
    stride: int,
    min_slice_tokens: int,
    max_slices: int,
) -> list[dict[str, Any]]:
    """Split text into fixed token windows."""
    tokens = tokenize(text)
    if len(tokens) < min_slice_tokens:
        return []
    rows: list[dict[str, Any]] = []
    for slice_index, start in enumerate(range(0, max(1, len(tokens)), stride)):
        if max_slices and len(rows) >= max_slices:
            break
        end = min(len(tokens), start + slice_tokens)
        if end - start < min_slice_tokens:
            continue
        row = _slice_tokens(tokens, start=start, end=end)
        row["slice_index"] = int(slice_index)
        row["slice_strategy_detail"] = f"fixed_{slice_tokens}_stride_{stride}"
        rows.append(row)
        if end >= len(tokens):
            break
    return rows


def build_fixed_slices(users: pd.DataFrame, config: SuicaSliceConfig) -> pd.DataFrame:
    """Build fixed-window slices for all users."""
    rows: list[dict[str, Any]] = []
    for user_id, text in zip(users["user_id"].astype(str), users["text"].fillna("").astype(str)):
        for row in fixed_token_slices_for_text(
            text,
            slice_tokens=config.slice_tokens,
            stride=config.slice_stride,
            min_slice_tokens=config.min_slice_tokens,
            max_slices=config.max_slices_per_user,
        ):
            row["user_id"] = user_id
            row["global_slice_id"] = len(rows)
            rows.append(row)
    return _finalize_slices(rows, config.strategy)


def build_semantic_shift_slices(users: pd.DataFrame, config: SuicaSliceConfig) -> pd.DataFrame:
    """Build slices at group-common semantic-shift boundaries.

    The global shift threshold is learned from all adjacent sentence pairs,
    making the boundary rule collective rather than per-user bespoke.
    """
    sentence_rows: list[dict[str, Any]] = []
    for user_id, text in zip(users["user_id"].astype(str), users["text"].fillna("").astype(str)):
        for sent_idx, sentence in enumerate(sentence_split(text)):
            tokens = tokenize(sentence)
            if not tokens:
                continue
            sentence_rows.append(
                {
                    "user_id": user_id,
                    "sentence_index": int(sent_idx),
                    "sentence_text": " ".join(tokens),
                    "token_count": int(len(tokens)),
                }
            )
    if len(sentence_rows) < 20:
        return build_fixed_slices(users, config)
    sent = pd.DataFrame(sentence_rows)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        max_features=12000,
        min_df=2,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(sent["sentence_text"].astype(str))
    dims = int(min(32, max(2, tfidf.shape[0] - 1), max(2, tfidf.shape[1] - 1)))
    dense = TruncatedSVD(n_components=dims, random_state=config.random_seed).fit_transform(tfidf)
    dense = StandardScaler().fit_transform(dense)
    sent["row_index"] = np.arange(len(sent))
    shift_distance = np.zeros(len(sent), dtype=float)
    distances = []
    for _uid, group in sent.sort_values(["user_id", "sentence_index"]).groupby("user_id"):
        idxs = group["row_index"].to_numpy(int)
        for left, right in zip(idxs[:-1], idxs[1:]):
            denom = float(np.linalg.norm(dense[left]) * np.linalg.norm(dense[right]))
            dist = 1.0 if denom < 1e-12 else float(1.0 - np.dot(dense[left], dense[right]) / denom)
            shift_distance[right] = dist
            distances.append(dist)
    threshold = float(np.quantile(distances, config.semantic_shift_quantile)) if distances else 1.0
    sent["semantic_shift_distance"] = shift_distance

    rows: list[dict[str, Any]] = []
    for user_id, group in sent.sort_values(["user_id", "sentence_index"]).groupby("user_id"):
        current: list[str] = []
        current_tokens = 0
        slice_index = 0
        for row in group.itertuples(index=False):
            tokens = tokenize(row.sentence_text)
            should_cut = (
                bool(current)
                and current_tokens >= config.min_slice_tokens
                and (
                    current_tokens + len(tokens) > config.semantic_max_tokens
                    or float(row.semantic_shift_distance) >= threshold
                )
            )
            if should_cut:
                rows.append(
                    {
                        "user_id": str(user_id),
                        "global_slice_id": len(rows),
                        "slice_index": int(slice_index),
                        "token_start": -1,
                        "token_end": -1,
                        "token_count": int(current_tokens),
                        "token_mid_frac": float(min(1.0, (slice_index + 0.5) / max(1, config.max_slices_per_user))),
                        "slice_text": " ".join(current),
                        "slice_strategy_detail": f"semantic_shift_q{config.semantic_shift_quantile:.2f}",
                    }
                )
                slice_index += 1
                current = []
                current_tokens = 0
                if config.max_slices_per_user and slice_index >= config.max_slices_per_user:
                    break
            current.extend(tokens)
            current_tokens += len(tokens)
        if current_tokens >= config.min_slice_tokens and (not config.max_slices_per_user or slice_index < config.max_slices_per_user):
            rows.append(
                {
                    "user_id": str(user_id),
                    "global_slice_id": len(rows),
                    "slice_index": int(slice_index),
                    "token_start": -1,
                    "token_end": -1,
                    "token_count": int(current_tokens),
                    "token_mid_frac": float(min(1.0, (slice_index + 0.5) / max(1, config.max_slices_per_user))),
                    "slice_text": " ".join(current),
                    "slice_strategy_detail": f"semantic_shift_q{config.semantic_shift_quantile:.2f}",
                }
            )
    out = _finalize_slices(rows, config.strategy)
    out.attrs["semantic_shift_threshold"] = threshold
    return out


def _rind_terms(prior_run_dir: str | Path, *, max_terms: int = 80) -> set[str]:
    """Load terms from prior rind/candidate promotion rows for recursive slicing."""
    prior = Path(prior_run_dir)
    candidates_path = prior / "breakthrough_promotion_candidates.csv"
    ledger_path = prior / "node_residual_ledger.csv"
    terms: list[str] = []
    if candidates_path.exists():
        cand = pd.read_csv(candidates_path)
        cand = cand.loc[cand["ledger_category"].isin(["rind_unstable", "candidate_seed"])].head(40)
        terms.extend(cand.get("top_relative_terms", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    elif ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        ledger = ledger.loc[ledger["ledger_category"].isin(["rind_unstable", "candidate_seed"])].head(40)
        terms.extend(ledger.get("top_relative_terms", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    out: list[str] = []
    for item in terms:
        for term in str(item).split(","):
            term = term.strip().lower()
            if len(term) >= 3 and not METHOD_RE.search(term):
                out.append(term)
    return set(out[:max_terms])


def build_rind_recursive_slices(users: pd.DataFrame, config: SuicaSliceConfig) -> pd.DataFrame:
    """Split prior-rind-like fixed slices into finer local sub-slices."""
    base = build_fixed_slices(
        users,
        SuicaSliceConfig(
            strategy="fixed_base_for_rind",
            slice_tokens=96,
            slice_stride=96,
            min_slice_tokens=config.min_slice_tokens,
            max_slices_per_user=config.max_slices_per_user,
            random_seed=config.random_seed,
            prior_run_dir=config.prior_run_dir,
        ),
    )
    terms = _rind_terms(config.prior_run_dir)
    rows: list[dict[str, Any]] = []
    per_user_counts: dict[str, int] = {}
    for row in base.itertuples(index=False):
        user_id = str(row.user_id)
        if config.max_slices_per_user and per_user_counts.get(user_id, 0) >= config.max_slices_per_user + 12:
            continue
        text = str(row.slice_text)
        lower = text.lower()
        should_recurse = bool(terms) and any(term in lower for term in terms) and int(row.token_count) >= 72
        if should_recurse:
            tokens = tokenize(text)
            for sub in fixed_token_slices_for_text(
                " ".join(tokens),
                slice_tokens=48,
                stride=48,
                min_slice_tokens=config.min_slice_tokens,
                max_slices=2,
            ):
                sub["user_id"] = user_id
                sub["global_slice_id"] = len(rows)
                sub["slice_index"] = int(per_user_counts.get(user_id, 0))
                sub["slice_strategy_detail"] = "rind_recursive_48_from_prior_terms"
                rows.append(sub)
                per_user_counts[user_id] = per_user_counts.get(user_id, 0) + 1
        else:
            rows.append(
                {
                    "user_id": user_id,
                    "global_slice_id": len(rows),
                    "slice_index": int(per_user_counts.get(user_id, 0)),
                    "token_start": int(getattr(row, "token_start", -1)),
                    "token_end": int(getattr(row, "token_end", -1)),
                    "token_count": int(row.token_count),
                    "token_mid_frac": float(row.token_mid_frac),
                    "slice_text": text,
                    "slice_strategy_detail": "rind_recursive_kept_96",
                }
            )
            per_user_counts[user_id] = per_user_counts.get(user_id, 0) + 1
    return _finalize_slices(rows, config.strategy)


def _finalize_slices(rows: list[dict[str, Any]], strategy: str) -> pd.DataFrame:
    """Finalize and leakage-mark a slice table."""
    if not rows:
        raise ValueError(f"SUICA strategy {strategy} produced no slices")
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["global_slice_id"] = np.arange(len(out))
    out["slice_strategy"] = strategy
    out["personality_leakage_clue"] = out["slice_text"].fillna("").astype(str).map(lambda text: bool(PERSONALITY_LEAK_RE.search(text)))
    before = len(out)
    leakage = int(out["personality_leakage_clue"].sum())
    out = out.loc[~out["personality_leakage_clue"]].reset_index(drop=True)
    out.attrs["token_slices_before_filter"] = before
    out.attrs["personality_leakage_slices_before_filter"] = leakage
    if out.empty:
        raise ValueError(f"SUICA strategy {strategy} removed all slices during leakage filtering")
    return out


def build_slices(users: pd.DataFrame, config: SuicaSliceConfig) -> pd.DataFrame:
    """Dispatch SUICA slicing strategy."""
    if config.strategy.startswith("fixed_"):
        if config.strategy != f"fixed_{config.slice_tokens}":
            width = int(config.strategy.split("_", 1)[1])
            config = SuicaSliceConfig(
                strategy=config.strategy,
                slice_tokens=width,
                slice_stride=width,
                min_slice_tokens=config.min_slice_tokens,
                max_slices_per_user=config.max_slices_per_user,
                semantic_max_tokens=config.semantic_max_tokens,
                semantic_shift_quantile=config.semantic_shift_quantile,
                random_seed=config.random_seed,
                prior_run_dir=config.prior_run_dir,
            )
        return build_fixed_slices(users, config)
    if config.strategy == "semantic_shift":
        return build_semantic_shift_slices(users, config)
    if config.strategy == "rind_recursive":
        return build_rind_recursive_slices(users, config)
    raise ValueError(f"unknown SUICA slice strategy: {config.strategy}")


def build_slices_for_strategy(users: pd.DataFrame, strategy: str, *, base_config: SuicaSliceConfig) -> pd.DataFrame:
    """Build slices while deriving token widths from strategy names."""
    if strategy.startswith("fixed_"):
        width = int(strategy.split("_", 1)[1])
        cfg = SuicaSliceConfig(
            strategy=strategy,
            slice_tokens=width,
            slice_stride=width,
            min_slice_tokens=base_config.min_slice_tokens,
            max_slices_per_user=base_config.max_slices_per_user,
            random_seed=base_config.random_seed,
            prior_run_dir=base_config.prior_run_dir,
        )
        return build_fixed_slices(users, cfg)
    cfg = SuicaSliceConfig(
        strategy=strategy,
        slice_tokens=base_config.slice_tokens,
        slice_stride=base_config.slice_stride,
        min_slice_tokens=base_config.min_slice_tokens,
        max_slices_per_user=base_config.max_slices_per_user,
        semantic_max_tokens=base_config.semantic_max_tokens,
        semantic_shift_quantile=base_config.semantic_shift_quantile,
        random_seed=base_config.random_seed,
        prior_run_dir=base_config.prior_run_dir,
    )
    return build_slices_dispatch(users, cfg)


def build_slices_dispatch(users: pd.DataFrame, config: SuicaSliceConfig) -> pd.DataFrame:
    """Dispatch helper without fixed strategy parser."""
    if config.strategy == "semantic_shift":
        return build_semantic_shift_slices(users, config)
    if config.strategy == "rind_recursive":
        return build_rind_recursive_slices(users, config)
    if config.strategy.startswith("fixed_"):
        return build_fixed_slices(users, config)
    raise ValueError(f"unknown SUICA slice strategy: {config.strategy}")


def controlled_slice_text(text: str, mode: str) -> str:
    """Apply optional text-level control before vectorization.

    This is not the main SUICA idea; it is a comparison branch for checking
    whether method/platform rind terms dominate the discovered factors.
    """
    text = str(text or "")
    if mode in {"method_stripped", "method_stripped_residual", "method_position_stripped_residual"}:
        text = URL_RE.sub(" ", text)
        text = METHOD_RE.sub(" ", text)
        text = DIGIT_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


def slice_control_design(slices: pd.DataFrame, mode: str) -> tuple[np.ndarray | None, list[str]]:
    """Build nuisance-condition matrix for vector residualization."""
    columns: list[np.ndarray] = []
    names: list[str] = []
    texts = slices["slice_text"].fillna("").astype(str)
    tokens = texts.map(tokenize)
    token_counts = slices["token_count"].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    safe_counts = np.maximum(token_counts, 1.0)
    lower = texts.str.lower()
    if "method" in mode:
        has_url = lower.map(lambda text: 1.0 if URL_RE.search(text) else 0.0).to_numpy(float)
        method_rate = lower.map(lambda text: len(METHOD_RE.findall(text))).to_numpy(float) / safe_counts
        digit_rate = lower.map(lambda text: len(DIGIT_RE.findall(text))).to_numpy(float) / safe_counts
        columns.extend([has_url, method_rate, digit_rate])
        names.extend(["has_url", "method_term_rate", "digit_rate"])
    if "position" in mode:
        slice_index = slices["slice_index"].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        token_mid = slices["token_mid_frac"].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        columns.extend([np.log1p(token_counts), slice_index, token_mid])
        names.extend(["log_token_count", "slice_index", "token_mid_frac"])
    if "pronoun" in mode:
        first_rate = tokens.map(lambda items: sum(str(tok).lower() in {"i", "me", "my", "mine", "myself"} for tok in items)).to_numpy(float) / safe_counts
        second_rate = tokens.map(lambda items: sum(str(tok).lower() in {"you", "your", "yours", "yourself"} for tok in items)).to_numpy(float) / safe_counts
        third_rate = tokens.map(lambda items: sum(str(tok).lower() in {"he", "she", "they", "him", "her", "them", "his", "their"} for tok in items)).to_numpy(float) / safe_counts
        columns.extend([first_rate, second_rate, third_rate])
        names.extend(["first_person_rate", "second_person_rate", "third_person_rate"])
    if not columns:
        return None, []
    design = np.column_stack(columns)
    design = _zscore_matrix(design)
    design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)
    return design, names


def residualize_vectors(vectors: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, float]:
    """Residualize slice vectors against nuisance-condition design."""
    if design.size == 0:
        return vectors, 1.0
    z = np.column_stack([np.ones(len(design)), design])
    beta, *_ = np.linalg.lstsq(z, vectors, rcond=None)
    fitted = z @ beta
    residual = vectors - fitted
    before = float(np.nanvar(vectors))
    after = float(np.nanvar(residual))
    retained = after / before if before > 1e-12 else 1.0
    return StandardScaler().fit_transform(residual), retained


def fit_slice_vectors(slices: pd.DataFrame, config: SuicaFactorConfig) -> tuple[csr_matrix, np.ndarray, TfidfVectorizer, dict[str, Any]]:
    """Fit TF-IDF/SVD vectors on SUICA slices."""
    texts = slices["slice_text"].fillna("").astype(str).map(lambda text: controlled_slice_text(text, config.text_control))
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        max_features=config.tfidf_max_features,
        min_df=config.tfidf_min_df,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    dims = int(min(config.svd_dims, max(2, tfidf.shape[0] - 1), max(2, tfidf.shape[1] - 1)))
    svd = TruncatedSVD(n_components=dims, random_state=config.random_seed)
    dense = svd.fit_transform(tfidf)
    dense = StandardScaler().fit_transform(dense)
    control_design, control_names = slice_control_design(slices, config.text_control)
    residual_variance_retained = 1.0
    if "residual" in config.text_control and control_design is not None:
        dense, residual_variance_retained = residualize_vectors(dense, control_design)
    diag = {
        "vector_backend": "suica_word_tfidf_1_2_svd",
        "text_control": config.text_control,
        "control_columns": control_names,
        "residual_variance_retained": float(residual_variance_retained),
        "slice_rows": int(tfidf.shape[0]),
        "tfidf_features": int(tfidf.shape[1]),
        "svd_dims": int(dims),
        "svd_explained_variance": float(np.sum(svd.explained_variance_ratio_)),
    }
    return tfidf, dense, vectorizer, diag


def _zscore_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std < 1e-12] = 1.0
    return (values - mean) / std


def hard_node_scores(
    slices: pd.DataFrame,
    labels: np.ndarray,
    node_ids: list[str],
    *,
    row_mask: np.ndarray | None = None,
    eps: float = 1e-4,
) -> pd.DataFrame:
    """Score users by log over/under-selection of hard slice clusters."""
    if row_mask is None:
        row_mask = np.ones(len(slices), dtype=bool)
    users = sorted(slices.loc[row_mask, "user_id"].astype(str).unique().tolist())
    user_pos = {user_id: idx for idx, user_id in enumerate(users)}
    node_pos = {node_id: idx for idx, node_id in enumerate(node_ids)}
    counts = np.zeros((len(users), len(node_ids)), dtype=float)
    totals = np.zeros(len(users), dtype=float)
    pop = np.zeros(len(node_ids), dtype=float)
    for idx, keep in enumerate(row_mask):
        if not keep:
            continue
        user_id = str(slices.iloc[idx]["user_id"])
        node_id = node_ids[int(labels[idx])]
        u = user_pos[user_id]
        j = node_pos[node_id]
        totals[u] += 1.0
        counts[u, j] += 1.0
        pop[j] += 1.0
    totals = np.maximum(totals, 1.0)
    user_share = counts / totals[:, None]
    pop_share = pop / max(1.0, float(row_mask.sum()))
    scores = np.log((user_share + eps) / (pop_share[None, :] + eps))
    scores = _zscore_matrix(scores)
    out = pd.DataFrame(scores, columns=[f"suica_node::{node_id}" for node_id in node_ids])
    out.insert(0, "user_id", users)
    out.insert(1, "token_slice_count", totals.astype(int))
    return out


def soft_node_scores(
    slices: pd.DataFrame,
    probs: np.ndarray,
    node_ids: list[str],
    *,
    row_mask: np.ndarray | None = None,
    eps: float = 1e-4,
) -> pd.DataFrame:
    """Score users by log over/under-selection of soft cluster memberships."""
    if row_mask is None:
        row_mask = np.ones(len(slices), dtype=bool)
    users = sorted(slices.loc[row_mask, "user_id"].astype(str).unique().tolist())
    user_pos = {user_id: idx for idx, user_id in enumerate(users)}
    sums = np.zeros((len(users), len(node_ids)), dtype=float)
    totals = np.zeros(len(users), dtype=float)
    selected_probs = probs[row_mask]
    pop_share = selected_probs.mean(axis=0)
    for local_idx, idx in enumerate(np.where(row_mask)[0]):
        user_id = str(slices.iloc[idx]["user_id"])
        u = user_pos[user_id]
        totals[u] += 1.0
        sums[u, :] += selected_probs[local_idx]
    totals = np.maximum(totals, 1.0)
    user_share = sums / totals[:, None]
    scores = np.log((user_share + eps) / (pop_share[None, :] + eps))
    scores = _zscore_matrix(scores)
    out = pd.DataFrame(scores, columns=[f"suica_node::{node_id}" for node_id in node_ids])
    out.insert(0, "user_id", users)
    out.insert(1, "token_slice_count", totals.astype(int))
    return out


def safe_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, *, method: str = "pearson") -> tuple[float, float, int]:
    """Safe Pearson/Spearman correlation."""
    xv = pd.to_numeric(pd.Series(x), errors="coerce")
    yv = pd.to_numeric(pd.Series(y), errors="coerce")
    if method == "spearman":
        xv = xv.rank(method="average")
        yv = yv.rank(method="average")
    mask = xv.notna() & yv.notna()
    n = int(mask.sum())
    if n < 4:
        return float("nan"), float("nan"), n
    xa = xv.loc[mask].to_numpy(float)
    ya = yv.loc[mask].to_numpy(float)
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return float("nan"), float("nan"), n
    res = stats.pearsonr(xa, ya)
    return float(res.statistic), float(res.pvalue), n


def split_half_reliability(scores_a: pd.DataFrame, scores_b: pd.DataFrame, *, prefix: str = "suica_node::") -> pd.DataFrame:
    """Compute split-half reliability for node scores."""
    merged = scores_a.merge(scores_b, on="user_id", suffixes=("_a", "_b"))
    rows: list[dict[str, Any]] = []
    for col in [c for c in scores_a.columns if c.startswith(prefix)]:
        r, p, n = safe_corr(merged[f"{col}_a"], merged[f"{col}_b"])
        rows.append({"feature": col, "node_id": col.replace(prefix, ""), "split_half_r": r, "split_half_p": p, "split_half_n": n})
    return pd.DataFrame(rows)


def varimax_rotation(loadings: np.ndarray, *, gamma: float = 1.0, max_iter: int = 80, tol: float = 1e-6) -> np.ndarray:
    """Return an orthogonal varimax rotation matrix."""
    p, k = loadings.shape
    rotation = np.eye(k)
    last = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        basis = loadings.T @ (rotated**3 - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        u, s, vh = np.linalg.svd(basis, full_matrices=False)
        rotation = u @ vh
        current = float(np.sum(s))
        if last and current < last * (1.0 + tol):
            break
        last = current
    return rotation


def fit_factors(
    node_scores: pd.DataFrame,
    reliability: pd.DataFrame,
    node_terms: pd.DataFrame,
    config: SuicaFactorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit PCA/varimax factors from reliable node scores."""
    all_features = [col for col in node_scores.columns if col.startswith("suica_node::")]
    rel_map = reliability.set_index("feature")["split_half_r"].to_dict() if not reliability.empty else {}
    selected = [feature for feature in all_features if float(rel_map.get(feature, 0.0) or 0.0) >= config.factor_min_reliability]
    if len(selected) < config.factor_min_features:
        ranked = sorted(all_features, key=lambda feature: float(rel_map.get(feature, 0.0) or 0.0), reverse=True)
        selected = ranked[: min(len(ranked), max(config.factor_min_features, config.factor_count + 1))]
    if not selected:
        return pd.DataFrame({"user_id": node_scores["user_id"].astype(str)}), pd.DataFrame(), {
            "factor_status": "skipped_no_features",
            "selected_node_feature_count": 0,
            "factor_count": 0,
        }
    n_components = int(min(config.factor_count, max(1, len(selected)), max(1, len(node_scores) - 1)))
    x = node_scores[selected].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    xz = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components, random_state=config.random_seed)
    raw_scores = pca.fit_transform(xz)
    loadings = pca.components_.T * np.sqrt(np.maximum(pca.explained_variance_, 0.0))[None, :]
    if n_components > 1:
        rotation = varimax_rotation(loadings)
        loadings = loadings @ rotation
        raw_scores = raw_scores @ rotation
    factor_names = [f"suica_factor_{idx + 1:02d}" for idx in range(n_components)]
    for idx in range(n_components):
        strongest = int(np.nanargmax(np.abs(loadings[:, idx])))
        if loadings[strongest, idx] < 0:
            loadings[:, idx] *= -1
            raw_scores[:, idx] *= -1
    factor_scores = pd.DataFrame(StandardScaler().fit_transform(raw_scores), columns=factor_names)
    factor_scores.insert(0, "user_id", node_scores["user_id"].astype(str).to_numpy())
    term_map = node_terms.set_index("node_id")["top_relative_terms"].to_dict() if not node_terms.empty else {}
    rows: list[dict[str, Any]] = []
    for feature_idx, feature in enumerate(selected):
        node_id = feature.replace("suica_node::", "")
        for factor_idx, factor_name in enumerate(factor_names):
            loading = float(loadings[feature_idx, factor_idx])
            rows.append(
                {
                    "factor": factor_name,
                    "feature": feature,
                    "node_id": node_id,
                    "loading": loading,
                    "abs_loading": abs(loading),
                    "node_split_half_r": float(rel_map.get(feature, float("nan"))),
                    "node_top_terms": term_map.get(node_id, ""),
                }
            )
    loading_frame = pd.DataFrame(rows).sort_values(["factor", "abs_loading"], ascending=[True, False]).reset_index(drop=True)
    diagnostics = {
        "factor_status": "built",
        "selected_node_feature_count": int(len(selected)),
        "factor_count": int(n_components),
        "factor_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "factor_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }
    return factor_scores, loading_frame, diagnostics


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray([1.0 if not np.isfinite(v) else float(v) for v in p_values], dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        running = min(running, ranked[idx] * n / rank)
        adjusted[order[idx]] = running
    return np.clip(adjusted, 0.0, 1.0).tolist()


def external_correlations(scores: pd.DataFrame, labels: pd.DataFrame, label_cols: tuple[str, ...], label_space: str) -> pd.DataFrame:
    """Correlate factor scores with external labels."""
    merged = scores.merge(labels[["user_id", *label_cols]], on="user_id", how="inner")
    features = [col for col in scores.columns if col.startswith("suica_factor_")]
    rows: list[dict[str, Any]] = []
    for label in label_cols:
        start = len(rows)
        p_values = []
        for feature in features:
            r, p, n = safe_corr(merged[feature], merged[label])
            rho, sp, _ = safe_corr(merged[feature], merged[label], method="spearman")
            rows.append(
                {
                    "label_space": label_space,
                    "label": label,
                    "feature": feature,
                    "n": n,
                    "pearson_r": r,
                    "pearson_p": p,
                    "spearman_rho": rho,
                    "spearman_p": sp,
                    "abs_pearson_r": abs(r) if np.isfinite(r) else float("nan"),
                }
            )
            p_values.append(p)
        for offset, q in enumerate(bh_fdr(p_values)):
            rows[start + offset]["pearson_q_bh_within_label"] = q
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["pearson_q_bh_within_label", "abs_pearson_r"], ascending=[True, False]).reset_index(drop=True)


def big5_cv(scores: pd.DataFrame, users: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    """Official-fold Big5 CV from SUICA factor scores."""
    data = scores.merge(users[["user_id", "official_fold", *BIG5_TRAITS]], on="user_id", how="inner")
    features = [col for col in scores.columns if col.startswith("suica_factor_")]
    rows: list[dict[str, Any]] = []
    for fold in sorted(data["official_fold"].dropna().unique().tolist()):
        train = data["official_fold"].ne(fold)
        test = data["official_fold"].eq(fold)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(data.loc[train, features].to_numpy(float))
        x_test = scaler.transform(data.loc[test, features].to_numpy(float))
        for trait in BIG5_TRAITS:
            model = Ridge(alpha=alpha)
            model.fit(x_train, data.loc[train, trait].to_numpy(float))
            pred = model.predict(x_test)
            true = data.loc[test, trait].to_numpy(float)
            r, p, n = safe_corr(pred, true)
            rho, _, _ = safe_corr(pred, true, method="spearman")
            rows.append(
                {
                    "fold": int(fold),
                    "trait": trait,
                    "n": n,
                    "pearson_r": r,
                    "pearson_p": p,
                    "spearman_rho": rho,
                    "mae": float(mean_absolute_error(true, pred)),
                    "rmse": float(math.sqrt(mean_squared_error(true, pred))),
                }
            )
    return pd.DataFrame(rows)


def top_terms_for_hard_nodes(tfidf: csr_matrix, vectorizer: TfidfVectorizer, labels: np.ndarray, node_ids: list[str], *, top_terms: int = 12) -> pd.DataFrame:
    """Top relative TF-IDF terms for hard nodes."""
    terms = np.asarray(vectorizer.get_feature_names_out())
    overall = np.asarray(tfidf.mean(axis=0)).ravel()
    rows = []
    for idx, node_id in enumerate(node_ids):
        mask = labels == idx
        if not np.any(mask):
            continue
        local = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        diff = local - overall
        pos = np.argsort(diff)[-top_terms:][::-1]
        rows.append({"node_id": node_id, "top_relative_terms": ", ".join(terms[pos].tolist()), "slice_count": int(mask.sum())})
    return pd.DataFrame(rows)


def top_terms_for_soft_nodes(tfidf: csr_matrix, vectorizer: TfidfVectorizer, probs: np.ndarray, node_ids: list[str], *, top_terms: int = 12) -> pd.DataFrame:
    """Top relative TF-IDF terms for soft nodes using posterior weights."""
    terms = np.asarray(vectorizer.get_feature_names_out())
    overall = np.asarray(tfidf.mean(axis=0)).ravel()
    rows = []
    for idx, node_id in enumerate(node_ids):
        weights = probs[:, idx]
        denom = max(1e-12, float(weights.sum()))
        local = np.asarray((tfidf.multiply(weights[:, None]).sum(axis=0) / denom)).ravel()
        diff = local - overall
        pos = np.argsort(diff)[-top_terms:][::-1]
        rows.append({"node_id": node_id, "top_relative_terms": ", ".join(terms[pos].tolist()), "slice_count": float(denom)})
    return pd.DataFrame(rows)


def build_tree_labels(vectors: np.ndarray, slices: pd.DataFrame, config: SuicaFactorConfig) -> tuple[np.ndarray, pd.DataFrame]:
    """Build recursive hard labels, returning final path-node labels."""
    labels = np.full(len(slices), -1, dtype=int)
    rows: list[dict[str, Any]] = []
    next_label = 0

    def recurse(indices: np.ndarray, parent_id: str, depth: int) -> None:
        nonlocal next_label
        if depth > config.max_depth:
            return
        desired_k = config.root_k if depth == 1 else config.child_k
        max_k = int(len(indices) // max(1, config.min_node_slices))
        k = int(min(desired_k, max_k))
        if k < 2:
            if parent_id:
                # Existing parent remains the terminal node.
                return
            return
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=config.random_seed + depth * 1009 + next_label,
            batch_size=min(2048, max(256, len(indices))),
            n_init=5,
            max_iter=100,
            reassignment_ratio=0.01,
        )
        local_labels = model.fit_predict(vectors[indices])
        for child in range(k):
            child_indices = indices[local_labels == child]
            if len(child_indices) < config.min_node_slices:
                continue
            node_id = f"{parent_id}.{child:02d}" if parent_id else f"n{child:02d}"
            terminal = depth == config.max_depth or len(child_indices) < config.min_node_slices * 2
            if terminal:
                label_idx = next_label
                next_label += 1
                labels[child_indices] = label_idx
                sub = slices.iloc[child_indices]
                rows.append(
                    {
                        "node_id": node_id,
                        "label_index": label_idx,
                        "parent_id": parent_id or "ROOT",
                        "depth": int(depth),
                        "slice_count": int(len(child_indices)),
                        "user_count": int(sub["user_id"].nunique()),
                    }
                )
            else:
                recurse(child_indices, node_id, depth + 1)

    recurse(np.arange(len(slices)), "", 1)
    if np.any(labels < 0):
        unassigned = np.where(labels < 0)[0]
        label_idx = next_label
        labels[unassigned] = label_idx
        rows.append(
            {
                "node_id": "unassigned",
                "label_index": label_idx,
                "parent_id": "ROOT",
                "depth": 0,
                "slice_count": int(len(unassigned)),
                "user_count": int(slices.iloc[unassigned]["user_id"].nunique()),
            }
        )
    node_table = pd.DataFrame(rows).sort_values("label_index").reset_index(drop=True)
    return labels, node_table


def run_factor_method(
    slices: pd.DataFrame,
    users: pd.DataFrame,
    bridge: pd.DataFrame,
    config: SuicaFactorConfig,
) -> dict[str, Any]:
    """Run one SUICA factorization method and return artifacts."""
    tfidf, vectors, vectorizer, vector_diag = fit_slice_vectors(slices, config)
    even_mask = slices["slice_index"].astype(int).mod(2).eq(0).to_numpy()
    odd_mask = ~even_mask
    if config.method == "kmeans_tree":
        labels, node_table = build_tree_labels(vectors, slices, config)
        node_ids = node_table.sort_values("label_index")["node_id"].astype(str).tolist()
        node_scores = hard_node_scores(slices, labels, node_ids)
        reliability = split_half_reliability(
            hard_node_scores(slices, labels, node_ids, row_mask=even_mask),
            hard_node_scores(slices, labels, node_ids, row_mask=odd_mask),
        )
        node_terms = top_terms_for_hard_nodes(tfidf, vectorizer, labels, node_ids)
    elif config.method == "pca_varimax":
        k = int(min(config.flat_k, max(2, len(slices) // max(1, config.min_node_slices))))
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=config.random_seed,
            batch_size=min(2048, max(256, len(slices))),
            n_init=5,
            max_iter=120,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(vectors)
        node_ids = [f"flat_{idx:02d}" for idx in range(k)]
        node_table = pd.DataFrame(
            {
                "node_id": node_ids,
                "label_index": list(range(k)),
                "parent_id": "ROOT",
                "depth": 1,
                "slice_count": [int((labels == idx).sum()) for idx in range(k)],
                "user_count": [int(slices.loc[labels == idx, "user_id"].nunique()) for idx in range(k)],
            }
        )
        node_scores = hard_node_scores(slices, labels, node_ids)
        reliability = split_half_reliability(
            hard_node_scores(slices, labels, node_ids, row_mask=even_mask),
            hard_node_scores(slices, labels, node_ids, row_mask=odd_mask),
        )
        node_terms = top_terms_for_hard_nodes(tfidf, vectorizer, labels, node_ids)
    elif config.method == "gmm_soft":
        k = int(min(config.gmm_k, max(2, len(slices) // max(1, config.min_node_slices))))
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=config.random_seed,
            max_iter=120,
            reg_covar=1e-5,
        )
        model.fit(vectors)
        probs = model.predict_proba(vectors)
        node_ids = [f"gmm_{idx:02d}" for idx in range(k)]
        node_table = pd.DataFrame(
            {
                "node_id": node_ids,
                "label_index": list(range(k)),
                "parent_id": "ROOT",
                "depth": 1,
                "slice_count": probs.sum(axis=0).tolist(),
                "user_count": [int((pd.Series(probs[:, idx]).gt(0.10).groupby(slices["user_id"].astype(str)).any()).sum()) for idx in range(k)],
            }
        )
        node_scores = soft_node_scores(slices, probs, node_ids)
        reliability = split_half_reliability(
            soft_node_scores(slices, probs, node_ids, row_mask=even_mask),
            soft_node_scores(slices, probs, node_ids, row_mask=odd_mask),
        )
        node_terms = top_terms_for_soft_nodes(tfidf, vectorizer, probs, node_ids)
    else:
        raise ValueError(f"unknown SUICA factor method: {config.method}")
    factor_scores, factor_loadings, factor_diag = fit_factors(node_scores, reliability, node_terms, config)
    big5_corr = external_correlations(factor_scores, users, BIG5_TRAITS, "Big5_suica_factor")
    mbti_corr = external_correlations(factor_scores, bridge, MBTI_AXES, "MBTI_bridge_suica_factor")
    cv = big5_cv(factor_scores, users, alpha=config.ridge_alpha) if any(col.startswith("suica_factor_") for col in factor_scores.columns) else pd.DataFrame()
    return {
        "tfidf": tfidf,
        "vectors": vectors,
        "vectorizer": vectorizer,
        "vector_diag": vector_diag,
        "node_table": node_table,
        "node_terms": node_terms,
        "node_scores": node_scores,
        "reliability": reliability,
        "factor_scores": factor_scores,
        "factor_loadings": factor_loadings,
        "factor_diag": factor_diag,
        "big5_corr": big5_corr,
        "mbti_corr": mbti_corr,
        "cv": cv,
    }


def q05_count(frame: pd.DataFrame) -> int:
    """Count FDR-significant rows."""
    if frame.empty or "pearson_q_bh_within_label" not in frame:
        return 0
    return int(pd.to_numeric(frame["pearson_q_bh_within_label"], errors="coerce").le(0.05).sum())


def max_abs_r(frame: pd.DataFrame) -> float:
    """Largest absolute Pearson r in a correlation frame."""
    if frame.empty or "abs_pearson_r" not in frame:
        return 0.0
    val = pd.to_numeric(frame["abs_pearson_r"], errors="coerce").max(skipna=True)
    return float(val) if np.isfinite(val) else 0.0


def mean_big5_cv(cv: pd.DataFrame) -> float:
    """Mean official-fold Big5 Pearson over traits."""
    if cv.empty or "trait" not in cv or "pearson_r" not in cv:
        return 0.0
    return float(cv.groupby("trait")["pearson_r"].mean().mean())


def mean_reliability(reliability: pd.DataFrame) -> float:
    """Mean split-half reliability."""
    if reliability.empty:
        return 0.0
    rel = pd.to_numeric(reliability["split_half_r"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(rel.mean()) if len(rel) else 0.0


def method_factor_rate(loadings: pd.DataFrame, *, top_k: int = 8) -> float:
    """Share of factors whose strongest nodes are method/platform dominated."""
    if loadings.empty or "node_top_terms" not in loadings:
        return 0.0
    hits = 0
    factors = 0
    for _factor, group in loadings.sort_values("abs_loading", ascending=False).groupby("factor"):
        factors += 1
        text = " ".join(group.head(top_k)["node_top_terms"].fillna("").astype(str).tolist())
        if METHOD_RE.search(text):
            hits += 1
    return hits / max(1, factors)


def mean_offdiag_abs_corr(frame: pd.DataFrame) -> float:
    """Mean absolute off-diagonal correlation."""
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return 0.0
    corr = numeric.corr().to_numpy(float)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = np.abs(corr[mask])
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else 0.0


def summarize_run(slice_strategy: str, factor_method: str, slices: pd.DataFrame, artifacts: dict[str, Any]) -> dict[str, Any]:
    """Summarize one SUICA strategy-method run."""
    big5_corr = artifacts["big5_corr"]
    mbti_corr = artifacts["mbti_corr"]
    cv = artifacts["cv"]
    reliability = artifacts["reliability"]
    factor_scores = artifacts["factor_scores"]
    factor_loadings = artifacts["factor_loadings"]
    factor_diag = artifacts["factor_diag"]
    node_table = artifacts["node_table"]
    mean_rel = mean_reliability(reliability)
    stable_node_rate = float(pd.to_numeric(reliability.get("split_half_r", pd.Series(dtype=float)), errors="coerce").ge(0.20).mean()) if not reliability.empty else 0.0
    factor_var = float(factor_diag.get("factor_explained_variance_sum", 0.0) or 0.0)
    cv_mean = mean_big5_cv(cv)
    top_big5 = max_abs_r(big5_corr)
    top_mbti = max_abs_r(mbti_corr)
    redundancy = mean_offdiag_abs_corr(factor_scores.drop(columns=["user_id"], errors="ignore")) if not factor_scores.empty else 0.0
    method_rate = method_factor_rate(factor_loadings)
    leakage_rate = float(slices.attrs.get("personality_leakage_slices_before_filter", 0) / max(1, slices.attrs.get("token_slices_before_filter", len(slices))))
    measurement = (
        0.35 * min(1.0, max(0.0, mean_rel))
        + 0.20 * min(1.0, stable_node_rate)
        + 0.20 * min(1.0, factor_var / 0.40)
        + 0.15 * (1.0 - min(1.0, redundancy))
        + 0.10 * min(1.0, len(node_table) / 80.0)
        - 0.20 * method_rate
        - 0.20 * leakage_rate
    )
    anchor = (
        0.35 * min(1.0, abs(top_big5) / 0.15)
        + 0.25 * min(1.0, abs(top_mbti) / 0.25)
        + 0.25 * min(1.0, abs(cv_mean) / 0.12)
        + 0.15 * min(1.0, (q05_count(big5_corr) + q05_count(mbti_corr)) / 6.0)
    )
    objective = float(max(0.0, min(1.0, 0.65 * measurement + 0.35 * anchor)))
    return {
        "slice_strategy": slice_strategy,
        "factor_method": factor_method,
        "suica_objective": objective,
        "measurement_score": float(max(0.0, min(1.0, measurement))),
        "anchor_score": float(max(0.0, min(1.0, anchor))),
        "slice_count": int(len(slices)),
        "users_with_slices": int(slices["user_id"].nunique()),
        "mean_slices_per_user": float(slices.groupby("user_id").size().mean()),
        "node_count": int(len(node_table)),
        "factor_count": int(factor_diag.get("factor_count", 0) or 0),
        "selected_node_feature_count": int(factor_diag.get("selected_node_feature_count", 0) or 0),
        "mean_split_half_r": mean_rel,
        "stable_node_rate_r_ge_0p20": stable_node_rate,
        "factor_explained_variance_sum": factor_var,
        "factor_redundancy_mean_abs_corr": redundancy,
        "method_factor_rate": method_rate,
        "personality_leakage_slice_rate": leakage_rate,
        "top_big5_abs_r": top_big5,
        "top_mbti_abs_r": top_mbti,
        "big5_q05_count": q05_count(big5_corr),
        "mbti_q05_count": q05_count(mbti_corr),
        "big5_cv_mean_r": cv_mean,
    }
