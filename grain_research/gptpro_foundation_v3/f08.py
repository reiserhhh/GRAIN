"""Synthetic text renderer and frozen pole/qualification readers for F08."""

from __future__ import annotations

import numpy as np


STATUSES = ("actual", "quoted", "hypothetical", "negated")
FAMILIES = ("plain", "conditional", "repair")


def render_event(observation: int, subject_status: str, grammar_rng: np.random.Generator, *, family: str = "plain", variant: int = 0) -> str:
    """Render a status-controlled sentence without author/world metadata."""
    if subject_status not in STATUSES:
        raise ValueError(f"unknown subject status {subject_status}")
    polarity = "I agree with this plan" if int(observation) else "I do not agree with this plan"
    templates = {
        "plain": ["I said: {p}.", "My response was that {p}.", "I wrote that {p}.", "For me, {p}."],
        "conditional": ["If asked, I would say {p}.", "In this case I think {p}.", "The condition is that {p}.", "Suppose this happens: {p}."],
        "repair": ["To clarify, I meant {p}.", "Correction: {p}.", "After reconsidering, {p}.", "What I mean is {p}."],
    }
    prefix = {"actual": "I actually said", "quoted": "Someone else said", "hypothetical": "I might say", "negated": "I deny saying"}[subject_status]
    body = templates[family][variant % len(templates[family])].format(p=polarity)
    return f"{prefix}: {body}"


def fit_reader(texts: list[str], poles: list[int], qualifications: list[int]) -> dict:
    """Fit deterministic word/char TF-IDF readers for pole and qualification."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from scipy.sparse import hstack

    word = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=5000)
    features = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    pole = Ridge(alpha=1.0).fit(features, np.asarray(poles, dtype=float))
    qualification = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0).fit(features, np.asarray(qualifications, dtype=int))
    return {"word": word, "char": char, "pole": pole, "qualification": qualification, "n_features": features.shape[1]}


def predict_reader(reader: dict, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped pole score and qualification probability."""
    from scipy.sparse import hstack

    features = hstack([reader["word"].transform(texts), reader["char"].transform(texts)]).tocsr()
    pole = np.clip(reader["pole"].predict(features), 0.0, 1.0)
    qualification = reader["qualification"].predict_proba(features)[:, 1]
    return pole, qualification


def evidence_card(event: dict, pole: float | None, qualification: float | None) -> dict:
    """Create an inspectable evidence card with null-safe scores."""
    return {"event_id": event.get("event_id"), "text": event.get("text"), "subject_status": event.get("subject_status"), "reader_id": event.get("reader_id", "synthetic_tfidf_reader"), "raw_score": pole, "qualification": qualification, "target": event.get("target"), "contribution": None, "uncertainty": None}
