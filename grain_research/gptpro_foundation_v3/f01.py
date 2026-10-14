"""F01 ontology and predecessor ledger helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


OBJECT_CLASSES = [
    "reflective_latent",
    "formative_index",
    "conditional_response_function",
    "dynamic_state",
    "network_interaction",
    "predictive_state",
    "reader_judgment",
]

GRAPH_NODES = [
    {"id": "O", "label": "opportunity", "status": "assigned_or_observed"},
    {"id": "C", "label": "choice", "status": "latent_or_observed"},
    {"id": "E", "label": "behavior_or_experience", "status": "target_dependent"},
    {"id": "T", "label": "text", "status": "observed"},
    {"id": "R", "label": "reader_evidence", "status": "model_output"},
    {"id": "S", "label": "score", "status": "constructed"},
    {"id": "Q", "label": "query_or_intervention", "status": "assigned"},
    {"id": "L", "label": "LLM_state_and_output", "status": "model_latent_or_observed"},
]


def mechanism_graph() -> dict:
    """Return the frozen three-arrow measurement graph."""
    return {
        "nodes": GRAPH_NODES,
        "edges": [
            {"from": "O", "to": "C", "status": "latent_or_observed", "meaning": "exposure changes available choices"},
            {"from": "C", "to": "E", "status": "latent_or_observed", "meaning": "selection and response are distinct"},
            {"from": "E", "to": "T", "status": "observed", "meaning": "language is an implementation channel"},
            {"from": "T", "to": "R", "status": "estimated", "meaning": "reader extracts evidence with error"},
            {"from": "R", "to": "S", "status": "constructed", "meaning": "score applies a declared target function"},
            {"from": "Q", "to": "L", "status": "assigned", "meaning": "query can alter model state and output"},
            {"from": "L", "to": "T", "status": "observed", "meaning": "output is not a human psychological causal edge"},
        ],
        "cross_cutting": ["author_parameters", "current_state", "goal", "audience", "knowledge", "reader_method", "reference_population"],
        "forbidden_shortcuts": [
            "T_to_personality_truth_without_reader_and_target_contract",
            "attention_edge_to_human_psychological_causality",
            "unobserved_opportunity_as_zero_expression",
            "reader_consensus_as_external_validity",
        ],
    }


def object_dictionary() -> list[dict]:
    """Return seven object classes with units, operations, and counterexamples."""
    return [
        {"class": "reflective_latent", "input": "multiple indicators", "unit": "reference-scaled latent", "allowed": "factor/SEM with local independence contract", "counterexample": "same covariance from method factor", "evidence": "external/behavioral linkage"},
        {"class": "formative_index", "input": "weighted evidence components", "unit": "declared index unit", "allowed": "component aggregation and sensitivity", "counterexample": "component omission changes target", "evidence": "content and criterion map"},
        {"class": "conditional_response_function", "input": "author, condition, opportunity, history", "unit": "function output", "allowed": "intervention/transport within support", "counterexample": "same mean, different slopes", "evidence": "held-condition response"},
        {"class": "dynamic_state", "input": "ordered history", "unit": "state-space coordinate or predictive law", "allowed": "filter/forecast", "counterexample": "coordinate gauge with same observed law", "evidence": "future prediction and state uncertainty"},
        {"class": "network_interaction", "input": "author-by-partner/event relation", "unit": "relational response", "allowed": "partner perturbation and cross-level model", "counterexample": "constant partner confounds author and partner", "evidence": "independent interaction outcome"},
        {"class": "predictive_state", "input": "history", "unit": "equivalence class of future laws", "allowed": "finite-horizon prediction/update", "counterexample": "one-step equality but multi-step divergence", "evidence": "future-law certificate"},
        {"class": "reader_judgment", "input": "text and context", "unit": "reader-specific probability/score", "allowed": "calibration and method comparison", "counterexample": "shared reader bias", "evidence": "multi-reader/independent anchor"},
    ]


def source_sha(path: Path) -> str:
    """Hash a source ledger used by the audit."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predecessor_ledger() -> list[dict]:
    """Return the currently verified primary-method ledger from the project notes."""
    return [
        {"id": "L01", "method": "predictive representations of state", "primary_locator": "NeurIPS 2001 abstract", "url": "https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html", "adopt": "finite future-law state", "not_claimed": "human personality meaning"},
        {"id": "L02", "method": "finite MDP metrics", "primary_locator": "arXiv:1207.4114", "url": "https://arxiv.org/abs/1207.4114", "adopt": "transition-aware equivalence", "not_claimed": "one-step text response is bisimulation"},
        {"id": "L03", "method": "adaptive design optimization", "primary_locator": "PubMed PMID 20028226", "url": "https://pubmed.ncbi.nlm.nih.gov/20028226/", "adopt": "information-directed design comparator", "not_claimed": "new attention mechanism"},
        {"id": "L04", "method": "ALBA", "primary_locator": "NAACL 2024 paper", "url": "https://aclanthology.org/2024.naacl-long.136.pdf", "adopt": "language-adaptive assessment comparison", "not_claimed": "same estimand as GRAIN"},
        {"id": "L05", "method": "competing social goals", "primary_locator": "PMC7672308", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7672308/", "adopt": "competing-generation rival", "not_claimed": "goal weight is a personality trait"},
        {"id": "L06", "method": "nonlinear ICA with auxiliary variables", "primary_locator": "PMLR 89 (2019)", "url": "https://proceedings.mlr.press/v89/hyvarinen19a.html", "adopt": "finite auxiliary-identification certificate", "not_claimed": "topic metadata alone satisfies assumptions"},
        {"id": "L07", "method": "sloppy parameter combinations", "primary_locator": "PMC2000971", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2000971/", "adopt": "function versus parameter uncertainty", "not_claimed": "predictive success identifies every parameter"},
        {"id": "L08", "method": "free-form user interactions", "primary_locator": "arXiv:2405.13052", "url": "https://arxiv.org/abs/2405.13052", "adopt": "controlled versus natural input comparison", "not_claimed": "our replication or human validity"},
    ]
