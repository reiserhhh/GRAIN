#!/usr/bin/env python
"""Run the frozen, local-only MEPS fixed-condition vector profile pilot.

Raw participant text remains in process memory only.  The script writes just
aggregate metrics and model/protocol metadata.  It deliberately does not load
the adjacent questionnaires, train a model, score assistant turns, or make a
personality/clinical/cross-language claim.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suica_core.meps_ai_conv import load_meps_ai_conv  # noqa: E402
from suica_core.meps_fixed_profile import paired_embedding_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path,
        default=ROOT / "data_sets" / "MEPS+AI_conv_experiment",
        help="Private MEPS bundle. Only decrypted_output user text is read.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/meps_fixed_condition_profile_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/meps_fixed_condition_profile_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/MEPS_V6_FIXED_CONDITION_PROFILE.md")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """Choose a local accelerator without routing private text off-host."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_frozen_encoder(cfg: dict[str, Any], device: str):
    """Load a pinned local model; no text can trigger a network download."""
    from sentence_transformers import SentenceTransformer

    spec = cfg["embedding"]
    model = SentenceTransformer(
        spec["model_id"],
        revision=spec["revision"],
        device=device,
        trust_remote_code=bool(spec["trust_remote_code"]),
        local_files_only=bool(spec["local_files_only"]),
    )
    model.max_seq_length = int(spec["max_seq_length"])
    return model


def _answer_views(views: pd.DataFrame, conditions: list[str]) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Return exactly one aggregate direct answer per participant and prompt."""
    answers = views.loc[
        views["view_type"].eq("meps_answer") & views["slot"].eq("aggregate")
    ].copy()
    grouped = {
        condition: answers.loc[answers["condition"].eq(condition)].copy()
        for condition in conditions
    }
    user_sets = [set(table["participant_code"].astype(str)) for table in grouped.values()]
    common = sorted(set.intersection(*user_sets)) if user_sets else []
    if len(common) < 4:
        raise ValueError("fewer than four participants have every primary MEPS answer")
    output: dict[str, pd.DataFrame] = {}
    for condition, table in grouped.items():
        table = table.loc[table["participant_code"].astype(str).isin(common)].copy()
        if table["participant_code"].duplicated().any():
            raise ValueError(f"duplicate aggregate direct answers in {condition}")
        output[condition] = table.set_index("participant_code").loc[common].reset_index()
    return common, output


def _free_views(views: pd.DataFrame, common: list[str]) -> pd.DataFrame:
    """Recover one participant-authored free-chat view without assistant turns."""
    free = views.loc[
        views["view_type"].eq("free_ai_chat") & views["slot"].eq("aggregate")
    ].copy()
    free = free.loc[free["participant_code"].astype(str).isin(common)].copy()
    if free["participant_code"].duplicated().any():
        raise ValueError("duplicate aggregate free-chat views")
    return free.set_index("participant_code").loc[common].reset_index()


def _token_length_audit(model: Any, prefixed_texts: list[str], *, max_length: int) -> dict[str, float | int]:
    """Record truncation risk before encode; no text or token IDs are retained."""
    encoded = model.tokenizer(
        prefixed_texts, truncation=False, add_special_tokens=True, verbose=False
    )
    lengths = np.array([len(ids) for ids in encoded["input_ids"]], dtype=int)
    return {
        "n_texts": int(len(lengths)),
        "median_tokens_before_truncation": float(np.median(lengths)),
        "max_tokens_before_truncation": int(np.max(lengths)),
        "n_texts_over_max_seq_length": int(np.sum(lengths > max_length)),
        "max_seq_length": int(max_length),
    }


def _encode(model: Any, texts: pd.Series, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, float | int]]:
    """Encode one condition with the frozen symmetric E5 query convention."""
    prefixed = [f'{spec["input_prefix"]}{text}' for text in texts.astype(str).tolist()]
    audit = _token_length_audit(model, prefixed, max_length=int(spec["max_seq_length"]))
    embeddings = model.encode(
        prefixed,
        batch_size=int(spec["batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(spec["normalize_embeddings"]),
    )
    return np.asarray(embeddings, dtype=float), audit


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return cosine matrix; embeddings are expected to be already L2-normalized."""
    return np.asarray(left, dtype=float) @ np.asarray(right, dtype=float).T


def _holm_pass(p_values: list[float], alpha: float) -> list[bool]:
    """Return Holm familywise decisions in the original hypothesis order."""
    order = np.argsort(np.asarray(p_values, dtype=float))
    out = [False] * len(p_values)
    for rank, position in enumerate(order):
        if p_values[position] <= alpha / (len(p_values) - rank):
            out[int(position)] = True
        else:
            break
    return out


def _write_report(
    path: Path,
    *,
    cfg: dict[str, Any],
    result: dict[str, Any],
    metrics: pd.DataFrame,
    token_audit: pd.DataFrame,
    primary_pass: bool,
) -> None:
    """Write a report without participant IDs, text, embeddings, or labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = metrics.loc[metrics["analysis_role"].eq("primary_fixed_prompt")].copy()
    descriptive = metrics.loc[metrics["analysis_role"].eq("descriptive_free_vs_fixed")].copy()
    lines = [
        "# SUICA V6 MEPS Fixed-Condition Vector Profile v1",
        "",
        "## Decision boundary",
        "",
        "This is a local-only, same-session Japanese **vector correspondence** pilot. It tests whether the frozen multilingual embedding places a participant's answers to two shared MEPS prompts closer than other participants' answers. It does not estimate personality, clinical state, external validity, cross-language numerical equivalence, or a cross-session response operator.",
        "",
        "- source text: `decrypted_output/` user-authored direct MEPS answers only",
        "- assistant turns: excluded from embeddings; retained nowhere in this output",
        "- questionnaires/mood: not loaded as labels or outcomes",
        "- raw text, participant IDs, token IDs, and embeddings: not persisted",
        "",
        "## Frozen representation",
        "",
        f"- model: `{cfg['embedding']['model_id']}`",
        f"- revision: `{cfg['embedding']['revision']}`",
        f"- input convention: `{cfg['embedding']['input_prefix']}` for every symmetric text view",
        f"- max sequence length: `{cfg['embedding']['max_seq_length']}`",
        f"- device used: `{result['runtime']['device']}`",
        "",
        "## Primary fixed-prompt comparisons",
        "",
        primary[
            [
                "condition_a", "condition_b", "n_participants", "same_person_auc",
                "same_person_auc_ci_low", "same_person_auc_ci_high", "linkage_permutation_p",
                "holm_pass", "length_matched_coverage", "length_matched_same_person_auc",
                "length_matched_auc_ci_low", "length_matched_auc_ci_high",
            ]
        ].round(4).to_markdown(index=False),
        "",
        "## Length / truncation audit",
        "",
        token_audit.round(3).to_markdown(index=False),
        "",
        "## Descriptive free-versus-fixed contrast",
        "",
        descriptive[
            [
                "condition_a", "condition_b", "n_participants", "same_person_auc",
                "same_person_auc_ci_low", "same_person_auc_ci_high", "linkage_permutation_p",
            ]
        ].round(4).to_markdown(index=False) if not descriptive.empty else "No complete free-chat comparison.",
        "",
        "The free-chat comparison is descriptive only. Its token audit is shown above; any text above the frozen 512-token cap is encoded with the model's standard truncation and cannot affect the primary direct-answer decision.",
        "",
        "## Registered interpretation",
        "",
        f"Primary promotion rule: every one of the three fixed-prompt comparisons must have AUC >= `{cfg['analysis']['promotion_min_auc']:.2f}`, participant-bootstrap lower CI > `{cfg['analysis']['promotion_ci_floor']:.2f}`, and Holm-corrected linkage permutation support at familywise alpha `{cfg['analysis']['familywise_alpha']:.2f}`. Result: `{'PASS' if primary_pass else 'NOT PROMOTED'}`.",
        "",
        "A PASS licenses only the statement that this frozen multilingual vector representation preserves protocol-conditioned same-person correspondence in this Japanese corpus. It does not license language equivalence, a stable trait claim, or generalization outside this single session.",
        "",
        "## Artifacts",
        "",
        "- `results/meps_fixed_condition_profile_v1/metrics.csv`",
        "- `results/meps_fixed_condition_profile_v1/token_audit.csv`",
        "- `results/meps_fixed_condition_profile_v1/run_manifest.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    spec = cfg["embedding"]
    analysis = cfg["analysis"]
    # Once the separately prefetched revision is present, force all MEPS text
    # processing offline. This is a defense-in-depth privacy control.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = resolve_device(args.device)
    model = load_frozen_encoder(cfg, device)
    views, _mood, loader_metadata = load_meps_ai_conv(args.input_root, include_mood=False)
    participant_codes, answer_tables = _answer_views(views, list(analysis["primary_conditions"]))

    embeddings: dict[str, np.ndarray] = {}
    audits: list[dict[str, Any]] = []
    for condition, table in answer_tables.items():
        embeddings[condition], audit = _encode(model, table["text"], spec)
        audits.append({"condition": condition, "analysis_role": "primary_fixed_prompt", **audit})

    metric_rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    pair_rows: list[dict[str, Any]] = []
    for index, (condition_a, condition_b) in enumerate(combinations(analysis["primary_conditions"], 2)):
        right_chars = answer_tables[condition_b]["user_chars"].to_numpy(float)
        row = paired_embedding_summary(
            _cosine(embeddings[condition_a], embeddings[condition_b]),
            right_chars,
            bootstrap_draws=int(analysis["bootstrap_draws"]),
            permutation_draws=int(analysis["permutation_draws"]),
            log_length_tolerance=float(analysis["log_length_tolerance"]),
            seed=int(cfg["seed"]) + index * 100,
        )
        row.update({"analysis_role": "primary_fixed_prompt", "condition_a": condition_a, "condition_b": condition_b})
        pair_rows.append(row)
        p_values.append(float(row["linkage_permutation_p"]))
    for row, holm_pass in zip(pair_rows, _holm_pass(p_values, float(analysis["familywise_alpha"])), strict=True):
        row["holm_pass"] = bool(holm_pass)
        metric_rows.append(row)

    # Free conversation is a separate condition. It is descriptive because it
    # changes both prompt/opportunity structure and interaction format.
    free = _free_views(views, participant_codes)
    if len(free) == len(participant_codes):
        free_embeddings, audit = _encode(model, free["text"], spec)
        audits.append({"condition": "free_chat", "analysis_role": "descriptive_free_vs_fixed", **audit})
        fixed_embeddings = np.mean(
            np.stack([embeddings[condition] for condition in analysis["primary_conditions"]]), axis=0
        )
        fixed_embeddings = fixed_embeddings / np.linalg.norm(fixed_embeddings, axis=1, keepdims=True)
        free_row = paired_embedding_summary(
            _cosine(fixed_embeddings, free_embeddings),
            free["user_chars"].to_numpy(float),
            bootstrap_draws=int(analysis["bootstrap_draws"]),
            permutation_draws=int(analysis["permutation_draws"]),
            log_length_tolerance=float(analysis["log_length_tolerance"]),
            seed=int(cfg["seed"]) + 999,
        )
        free_row.update({
            "analysis_role": "descriptive_free_vs_fixed",
            "condition_a": "fixed_prompt_mean_q1_q2_q3",
            "condition_b": "free_chat",
            "holm_pass": False,
        })
        metric_rows.append(free_row)

    metrics = pd.DataFrame(metric_rows)
    token_audit = pd.DataFrame(audits)
    primary = metrics.loc[metrics["analysis_role"].eq("primary_fixed_prompt")]
    primary_pass = bool(
        len(primary) == 3
        and primary["same_person_auc"].ge(float(analysis["promotion_min_auc"])).all()
        and primary["same_person_auc_ci_low"].gt(float(analysis["promotion_ci_floor"])).all()
        and primary["holm_pass"].all()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    token_audit.to_csv(args.output_dir / "token_audit.csv", index=False)
    manifest = {
        "run_name": cfg["run_name"],
        "config": cfg,
        "runtime": {
            "device": device,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sentence_transformers_version": __import__("sentence_transformers").__version__,
            "model_max_seq_length": int(model.max_seq_length),
        },
        "input": {
            "layout_manifest": loader_metadata["layout_manifest"],
            "n_loaded_participants": int(loader_metadata["n_loaded_participants"]),
            "n_primary_complete_participants": int(len(participant_codes)),
            "raw_text_persisted": False,
            "participant_ids_persisted": False,
            "embeddings_persisted": False,
            "external_labels_loaded": False,
        },
        "decision": "PASS_PROTOCOL_CONDITIONED_CORRESPONDENCE" if primary_pass else "NOT_PROMOTED",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(args.report, cfg=cfg, result={"runtime": manifest["runtime"]}, metrics=metrics,
                  token_audit=token_audit, primary_pass=primary_pass)
    print(json.dumps({
        "decision": manifest["decision"],
        "n_primary_complete_participants": len(participant_codes),
        "metrics_path": str(args.output_dir / "metrics.csv"),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
