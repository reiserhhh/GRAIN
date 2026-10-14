#!/usr/bin/env python3
"""One bounded new joint-target observation; no fit, retries or precision search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_target_binding as M
from grain_research.vector_mechanisms.navigation import MODEL

SPEC = ROOT / "configs/grain_joint_target_binding_hm198.json"
CODE = (
    "configs/grain_joint_target_binding_hm198.json",
    "docs/GRAIN_JOINT_TARGET_BINDING_PLAN_HM198.md",
    "grain_research/construction/math_joint_target_binding.py",
    "scripts/run_grain_joint_target_binding_hm198.py",
    "tests/test_grain_joint_target_binding.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/vector_mechanisms/trace.py",
)


def read(path):
    """Read a JSON artifact without accepting executable pickle input."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash input bytes incrementally, including frozen model weights."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Create a scientific artifact exclusively; do not overwrite failed results."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def status_write(out, status):
    """Atomically update only the mutable execution receipt."""
    path = out / "status.tmp"
    path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    path.replace(out / "status.json")


def save_npz(path, **values):
    """Write finite numeric arrays with an exclusive file descriptor."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def check_hashes(root, hashes):
    """Verify sealed code or artifacts instead of trusting their presence."""
    for name, expected in hashes.items():
        if sha(root / name) != expected:
            raise ValueError("Changed sealed bytes: " + name)


def seal(out, name, paths):
    """Commit a stage receipt containing exactly its output bytes."""
    write(out / name, {p: sha(out / p) for p in paths})


def prepare(out):
    """Freeze new texts, question roles, native tokens and dependencies before calls."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement or automatic r1 restart")
    spec, rows, query = read(SPEC), M.documents(), M.queries()
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    answers = [tok.encode(s, add_special_tokens=False) for s in ("N", "Y")]
    if any(len(x) != 1 for x in answers) or answers[0] == answers[1]:
        raise ValueError("Distinct native single-token N/Y alphabet required")
    units = []
    for row in rows:
        shared = None
        for qi, q in enumerate(query):
            user = f"Record:\n{row['text']}\n\nQuestion: {q['question']}\nAnswer:"
            messages = [
                {"role": "system", "content": M.SYSTEM},
                {"role": "user", "content": user},
            ]
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            encoded = tok(
                rendered, add_special_tokens=False, return_offsets_mapping=True
            )
            if encoded["input_ids"] != ids or rendered.count(row["text"]) != 1:
                raise ValueError("Native template or unique body anchor changed")
            start = rendered.index(row["text"])
            end = start + len(row["text"])
            eligible = [
                i
                for i, (a, b) in enumerate(encoded["offset_mapping"])
                if end >= b > a >= start
            ]
            position = eligible[-1]
            prefix = ids[: position + 1]
            if shared is not None and prefix != shared:
                raise ValueError("A later query changed the actual causal body prefix")
            shared = prefix
            if len(ids) + 16 > spec["max_tokens"]:
                raise ValueError("No implicit truncation")
            units.append(
                {
                    "document": row["document"],
                    "query": qi,
                    "input_ids": ids,
                    "body_position": position,
                }
            )
    if len(rows) != spec["documents"] or len(query) != spec["queries"]:
        raise ValueError("Fixed support changed")
    model_files = {
        p.name: sha(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    (out / "00").mkdir(parents=True)
    (out / "01").mkdir()
    for name, value in (("rows", rows), ("queries", query), ("units", units)):
        write(out / f"00/{name}.json", value)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "answer_ids": [a[0] for a in answers],
            "model_path": str(MODEL),
            "model_files": model_files,
            "code": {p: sha(ROOT / p) for p in CODE},
            "new_documents": 16,
            "new_enquiries": 224,
            "frozen_before_new_calls": True,
            "psychological_truth": False,
            "human_labels_or_private_text": False,
            "new_fit": False,
        },
    )
    seal(
        out,
        "prepared_seal.json",
        ["00/rows.json", "00/queries.json", "00/units.json", "00/manifest.json"],
    )
    status_write(
        out,
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
        },
    )
    print("PREPARED HM198:16new records/224 enquiries", flush=True)


def check(out):
    """Authenticate the material and every scientific code dependency."""
    check_hashes(out, read(out / "prepared_seal.json"))
    m = read(out / "00/manifest.json")
    check_hashes(ROOT, m["code"])
    return m


def collect(out):
    """Collect sixteen fixed batches once, with safe pending/failure receipts."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest, status = check(out), read(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No uncertain, terminal or cap-renewed replay")
    check_hashes(MODEL, manifest["model_files"])
    spec, units = manifest["spec"], read(out / "00/units.json")
    lock = out / "collector.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        torch.set_num_threads(2)
        start = time.monotonic()
        try:
            observer = DecoderObserver.load(
                str(MODEL),
                device=spec["model_device"],
                dtype=spec["model_dtype"],
                max_tokens=spec["max_tokens"],
            )
            observer.model.requires_grad_(False)
        finally:
            status["model_seconds"] += time.monotonic() - start
            status_write(out, status)
        write(
            out / "runtime.json",
            {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(next(observer.model.parameters()).device),
                "dtype": str(next(observer.model.parameters()).dtype),
                "pid": os.getpid(),
            },
        )
        indices = manifest["answer_ids"]
        head = observer.model.lm_head
        w = head.weight.detach()[indices].float().cpu().numpy().astype(float)
        bias = (
            np.zeros(2)
            if head.bias is None
            else head.bias.detach()[indices].float().cpu().numpy().astype(float)
        )
        save_npz(out / "01/head.npz", weights=w, bias=bias)
        for doc in range(spec["documents"]):
            if status["model_seconds"] >= spec["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Finite cap; no automatic renewal")
            batch = [u for u in units if u["document"] == doc]
            status.update(state="RUNNING", attempts=status["attempts"] + 1, pending=doc)
            status_write(out, status)
            start = time.monotonic()
            try:
                device = next(observer.model.parameters()).device
                length = max(len(u["input_ids"]) for u in batch)
                padded = [
                    u["input_ids"]
                    + [observer.tokenizer.pad_token_id] * (length - len(u["input_ids"]))
                    for u in batch
                ]
                ids = torch.tensor(padded, dtype=torch.long, device=device)
                mask = torch.tensor(
                    [
                        [1] * len(u["input_ids"]) + [0] * (length - len(u["input_ids"]))
                        for u in batch
                    ],
                    dtype=torch.long,
                    device=device,
                )
                arange = torch.arange(len(batch), device=device)
                last = torch.tensor(
                    [len(u["input_ids"]) - 1 for u in batch], device=device
                )
                body_pos = torch.tensor(
                    [u["body_position"] for u in batch], device=device
                )
                captures = {}

                def capture(
                    _module,
                    _args,
                    output,
                    captures=captures,
                    arange=arange,
                    body_pos=body_pos,
                ):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captures["body"] = (
                        hidden[arange, body_pos].detach().float().cpu().numpy()
                    )

                handle = observer.model.model.layers[
                    spec["capture_layer"]
                ].register_forward_hook(capture)
                try:
                    with torch.inference_mode():
                        result = observer.model.model(
                            input_ids=ids,
                            attention_mask=mask,
                            use_cache=False,
                            return_dict=True,
                        )
                        h = result.last_hidden_state[arange, last]
                        full = (
                            observer.model.lm_head(h)
                            .float()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        h = h.detach().float().cpu().numpy()
                finally:
                    handle.remove()
                if np.max(abs(captures["body"] - captures["body"][0])) > 3e-4:
                    raise ValueError(
                        "Future question modified the common body observation"
                    )
                candidate = full[:, indices]
                normalizer = logsumexp(full, axis=1)
                conditional = np.exp(candidate - logsumexp(candidate, axis=1)[:, None])
                logodds = candidate[:, 1] - candidate[:, 0]
                mass = np.exp(logsumexp(candidate, axis=1) - normalizer)
                entropy = -(conditional * np.log(conditional)).sum(axis=1)
                values = np.column_stack((conditional[:, 1], logodds, mass, entropy))
                path = out / f"01/b{doc:02d}.npz"
                save_npz(
                    path,
                    candidate_logits=candidate,
                    normalizer=normalizer,
                    readings=values,
                    final_state=h,
                    body=captures["body"],
                )
                write(
                    out / f"01/b{doc:02d}.json",
                    {
                        "document": doc,
                        "queries": len(batch),
                        "sha256": sha(path),
                        "body_query_max_difference": float(
                            np.max(abs(captures["body"] - captures["body"][0]))
                        ),
                    },
                )
            finally:
                status["model_seconds"] += time.monotonic() - start
                status_write(out, status)
            status.update(completed=doc + 1, pending=None)
            status_write(out, status)
            print(
                f"COLLECTED HM198 {doc + 1}/16;{status['model_seconds']:.3f}s",
                flush=True,
            )
        seal(
            out,
            "collected_seal.json",
            [p.relative_to(out).as_posix() for p in sorted((out / "01").iterdir())]
            + ["runtime.json"],
        )
        status.update(state="COLLECTED", pending=None)
        status_write(out, status)
    except Exception as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = repr(exc)
        status_write(out, status)
        raise
    finally:
        lock.unlink()


def material(out):
    """Read complete sealed native observations, not a partial running packet."""
    check(out)
    status = read(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 16
        or status["pending"] is not None
    ):
        raise ValueError("All sixteen actual batches required")
    check_hashes(out, read(out / "collected_seal.json"))
    blocks = []
    for i in range(16):
        with np.load(out / f"01/b{i:02d}.npz", allow_pickle=False) as z:
            blocks.append({n: z[n] for n in z.files})
    return read(out / "00/rows.json"), {
        n: np.stack([b[n] for b in blocks]) for n in blocks[0]
    }


def csv_write(path, rows):
    """Create a complete inspectable table; keep signed failures and controls."""
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(out):
    """Test frozen composition mechanisms, without fitting any native output."""
    rows, z = material(out)
    query = read(out / "00/queries.json")
    spec = read(SPEC)
    y = z["readings"]
    coefficients = M.coefficients(rows, y)
    fields = M.binding_fields(y[:, :, 0])
    field_names = list(fields)
    field_values = np.stack(list(fields.values()), axis=1)
    field_coefficients = M.coefficients(rows, field_values)
    logic = M.coefficients(rows, M.truth_table(rows))
    rows_csv, contrast_csv, field_csv = [], [], []
    for bank in range(2):
        for order in range(2):
            for qi, q in enumerate(query):
                for ui, unit in enumerate(
                    ("score", "logodds", "answer_mass", "entropy")
                ):
                    c = coefficients[bank, order, :, qi, ui]
                    rows_csv.append(
                        {
                            "bank": bank,
                            "order": order,
                            "query": qi,
                            "kind": q["kind"],
                            "unit": unit,
                            **dict(zip(("common", "E", "N", "EN"), c.tolist())),
                        }
                    )
                contrast_csv.append(
                    {
                        "bank": bank,
                        "order": order,
                        "query": qi,
                        "kind": q["kind"],
                        "score_difference": float(
                            0.8 * coefficients[bank, order, 3, qi, 0]
                        ),
                        "logodds_difference": float(
                            0.8 * coefficients[bank, order, 3, qi, 1]
                        ),
                        "designed_logic_difference": float(
                            0.8 * logic[bank, order, 3, qi]
                        ),
                    }
                )
            for ni, name in enumerate(field_names):
                c = field_coefficients[bank, order, :, ni]
                field_csv.append(
                    {
                        "bank": bank,
                        "order": order,
                        "field": name,
                        **dict(zip(("common", "E", "N", "EN"), c.tolist())),
                    }
                )
    csv_write(out / "characters.csv", rows_csv)
    csv_write(out / "author_law_contrasts.csv", contrast_csv)
    csv_write(out / "composition_fields.csv", field_csv)
    save_npz(
        out / "analysis_arrays.npz",
        coefficients=coefficients,
        field_values=field_values,
        field_coefficients=field_coefficients,
        author_contrasts=M.author_contrast(rows, y, spec["author_law_rho"]),
    )
    j, p = fields["logical_opposition"], fields["atomic_product"]
    summary = {
        "native_joint_AND_EN": field_coefficients[:, :, 3, 0].tolist(),
        "native_joint_OR_EN": field_coefficients[:, :, 3, 1].tolist(),
        "logical_opposition_EN": field_coefficients[:, :, 3, 2].tolist(),
        "logical_common_EN": field_coefficients[:, :, 3, 3].tolist(),
        "atomic_product_EN": field_coefficients[:, :, 3, 4].tolist(),
        "composition_remainder_EN": field_coefficients[:, :, 3, 5].tolist(),
        "joint_remainder_RMS": float(np.sqrt(np.mean((j - p) ** 2))),
        "native_joint_RMS": float(np.sqrt(np.mean(j**2))),
        "atomic_product_RMS": float(np.sqrt(np.mean(p**2))),
        "same_direction_frames_AND": int((field_coefficients[:, :, 3, 0] > 0).sum()),
        "same_direction_frames_OR": int((field_coefficients[:, :, 3, 1] < 0).sum()),
        "single_query_EN_RMS": float(
            np.sqrt(np.mean(coefficients[:, :, 3, :4, 0] ** 2))
        ),
        "unchanged_fact_EN_RMS": float(
            np.sqrt(np.mean(coefficients[:, :, 3, 12:, 0] ** 2))
        ),
        "answer_mass_min": float(y[:, :, 2].min()),
        "answer_mass_mean": float(y[:, :, 2].mean()),
        "body_query_max_difference": max(
            read(out / f"01/b{i:02d}.json")["body_query_max_difference"]
            for i in range(16)
        ),
        "human_personality_accuracy": None,
        "original_scope_complete": False,
        "new_fits_or_grids": 0,
    }
    write(out / "summary.json", summary)
    seal(
        out,
        "analyzed_seal.json",
        [
            "characters.csv",
            "author_law_contrasts.csv",
            "composition_fields.csv",
            "analysis_arrays.npz",
            "summary.json",
        ],
    )
    print(json.dumps(summary, indent=2), flush=True)


def verify(out):
    """Alternate direct-query/mixture sums and CPU64 output-head evaluation."""
    rows, z = material(out)
    check_hashes(out, read(out / "analyzed_seal.json"))
    with np.load(out / "01/head.npz", allow_pickle=False) as head:
        reconstructed = (
            z["final_state"].astype(float) @ head["weights"].T + head["bias"]
        )
    head_error = float(np.max(abs(reconstructed - z["candidate_logits"])))
    y = z["readings"]
    score_error = float(
        np.max(
            abs(
                expit(z["candidate_logits"][:, :, 1] - z["candidate_logits"][:, :, 0])
                - y[:, :, 0]
            )
        )
    )
    with np.load(out / "analysis_arrays.npz", allow_pickle=False) as a:
        stored = {n: a[n] for n in a.files}
    error = 0.0
    for bank in range(2):
        for order in range(2):
            ids = [
                i
                for i, r in enumerate(rows)
                if (r["bank"], r["order"]) == (bank, order)
            ]
            direct = sum(rows[i]["e"] * rows[i]["n"] * y[i] / 4 for i in ids)
            error = max(
                error,
                float(np.max(abs(direct - stored["coefficients"][bank, order, 3]))),
            )
            plus = sum((1 + 0.4 * rows[i]["e"] * rows[i]["n"]) * y[i] / 4 for i in ids)
            minus = sum((1 - 0.4 * rows[i]["e"] * rows[i]["n"]) * y[i] / 4 for i in ids)
            error = max(
                error,
                float(
                    np.max(abs(plus - minus - stored["author_contrasts"][bank, order]))
                ),
            )
    f = M.binding_fields(y[:, :, 0])
    j = np.array(
        [
            (v[4] - v[5] - v[6] + v[7] - v[8] + v[9] + v[10] - v[11]) / 8
            for v in y[:, :, 0]
        ]
    )
    error = max(error, float(np.max(abs(j - f["logical_opposition"]))))
    if head_error > 3e-4 or score_error > 1e-12 or error > 1e-12:
        raise ValueError(
            f"Registered numeric audit failed:head={head_error};score={score_error};account={error}"
        )
    result = {
        "state": "PASS",
        "CPU64_head_max_error": head_error,
        "score_max_error": score_error,
        "direct_account_max_error": error,
        "native_enquiries": 224,
        "author_law_query_accounts": 56,
        "same_agent_alternate_arithmetic": True,
        "independent_blind_replication": False,
    }
    write(out / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    """Execute one explicit stage; never restart a terminal scientific collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / read(SPEC)["output"]
    globals()[args.stage](out)


if __name__ == "__main__":
    main()
