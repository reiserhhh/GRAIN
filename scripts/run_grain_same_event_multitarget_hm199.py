#!/usr/bin/env python3
"""Bounded same-event SOURCE/GRAIN/five-query observations, without answer fits."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_same_event_multitarget as M
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_joint_target_binding_hm198 as H

SPEC = ROOT / "configs/grain_same_event_multitarget_hm199.json"
CODE = (
    "configs/grain_same_event_multitarget_hm199.json",
    "docs/GRAIN_SAME_EVENT_MULTITARGET_PLAN_HM199.md",
    "grain_research/construction/math_same_event_multitarget.py",
    "scripts/run_grain_same_event_multitarget_hm199.py",
    "tests/test_grain_same_event_multitarget.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/navigation.py",
    "scripts/run_grain_joint_target_binding_hm198.py",
    "docs/GRAIN_VM05_SOURCE_EXPANSION_PROTOCOL.md",
    "scripts/run_grain_real_author_path.py",
)


def arrays(path):
    """Read numeric archives without pickle or printing raw local material."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def table(path, rows):
    """Persist all addressed contacts; convert undefined values to blank cells."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(x):
    """Undefined contacts remain absent in strict JSON and CSV."""
    return float(x) if np.isfinite(x) else None


def prepare(out):
    """Bind old public PANDORA units and new frozen prompts before observations."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement or automatic restart")
    spec = H.read(SPEC)
    p, e, n = (ROOT / spec[k] for k in ("parent", "event_parent", "native_parent"))
    inputs = {}
    for parent, seals in (
        (p, ("prepared_seal.json", "compiled_seal.json")),
        (e, ("prepared_seal.json", "compiled_seal.json")),
    ):
        for seal in seals:
            registered = H.read(parent / seal)
            H.check_hashes(parent, registered)
            for name in (seal, *registered):
                key = str((parent / name).relative_to(ROOT))
                inputs[key] = H.sha(ROOT / key)
    em, pm = arrays(e / "observations.npz"), arrays(p / "observations.npz")
    fields = arrays(p / "reader_bundle.npz")["fields"]
    if fields.shape != (2, 5, 4, 3, 1024):
        raise ValueError("Incomplete SOURCE target family")
    if not np.array_equal(pm["selected"], em["selected"]) or not np.array_equal(
        pm["control"], em["control"]
    ):
        raise ValueError("Different event selection interfaces")
    nm = H.read(n / "manifest.json")
    if H.sha(n / "private_material.json") != nm["material_sha256"]:
        raise ValueError("Changed historical public-dataset material")
    if H.read(n / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Historical observations are not terminal")
    old_rows = H.read(n / "private_material.json")
    lookup = {(r["author_id"], r["view"]): r for r in old_rows}
    if len(lookup) != len(old_rows):
        raise ValueError("Duplicated historical unit")
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    alphabet = D.candidates("neuroticism", tok)
    units, legacy_logits, legacy_z, legacy_h = [], [], [], []
    for u, uid in enumerate(em["author_ids"]):
        author_logits, author_z, author_h = [], [], []
        for view in (0, 1):
            row = lookup[(str(uid), view)]
            if row["task"] != "neuroticism" or row["source_group"] != "big5_only":
                raise ValueError("Unexpected author source")
            if any(
                not np.array_equal(row[k], em[k][u, view])
                for k in ("selected", "control")
            ):
                raise ValueError("Different S/C evidence identities")
            packet_path = n / "private_packets" / f"{uid}_{view}.json"
            state_path = packet_path.with_suffix(".npz")
            packet, states = H.read(packet_path), arrays(state_path)
            if H.sha(state_path) != packet["state_sha256"]:
                raise ValueError("Changed historical N states")
            if (packet["author_id"], packet["view"], packet["task"]) != (
                str(uid),
                view,
                "neuroticism",
            ):
                raise ValueError("Historical N endpoint identity mismatch")
            for path in (packet_path, state_path):
                inputs[str(path.relative_to(ROOT))] = H.sha(path)
            groups = {
                "full": list(range(12)),
                "remove_selected": [i for i in range(12) if i not in row["selected"]],
                "remove_control": [i for i in range(12) if i not in row["control"]],
                "selected_only": row["selected"],
                "control_only": row["control"],
            }
            vl, vz, vh = [], [], []
            for arm, slots in groups.items():
                texts = [row["texts"][i] for i in slots]
                old = packet["readouts"][arm]
                old_ids, _, _ = D.tokenize_prompt(texts, "neuroticism", tok)
                if len(old_ids) != old["tokens"]:
                    raise ValueError("Reconstructed historical N token length changed")
                vl.append(old["candidate_logits"])
                vz.append(old["full_vocab_log_normalizer"])
                vh.append(states[arm + "__final_norm"])
                shared = None
                for target in spec["new_target_indices"]:
                    user, _ = D.render(texts, M.TASKS[target])
                    messages = [
                        {"role": "system", "content": D.SYSTEM},
                        {"role": "user", "content": user},
                    ]
                    rendered = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    ids = tok.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                    encoded = tok(
                        rendered, add_special_tokens=False, return_offsets_mapping=True
                    )
                    boundary = rendered.rindex("\nEvidence ends here.")
                    position = max(
                        i
                        for i, (a, b) in enumerate(encoded["offset_mapping"])
                        if boundary >= b > a
                    )
                    if (
                        encoded["input_ids"] != ids
                        or len(ids) + 16 > spec["max_tokens"]
                    ):
                        raise ValueError(
                            "Changed native template or unsupported length"
                        )
                    prefix = ids[: position + 1]
                    if shared is not None and shared != prefix:
                        raise ValueError(
                            "Query changed the common causal evidence prefix"
                        )
                    shared = prefix
                    units.append(
                        {
                            "author": u,
                            "view": view,
                            "arm": M.ARMS.index(arm),
                            "target": target,
                            "input_ids": ids,
                            "body_position": position,
                        }
                    )
            author_logits.append(vl)
            author_z.append(vz)
            author_h.append(vh)
        legacy_logits.append(author_logits)
        legacy_z.append(author_z)
        legacy_h.append(author_h)
    if len(units) != spec["new_enquiries"] or len(em["author_ids"]) != spec["authors"]:
        raise ValueError("Incomplete declared support")
    for name in ("manifest.json", "run_status.json", "private_material.json"):
        path = n / name
        inputs[str(path.relative_to(ROOT))] = H.sha(path)
    model_files = {
        path.name: H.sha(path)
        for path in MODEL.iterdir()
        if path.is_file() and not path.name.startswith("._")
    }
    if model_files["model.safetensors"] != nm["model_sha256"]:
        raise ValueError("Different frozen native observer")
    (out / "00").mkdir(parents=True)
    (out / "01").mkdir()
    H.save_npz(
        out / "00/observations.npz",
        fields=fields,
        panel_vectors=pm["panel_vectors"][:, 0],
        selected=em["selected"],
        control=em["control"],
        author_ids=em["author_ids"],
        splits=em["splits"],
        legacy_logits=np.asarray(legacy_logits),
        legacy_z=np.asarray(legacy_z),
        legacy_states=np.asarray(legacy_h),
    )
    H.write(out / "00/units.json", units)
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "alphabet": alphabet,
            "code": {k: H.sha(ROOT / k) for k in CODE},
            "inputs": inputs,
            "model_files": model_files,
            "model_path": str(MODEL),
            "frozen_before_new_observations": True,
            "public_material": "PANDORA cached comments",
            "new_human_authors": 0,
            "new_psychological_labels_read": 0,
            "original_scope_complete": False,
        },
    )
    H.seal(
        out,
        "prepared_seal.json",
        ("00/observations.npz", "00/units.json", "00/manifest.json"),
    )
    H.status_write(
        out,
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
        },
    )
    print(f"PREPARED HM199:{len(units)}new inquiries/240 batches", flush=True)


def check(out):
    """Verify declared input/code bytes, not merely existence of receipts."""
    H.check_hashes(out, H.read(out / "prepared_seal.json"))
    m = H.read(out / "00/manifest.json")
    H.check_hashes(ROOT, m["code"])
    H.check_hashes(ROOT, m["inputs"])
    return m


def collect(out):
    """Run each new four-query batch once; preserve incomplete/failure states."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    m, status = check(out), H.read(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No uncertain, terminal or cap-renewed restart")
    H.check_hashes(MODEL, m["model_files"])
    units, spec = H.read(out / "00/units.json"), m["spec"]
    lock = out / "collector.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        torch.set_num_threads(2)
        start = time.monotonic()
        observer = DecoderObserver.load(
            str(MODEL),
            device=spec["device"],
            dtype=spec["dtype"],
            max_tokens=spec["max_tokens"],
        )
        status["model_seconds"] += time.monotonic() - start
        H.status_write(out, status)
        H.write(
            out / "runtime.json",
            {
                "pid": os.getpid(),
                "torch": torch.__version__,
                "device": spec["device"],
                "dtype": spec["dtype"],
            },
        )
        device = next(observer.model.parameters()).device
        alphabet = m["alphabet"]["token_ids"]
        head = observer.model.lm_head
        H.save_npz(
            out / "01/head.npz",
            weights=head.weight.detach()[alphabet].float().cpu().numpy(),
            bias=np.zeros(10)
            if head.bias is None
            else head.bias.detach()[alphabet].float().cpu().numpy(),
        )
        for b in range(spec["new_batches"]):
            if status["model_seconds"] >= spec["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Finite budget; no automatic extension")
            batch = units[4 * b : 4 * b + 4]
            address = {(u["author"], u["view"], u["arm"]) for u in batch}
            if (
                len(address) != 1
                or [u["target"] for u in batch] != spec["new_target_indices"]
            ):
                raise ValueError("Batch grouping changed")
            status.update(state="RUNNING", attempts=status["attempts"] + 1, pending=b)
            H.status_write(out, status)
            start = time.monotonic()
            try:
                length = max(len(u["input_ids"]) for u in batch)
                ids = torch.tensor(
                    [
                        u["input_ids"]
                        + [observer.tokenizer.pad_token_id]
                        * (length - len(u["input_ids"]))
                        for u in batch
                    ],
                    device=device,
                )
                mask = torch.tensor(
                    [
                        [1] * len(u["input_ids"]) + [0] * (length - len(u["input_ids"]))
                        for u in batch
                    ],
                    device=device,
                )
                ii = torch.arange(4, device=device)
                last = torch.tensor(
                    [len(u["input_ids"]) - 1 for u in batch], device=device
                )
                body_pos = torch.tensor(
                    [u["body_position"] for u in batch], device=device
                )
                captures = {}

                def capture(
                    _module, _args, output, captures=captures, ii=ii, body_pos=body_pos
                ):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captures["body"] = (
                        hidden[ii, body_pos].detach().float().cpu().numpy()
                    )

                hook = observer.model.model.layers[
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
                        states = result.last_hidden_state[ii, last]
                        full = (
                            observer.model.lm_head(states)
                            .float()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        states = states.detach().float().cpu().numpy()
                finally:
                    hook.remove()
                prefix_error = float(
                    np.max(abs(captures["body"] - captures["body"][0]))
                )
                if prefix_error > 3e-4:
                    raise ValueError("Future query changed causal prefix")
                path = out / f"01/b{b:03d}.npz"
                H.save_npz(
                    path,
                    candidate_logits=full[:, alphabet],
                    normalizer=logsumexp(full, axis=1),
                    final_states=states,
                    body=captures["body"],
                )
                H.write(
                    path.with_suffix(".json"),
                    {
                        "batch": b,
                        "address": list(next(iter(address))),
                        "sha256": H.sha(path),
                        "body_error": prefix_error,
                    },
                )
            finally:
                status["model_seconds"] += time.monotonic() - start
                H.status_write(out, status)
            status.update(completed=b + 1, pending=None)
            H.status_write(out, status)
            if (b + 1) % 10 == 0:
                print(
                    f"COLLECTED HM199 {b + 1}/240;{status['model_seconds']:.3f}s",
                    flush=True,
                )
        H.seal(
            out,
            "collected_seal.json",
            [p.relative_to(out).as_posix() for p in sorted((out / "01").iterdir())]
            + ["runtime.json"],
        )
        status.update(state="COLLECTED", pending=None)
        H.status_write(out, status)
    except Exception as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = repr(exc)
        H.status_write(out, status)
        raise
    finally:
        lock.unlink()


def complete(out):
    """Load all five targets with explicit provenance, never partial success."""
    m, status = check(out), H.read(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != m["spec"]["new_batches"]
    ):
        raise RuntimeError("Collection incomplete")
    H.check_hashes(out, H.read(out / "collected_seal.json"))
    z, units = arrays(out / "00/observations.npz"), H.read(out / "00/units.json")
    logits, normalizer, states = (
        np.empty((24, 2, 5, 5, 10)),
        np.empty((24, 2, 5, 5)),
        np.empty((24, 2, 5, 5, 1536)),
    )
    logits[:, :, 1], normalizer[:, :, 1], states[:, :, 1] = (
        z["legacy_logits"],
        z["legacy_z"],
        z["legacy_states"],
    )
    for b in range(240):
        a = arrays(out / f"01/b{b:03d}.npz")
        for j, u in enumerate(units[4 * b : 4 * b + 4]):
            k = u["author"], u["view"], u["target"], u["arm"]
            logits[k], normalizer[k], states[k] = (
                a["candidate_logits"][j],
                a["normalizer"][j],
                a["final_states"][j],
            )
    return m, z, logits, normalizer, states


def groups(z):
    """Keep original author clustering, each application view and historical roles."""
    return ["author_mean", "view0", "view1"] + [
        "split_" + s for s in sorted(set(z["splits"].tolist()))
    ]


def take(x, z, group):
    """Select whole authors, averaging views only when requested."""
    if group.startswith("view"):
        return x[:, int(group[-1])]
    ix = np.ones(24, bool) if group == "author_mean" else z["splits"] == group[6:]
    return x[ix].mean(1)


def analyze(out):
    """Build SOURCE/native contact matrices without choosing a predictor."""
    start = time.monotonic()
    m, z, logits, normalizer, states = complete(out)
    readings = M.digit_units(logits, normalizer)
    actions = M.natural_actions(readings).transpose(0, 1, 3, 2, 4)
    dh = M.natural_actions(states).transpose(0, 1, 3, 2, 4)
    common, specific = M.target_account(dh)
    head = arrays(out / "01/head.npz")["weights"].astype(float)
    w = head[5:].mean(0) - head[:5].mean(0)
    common_l, specific_l = common @ w, specific @ w
    dx = np.stack(
        [
            [
                M.vector_actions(
                    z["panel_vectors"][u, v], z["selected"][u, v], z["control"][u, v]
                )
                for v in (0, 1)
            ]
            for u in range(24)
        ]
    )
    source = np.einsum("uvod,brcsd->uvobrcs", dx, z["fields"])
    rows, contacts, accounts, diag = [], [], [], []
    for u, v, q, arm in np.ndindex(24, 2, 5, 5):
        values = readings[u, v, q, arm]
        rows.append(
            {
                "author": u,
                "view": v,
                "target": M.TASKS[q],
                "arm": M.ARMS[arm],
                "provenance": "inherited_N" if q == 1 else "new_HM199",
                **{
                    k: float(values[j])
                    for j, k in enumerate(
                        ("score", "log_odds", "geometric_logit", "candidate_mass")
                    )
                },
            }
        )
    for op, b, c, s in np.ndindex(2, 2, 4, 3):
        x = source[:, :, op, b, :, c, s]
        for group in groups(z):
            xx = take(x, z, group)
            for unit in range(3):
                yy = take(actions[:, :, op, :, unit], z, group)
                cv, cr = M.covariances(xx, yy), M.correlations(xx, yy)
                for r, q in np.ndindex(5, 5):
                    contacts.append(
                        {
                            "operation": ("deletion", "isolated")[op],
                            "source_view": b,
                            "component": (
                                "marginal",
                                "matched",
                                "reference_shift",
                                "profile_composition",
                            )[c],
                            "space": ("full", "retained", "complement")[s],
                            "group": group,
                            "unit": ("score", "log_odds", "geometric_logit")[unit],
                            "source_target": M.TASKS[r],
                            "native_query": M.TASKS[q],
                            "covariance": float(cv[r, q]),
                            "r": number(cr[r, q]),
                        }
                    )
            yy, cc, ss = (
                take(a[:, :, op], z, group)
                for a in (actions[..., 2], common_l, specific_l)
            )
            matrices = [M.covariances(xx, a) for a in (yy, cc, ss)]
            for r, q in np.ndindex(5, 5):
                accounts.append(
                    {
                        "operation": ("deletion", "isolated")[op],
                        "source_view": b,
                        "component": (
                            "marginal",
                            "matched",
                            "reference_shift",
                            "profile_composition",
                        )[c],
                        "space": ("full", "retained", "complement")[s],
                        "group": group,
                        "source_target": M.TASKS[r],
                        "native_query": M.TASKS[q],
                        "total": float(matrices[0][r, q]),
                        "common": float(matrices[1][r, q]),
                        "specific": float(matrices[2][r, q]),
                    }
                )
        if c == 1 and op == 0:
            xx, yy = x.mean(1), specific_l[:, :, op].mean(1)
            ci = M.bootstrap_diagonal(
                xx, yy, m["spec"]["bootstrap_draws"], m["spec"]["bootstrap_seed"]
            )
            r = M.correlations(xx, actions[:, :, op, :, 0].mean(1))
            cv = M.covariances(xx, yy)
            vcv = [M.covariances(x[:, v], specific_l[:, v, op]) for v in (0, 1)]
            for q in range(5):
                diag.append(
                    {
                        "source_view": b,
                        "space": ("full", "retained", "complement")[s],
                        "target": M.TASKS[q],
                        "native_score_r": number(r[q, q]),
                        "specific_geometric_covariance": float(cv[q, q]),
                        "specific_view0_covariance": float(vcv[0][q, q]),
                        "specific_view1_covariance": float(vcv[1][q, q]),
                        "specific_covariance_CI_low": float(ci[q, 0]),
                        "specific_covariance_CI_high": float(ci[q, 1]),
                    }
                )
    direct = (
        np.einsum("uvqad,kd->uvqak", states, head) + arrays(out / "01/head.npz")["bias"]
    )
    head_error = float(np.max(abs(direct - logits)))
    linear_error = float(np.max(abs(actions[..., 2] - common_l - specific_l)))
    if max(head_error, linear_error) > 3e-4:
        raise ValueError("Frozen output head cannot account for the observations")
    for name, data in (
        ("native_readings.csv", rows),
        ("all_contacts.csv", contacts),
        ("common_specific_accounts.csv", accounts),
        ("matched_diagonals.csv", diag),
    ):
        table(out / name, data)
    H.save_npz(
        out / "analysis.npz",
        source_actions=source,
        native_actions=actions,
        common_logit_action=common_l,
        specific_logit_action=specific_l,
    )
    summary = {
        "new_batches": 240,
        "new_enquiries": 960,
        "inherited_N_observations": 240,
        "authors": 24,
        "native_readings": len(rows),
        "matrix_contacts": len(contacts),
        "common_specific_accounts": len(accounts),
        "matched_diagonals": diag,
        "head_account_error": head_error,
        "common_specific_logit_error": linear_error,
        "candidate_mass_min_mean": [
            float(readings[..., 3].min()),
            float(readings[..., 3].mean()),
        ],
        "model_seconds": H.read(out / "status.json")["model_seconds"],
        "analysis_seconds": time.monotonic() - start,
        "native_fit": False,
        "psychological_accuracy": False,
        "original_scope_complete": False,
    }
    H.write(out / "summary.json", summary)
    H.seal(
        out,
        "analyzed_seal.json",
        (
            "native_readings.csv",
            "all_contacts.csv",
            "common_specific_accounts.csv",
            "matched_diagonals.csv",
            "analysis.npz",
            "summary.json",
        ),
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "matched_diagonals"}),
        flush=True,
    )


def verify(out):
    """Replay direct event means and covariances without new model observations."""
    _, z, logits, _normalizer, states = complete(out)
    H.check_hashes(out, H.read(out / "analyzed_seal.json"))
    a = arrays(out / "analysis.npz")
    head = arrays(out / "01/head.npz")
    head_error = 0.0
    for u, v, q, arm in np.ndindex(24, 2, 5, 5):
        head_error = max(
            head_error,
            float(
                np.max(
                    abs(
                        head["weights"].astype(float) @ states[u, v, q, arm]
                        + head["bias"]
                        - logits[u, v, q, arm]
                    )
                )
            ),
        )
    direct_error = 0.0
    for u, v in np.ndindex(24, 2):
        x, s, c = z["panel_vectors"][u, v], z["selected"][u, v], z["control"][u, v]
        s_left, c_left = (
            [i for i in range(12) if i not in s],
            [i for i in range(12) if i not in c],
        )
        for op, shift in enumerate(
            (x[s_left].mean(0) - x[c_left].mean(0), x[s].mean(0) - x[c].mean(0))
        ):
            for b, r, k, space in np.ndindex(2, 5, 4, 3):
                expected = float(z["fields"][b, r, k, space] @ shift)
                direct_error = max(
                    direct_error,
                    abs(expected - a["source_actions"][u, v, op, b, r, k, space]),
                )
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    score = p @ (np.arange(10) / 9)
    direct_actions = np.stack(
        (score[..., 1] - score[..., 2], score[..., 3] - score[..., 4]), -1
    ).transpose(0, 1, 3, 2)
    score_error = float(np.max(abs(direct_actions - a["native_actions"][..., 0])))
    covariance_error = 0.0
    with (out / "common_specific_accounts.csv").open() as stream:
        for row in csv.DictReader(stream):
            op = ("deletion", "isolated").index(row["operation"])
            b, r, q = (
                int(row["source_view"]),
                M.TASKS.index(row["source_target"]),
                M.TASKS.index(row["native_query"]),
            )
            k = ("marginal", "matched", "reference_shift", "profile_composition").index(
                row["component"]
            )
            space = ("full", "retained", "complement").index(row["space"])
            x = take(a["source_actions"][:, :, op, b, :, k, space], z, row["group"])[
                :, r
            ]
            for name, values in (
                ("total", a["native_actions"][..., 2]),
                ("common", a["common_logit_action"]),
                ("specific", a["specific_logit_action"]),
            ):
                y = take(values[:, :, op], z, row["group"])[:, q]
                expected = np.cov(x, y, ddof=1)[0, 1]
                covariance_error = max(
                    covariance_error, abs(float(row[name]) - expected)
                )
    all_rows = list(csv.DictReader((out / "common_specific_accounts.csv").open()))
    addition = max(
        abs(float(r["total"]) - float(r["common"]) - float(r["specific"]))
        for r in all_rows
    )
    if (
        max(direct_error, score_error, covariance_error) > 3e-12
        or max(head_error, addition) > 3e-4
    ):
        raise ValueError("Direct alternate-formula audit failed")
    receipt = {
        "PASS": True,
        "native_accounts": 1200,
        "common_specific_covariance_accounts": len(all_rows),
        "head_error": head_error,
        "event_mean_error": direct_error,
        "score_error": score_error,
        "covariance_error": covariance_error,
        "addition_error": addition,
        "same_agent_not_blind": True,
    }
    H.write(out / "verification.json", receipt)
    H.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/same_event_multitarget_r1",
    )
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
        args.stage
    ](args.output)
