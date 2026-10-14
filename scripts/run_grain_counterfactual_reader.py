#!/usr/bin/env python
"""Counterfactual evidence -> reader -> internal site packet V1 (docs/GRAIN_COUNTERFACTUAL_READER_PACKET_V1.md).

Stages:
  prepare  tokenizer-only input checkpoint (no model forward): units, hashes, assembly, alignment, truncation
  pilot    units 1-8, all 17 rows each (136 calls), then the frozen gates
  full     units 9-40 (544 calls), only after a PASS pilot that is not non-responsive
Every attempt goes through the append-only Accounting ledger (cap 680, checked before each call, no retry).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROTOCOL = REPO / "docs/GRAIN_COUNTERFACTUAL_READER_PACKET_V1.md"
PROTOCOL_SHA = "616fb2d2656998cdc94d54ddbe3a49574a782f28e7ec18c24d20378eaee8bf0d"
AMENDMENT = REPO / "docs/GRAIN_COUNTERFACTUAL_READER_PACKET_V1_A1.md"
AMENDMENT_SHA = "db1fc0963ff0fac829398c1e6fc9117f077c3a974dcbf71cfd5daf22b8aa5eac"
OUT = REPO / "results/grain_gptpro_foundation_v3/R14/counterfactual_reader_r1"
R17 = REPO / "results/grain_gptpro_foundation_v3/R17/natural_sensitivity_r1"
CAP = 680
TAU = 1e-6
PILOT_CAP_S, TOTAL_CAP_S = 45 * 60, 90 * 60
LAYER, SITE, CONTROL = 12, 0, 6
NLI_MODEL, NLI_REV = "cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
HYPOTHESIS = "I feel comfortable around people."
INSERTS = {
    1: {"pos": "\nI really love meeting new people and chatting with everyone at big parties.",
        "neg": "\nI really prefer quiet evenings alone and avoid meeting people at big parties.",
        "neu": "\nI need to replace the old light bulbs in the kitchen this weekend.",
        "oth": "\nMy friend loves meeting new people and chatting with everyone at big parties."},
    2: {"pos": "\nBeing around lots of people gives me energy, and I talk to strangers easily.",
        "neg": "\nBeing around lots of people drains my energy, and I avoid talking to strangers.",
        "neu": "\nThe bus schedule changed last month, so I now leave home ten minutes earlier.",
        "oth": "\nBeing around lots of people gives her energy, and she talks to strangers easily."},
}
CONDS = ("orig", "neu", "pos", "neg", "oth")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_text(t: str) -> str:
    return sha(t.encode("utf-8"))


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def load_units() -> list[dict]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("r17", REPO / "scripts/run_grain_r17_natural_sensitivity.py")
    r17 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r17)
    material = r17.load_material()
    manifest = r17.load_manifest(R17)
    units = []
    for corpus in ("pandora", "essays"):
        sub = manifest[manifest.corpus == corpus].head(20).reset_index(drop=True)
        for i, rec in sub.iterrows():
            text = material[corpus]["texts"][int(rec["unit_row"])]
            if sha_text(text) != rec["base_sha256"]:
                raise ValueError(f"base text hash drift {corpus} {rec['author_index']}")
            units.append({"key": f"{corpus}:{int(rec['author_index'])}", "corpus": corpus, "pos": int(i),
                          "author_index": int(rec["author_index"]), "set": 1 if i % 2 == 0 else 2,
                          "pilot": i < 4, "text": text, "text_sha256": rec["base_sha256"]})
    return units


def qwen_template(observer):
    from grain_research.vector_mechanisms.direct_traits import SYSTEM, question, render
    marker = "REPLACEABLEARTICLEMARKER"
    user, _ = render([marker], "extraversion")
    ids = observer.prompt_ids(SYSTEM, user)
    mpos = observer.content_indices(SYSTEM, user, marker)
    qpos = observer.content_indices(SYSTEM, user, question("extraversion"))
    if mpos != list(range(mpos[0], mpos[-1] + 1)):
        raise ValueError("noncontiguous marker")
    prefix, suffix = ids[: mpos[0]], ids[mpos[-1] + 1:]
    q_off = qpos[0] - (mpos[-1] + 1)
    head = observer.tokenizer.decode(suffix[q_off:])
    if not head.startswith(" the writer's extraversion"):  # A1: V2 content_indices row definition
        raise ValueError("question offset misaligned")
    return prefix, suffix, q_off


def assemble(tok, prefix, suffix, q_off, text, insert, budget):
    a = tok.encode(text, add_special_tokens=False)
    kept = a[:budget]
    ins = tok.encode(insert, add_special_tokens=False) if insert else []
    ids = prefix + kept + ins + suffix
    qstart = len(prefix) + len(kept) + len(ins) + q_off
    return {"ids": ids, "qrows": list(range(qstart, len(ids))), "prefix_rows": list(range(0, len(prefix) + len(kept))),
            "unit_tokens": len(a), "unit_tokens_kept": len(kept), "insert_tokens": len(ins)}


def nli_setup():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = snapshot_download(NLI_MODEL, revision=NLI_REV, local_files_only=True,
                             allow_patterns=["*.json", "*.txt", "spm.model", "model.safetensors"])
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, use_safetensors=True, dtype=torch.float32).eval()
    mapping = {int(i): str(v).lower() for i, v in model.config.id2label.items()}
    if set(mapping.values()) != {"contradiction", "entailment", "neutral"}:
        raise ValueError("unknown NLI labels")
    return tok, model, mapping


def nli_truncate(tok, text):
    hyp = len(tok.encode(HYPOTHESIS, add_special_tokens=False))
    longest = max(len(tok.encode(s, add_special_tokens=False)) for d in INSERTS.values() for s in d.values())
    budget = 512 - 3 - hyp - longest
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    n = len(enc["input_ids"])
    if n <= budget:
        return text, {"nli_unit_tokens": n, "nli_unit_tokens_kept": n, "nli_truncated": False}
    cut = enc["offset_mapping"][budget - 1][1]
    return text[:cut], {"nli_unit_tokens": n, "nli_unit_tokens_kept": budget, "nli_truncated": True}


def prepare():
    from scripts.run_grain_persona_generation_trace import load_observer
    OUT.mkdir(parents=True, exist_ok=True)
    units = load_units()
    authors = {}
    for u in units:
        authors.setdefault((u["corpus"], u["author_index"]), []).append(u["key"])
    observer = load_observer(max_tokens=1024, layer_count=2)
    tok = observer.tokenizer
    calls_before = observer.calls
    prefix, suffix, q_off = qwen_template(observer)
    ins_len = {s: {c: len(tok.encode(t, add_special_tokens=False)) for c, t in d.items()} for s, d in INSERTS.items()}
    if any(len(set(v.values())) != 1 for v in ins_len.values()):
        return write(OUT / "input_checkpoint.json", {"status": "STOP_INPUT", "reason": "insert lengths unequal", "ins_len": ins_len})
    budget = 1024 - 16 - len(prefix) - len(suffix) - max(max(v.values()) for v in ins_len.values())
    ntok, nmod, _ = nli_setup()
    rows, problems = [], []
    for u in units:
        arms = {c: assemble(tok, prefix, suffix, q_off, u["text"], None if c == "orig" else INSERTS[u["set"]][c], budget) for c in CONDS}
        ins_arms = [arms[c] for c in ("neu", "pos", "neg", "oth")]
        same_len = len({len(a["ids"]) for a in ins_arms}) == 1
        same_q = all(a["qrows"] == ins_arms[0]["qrows"] and [a["ids"][i] for i in a["qrows"]] == [ins_arms[0]["ids"][i] for i in ins_arms[0]["qrows"]] for a in ins_arms)
        same_prefix = all(a["ids"][: len(a["prefix_rows"])] == arms["orig"]["ids"][: len(arms["orig"]["prefix_rows"])] for a in ins_arms)
        if not (same_len and same_q and same_prefix):
            problems.append(u["key"])
        nli_text, nli_info = nli_truncate(ntok, u["text"])
        nli_lens = {c: len(ntok(nli_text + ("" if c == "orig" else INSERTS[u["set"]][c]), HYPOTHESIS)["input_ids"]) for c in CONDS}
        rows.append({"key": u["key"], "corpus": u["corpus"], "pos": u["pos"], "set": u["set"], "pilot": u["pilot"],
                     "text_sha256": u["text_sha256"], "qwen_len": {c: len(arms[c]["ids"]) for c in CONDS},
                     "qwen_unit_tokens": arms["orig"]["unit_tokens"], "qwen_unit_tokens_kept": arms["orig"]["unit_tokens_kept"],
                     "qwen_truncated": arms["orig"]["unit_tokens_kept"] < arms["orig"]["unit_tokens"],
                     "question_rows": len(ins_arms[0]["qrows"]), "prefix_rows": len(ins_arms[0]["prefix_rows"]),
                     "ids_sha256": {c: sha(json.dumps(arms[c]["ids"]).encode()) for c in CONDS},
                     "aligned": same_len and same_q and same_prefix, "nli_lens": nli_lens, **nli_info,
                     "nli_premise_sha256": sha_text(nli_text)})
    cfg = observer.model.config
    status = "PASS" if not problems and observer.calls == calls_before and max(max(r["nli_lens"].values()) for r in rows) <= 512 else "STOP_INPUT"
    out = {"status": status, "protocol_sha256": PROTOCOL_SHA, "units": len(units), "unique_authors": len(authors),
           "repeated_authors": [k for k, v in authors.items() if len(v) > 1], "pilot_units": sum(u["pilot"] for u in units),
           "sets_per_corpus": {c: {s: sum(1 for u in units if u["corpus"] == c and u["set"] == s) for s in (1, 2)} for c in ("pandora", "essays")},
           "insert_token_lengths": ins_len, "insert_sha256": {s: {c: sha_text(t) for c, t in d.items()} for s, d in INSERTS.items()},
           "hypothesis": HYPOTHESIS, "prefix_len": len(prefix), "suffix_len": len(suffix), "question_offset_in_suffix": q_off,
           "unit_token_budget": budget, "qwen_truncated_units": sum(r["qwen_truncated"] for r in rows),
           "nli_truncated_units": sum(r["nli_truncated"] for r in rows), "alignment_problems": problems,
           "model_forwards_in_prepare": observer.calls - calls_before,
           "config": {"layers": cfg.num_hidden_layers, "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
                      "hidden": cfg.hidden_size, "site": f"L{LAYER}.H{SITE}", "control": f"L{LAYER}.H{CONTROL}"},
           "rows": rows}
    write(OUT / "input_checkpoint.json", out)
    return out


class Runner:
    def __init__(self):
        from grain_research.vector_mechanisms.direct_traits import candidates
        from grain_research.vector_mechanisms.native_action import Accounting
        from scripts.run_grain_persona_generation_trace import load_observer
        chk = json.loads((OUT / "input_checkpoint.json").read_text())
        if chk["status"] != "PASS":
            raise RuntimeError("input checkpoint did not pass")
        self.units = load_units()
        self.obs = load_observer(max_tokens=1024, layer_count=2)
        self.tok = self.obs.tokenizer
        self.alpha = candidates("extraversion", self.tok)
        self.attn = self.obs.model.model.layers[LAYER].self_attn
        self.dim = self.obs.model.config.hidden_size // self.obs.model.config.num_attention_heads
        self.prefix, self.suffix, self.q_off = qwen_template(self.obs)
        self.budget = chk["unit_token_budget"]
        self.acc = Accounting(OUT / "accounting.jsonl", CAP)
        self.ntok, self.nmod, self.nmap = nli_setup()
        self.chk = {r["key"]: r for r in chk["rows"]}

    def call(self, item, step, fn):
        self.acc.start(item, step, 1)
        try:
            res = fn()
        except Exception as exc:
            self.acc.fail(item, repr(exc))
            raise
        self.acc.finish(item, **{k: v for k, v in res.items() if not k.startswith("_")})
        return res

    def qwen_score(self, ids, capture=False):
        from scripts.run_grain_persona_reverse_trace import read_score
        box = {}
        handle = None
        if capture:
            handle = self.attn.o_proj.register_forward_pre_hook(lambda _m, inp: box.__setitem__("x", inp[0].detach().clone()))
        try:
            r = read_score(self.obs, ids, self.alpha)
        finally:
            if handle is not None:
                handle.remove()
        if not np.isfinite(r["score"]):
            raise ValueError("non-finite score")
        return {"score": float(r["score"]), "mass": float(r["mass"]), "_x": box.get("x")}

    def qwen_patch(self, ids, donor, rows, head):
        from scripts.run_grain_persona_reverse_trace import head_score
        r = head_score(self.obs, ids, self.alpha, LAYER, donor, rows, head)
        if not np.isfinite(r["score"]):
            raise ValueError("non-finite score")
        return {"score": float(r["score"]), "mass": float(r["mass"])}

    def nli_score(self, premise):
        import torch
        batch = self.ntok(premise, HYPOTHESIS, return_tensors="pt", truncation=False)
        if batch["input_ids"].shape[1] > 512:
            raise ValueError("NLI input exceeds 512")
        with torch.inference_mode():
            p = torch.softmax(self.nmod(**batch).logits, dim=-1)[0].numpy()
        probs = {self.nmap[j]: float(p[j]) for j in self.nmap}
        return {"score": probs["entailment"] - probs["contradiction"], **{"p_" + k: v for k, v in probs.items()},
                "tokens": int(batch["input_ids"].shape[1])}

    def unit(self, u):
        key = u["key"]
        arms = {c: assemble(self.tok, self.prefix, self.suffix, self.q_off, u["text"], None if c == "orig" else INSERTS[u["set"]][c], self.budget) for c in CONDS}
        for c in CONDS:
            if sha(json.dumps(arms[c]["ids"]).encode()) != self.chk[key]["ids_sha256"][c]:
                raise ValueError("ids drift from input checkpoint")
        res = {"key": key, "corpus": u["corpus"], "pos": u["pos"], "set": u["set"], "pilot": u["pilot"]}
        xs = {}
        for c in CONDS:
            r = self.call(f"qwen|{key}|{c}", "qwen_condition", lambda c=c: self.qwen_score(arms[c]["ids"], capture=c in ("neu", "pos", "neg")))
            res[f"qwen_{c}"] = r["score"]
            res[f"mass_{c}"] = r["mass"]
            if r["_x"] is not None:
                xs[c] = r["_x"]
        q, pre = arms["neu"]["qrows"], arms["neu"]["prefix_rows"]
        res["prefix_max_abs_diff"] = float(max((xs[d][0, pre] - xs["neu"][0, pre]).abs().max().item() for d in ("pos", "neg")))
        sl = lambda h: slice(h * self.dim, (h + 1) * self.dim)  # noqa: E731
        tensors = {f"{c}_q_h{h}": xs[c][0, q, sl(h)].float().cpu().numpy() for c in ("neu", "pos", "neg") for h in (SITE, CONTROL)}
        tensors.update({f"{c}_prefix_h{SITE}": xs[c][0, pre, sl(SITE)].float().cpu().numpy() for c in ("neu", "pos", "neg")})
        tdir = OUT / "private_tensors"
        tdir.mkdir(exist_ok=True)
        np.savez(tdir / f"{key.replace(':', '_')}.npz", **tensors)
        res["tensor_sha256"] = {k: sha(v.tobytes()) for k, v in tensors.items()}
        res["question_rows"], res["prefix_rows"] = [q[0], q[-1]], [pre[0], pre[-1]]
        for d in ("pos", "neg"):
            for h, name in ((SITE, "h0"), (CONTROL, "hc")):
                r = self.call(f"patch|{key}|{d}|{name}", "transplant", lambda d=d, h=h: self.qwen_patch(arms["neu"]["ids"], xs[d], q, h))
                res[f"patch_{d}_{name}"] = r["score"]
        r = self.call(f"self|{key}", "self_patch", lambda: self.qwen_patch(arms["neu"]["ids"], xs["neu"], q, SITE))
        res["self_patch"] = r["score"]
        for d in ("pos", "neg"):
            r = self.call(f"placebo|{key}|{d}", "placebo_prefix", lambda d=d: self.qwen_patch(arms["neu"]["ids"], xs[d], pre, SITE))
            res[f"placebo_{d}"] = r["score"]
        nli_text, _info = nli_truncate(self.ntok, u["text"])
        for c in CONDS:
            premise = nli_text + ("" if c == "orig" else INSERTS[u["set"]][c])
            r = self.call(f"nli|{key}|{c}", "nli_condition", lambda p=premise: self.nli_score(p))
            res[f"nli_{c}"] = r["score"]
            res[f"nli_tokens_{c}"] = r["tokens"]
        res["qwen_calls_total"] = self.obs.calls
        return res


def gates(rows):
    self_err = max(abs(r["self_patch"] - r["qwen_neu"]) for r in rows)
    plac_err = max(max(abs(r["placebo_pos"] - r["qwen_neu"]), abs(r["placebo_neg"] - r["qwen_neu"])) for r in rows)
    prefix = max(r["prefix_max_abs_diff"] for r in rows)
    finite = all(np.isfinite([v for k, v in r.items() if k.startswith(("qwen_", "patch_", "nli_", "placebo_", "self_")) and isinstance(v, float)]).all() for r in rows)
    contrasts = [abs(r["qwen_pos"] - r["qwen_neu"]) for r in rows] + [abs(r["qwen_neg"] - r["qwen_neu"]) for r in rows] + [abs(r["qwen_oth"] - r["qwen_pos"]) for r in rows]
    plumbing = {"self_patch_max_abs": self_err, "placebo_max_abs": plac_err, "prefix_activation_max_abs": prefix, "finite": finite,
                "self_ok": self_err <= TAU, "placebo_ok": plac_err <= TAU, "prefix_ok": prefix <= 1e-6}
    ok = plumbing["self_ok"] and plumbing["placebo_ok"] and plumbing["prefix_ok"] and finite
    responsive = max(contrasts) > 10 * TAU
    return ok, responsive, {**plumbing, "max_abs_insert_contrast": max(contrasts)}


def run_units(stage):
    r = Runner()
    rows_path = OUT / "results.jsonl"
    done = {json.loads(x)["key"] for x in rows_path.read_text().splitlines()} if rows_path.exists() else set()
    if stage == "pilot":
        budget = {"pilot_start_epoch": time.time()}
        write(OUT / "budget.json", budget)
        cap, todo = PILOT_CAP_S, [u for u in r.units if u["pilot"]]
    else:
        pc = json.loads((OUT / "pilot_check.json").read_text())
        if pc["decision"] != "CONTINUE_FULL":
            raise RuntimeError("pilot did not authorize the full table")
        budget = json.loads((OUT / "budget.json").read_text())
        cap, todo = TOTAL_CAP_S, [u for u in r.units if not u["pilot"]]
    start = budget["pilot_start_epoch"]
    per_unit = []
    stopped = None
    for u in todo:
        if u["key"] in done:
            continue
        est = max(per_unit) if per_unit else 60.0
        if time.time() - start + 1.2 * est > cap:
            stopped = {"reason": "wall_cap", "next_unit": u["key"]}
            break
        t0 = time.time()
        res = r.unit(u)
        res["seconds"] = time.time() - t0
        per_unit.append(res["seconds"])
        with rows_path.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(stage, u["key"], round(res["seconds"], 1), "calls", r.acc.passes_attempted(), flush=True)
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()]
    info = {"stage": stage, "attempted_calls": r.acc.passes_attempted(), "units_done": len(rows), "stopped": stopped,
            "elapsed_since_pilot_start_s": time.time() - start, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
            "qwen_forwards_observer": r.obs.calls}
    if stage == "pilot":
        prows = [x for x in rows if x["pilot"]]
        ok, responsive, g = gates(prows)
        decision = "STOP_PLUMBING" if not ok else ("STOP_NONRESPONSIVE_PILOT" if not responsive else "CONTINUE_FULL")
        if stopped:
            decision = "STOP_WALL_CAP"
        write(OUT / "pilot_check.json", {**info, "gates": g, "decision": decision, "tau": TAU})
        print(json.dumps({**info, "gates": g, "decision": decision}, indent=1))
    else:
        ok, _resp, g = gates(rows)
        write(OUT / "full_status.json", {**info, "gates_all_units": g, "status": "COMPLETE" if not stopped and len(rows) == 40 else "PARTIAL"})
        print(json.dumps({**info, "gates_all_units": g}, indent=1))


def main() -> int:
    if sha(PROTOCOL.read_bytes()) != PROTOCOL_SHA or sha(AMENDMENT.read_bytes()) != AMENDMENT_SHA:
        raise RuntimeError("protocol sha mismatch")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "pilot", "full"], required=True)
    stage = ap.parse_args().stage
    if stage == "prepare":
        out = prepare()
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    else:
        run_units(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
