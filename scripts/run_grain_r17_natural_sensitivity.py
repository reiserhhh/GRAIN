#!/usr/bin/env python3
"""R17 packet 1: bounded natural-material geometric sensitivity (geometric only).

Stages: prepare (manifest, validation, no reader scores), encode (capped item
forwards), analyze (no forwards). See docs/GRAIN_R17_NATURAL_SENSITIVITY_PROTOCOL_V1.md.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / "docs/GRAIN_R17_NATURAL_SENSITIVITY_PROTOCOL_V1.md"
PROTOCOL_SHA = "cee18f6e32a45ac0de1548c0ea34c78d5a8830c77dffd3f76e0fae5905e5a5b1"
PROTOCOL_SHA_A1 = "6968d5a9926d83c11a81f92cb17400a65c06fbfeb91439671acb39a4e58d6e42"
LB = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2"
OUT = ROOT / "results/grain_gptpro_foundation_v3/R17/natural_sensitivity_r1"
CAP_TOTAL = 1000
SEED = 20260926
BOOT_SEED = 20260915
PER_CORPUS = 50
TOKEN_LIMIT = 510
MIN_WS = 12
CORPUS_CODE = {"pandora": 1, "essays": 2}
EDITS = ("case_fold", "contraction_expand", "sentence_reverse")
EDITS_A1 = (*EDITS, "punctuation_strip")
READERS = {
    "e5": {
        "model": "intfloat/e5-large-v2",
        "revision": "f169b11e22de13617baa190a028a32f3493550b6",
        "prefix": "query: ",
        "pooling": "mean",
        "cap": 512,
        "versions": ("base", *EDITS_A1),
    },
    "bge": {
        "model": "BAAI/bge-large-en-v1.5",
        "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "prefix": "",
        "pooling": "cls",
        "cap": 512,
        "versions": ("base", *EDITS_A1, "partner"),
    },
}
PLANNED = {"e5": 500, "bge": 600}  # A1 ceilings; total cap binds
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "can't": "cannot", "couldn't": "could not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "you're": "you are",
    "we're": "we are", "they're": "they are", "you've": "you have",
    "we've": "we have", "they've": "they have", "you'll": "you will",
    "we'll": "we will", "they'll": "they will",
}
_ALT = "|".join(
    re.escape(k).replace("'", "['’]")
    for k in sorted(CONTRACTIONS, key=len, reverse=True)
)
CONTRACTION_RE = re.compile(rf"(?<![\w'’])({_ALT})(?![\w'’])", re.IGNORECASE)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def _case_like(src, rep):
    if src.isupper() and len(src) > 1:
        return rep.upper()
    if src[:1].isupper():
        return rep[:1].upper() + rep[1:]
    return rep


def edit_text(text, edit):
    """Return (edited, applicable, retained, detail) for one declared edit."""
    if edit == "case_fold":
        new = text.lower()
        applicable = new != text
        retained = collections.Counter(text.lower().split()) == collections.Counter(
            new.split()
        )
        return new, applicable, retained, {"changed_chars": sum(
            a != b for a, b in zip(text, new, strict=True))}
    if edit == "contraction_expand":
        count = 0

        def repl(m):
            nonlocal count
            count += 1
            key = m.group(0).lower().replace("’", "'")
            return _case_like(m.group(0), CONTRACTIONS[key])

        new = CONTRACTION_RE.sub(repl, text)
        outside_old = CONTRACTION_RE.split(text)[0::2]
        outside_new_ok = CONTRACTION_RE.sub("\x00", text).split("\x00") == outside_old
        return new, count > 0, bool(outside_new_ok), {"substitutions": count}
    if edit == "sentence_reverse":
        parts = [p for p in SENT_SPLIT.split(text.strip()) if p]
        good = [p for p in parts if len(re.findall(r"\w+", p)) >= 2]
        applicable = len(parts) >= 2 and len(good) == len(parts)
        new = " ".join(reversed(parts)) if applicable else text
        retained = collections.Counter(text.split()) == collections.Counter(new.split())
        return new, applicable and new != text, retained, {"segments": len(parts)}
    if edit == "punctuation_strip":
        new = re.sub(r"[ \t]+", " ", re.sub(r"[^\w\s'\u2019]", " ", text))
        retained = re.findall(r"\w+", text) == re.findall(r"\w+", new)
        return new, new != text, retained, {"removed_chars": len(text) - len(new)}
    raise ValueError(edit)


def load_material():
    """Verified unit texts and addresses from the local_bilinear_r2 material."""
    from grain_research.local_bilinear import chunk_spans
    from scripts.run_grain_local_construction import load_texts

    datasets = load_texts()
    out = {}
    for corpus in ("pandora", "essays"):
        info = json.loads((LB / corpus / "material.json").read_text())
        for name, h in info["files"].items():
            if digest(LB / corpus / name) != h:
                raise ValueError(f"Changed cached material {corpus}/{name}")
        addresses = pd.read_csv(LB / corpus / "private_addresses.csv")
        d = datasets[corpus]
        texts = []
        for u, document in enumerate(d["documents"]):
            if corpus == "pandora":
                texts.extend(d["units"][u])
            else:
                texts.extend(document[a:b] for a, b in chunk_spans(document))
        if len(texts) != len(addresses):
            raise ValueError("Unit/address mismatch")
        for i in range(0, len(texts), 997):
            row = addresses.iloc[i]
            doc = d["documents"][int(row.author_index)]
            if doc[int(row.start) : int(row.end)] != texts[i]:
                raise ValueError("Address replay mismatch")
        with np.load(LB / corpus / "private_blocks.npz", allow_pickle=False) as z:
            held = z["held"].copy()
        out[corpus] = {"texts": texts, "addresses": addresses, "held": held}
    return out


def tokenizers():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    toks = {}
    for name, spec in READERS.items():
        path = snapshot_download(
            spec["model"], revision=spec["revision"], local_files_only=True,
            allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"],
        )
        toks[name] = (path, AutoTokenizer.from_pretrained(path))
    return toks


def ntok(toks, text):
    return {
        name: len(tok(READERS[name]["prefix"] + text, truncation=False)["input_ids"])
        for name, (_, tok) in toks.items()
    }


def prepare(args):
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed after freeze")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise SystemExit("Manifest already frozen; refusing to rebuild")
    material = load_material()
    toks = tokenizers()
    rows, scans, refusals = [], [], []
    for corpus, m in material.items():
        addr, texts = m["addresses"], m["texts"]
        by_author = {a: g.index.to_numpy() for a, g in addr.groupby("author_index")}
        rng = np.random.default_rng(np.random.SeedSequence([SEED, CORPUS_CODE[corpus]]))
        order = rng.permutation(np.sort(m["held"]))
        chosen = 0
        for rank, author in enumerate(order):
            if chosen >= PER_CORPUS:
                break
            idx = by_author[int(author)]
            picked, reasons = None, collections.Counter()
            for pos, row in enumerate(idx):
                text = texts[row]
                if len(text.split()) < MIN_WS:
                    reasons["base_under_12_ws_tokens"] += 1
                    continue
                versions, bad = {"base": text}, None
                details = {}
                for e in EDITS:
                    new, app, ret, det = edit_text(text, e)
                    details[e] = det
                    if not app:
                        bad = f"{e}_not_applicable"
                        break
                    if not ret:
                        bad = f"{e}_retention_failed"
                        break
                    versions[e] = new
                if bad:
                    reasons[bad] += 1
                    continue
                counts = {v: ntok(toks, t) for v, t in versions.items()}
                if any(n > TOKEN_LIMIT for c in counts.values() for n in c.values()):
                    reasons["version_over_510_tokens"] += 1
                    continue
                partner = None
                for step in range(1, len(idx)):
                    prow = idx[(pos + step) % len(idx)]
                    ptext = texts[prow]
                    if len(ptext.split()) < MIN_WS or sha(ptext) == sha(text):
                        continue
                    pc = ntok(toks, ptext)
                    if max(pc.values()) <= TOKEN_LIMIT:
                        partner = (prow, ptext, pc)
                        break
                if partner is None:
                    reasons["no_valid_partner"] += 1
                    continue
                picked = (row, versions, counts, partner, details)
                break
            scans.append({
                "corpus": corpus, "scan_rank": rank, "author_index": int(author),
                "units_available": len(idx), "selected": picked is not None,
                **{f"rejected_{k}": v for k, v in reasons.items()},
            })
            if picked is None:
                refusals.append({"corpus": corpus, "author_index": int(author),
                                 "reason": "no_eligible_unit", **dict(reasons)})
                continue
            row, versions, counts, (prow, ptext, pc), details = picked
            rec = {
                "corpus": corpus, "author_index": int(author),
                "unit_row": int(row), "unit": int(addr.loc[row, "unit"]),
                "partner_row": int(prow), "partner_unit": int(addr.loc[prow, "unit"]),
                "author_units": len(idx),
                "partner_sha256": sha(ptext),
                "partner_ws": len(ptext.split()),
                **{f"partner_tok_{k}": v for k, v in pc.items()},
            }
            for v, t in versions.items():
                rec[f"{v}_sha256"] = sha(t)
                rec[f"{v}_ws"] = len(t.split())
                rec[f"{v}_chars"] = len(t)
                for k, n in counts[v].items():
                    rec[f"{v}_tok_{k}"] = n
            rec["contraction_substitutions"] = details["contraction_expand"]["substitutions"]
            rec["sentence_segments"] = details["sentence_reverse"]["segments"]
            rec["case_changed_chars"] = details["case_fold"]["changed_chars"]
            rows.append(rec)
            chosen += 1
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(scans).fillna(0).to_csv(out / "author_scan.csv", index=False)
    pd.DataFrame(refusals).fillna(0).to_csv(out / "refusals.csv", index=False)
    items = [
        {"reader": r, "corpus": x.corpus, "author_index": int(x.author_index),
         "version": v}
        for r, spec in READERS.items() for x in manifest.itertuples()
        for v in spec["versions"]
    ]
    planned = collections.Counter(i["reader"] for i in items)
    if sum(planned.values()) > CAP_TOTAL:
        raise ValueError("Planned items exceed cap")
    write(out / "manifest.json", {
        "protocol_sha256": PROTOCOL_SHA,
        "code_sha256": digest(Path(__file__)),
        "manifest_sha256": digest(out / "manifest.csv"),
        "selected_by_corpus": manifest.corpus.value_counts().to_dict(),
        "authors_scanned": collections.Counter(s["corpus"] for s in scans),
        "planned_item_forwards": dict(planned),
        "planned_total": sum(planned.values()),
        "cap_total": CAP_TOTAL,
        "outcome_scores_examined": False,
        "reader_scores_examined": False,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    })
    print(json.dumps(json.loads((out / "manifest.json").read_text())), flush=True)


def amend(args):
    """A1: validate punctuation_strip and case_fold input-ID identity; no reader scores."""
    if digest(PROTOCOL) != PROTOCOL_SHA_A1:
        raise ValueError("Protocol changed after A1")
    out = args.output
    if (out / "manifest_a1.json").exists():
        raise SystemExit("A1 manifest already frozen")
    meta = json.loads((out / "manifest.json").read_text())
    if digest(out / "manifest.csv") != meta["manifest_sha256"]:
        raise ValueError("Manifest changed")
    manifest = pd.read_csv(out / "manifest.csv")
    material = load_material()
    toks = tokenizers()
    rows = []
    for rec in manifest.to_dict("records"):
        text = material[rec["corpus"]]["texts"][int(rec["unit_row"])]
        if sha(text) != rec["base_sha256"]:
            raise ValueError("Base hash drift")
        new, app, ret, det = edit_text(text, "punctuation_strip")
        counts = ntok(toks, new)
        row = {"corpus": rec["corpus"], "author_index": rec["author_index"],
               "punctuation_strip_sha256": sha(new),
               "punctuation_strip_applicable": bool(app),
               "punctuation_strip_retained": bool(ret),
               "punctuation_strip_removed_chars": det["removed_chars"],
               "punctuation_strip_ws": len(new.split()),
               **{f"punctuation_strip_tok_{k}": v for k, v in counts.items()},
               "punctuation_strip_valid": bool(app and ret
                                               and max(counts.values()) <= TOKEN_LIMIT)}
        low = edit_text(text, "case_fold")[0]
        for name, (_, tok) in toks.items():
            pre = READERS[name]["prefix"]
            row[f"case_fold_ids_identical_{name}"] = (
                tok(pre + text)["input_ids"] == tok(pre + low)["input_ids"])
        rows.append(row)
    a1 = pd.DataFrame(rows)
    a1.to_csv(out / "manifest_a1.csv", index=False)
    write(out / "manifest_a1.json", {
        "protocol_sha256": PROTOCOL_SHA_A1, "code_sha256": digest(Path(__file__)),
        "manifest_a1_sha256": digest(out / "manifest_a1.csv"),
        "punctuation_strip_valid": int(a1.punctuation_strip_valid.sum()),
        "case_fold_ids_identical": {n: int(a1[f"case_fold_ids_identical_{n}"].sum())
                                    for n in READERS},
        "units": len(a1), "reader_scores_examined": False,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    })
    print((out / "manifest_a1.json").read_text())


def load_manifest(out):
    meta = json.loads((out / "manifest.json").read_text())
    if digest(out / "manifest.csv") != meta["manifest_sha256"]:
        raise ValueError("Manifest changed")
    a1meta = json.loads((out / "manifest_a1.json").read_text())
    if digest(out / "manifest_a1.csv") != a1meta["manifest_a1_sha256"]:
        raise ValueError("A1 manifest changed")
    m = pd.read_csv(out / "manifest.csv").merge(
        pd.read_csv(out / "manifest_a1.csv"), on=["corpus", "author_index"],
        validate="one_to_one")
    if len(m) != 100:
        raise ValueError("Unexpected manifest size")
    return m


def _versions_for(material, rec):
    text = material[rec["corpus"]]["texts"][int(rec["unit_row"])]
    versions = {"base": text}
    for e in EDITS_A1:
        versions[e] = edit_text(text, e)[0]
    versions["partner"] = material[rec["corpus"]]["texts"][int(rec["partner_row"])]
    for v, t in versions.items():
        key = "partner_sha256" if v == "partner" else f"{v}_sha256"
        if sha(t) != rec[key]:
            raise ValueError(f"Text hash drift {rec['corpus']} {rec['author_index']} {v}")
    return versions


def _ledger(out):
    path = out / "accounting" / "attempts.jsonl"
    events = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    started = [e for e in events if e["status"] == "STARTED"]
    ok = {e["item"] for e in events if e["status"] == "OK"}
    failed = [e for e in events if e["status"] == "FAILED"]
    return path, started, ok, failed


def encode(args):
    import torch
    import transformers
    from transformers import AutoModel

    out = args.output
    if digest(PROTOCOL) != PROTOCOL_SHA_A1:
        raise ValueError("Protocol changed after A1")
    manifest = load_manifest(out)
    material = load_material()
    (out / "accounting").mkdir(exist_ok=True)
    path, started, ok, failed = _ledger(out)
    unfinished = {e["item"] for e in started} - ok
    if failed or unfinished:
        raise SystemExit(f"Prior failed/interrupted items {sorted(unfinished)[:5]}; no automatic retry")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    toks = tokenizers()
    new_calls = 0
    for reader, spec in READERS.items():
        model = None
        vdir = out / "private_vectors" / reader
        vdir.mkdir(parents=True, exist_ok=True)
        for rec in manifest.to_dict("records"):
            versions = _versions_for(material, rec)
            for v in spec["versions"]:
                item = f"{reader}:{rec['corpus']}:{rec['author_index']}:{v}"
                if item in ok:
                    continue
                if v == "punctuation_strip" and not rec["punctuation_strip_valid"]:
                    continue
                target = vdir / f"{rec['corpus']}_{rec['author_index']}_{v}.npy"
                if v == "case_fold":
                    pre = spec["prefix"]
                    tok0 = toks[reader][1]
                    same = (tok0(pre + versions["base"])["input_ids"]
                            == tok0(pre + versions[v])["input_ids"])
                    if same != bool(rec[f"case_fold_ids_identical_{reader}"]):
                        raise ValueError("A1 identity flag drift")
                    if same:
                        base = vdir / f"{rec['corpus']}_{rec['author_index']}_base.npy"
                        np.save(target, np.load(base))
                        with path.open("a") as fh:
                            fh.write(json.dumps({"item": item, "reader": reader,
                                                 "status": "OK",
                                                 "mode": "TOKENIZER_IDENTICAL",
                                                 "forward": False,
                                                 "sha256": digest(target),
                                                 "t": time.time()}) + "\n")
                        ok.add(item)
                        continue
                if args.limit is not None and new_calls >= args.limit:
                    print(json.dumps({"stage": "encode", "status": "LIMIT", "new_calls": new_calls}))
                    return
                _, started, ok, failed = _ledger(out)
                total = len(started)
                by_reader = sum(e["reader"] == reader for e in started)
                if total >= CAP_TOTAL or by_reader >= PLANNED[reader]:
                    write(out / "accounting" / "status.json", {"status": "CAP_REACHED",
                          "total": total, "by_reader": by_reader})
                    raise SystemExit("Cap reached before call")
                if model is None:
                    model = AutoModel.from_pretrained(
                        toks[reader][0], use_safetensors=True, dtype=torch.float32
                    ).eval().to(device)
                tok = toks[reader][1]
                with path.open("a") as fh:
                    fh.write(json.dumps({"item": item, "reader": reader,
                                         "status": "STARTED", "attempt": total + 1,
                                         "t": time.time()}) + "\n")
                try:
                    batch = tok(spec["prefix"] + versions[v], truncation=True,
                                max_length=spec["cap"], return_tensors="pt").to(device)
                    if batch["input_ids"].shape[1] > TOKEN_LIMIT + 2:
                        raise ValueError("Unexpected length")
                    with torch.inference_mode():
                        h = model(**batch).last_hidden_state
                        if spec["pooling"] == "mean":
                            mask = batch["attention_mask"][:, :, None]
                            vec = (h * mask).sum(1) / mask.sum(1)
                        else:
                            vec = h[:, 0]
                        vec = torch.nn.functional.normalize(vec, dim=1)
                    arr = vec.cpu().numpy()[0].astype(np.float32)
                    if not np.isfinite(arr).all():
                        raise ValueError("Non-finite vector")
                    np.save(target, arr)
                    with path.open("a") as fh:
                        fh.write(json.dumps({"item": item, "reader": reader,
                                             "status": "OK", "sha256": digest(target),
                                             "tokens": int(batch["input_ids"].shape[1]),
                                             "t": time.time()}) + "\n")
                    new_calls += 1
                except Exception as exc:  # recorded, no retry
                    with path.open("a") as fh:
                        fh.write(json.dumps({"item": item, "reader": reader,
                                             "status": "FAILED", "error": repr(exc),
                                             "t": time.time()}) + "\n")
                    raise
    _, started, ok, failed = _ledger(out)
    write(out / "accounting" / "status.json", {
        "status": "COMPLETE", "attempts_total": len(started),
        "attempts_by_reader": dict(collections.Counter(e["reader"] for e in started)),
        "ok": len(ok), "failed": len(failed), "cap_total": CAP_TOTAL,
        "device": device, "torch": torch.__version__,
        "transformers": transformers.__version__, "batch_size": 1, "dtype": "float32",
    })
    print(json.dumps(json.loads((out / "accounting" / "status.json").read_text())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "amend", "encode", "analyze"])
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "amend":
        amend(args)
    elif args.stage == "encode":
        encode(args)
    else:
        from scripts.analyze_grain_r17_natural_sensitivity import analyze

        analyze(args)


if __name__ == "__main__":
    main()
