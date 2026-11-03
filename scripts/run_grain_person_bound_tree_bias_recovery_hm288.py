#!/usr/bin/env python3
"""Equivalent absent-bias representation on existing native outputs,no recollection."""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_person_bound_tree_hm288 as P
from scripts.score_grain_author_belief_cold import load_bundle

PARENT = P.DEFAULT
DEFAULT = PARENT / "bias_export_recovery_r1"
PLAN = "docs/GRAIN_PERSON_BOUND_TREE_BIAS_EXPORT_RECOVERY_HM288.md"
ARCH = (
    ROOT.parent
    / "project persona/.venv/lib/python3.14/site-packages/transformers/models/qwen2/modeling_qwen2.py"
)


def prepare(out):
    """Certify no model bias exists and correct only its exported vector shape."""
    if out.exists():
        raise FileExistsError("No recovery overwrite or retry")
    old = R.check(PARENT, ("collected_seal.json",))
    head, cfg = load_bundle(PARENT / "head.npz"), R.read(J.MODEL / "config.json")
    if (
        head["weights"].shape != (3, 1536)
        or head["bias"].shape != (2,)
        or np.any(head["bias"] != 0)
    ):
        raise ValueError("Only the known absent-bias export shape may be recovered")
    if cfg["architectures"] != ["Qwen2ForCausalLM"] or not cfg["tie_word_embeddings"]:
        raise ValueError("Declared pinned tied Qwen2 architecture required")
    if (
        "self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)"
        not in ARCH.read_text()
    ):
        raise ValueError("No architectural zero-bias proof available")
    with safe_open(str(J.MODEL / "model.safetensors"), framework="np") as source:
        keys = set(source.keys())
        if any(k.endswith("lm_head.bias") for k in keys):
            raise ValueError("Cannot invent an unexported real model bias")
        tied_present = "model.embed_tokens.weight" in keys
        if not tied_present:
            raise ValueError("Pinned tied output/input weights absent")
    stages = R.read(PARENT / "collected_seal.json")
    paths = [
        PARENT / p
        for p in (
            "manifest.json",
            "prepared_seal.json",
            "collected_seal.json",
            "head.npz",
        )
    ]
    paths += [J.MODEL / "config.json", J.MODEL / "model.safetensors", ARCH]
    own = "scripts/run_grain_person_bound_tree_bias_recovery_hm288.py"
    code = dict(old["code"]) | {p: R.digest(ROOT / p) for p in (own, PLAN)}
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": old["config"],
            "code": code,
            "inputs": dict(old["inputs"]) | {str(p): R.digest(p) for p in paths},
            "alphabet": old["alphabet"],
            "model_weights_sha256": old["model_weights_sha256"],
            "reused_native_capture_no_new_model": True,
            "original_scope_complete": False,
        },
    )
    linked = (
        "units.json",
        "bank.json",
        "cases.json",
        "source.npz",
        "cache_guard.json",
        "collected_runtime.json",
    )
    for name in linked:
        (out / name).symlink_to(PARENT / name)
    (out / "batches").symlink_to(PARENT / "batches", target_is_directory=True)
    R.archive(out / "head.npz", weights=head["weights"], bias=np.zeros(3))
    R.write(
        out / "export_recovery.json",
        {
            "original_head_sha256": R.digest(PARENT / "head.npz"),
            "old_bias_shape": [2],
            "equivalent_bias_shape": [3],
            "architectural_bias_false": True,
            "checkpoint_head_bias_absent": True,
            "tied_embedding_present": tied_present,
            "weights_unchanged": True,
            "new_model_loads_forwards_API_private_training": 0,
            "primary_analyze_failure": "ValueError (50,3)+(2,) absent-bias export broadcast",
            "primary_verify_failure": "FileNotFoundError analyzed_seal.json after failed analyze",
            "source_question_material_parameters_unchanged": True,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "prepared_seal.json", linked + ("export_recovery.json",))
    R.seal(out, "collected_seal.json", tuple(stages))
    print(json.dumps(R.read(out / "export_recovery.json")))


def execute(out):
    """Prepare and replay once with a finite CPU cap;never call native collection."""
    began = time.monotonic()

    def expired(*_):
        """Stop without renewing the recovery audit budget."""
        raise TimeoutError("One180sCPU recovery budget exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    try:
        prepare(out)
        P.analyze(out)
        P.verify(out)
        R.check(
            out, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
        )
        R.write(
            out / "recovery_runtime.json",
            {
                "state": "COMPLETE",
                "CPU_seconds": time.monotonic() - began,
                "new_native_calls": 0,
            },
        )
        print(json.dumps(R.read(out / "recovery_runtime.json")))
    except Exception as error:
        if out.exists() and not (out / "recovery_failure.json").exists():
            R.write(
                out / "recovery_failure.json",
                {
                    "error": repr(error),
                    "seconds": time.monotonic() - began,
                    "no_auto_retry": True,
                },
            )
        raise
    finally:
        signal.alarm(0)


def main():
    """Run the one companion or verify its identities without scientific reruns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("execute", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(PARENT, ("collected_seal.json",))
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM288 original failure/collection and equivalent recovery seals PASS")
    else:
        execute(args.out)


if __name__ == "__main__":
    main()
