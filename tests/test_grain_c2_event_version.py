import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts/run_grain_c2_event_version_sensitivity.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def toy():
    base = {c: 0.0 for c in ev.dd.CONSUMED}
    rows = []
    for uid, cid, body, sub in [("u1", "a", "hello there", "news"), ("u1", "a", "hello there!", "news"), ("u1", "b", "x", "mbti"),
                                ("u2", "c", "y", "intp"), ("u2", "c", "y", "intp")]:
        r = dict(base, user_id=uid, comment_id=cid, body_raw=body, subreddit=sub)
        rows.append(r)
    return pd.DataFrame(rows)


def test_first_last_select_one_row_per_key_in_file_order():
    df = toy()
    f, l = ev.select_view(df, "first"), ev.select_view(df, "last")
    assert len(f) == len(l) == 3
    assert f.loc[(f.user_id == "u1") & (f.comment_id == "a"), "body_raw"].item() == "hello there"
    assert l.loc[(l.user_id == "u1") & (l.comment_id == "a"), "body_raw"].item() == "hello there!"


def test_content_hash_separates_views_with_different_content():
    df = toy()
    f, l = ev.select_view(df, "first"), ev.select_view(df, "last")
    kf, kl = ev.c2.row_keys(f), ev.c2.row_keys(l)
    assert (kf.values == kl.values).all()  # same event keys
    hf = ev.content_hash(f, kf, np.ones(len(f), bool))
    hl = ev.content_hash(l, kl, np.ones(len(l), bool))
    assert hf != hl
    same = ev.content_hash(f, kf, np.ones(len(f), bool))
    assert hf == same


def test_identical_versions_hash_equal():
    df = toy()
    sub = df[df.user_id == "u2"]
    f, l = ev.select_view(sub, "first"), ev.select_view(sub, "last")
    assert ev.content_hash(f, ev.c2.row_keys(f), np.ones(1, bool)) == ev.content_hash(l, ev.c2.row_keys(l), np.ones(1, bool))
