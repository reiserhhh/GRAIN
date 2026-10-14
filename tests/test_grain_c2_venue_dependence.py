import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c2", ROOT / "scripts/run_grain_c2_mcd_venue_dependence.py")
c2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2)


def toy():
    rows = []
    for u, subs in {"u1": ["mbti", "intp", "soccer", "soccer", "AskReddit"], "u2": ["INFJ", "introvert"],
                    "u3": ["jung", "typescript", "frankocean"], "u4": ["news"]}.items():
        for i, s in enumerate(subs):
            rows.append({"user_id": u, "comment_id": f"{u}c{i}", "subreddit": s})
    rows.append(dict(rows[2]))  # exact duplicate row, as in the real input
    return pd.DataFrame(rows)


def test_trait_named_rule():
    assert c2.is_trait_named("INTP") and c2.is_trait_named("entjs") and c2.is_trait_named("mbtiplus")
    assert c2.is_trait_named("objectivepersonality") and c2.is_trait_named("jungiantypology")
    assert not any(c2.is_trait_named(s) for s in ["jung", "typescript", "frankocean", "soccer", "intpx"])


def test_arm_masks_count_matched_and_deterministic():
    df = toy()
    masks, table = c2.build_arm_masks(df)
    masks2, _ = c2.build_arm_masks(df.copy())
    assert all(np.array_equal(masks[a], masks2[a]) for a in "ABC")
    assert masks["A"].all()
    assert (masks["B"] == ~df["subreddit"].map(c2.is_trait_named).to_numpy()).all()
    kept_b = df.loc[masks["B"]].groupby("user_id").size().reindex(table["author"]).fillna(0).to_numpy()
    kept_c = df.loc[masks["C"]].groupby("user_id").size().reindex(table["author"]).fillna(0).to_numpy()
    assert np.array_equal(kept_b, kept_c)
    t = table.set_index("author")
    assert bool(t.loc["u2", "mcd_unobserved_BC"]) and not bool(t.loc["u1", "mcd_unobserved_BC"])
    assert t.loc["u3", "n_removed_C"] == 0 and t.loc["u4", "n_removed_C"] == 0


def test_row_keys_unique_with_duplicates():
    keys = c2.row_keys(toy())
    assert not keys.duplicated().any()


def test_paired_seed_and_cache_key():
    s = c2.paired_seed("np_mcd", "openness", 2, 2634, 0.5)
    assert s == c2.paired_seed("np_mcd", "openness", 2, 2634, 0.5)
    assert s != c2.paired_seed("np_mcd", "openness", 2, 2634, 1.0)
    settings = {"k": 48}
    base = c2.mcd_cache_key("A", "h", ["a"], ["b"], ["c"], settings)
    assert base != c2.mcd_cache_key("B", "h", ["a"], ["b"], ["c"], settings)
    assert base != c2.mcd_cache_key("A", "h2", ["a"], ["b"], ["c"], settings)
    assert base != c2.mcd_cache_key("A", "h", ["a", "x"], ["b"], ["c"], settings)
