"""
从 ModelScope (cmriat/hotpotqa) 下载 HotpotQA validation 集，并随机采样 100 条。
"""

import json
import os
import random
import sys
from typing import List, Dict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MS_CACHE = os.path.join(DATA_DIR, "ms_cache")
SAMPLE_PATH = os.path.join(DATA_DIR, "hotpot_sample_100.json")


def _ensure_parquet() -> str:
    """从 ModelScope 下载（带缓存）并返回 validation parquet 路径。"""
    expected = os.path.join(
        MS_CACHE, "cmriat", "hotpotqa", "data", "validation-00000-of-00001.parquet"
    )
    if os.path.exists(expected):
        return expected

    from modelscope import snapshot_download

    print("[load_data] downloading cmriat/hotpotqa from ModelScope ...")
    snapshot_download(
        "cmriat/hotpotqa", repo_type="dataset", cache_dir=MS_CACHE
    )
    return expected


def sample(n: int = 100, seed: int = 42) -> List[Dict]:
    if os.path.exists(SAMPLE_PATH):
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    import pandas as pd

    pq = _ensure_parquet()
    df = pd.read_parquet(pq)
    print(f"[load_data] loaded {len(df)} validation examples")

    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), n)

    sub = []
    for i in idx:
        row = df.iloc[i]
        golds = list(row["golden_answers"])
        meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
        sub.append(
            {
                "_id": row["id"],
                "question": row["question"],
                "answer": golds[0] if golds else "",
                "golden_answers": golds,
                "type": meta.get("type"),
                "level": meta.get("level"),
            }
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False, indent=2)
    print(f"[load_data] sampled {n} -> {SAMPLE_PATH}")
    return sub


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    items = sample(n)
    print(f"got {len(items)} items; first:")
    print(json.dumps(items[0], indent=2, ensure_ascii=False))
