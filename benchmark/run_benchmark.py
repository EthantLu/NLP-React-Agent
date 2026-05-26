"""
在 HotpotQA 上跑 ReAct 智能体并计算 EM / F1。
用法:
    python -m benchmark.run_benchmark              # 默认跑 100 条
    python -m benchmark.run_benchmark --n 10       # 跑前 10 条做小测
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.evaluate import em_score, f1_score
from benchmark.hotpot_react import HotpotLLM, HotpotReActAgent
from benchmark.load_data import sample

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--min_interval",
        type=float,
        default=1.2,
        help="LLM 两次请求之间的最小间隔（秒），用于绕过 1 QPS 限速",
    )
    args = ap.parse_args()

    items = sample(args.n)[: args.n]
    print(f"[benchmark] running on {len(items)} examples")

    llm = HotpotLLM(min_interval=args.min_interval)
    agent = HotpotReActAgent(llm, max_steps=args.max_steps, verbose=args.verbose)
    print(f"[benchmark] LLM min_interval={llm.min_interval}s, max_retries={llm.max_retries}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        RESULTS_DIR, f"react_hotpot_n{args.n}_{int(time.time())}.json"
    )

    records = []
    em_sum = 0.0
    f1_sum = 0.0
    t0 = time.time()
    for i, ex in enumerate(items):
        q = ex["question"]
        gold = ex["answer"]
        print(f"\n===== [{i + 1}/{len(items)}] {q[:120]}")
        try:
            res = agent.run(q)
        except Exception as e:
            print(f"[error] {e}")
            res = {"answer": None, "steps": 0, "trace": [], "error": str(e)}
        pred = res.get("answer") or ""
        em = em_score(pred, gold)
        f1 = f1_score(pred, gold)
        em_sum += em
        f1_sum += f1
        rec = {
            "_id": ex["_id"],
            "question": q,
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
            "steps": res.get("steps"),
            "trace": res.get("trace"),
            "type": ex.get("type"),
            "level": ex.get("level"),
        }
        records.append(rec)
        print(f"  gold={gold!r} | pred={pred!r} | EM={em} F1={f1:.3f}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_done": len(records),
                    "em": em_sum / len(records),
                    "f1": f1_sum / len(records),
                    "records": records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    dt = time.time() - t0
    n = len(records)
    print("\n========= SUMMARY =========")
    print(f"N        : {n}")
    print(f"EM       : {em_sum / n:.4f}")
    print(f"F1       : {f1_sum / n:.4f}")
    print(f"elapsed  : {dt:.1f}s ({dt / n:.1f}s/example)")
    print(f"results  : {out_path}")


if __name__ == "__main__":
    main()
