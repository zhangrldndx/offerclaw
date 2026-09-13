# -*- coding: utf-8 -*-
"""eval_final_negatives.py — Final 拒答集运行器(trivial + adversarial)。

判定走 `_retrieve_and_classify` 的 in_kb(与 gated_query 同一真源,不触发答案合成,
不依赖 LLM 可用性)。期望全部 in_kb=False —— 沾边就答即失败。

每次调用 = 一个独立进程(配合指导 §3.3 的跨进程重复要求)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_path",
                    default=os.path.join("tests", "negative_final_set.json"))
    ap.add_argument("--save", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from rag_gate import _retrieve_and_classify
    items = json.load(open(a.set_path, encoding="utf-8"))["items"]
    rows = []
    for i, it in enumerate(items, 1):
        print(f"\r  拒答评测 {i}/{len(items)} [{it['id']}]   ",
              end="", file=sys.stderr, flush=True)
        try:
            g = _retrieve_and_classify(it["q"], top_k=5)
            accepted = bool(g.get("in_kb"))
        except Exception:
            accepted = False          # 检索异常按拒答处理(fail-soft 与生产一致)
        rows.append({"id": it["id"], "kind": it["kind"], "q": it["q"],
                     "accepted": accepted})
    print("", file=sys.stderr)

    out = {"label": a.label, "set": a.set_path, "rows": rows}
    for kind in ("trivial", "adversarial"):
        sub = [r for r in rows if r["kind"] == kind]
        rej = sum(1 for r in sub if not r["accepted"])
        out[kind] = {"reject": rej, "n": len(sub),
                     "false_accept_ids": [r["id"] for r in sub if r["accepted"]]}
        print(f"{kind:<12} 拒答 {rej}/{len(sub)}"
              + (f"   误纳: {out[kind]['false_accept_ids']}" if rej < len(sub) else ""))
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        try:
            from rag_tools import index_fingerprint
            out["_meta"] = {"index_fingerprint": index_fingerprint()}
        except Exception:
            pass
        json.dump(out, open(a.save, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[已存] {a.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
