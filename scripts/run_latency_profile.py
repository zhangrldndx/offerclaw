#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast/Quality 延迟分档固化(COMPLETE_TEST_PLAN §13A-5 / §10 简版)。

固定 15 题:tests/rag_bench_set.json 前 8(书面口径)+
tests/rag_bench_paraphrase_set.json 前 7(口语口径)。取题规则是**位置确定性**
的,没有挑题空间;题面在仓库内基准集里,本脚本只落聚合延迟。

每题走生产 reference 检索+生成链(与 /api/query 的 kb 路径同代码,不经语义
规划器),分段计时:retrieval_ms(检索+判据+门)/ synthesis_ms(生成)/
total_ms。首题含模型冷加载,单独记为 cold;p50/p95 用其余 14 题(warm)。

模式经产品开关 ``RAG_MODE`` 生效(rag_mode.py),**不在脚本里拼旋钮**——
计划明令禁止用测试脚本私自拼环境变量冒充产品模式。每个模式一个子进程,
避免进程内缓存跨模式串味。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fixed_questions() -> list[dict]:
    bench = json.loads((ROOT / "tests/rag_bench_set.json").read_text(encoding="utf-8"))
    para = json.loads((ROOT / "tests/rag_bench_paraphrase_set.json").read_text(encoding="utf-8"))
    rows = [{"id": i["id"], "style": "standard", "q": i["q"]} for i in bench["items"][:8]]
    rows += [{"id": i["id"], "style": "oral", "q": i["q"]} for i in para["items"][:7]]
    return rows


def _measure_one(question: str) -> dict:
    from rag_gate import _chat, _grounded_messages, _fallback_messages, _retrieve_and_classify
    from rag_multi_source import REFERENCE_EXCLUDES

    t0 = time.perf_counter()
    g = _retrieve_and_classify(
        question, 5, exclude_source_types=REFERENCE_EXCLUDES,
        metadata_filters={"owner_scope": {"curated"}}, allow_paper_route=False)
    t1 = time.perf_counter()
    if g.get("in_kb"):
        _chat(_grounded_messages(question, g["chunks"],
                                 g.get("answer_action", "answer")))
    else:
        _chat(_fallback_messages(question, g.get("chunks") or []))
    t2 = time.perf_counter()
    return {"in_kb": bool(g.get("in_kb")),
            "retrieval_ms": round((t1 - t0) * 1000, 1),
            "synthesis_ms": round((t2 - t1) * 1000, 1),
            "total_ms": round((t2 - t0) * 1000, 1)}


def worker(mode: str, out_path: Path) -> None:
    assert os.environ.get("RAG_MODE") == mode
    os.environ.setdefault("LLM_USAGE_LOG", "0")
    rows = []
    for item in fixed_questions():
        r = _measure_one(item["q"])
        rows.append({"id": item["id"], "style": item["style"], **r})
        print(f"[{mode}] {item['id']}: total {r['total_ms']:.0f}ms "
              f"(retr {r['retrieval_ms']:.0f} + synth {r['synthesis_ms']:.0f})",
              flush=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "p50_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 1),
        "max_ms": round(ordered[-1], 1),
    }


def run(raw_dir: Path, summary_path: Path | None, modes: list[str]) -> dict:
    raw_dir = raw_dir.expanduser().resolve()
    if raw_dir == ROOT or ROOT in raw_dir.parents:
        raise SystemExit("raw latency output must stay outside the repository")
    raw_dir.mkdir(parents=True, exist_ok=True)
    per_mode: dict[str, dict] = {}
    for mode in modes:
        out = raw_dir / f"latency_{mode}.json"
        env = {**os.environ, "RAG_MODE": mode, "LLM_USAGE_LOG": "0"}
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", mode, "--worker-output", str(out)],
            cwd=ROOT, env=env)
        if proc.returncode != 0 or not out.is_file():
            raise SystemExit(f"latency worker failed for mode {mode}")
        rows = json.loads(out.read_text(encoding="utf-8"))
        cold, warm = rows[0], rows[1:]
        per_mode[mode] = {
            "cases": len(rows),
            "in_kb_count": sum(r["in_kb"] for r in rows),
            "cold_first_case_total_ms": cold["total_ms"],
            "warm_total": _stats([r["total_ms"] for r in warm]),
            "warm_retrieval": _stats([r["retrieval_ms"] for r in warm]),
            "warm_synthesis": _stats([r["synthesis_ms"] for r in warm]),
        }
    questions = fixed_questions()
    summary = {
        "schema_version": "offerclaw-latency-profile-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "bench_set 前 8(standard)+ paraphrase_set 前 7(oral),位置确定性",
        "question_ids": [q["id"] for q in questions],
        "questions_sha256": hashlib.sha256(json.dumps(
            [q["q"] for q in questions], ensure_ascii=False).encode()).hexdigest(),
        "modes": per_mode,
        "note": "同机顺序实测;quality 首题含判据/嵌入模型冷加载。原始逐题数据在仓库外。",
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(json.dumps({m: v["warm_total"] for m, v in per_mode.items()},
                     ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--modes", default="fast,quality")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, args.worker_output)
    else:
        if args.raw_dir is None:
            raise SystemExit("--raw-dir is required")
        run(args.raw_dir, args.summary, [m.strip() for m in args.modes.split(",") if m.strip()])
