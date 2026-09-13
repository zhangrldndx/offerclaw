#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the three pre-registered Final v3 arms and report the full metric panel.

Arms:
  A1  production default (prefix none, judge off)         -- the "before"
  B0  pool28 + answerability d12 + early exit             -- Final v2's B, for attribution
  B3  full quality package: pool28 + HyDE dense channel + HyDE lexical channel
      + judge d12 early-exit + three-vote consensus gate  -- the candidate

Everything that failed once in this campaign is structural here: arms run in
separate processes (import-time env freezing), the median of three repeats is
reported (cold judge calls resample), per-row output stays outside the repo
(blind split), sentinels reject a run whose judge silently did nothing or
degraded, and the negative accounting is three-state -- a refused-then-corrected
false premise is the *correct* behaviour, not a false accept.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JUDGE_ENV = {
    "RAG_ANSWERABILITY": "1",
    "RAG_ANSWERABILITY_EARLY_EXIT": "1",
    "RAG_ANSWERABILITY_DEPTH": "12",
}
ARMS = {
    "A1": ("baseline", {}, False, False),
    "B0": ("compact32_pool28", dict(JUDGE_ENV), True, False),
    "B3": ("compact32_pool28_hydechan_bm25",
           {**JUDGE_ENV, "RAG_ANSWERABILITY_GATE": "1",
            "RAG_ANSWERABILITY_GATE_VOTES": "3"}, True, True),
}
FORCE_OFF = ("RAG_ANSWERABILITY_GATE", "RAG_ANSWERABILITY_GATE_VOTES",
             "RAG_ANSWERABILITY_TIEBREAK", "RAG_ANSWERABILITY_MODE",
             "RAG_HYDE", "RAG_QUERY_REWRITE", "RAG_DOC2QUERY",
             "RAG_EN_QUOTA", "RAG_EN_GATE_MIN", "RAG_RERANK_EN_ONNX_DIR")
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "MODELSCOPE_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def _percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def _check_sentinels(payload: dict, expect_judge: bool, name: str) -> None:
    rows = payload["positive"]["rows"] + payload["negative"]["rows"]
    diags = [(r.get("gate_features") or {}).get("answerability_rerank") or {} for r in rows]
    applied = sum(1 for d in diags if d.get("applied"))
    degraded = sum(1 for d in diags if d.get("graded", 0) < d.get("calls", 0))
    if expect_judge and applied != len(rows):
        raise SystemExit(f"[final-v3] 哨兵失败 {name}: 判据只在 {applied}/{len(rows)} 行生效")
    if not expect_judge and applied:
        raise SystemExit(f"[final-v3] 哨兵失败 {name}: 该臂不该开判据,却有 {applied} 行生效")
    if degraded:
        raise SystemExit(f"[final-v3] 哨兵失败 {name}: {degraded} 行裁判掉线")


def _row(repeat: int, payload: dict, expected: dict) -> dict:
    positive = payload["positive"]
    metrics = positive["metrics"]
    rows = positive["rows"]
    latencies = [r["latency_ms"] for r in rows[1:]]        # row 1 pays model load
    neg_rows = payload["negative"]["rows"]
    accepted = {r["query_id"] for r in neg_rows if r.get("gate_decision")}
    must_abstain = {q for q, kind in expected.items() if kind == "abstain_from_kb"}
    correctable = {q for q, kind in expected.items() if kind == "correct_premise"}
    return {
        "run": repeat,
        "r1": metrics["strict_ranking"]["recall@1"]["hits"],
        "r3": metrics["strict_ranking"]["recall@3"]["hits"],
        "r5": metrics["strict_ranking"]["recall@5"]["hits"],
        "mrr": round(statistics.fmean(r["reciprocal_rank_at_10"] for r in rows), 4),
        "ndcg5": round(statistics.fmean(r["ndcg_at_5"] for r in rows), 4),
        "candidate": metrics["funnel"]["rrf_candidate"]["hits"],
        "gate_pass": metrics["funnel"]["correct_top1_gate_pass"]["hits"],
        "effective": metrics["funnel"]["effective_evidence"]["hits"],
        "p50_ms": round(_percentile(latencies, 0.5)),
        "p95_ms": round(_percentile(latencies, 0.95)),
        "false_accept_abstain": sorted(accepted & must_abstain),
        "correctable_reached": len(accepted & correctable),
        "n_correctable": len(correctable),
        "top1": {r["query_id"]: (r.get("final_chunk_ids") or [""])[0] for r in rows},
    }


def _spread(values):
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values), "runs": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="docs/rag_eval/final_v3/final_v3.json")
    parser.add_argument("--arms", default="A1,B0,B3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outdir", default="~/.offerclaw/private_eval/final_v3/runs")
    parser.add_argument("--summary", default="docs/rag_eval/final_v3/runs/SUMMARY.json")
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    data = json.loads(dataset.read_text(encoding="utf-8"))
    if data.get("status") != "approved":
        raise SystemExit(f"[final-v3] 数据集 status={data.get('status')!r},需人工确认后 approved")
    expected = {i["query_id"]: i.get("expected_behavior")
                for i in data["items"] if i["case_kind"] == "negative"}

    verify = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "freeze_final_v3_config.py"), "--verify"],
        cwd=ROOT, capture_output=True, text=True)
    print(verify.stdout.strip() or verify.stderr.strip())
    if verify.returncode != 0:
        raise SystemExit("[final-v3] 冻结件与工作树不一致,拒绝运行")

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list] = {}
    for arm in args.arms.split(","):
        profile_arm, overrides, expect_judge, _gate = ARMS[arm]
        for repeat in range(1, args.repeats + 1):
            out = outdir / f"{arm}_run{repeat}.json"
            if out.exists():
                payload = json.loads(out.read_text(encoding="utf-8"))["runs"][0]
                try:
                    _check_sentinels(payload, expect_judge, out.name)
                except SystemExit:
                    out.unlink()
                else:
                    print(f"[final-v3] {arm} run {repeat} 已存在,跳过", flush=True)
                    results.setdefault(arm, []).append(_row(repeat, payload, expected))
                    continue
            env = dict(os.environ)
            for key in FORCE_OFF:
                env.pop(key, None)
            for key in JUDGE_ENV:
                env.pop(key, None)
            env.update(OFFLINE)
            env.update(overrides)
            print(f"[final-v3] {arm} run {repeat}/{args.repeats} (arm={profile_arm})", flush=True)
            proc = subprocess.run(
                [sys.executable, str(ROOT / "eval_colloquial_rag.py"),
                 "--cases", str(dataset), "--arm", profile_arm,
                 "--route-mode", "oracle", "--output", str(out)],
                cwd=ROOT, env=env)
            if proc.returncode != 0:
                raise SystemExit(f"[final-v3] {arm} run {repeat} 失败")
            payload = json.loads(out.read_text(encoding="utf-8"))["runs"][0]
            _check_sentinels(payload, expect_judge, out.name)
            results.setdefault(arm, []).append(_row(repeat, payload, expected))

    summary = {}
    for arm, runs in results.items():
        summary[arm] = {key: _spread([float(r[key]) for r in runs])
                        for key in ("r1", "r3", "r5", "mrr", "ndcg5", "candidate",
                                    "gate_pass", "effective", "p50_ms", "p95_ms",
                                    "correctable_reached")}
        summary[arm]["false_accept_abstain"] = {
            "median": statistics.median(len(r["false_accept_abstain"]) for r in runs),
            "runs": [r["false_accept_abstain"] for r in runs],
        }
    report = {"schema_version": "final-v3-report-v1", "repeats": args.repeats,
              "dataset": args.dataset, "summary": summary,
              "runs": {arm: [{k: v for k, v in r.items() if k != "top1"} for r in runs]
                       for arm, runs in results.items()}}
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"\n[final-v3] 汇总 -> {summary_path.relative_to(ROOT)}")
    for arm, stats in summary.items():
        print(f"  {arm}: R@1 {stats['r1']['median']:.0f} R@3 {stats['r3']['median']:.0f} "
              f"R@5 {stats['r5']['median']:.0f} MRR {stats['mrr']['median']:.3f} "
              f"有效 {stats['effective']['median']:.0f} 真误纳 {stats['false_accept_abstain']['median']:.0f}")


if __name__ == "__main__":
    main()
