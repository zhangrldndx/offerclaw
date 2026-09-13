#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the four pre-registered Final v2 arms, three times each.

Three properties matter more than convenience here.

*Arms run in separate processes.*  Several knobs on this path are read at import
time, so setting an env var in-process leaks into whatever ran before it -- this
round lost two measurement runs to exactly that.  A subprocess per (arm, repeat)
makes the arm's configuration a property of the process, not of call order.

*The median is reported, not the best run.*  Cold judge calls are not
reproducible: the same configuration flipped two negative decisions between cold
runs earlier in this round.  Picking the best of three would turn that noise into
a result.

*It refuses to run against an unconfirmed set.*  Final v2 is one-shot; scoring it
before a human has confirmed the qrels would spend the blind set on measuring the
authoring instead of the retrieval.
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

# (arm profile, env overrides, whether the answerability judge must be active)
ARMS = {
    "A1": ("baseline", {}, False),                      # production Fast default: prefix none
    "A2": ("compact32", {}, False),                     # the arm this round used as its baseline
    "B": ("compact32_pool28", {
        "RAG_ANSWERABILITY": "1",
        "RAG_ANSWERABILITY_EARLY_EXIT": "1",
        "RAG_ANSWERABILITY_DEPTH": "12",
    }, True),
    "C": ("compact32_pool28_hyde", {
        "RAG_ANSWERABILITY": "1",
        "RAG_ANSWERABILITY_EARLY_EXIT": "1",
        "RAG_ANSWERABILITY_DEPTH": "12",
        "RAG_HYDE": "1",
    }, True),
}
# Knobs that must be off in every arm, so a stale shell cannot change one.
FORCE_OFF = ("RAG_ANSWERABILITY_GATE", "RAG_ANSWERABILITY_TIEBREAK",
             "RAG_ANSWERABILITY_MODE", "RAG_QUERY_REWRITE", "RAG_DOC2QUERY",
             "RAG_EN_QUOTA", "RAG_EN_GATE_MIN", "RAG_RERANK_EN_ONNX_DIR")
# Every model this path needs is already in the local cache, so a hub round-trip
# can only add a stall (one A2 repeat hung 14 minutes at 0% CPU on a modelscope
# metadata call) and jitter into the latency numbers.
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "MODELSCOPE_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def _row(repeat: int, sentinel: dict, payload: dict) -> dict:
    metrics = payload["positive"]["metrics"]
    return {
        "run": repeat, "sentinel": sentinel,
        "r1": metrics["strict_ranking"]["recall@1"]["hits"],
        "r3": metrics["strict_ranking"]["recall@3"]["hits"],
        "r5": metrics["strict_ranking"]["recall@5"]["hits"],
        "candidate": metrics["funnel"]["rrf_candidate"]["hits"],
        "gate_pass": metrics["funnel"]["correct_top1_gate_pass"]["hits"],
        "false_accept": payload["negative"]["metrics"]["false_accept"]["hits"],
        "top1": {r["query_id"]: (r.get("final_chunk_ids") or [""])[0]
                 for r in payload["positive"]["rows"]},
    }


def _spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values), "runs": values}


def _check_sentinels(path: Path, expect_judge: bool) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))["runs"][0]
    rows = run["positive"]["rows"] + run["negative"]["rows"]
    diags = [(r.get("gate_features") or {}).get("answerability_rerank") or {} for r in rows]
    applied = sum(1 for d in diags if d.get("applied"))
    degraded = sum(1 for d in diags if d.get("graded", 0) < d.get("calls", 0))
    if expect_judge and applied != len(rows):
        raise SystemExit(f"[final-v2] 哨兵失败 {path.name}: 判据只在 {applied}/{len(rows)} 行生效")
    if not expect_judge and applied:
        raise SystemExit(f"[final-v2] 哨兵失败 {path.name}: 该臂不该开判据，却有 {applied} 行生效")
    if degraded:
        raise SystemExit(f"[final-v2] 哨兵失败 {path.name}: {degraded} 行裁判掉线，结果不可解读")
    return {"applied": applied, "rows": len(rows), "degraded": degraded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="docs/rag_eval/final_v2/final_v2.json")
    parser.add_argument("--arms", default="A1,A2,B,C")
    parser.add_argument("--repeats", type=int, default=3)
    # Blind-split runs may not write into the repository: per-row output would
    # put the sealed questions and their retrieved passages under version
    # control, which is how a blind set stops being blind.  Only the aggregate
    # summary comes back into the repo.
    parser.add_argument("--outdir", default="~/.offerclaw/private_eval/final_v2/runs")
    parser.add_argument("--summary", default="docs/rag_eval/final_v2/runs/SUMMARY.json")
    parser.add_argument("--allow-draft", action="store_true",
                        help="仅用于管道自检，绝不用于出结果")
    args = parser.parse_args()

    # Labels first: "your qrels are unconfirmed" is the more actionable of the
    # two refusals, and it is true regardless of what the code looks like.
    dataset = ROOT / args.dataset
    status = json.loads(dataset.read_text(encoding="utf-8")).get("status")
    if status != "approved" and not args.allow_draft:
        raise SystemExit(
            f"[final-v2] 数据集 status={status!r}，需人工确认 qrels 后改为 approved 才能运行")

    verify = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "freeze_final_v2_config.py"), "--verify"],
        cwd=ROOT, capture_output=True, text=True)
    print(verify.stdout.strip() or verify.stderr.strip())
    if verify.returncode != 0:
        raise SystemExit(
            "[final-v2] 冻结件与工作树不一致，拒绝运行。冻结件记录的是 Final v2 那次"
            "测量时的代码状态;树已经前进,重跑同一批臂必须先回到那个状态,"
            "新配置应当另建 Final v3 而不是复用本集。")

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list] = {}
    for arm in args.arms.split(","):
        profile_arm, overrides, expect_judge = ARMS[arm]
        for repeat in range(1, args.repeats + 1):
            out = outdir / f"{arm}_run{repeat}.json"
            env = dict(os.environ)
            for key in FORCE_OFF:
                env.pop(key, None)
            for key in ("RAG_ANSWERABILITY", "RAG_ANSWERABILITY_EARLY_EXIT",
                        "RAG_ANSWERABILITY_DEPTH", "RAG_HYDE"):
                env.pop(key, None)
            # Post default-flip (2026-08-29): unset RAG_ANSWERABILITY means ON,
            # so judge-off arms must say "0" -- deleting the variable would
            # silently run them with the judge.
            env["RAG_ANSWERABILITY"] = "0"
            env.update(OFFLINE)
            env.update(overrides)
            if out.exists():
                try:
                    sentinel = _check_sentinels(out, expect_judge)
                except SystemExit:
                    out.unlink()          # a bad run is not a resumable one
                else:
                    print(f"[final-v2] {arm} run {repeat}/{args.repeats} 已存在，跳过",
                          flush=True)
                    payload = json.loads(out.read_text(encoding="utf-8"))["runs"][0]
                    results.setdefault(arm, []).append(
                        _row(repeat, sentinel, payload))
                    continue
            print(f"[final-v2] {arm} run {repeat}/{args.repeats} "
                  f"(arm={profile_arm}, judge={'on' if expect_judge else 'off'})",
                  flush=True)
            proc = subprocess.run(
                [sys.executable, str(ROOT / "eval_colloquial_rag.py"),
                 "--cases", str(dataset), "--arm", profile_arm,
                 "--route-mode", "oracle", "--output", str(out)],
                cwd=ROOT, env=env)
            if proc.returncode != 0:
                raise SystemExit(f"[final-v2] {arm} run {repeat} 失败")
            sentinel = _check_sentinels(out, expect_judge)
            payload = json.loads(out.read_text(encoding="utf-8"))["runs"][0]
            results.setdefault(arm, []).append(_row(repeat, sentinel, payload))

    summary = {arm: {key: _spread([float(run[key]) for run in runs])
                     for key in ("r1", "r3", "r5", "candidate", "gate_pass", "false_accept")}
               for arm, runs in results.items()}
    report = {"schema_version": "final-v2-report-v1", "repeats": args.repeats,
              "dataset": args.dataset, "summary": summary,
              "runs": {arm: [{k: v for k, v in r.items() if k != "top1"} for r in runs]
                       for arm, runs in results.items()}}
    summary_path = Path(args.summary).expanduser()
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Aggregates only -- no question text, no retrieved chunk ids.
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[final-v2] 逐行产物 -> {outdir}（仓库外）")
    print(f"[final-v2] 汇总 -> {summary_path}")
    for arm, stats in summary.items():
        print(f"  {arm}: R@1 中位 {stats['r1']['median']:.0f} "
              f"(范围 {stats['r1']['min']:.0f}-{stats['r1']['max']:.0f})  "
              f"R@3 {stats['r3']['median']:.0f}  R@5 {stats['r5']['median']:.0f}  "
              f"误纳 {stats['false_accept']['median']:.0f}")


if __name__ == "__main__":
    main()
