#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动作合同专项集(COMPLETE_TEST_PLAN §7 Phase 1)。

走**生产 reference 检索+生成链**(与 /api/query 的 kb 路径同代码、不经语义
规划器),对 24 题固定集逐题判三层动作合同:

  1. pipeline_action   —— 检索结果上盖的章(rag_gate.answer_action_from_retrieval,
                          与生产/评测同一把尺子);
  2. contract_action   —— 实际传给生成的合同(修复后恒等于 1,记录以监督交接);
  3. final_sentence    —— 最终答案首句是否履行动作(correct_premise 须首句纠偏;
                          abstain 须走明确标注的 fallback,不冒充 KB 证据)。

"是否补充资料外内容"无法确定性自动判定,本脚本**不假装能判**——逐题落
原始答案到仓库外供人工抽查,汇总里如实标注该维度未自动判定(无静默截断)。

硬门(§7):pipeline 24/24;correct_premise 最终首句 ≥11/12;abstain 12/12。
未达门槛退出码非零。raw(题面+答案+证据)只写仓库外;仓库内汇总只含
聚合数与逐题布尔(题面本就在 tests/action_contract_v1.json)。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SET = ROOT / "tests" / "action_contract_v1.json"

# 纠偏合同要求"第一句明确指出前提不成立"(rag_gate._CORRECT_PREMISE_RULES)。
# 这里判的是**合同履行**,不是语义理解:首句必须含否定性纠偏措辞之一。
_CORRECTION_MARKERS = ("前提不成立", "前提有误", "不成立", "并不是", "并非",
                      "不是", "有误", "相矛盾", "说法不对", "恰恰相反")
_SENTENCE_END = re.compile(r"[。!！?？\n]")


class ActionContractError(ValueError):
    pass


def load_set(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "offerclaw-action-contract-v1":
        raise ActionContractError("unsupported action contract schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise ActionContractError("action contract v1 must contain exactly 24 cases")
    kinds = Counter(str(c.get("kind") or "") for c in cases)
    if kinds != {"correct_premise": 12, "abstain": 12}:
        raise ActionContractError(f"need 12 correct_premise + 12 abstain, got {dict(kinds)}")
    ids = [str(c.get("id") or "") for c in cases]
    questions = [str(c.get("question") or "").strip() for c in cases]
    if any(not v for v in ids + questions):
        raise ActionContractError("every case needs id and question")
    if len(set(ids)) != len(ids) or len(set(questions)) != len(questions):
        raise ActionContractError("ids and questions must be unique (no paraphrase padding)")
    for c in cases:
        if c["kind"] == "correct_premise" and not str(c.get("evidence_hint") or "").strip():
            raise ActionContractError(f"correct_premise case {c['id']} needs evidence_hint")
    return payload


def first_sentence(text: str) -> str:
    stripped = text.strip()
    match = _SENTENCE_END.search(stripped)
    return stripped[: match.end()] if match else stripped[:80]


def sentence_corrects_premise(text: str) -> bool:
    head = first_sentence(text)
    return any(marker in head for marker in _CORRECTION_MARKERS)


def _git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip())
    return {"git_head": head, "git_dirty": dirty}


def _inside_repo(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return resolved == ROOT or ROOT in resolved.parents


def run_case(question: str) -> dict:
    from rag_gate import (FALLBACK_LABEL, _chat, _fallback_messages,
                          _grounded_messages, _retrieve_and_classify,
                          answer_action_from_retrieval,
                          synthesize_fallback_answer)
    from rag_multi_source import REFERENCE_EXCLUDES

    started = time.perf_counter()
    retrieval = _retrieve_and_classify(
        question, 5,
        exclude_source_types=REFERENCE_EXCLUDES,
        metadata_filters={"owner_scope": {"curated"}},
        allow_paper_route=False,
    )
    retrieval_ms = (time.perf_counter() - started) * 1000
    stamped = str(retrieval.get("answer_action") or "")
    derived = answer_action_from_retrieval(retrieval)
    chunks = list(retrieval.get("chunks") or [])
    if retrieval.get("in_kb"):
        contract_action = stamped or "answer"
        answer = _chat(_grounded_messages(question, chunks, contract_action)) or ""
        mode = "kb_grounded"
    else:
        contract_action = "abstain"
        fallback = synthesize_fallback_answer(question, chunks)
        answer = FALLBACK_LABEL + (fallback or "(无 LLM key,无法作答)")
        mode = "general_fallback"
    return {
        "pipeline_action": stamped, "derived_action": derived,
        "contract_action": contract_action, "mode": mode,
        "in_kb": bool(retrieval.get("in_kb")),
        "sources": list(retrieval.get("sources") or []),
        "answer": answer, "retrieval_ms": round(retrieval_ms, 1),
        "fallback_labeled": answer.startswith(FALLBACK_LABEL),
    }


def run(set_path: Path, raw_dir: Path, summary_path: Path | None) -> dict:
    raw_dir = raw_dir.expanduser().resolve()
    if _inside_repo(raw_dir):
        raise ActionContractError("raw output must stay outside the repository")
    if summary_path is not None and not _inside_repo(summary_path.expanduser().resolve()):
        raise ActionContractError("published summary must stay in repository")
    payload = load_set(set_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("LLM_USAGE_LOG", "0")
    # 本 run 独占的 answerability cache(§5.2),不污染、不借力生产缓存。
    import rag_answerability
    rag_answerability.CACHE_PATH = raw_dir / "answerability_cache.json"

    rows = []
    for case in payload["cases"]:
        qid, kind, question = case["id"], case["kind"], case["question"]
        try:
            result = run_case(question)
        except Exception as exc:            # 网关抖动等:重试一次,再败如实记错
            time.sleep(3)
            try:
                result = run_case(question)
            except Exception:
                result = {"error": f"{type(exc).__name__}", "pipeline_action": "",
                          "contract_action": "", "mode": "error", "in_kb": False,
                          "sources": [], "answer": "", "retrieval_ms": 0.0,
                          "derived_action": "", "fallback_labeled": False}
        expected = kind
        pipeline_ok = result["pipeline_action"] == expected
        if kind == "correct_premise":
            final_ok = result["in_kb"] and sentence_corrects_premise(result["answer"])
        else:
            final_ok = (not result["in_kb"]) and result["fallback_labeled"]
        rows.append({
            "id": qid, "kind": kind, **result,
            "pipeline_ok": pipeline_ok,
            "handoff_ok": result["pipeline_action"] == result["contract_action"]
                          == result["derived_action"] or kind == "abstain",
            "final_ok": final_ok,
        })
        print(f"[{qid}] pipeline={result['pipeline_action'] or '-':16s} "
              f"final_ok={final_ok} ({result['retrieval_ms']:.0f}ms)")

    (raw_dir / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    cp = [r for r in rows if r["kind"] == "correct_premise"]
    ab = [r for r in rows if r["kind"] == "abstain"]
    gates = {
        "pipeline_24_of_24": sum(r["pipeline_ok"] for r in rows),
        "correct_premise_final": sum(r["final_ok"] for r in cp),
        "abstain_final": sum(r["final_ok"] for r in ab),
    }
    passed = (gates["pipeline_24_of_24"] == 24
              and gates["correct_premise_final"] >= 11
              and gates["abstain_final"] == 12)
    summary = {
        "schema_version": "offerclaw-action-contract-result-v1",
        "release_status": payload["release_status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        **_git_state(),
        "set": {"path": str(set_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(set_path.read_bytes()).hexdigest()},
        "gates": gates,
        "gate_thresholds": {"pipeline": "24/24", "correct_premise_final": ">=11/12",
                            "abstain_final": "12/12"},
        "passed": passed,
        "not_auto_judged": "是否补充资料外内容——raw rows.json 供人工抽查",
        "cases": [{k: r[k] for k in
                   ("id", "kind", "pipeline_action", "pipeline_ok", "handoff_ok",
                    "final_ok", "mode", "retrieval_ms")} for r in rows],
        "raw_location": "outside_repository",
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(json.dumps({"gates": gates, "passed": passed}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--set", type=Path, default=DEFAULT_SET)
    result.add_argument("--raw-dir", type=Path, required=True)
    result.add_argument("--summary", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.set, args.raw_dir, args.summary)
