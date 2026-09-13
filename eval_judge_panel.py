# -*- coding: utf-8 -*-
"""裁判团评测（judge panel）——修掉"每个裁判评的答案不一样"的方法漏洞。

背景（2026-07-04）：此前 qwen 自评 9.92、deepseek 交叉 8.83 的对比里，两次 eval
各自重新生成答案（生成 temp=0.2 有随机性），1.1 分差里混着"答案本身不同"的噪声。
本脚本把方法做干净：

  1. 答案只生成一次，连同检索片段一起存档；
  2. 多个裁判（可跨厂家/跨供应商）用**同一套 judge prompt、temp=0** 评**同一批答案**；
  3. 输出逐题×逐裁判分数表 + 每裁判均分 + 分歧最大的题——分歧题才是根因分析的入口。

用法：
    python eval_judge_panel.py --n 12 --out docs/rag_eval/judge_panel.json
默认使用当前进程的 LLM 配置；额外裁判必须通过 ``JUDGE_EXTRA_*`` 显式传入。
"""
import argparse
import json
import os
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

JUDGE_SYSTEM = (
    "你是严格的 RAG 答案质量评审。基于[问题][检索资料][答案]，三个维度各打 0-10 分（只看资料，不用外部知识）：\n"
    "1. faithfulness 忠实度：答案每个论断是否都能从检索资料找到支撑，有无资料外杜撰/幻觉。\n"
    "2. completeness 完整度：答案是否充分、准确地回答了问题。\n"
    "3. citation 引用准确性：答案末尾引用的资料编号是否对应真实支撑该论断的资料（无引用则 0）。\n"
    '只输出 JSON：{"faithfulness":N,"completeness":N,"citation":N,"reason":"一句话"}'
)


def _load_judges() -> list[dict]:
    """Build judge configs without probing sibling repositories or private files."""
    from day1_api_starter import get_llm_config, load_local_env

    load_local_env()
    active = get_llm_config()
    judges = []
    if active.get("api_key") and active.get("api_base") and active.get("model"):
        judges.append({
            "name": "active-primary",
            "url": active["api_base"].rstrip("/") + "/chat/completions",
            "key": active["api_key"],
            "model": active["model"],
            "pacing": float(os.environ.get("JUDGE_PRIMARY_PACING", "1")),
        })

    extra_key = os.environ.get("JUDGE_EXTRA_API_KEY", "").strip()
    extra_base = os.environ.get("JUDGE_EXTRA_BASE_URL", "").strip()
    extra_models = [
        value.strip() for value in os.environ.get("JUDGE_EXTRA_MODELS", "").split(",")
        if value.strip()
    ]
    if extra_key and extra_base and extra_models:
        for index, model in enumerate(extra_models, 1):
            judges.append({
                "name": f"extra-{index}",
                "url": extra_base.rstrip("/") + "/chat/completions",
                "key": extra_key,
                "model": model,
                "pacing": float(os.environ.get("JUDGE_EXTRA_PACING", "1")),
            })
    return judges


def _judge_once(judge: dict, question: str, chunks: list, answer: str):
    context = "\n\n".join(f"[资料{i + 1}]\n{c[:700]}" for i, c in enumerate(chunks))
    payload = {
        "model": judge["model"],
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",
             "content": f"[问题]\n{question}\n\n[检索资料]\n{context}\n\n[答案]\n{answer}"},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    from day1_api_starter import chat_completion, extract_content

    data = chat_completion(judge["url"], {"Authorization": f"Bearer {judge['key']}"},
                           payload, timeout=90)
    raw = extract_content(data) or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {k: float(d.get(k, 0)) for k in ("faithfulness", "completeness", "citation")} | {
            "reason": str(d.get("reason", ""))[:120]
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="docs/rag_eval/judge_panel.json")
    ap.add_argument("--artifacts", default="~/.offerclaw/private_eval/judge_panel_artifacts.json",
                    help="已有答案存档则直接复用（保证多裁判评同一批）")
    args = ap.parse_args()

    from eval_rag_answer import _pick_subset
    from rag_gate import _retrieve_and_classify, synthesize_grounded_answer

    # ── 1. 答案生成（一次，或复用存档）─────────────────────────────
    if os.path.exists(args.artifacts):
        artifacts = json.load(open(args.artifacts, encoding="utf-8"))
        print(f"[复用] {args.artifacts} 里的 {len(artifacts)} 份答案")
    else:
        items = json.load(open(os.path.join(BASE, "tests/rag_bench_set.json"),
                               encoding="utf-8"))["items"]
        sel = _pick_subset(items, args.n)
        artifacts = []
        for q in sel:
            g = _retrieve_and_classify(q["q"], 5)
            if not g["in_kb"] or not g["chunks"]:
                continue
            time.sleep(4)  # dashscope 免费档节流
            ans = synthesize_grounded_answer(q["q"], g["chunks"])
            if not ans:
                continue
            artifacts.append({"id": q["id"], "domain": q["domain"], "q": q["q"],
                              "chunks": g["chunks"], "answer": ans})
            print(f"  [生成] {q['id']} ok", flush=True)
        os.makedirs(os.path.dirname(args.artifacts) or ".", exist_ok=True)
        json.dump(artifacts, open(args.artifacts, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[存档] {len(artifacts)} 份答案 → {args.artifacts}")

    # ── 2. 裁判团逐题打分（同一批答案）───────────────────────────────
    judges = _load_judges()
    table: dict = {a["id"]: {} for a in artifacts}
    means: dict = {}
    for j in judges:
        scores = []
        for a in artifacts:
            time.sleep(j["pacing"])
            v = _judge_once(j, a["q"], a["chunks"], a["answer"])
            table[a["id"]][j["name"]] = v
            if v:
                scores.append(v)
            print(f"  [{j['name']}] {a['id']} "
                  f"{'忠' + str(int(v['faithfulness'])) if v else 'FAIL'}", flush=True)
        if scores:
            means[j["name"]] = {
                k: round(sum(s[k] for s in scores) / len(scores), 2)
                for k in ("faithfulness", "completeness", "citation")
            } | {"n": len(scores)}

    # ── 3. 汇总 + 分歧题 ────────────────────────────────────────────
    print("\n=== 各裁判均分（同一批答案，temp=0）===")
    for name, m in means.items():
        print(f"  {name:<20} 忠实 {m['faithfulness']} · 完整 {m['completeness']} · 引用 {m['citation']} (n={m['n']})")

    spreads = []
    for qid, per in table.items():
        fs = [v["faithfulness"] for v in per.values() if v]
        if len(fs) >= 2:
            spreads.append((max(fs) - min(fs), qid))
    spreads.sort(reverse=True)
    print("\n=== 忠实度分歧最大的题（根因分析入口）===")
    for spread, qid in spreads[:4]:
        row = " | ".join(f"{n}:{int(v['faithfulness'])}" for n, v in table[qid].items() if v)
        print(f"  [{qid}] 分差 {spread:.0f} → {row}")

    out = {"means": means, "table": table,
           "note": "同一本地私密答案集，所有裁判同 prompt、temp=0。"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已存] {args.out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
