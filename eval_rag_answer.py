"""端到端答案质量评测（RAGAS 风格 LLM-as-judge）。

检索评测只回答「找得准不准」（Recall@k / MRR），但 RAG 的最终价值是「答案好不好」。
本评测对题走**生产主路径**：`_retrieve_and_classify`（含 rerank/BM25/route）→ grounded 生成 → LLM 裁判三维度：
- **忠实度 faithfulness**：答案每个论断是否都能从检索资料找到支撑、有无资料外杜撰（RAG 最核心，防幻觉）；
- **完整度 completeness**：是否充分、准确回答了问题；
- **引用准确性 citation**：末尾引用的资料编号是否对应真实支撑。

对应八股：RAGAS / LLM-as-judge / faithfulness / answer relevance。
局限：judge 与生成默认继承同一个 GPT 主模型，仍有自评偏差；`RAG_JUDGE_MODEL` 可换独立模型降偏。

用法：python eval_rag_answer.py [--n 12] [--save docs/rag_eval/answer_quality.json]
"""
import argparse
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def _judge(question: str, chunks: list, answer: str):
    """LLM 裁判，返回 {faithfulness, completeness, citation, reason} 或 None。"""
    from rag_gate import _chat
    context = "\n\n".join(f"[资料{i+1}]\n{c[:700]}" for i, c in enumerate(chunks))
    msg = [
        {"role": "system", "content": (
            "你是严格的 RAG 答案质量评审。基于[问题][检索资料][答案]，三个维度各打 0-10 分（只看资料，不用外部知识）：\n"
            "1. faithfulness 忠实度：答案每个论断是否都能从检索资料找到支撑，有无资料外杜撰/幻觉。\n"
            "2. completeness 完整度：答案是否充分、准确地回答了问题。\n"
            "3. citation 引用准确性：答案末尾引用的资料编号是否对应真实支撑该论断的资料（无引用则 0）。\n"
            '只输出 JSON：{"faithfulness":N,"completeness":N,"citation":N,"reason":"一句话"}')},
        {"role": "user", "content": f"[问题]\n{question}\n\n[检索资料]\n{context}\n\n[答案]\n{answer}"},
    ]
    # 裁判走 temp=0 求可复现；RAG_JUDGE_MODEL 设了就换裁判模型做交叉评审
    # （量化自评上浮偏差——default 时 judge==generator，是自评上限）。
    judge_model = os.environ.get("RAG_JUDGE_MODEL") or None
    raw = _chat(msg, max_tokens=300, temperature=0.0, model=judge_model)
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {k: float(d.get(k, 0)) for k in ("faithfulness", "completeness", "citation")} | {"reason": d.get("reason", "")}
    except Exception:
        return None


def _pick_subset(items, n):
    """各域均匀抽样 + 尽量含难题，保证子集代表性与可复现（按 id 排序，不随机）。"""
    from collections import defaultdict
    by = defaultdict(list)
    for q in items:
        by[q["domain"]].append(q)
    per = max(1, n // len(by))
    sel = []
    for dom in sorted(by):
        qs = sorted(by[dom], key=lambda x: (not x.get("hard"), x["id"]))  # 难题优先
        sel += qs[:per]
    return sel[:n]


def evaluate(n=12, save=None):
    from rag_gate import _retrieve_and_classify, synthesize_grounded_answer
    items = json.load(open(os.path.join(BASE, "tests/rag_bench_set.json")))["items"]
    sel = _pick_subset(items, n)
    rows = []
    # 免费档节流：LLM_EVAL_PACING 秒/题（默认 0）。dashscope 免费额度按时间窗
    # 刷新，连发 24 个请求会打爆突发限额（2026-07-04 实测）——付费账户无需设置。
    pacing = float(os.environ.get("LLM_EVAL_PACING", "0"))
    import time as _time
    for qi, q in enumerate(sel):
        if pacing and qi:
            _time.sleep(pacing)
        g = _retrieve_and_classify(q["q"], 5)
        if not g["in_kb"] or not g["chunks"]:
            rows.append({"id": q["id"], "domain": q["domain"], "status": "未命中/无片段"})
            continue
        answer = synthesize_grounded_answer(q["q"], g["chunks"])
        if not answer:
            rows.append({"id": q["id"], "domain": q["domain"], "status": "无 LLM key"})
            continue
        j = _judge(q["q"], g["chunks"], answer)
        if not j:
            rows.append({"id": q["id"], "domain": q["domain"], "status": "judge 失败"})
            continue
        rows.append({"id": q["id"], "domain": q["domain"], "q": q["q"], **j})
        print(f"  [{q['id']}/{q['domain']}] 忠实{j['faithfulness']:.0f} 完整{j['completeness']:.0f} 引用{j['citation']:.0f}  {q['q'][:24]}", flush=True)

    scored = [r for r in rows if "faithfulness" in r]
    summary = {}
    if scored:
        for k in ("faithfulness", "completeness", "citation"):
            summary[k] = round(sum(r[k] for r in scored) / len(scored), 2)
        print(f"\n=== 答案质量（{len(scored)}/{len(sel)} 题命中并评分）===")
        print(f"忠实度 {summary['faithfulness']}/10 · 完整度 {summary['completeness']}/10 · 引用准确性 {summary['citation']}/10")
    out = {"n_evaluated": len(scored), "n_total": len(sel), "summary": summary, "rows": rows}
    if save:
        json.dump(out, open(save, "w"), ensure_ascii=False, indent=2)
        print(f"[已存] {save}")
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--save")
    a = ap.parse_args()
    evaluate(n=a.n, save=a.save)
