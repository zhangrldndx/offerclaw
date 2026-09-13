"""Round 3：HyDE（Hypothetical Document Embeddings，假设性文档嵌入）查询侧增强。

原理：query 是「问题」形态，文档是「答案」形态，两者在向量空间存在表述鸿沟。
HyDE 先让 LLM 生成一段「假设答案」，再用假设答案的 embedding 去检索——
假设答案与真实文档同为「答案」形态，语义更贴近，召回更准。
适用边界：query 模糊 / 口语化 / 与文档用词差异大时收益最大；query 本身已含精确关键词时收益小。

实现取「原问题 + 假设答案」拼接：保留原问题关键词（与 BM25/精确匹配兼容），同时补语义。
对应知识库 all-in-rag ch4「查询构造 / 查询改写」。

默认关（需 LLM 调用，有延迟/成本）；RAG_HYDE=1 开启。任何失败 → 退回原 query（不影响主链路）。
"""
import os


def hyde_enabled() -> bool:
    return os.environ.get("RAG_HYDE", "0") == "1"


def rewrite_enabled() -> bool:
    return os.environ.get("RAG_QUERY_REWRITE", "0") == "1"


def rewrite_query(question: str, *, enabled: bool | None = None) -> str:
    """[Round 9 / P1.5] 口语 query → 书面检索式改写（查询侧,在线,+1 次 LLM 延迟）。

    与 doc2query 是「用户说人话、文档是书面语」这同一条鸿沟的两侧解法：
    doc2query 离线把文档翻成口语问题（查询零延迟）,本函数在线把口语翻成书面。
    谁有效由 held-out A/B 说话（docs/rag_eval/round9/）,输的一方记负结果。
    只影响向量检索文本；BM25 词法与拒答门控仍用原句。默认关；失败 → 原问题。
    """
    if not (rewrite_enabled() if enabled is None else enabled):
        return question
    try:
        from rag_gate import _chat  # 延迟 import 避免循环依赖
        msg = [{"role": "user", "content":
                "把下面这个口语化提问改写成一句简洁、书面、带准确技术术语的检索查询"
                "（30 字内，只输出改写结果，不要解释）：\n\n" + question}]
        out = _chat(msg, max_tokens=60, temperature=0.2,
                    extra_payload={"enable_thinking": False})
        if isinstance(out, str) and out.strip():
            # 拼接而非替换：保留原句关键词（与向量召回的字面信号兼容），补书面表述
            return question + "\n" + out.strip()
    except Exception:
        pass
    return question


def hyde_expand(question: str, *, enabled: bool | None = None) -> str:
    """生成「原问题 + LLM 假设答案」拼接文本用于检索；未启用或失败 → 原问题。

    ``enabled`` 让**调用方**说了算,`None` 才回落到 env。检索档位由 RetrievalProfile
    拥有,可这个函数原来只认 RAG_HYDE:于是一个显式开了 HyDE 的档会静默按原问题跑,
    漏斗逐字节不动、延迟反而更低,看起来像"HyDE 没用"。本项目已在 RAG_RECALL_N 上
    栽过同一款死旋钮,所以这里让 profile 的意图可以压过 env,而不是反过来。
    """
    if not (hyde_enabled() if enabled is None else enabled):
        return question
    try:
        from rag_gate import _chat  # 延迟 import 避免与 rag_gate 循环依赖
        msg = [{"role": "user", "content":
                "针对下面的技术问题，写一段简洁、专业的参考答案（120 字内，"
                "面向检索召回，可以泛但不要跑题，直接给内容、不要开场白）：\n\n" + question}]
        ans = _chat(msg, max_tokens=200)
        if ans and ans.strip():
            return question + "\n" + ans.strip()
    except Exception:
        pass
    return question


def hyde_with_terms(question: str) -> tuple[str, list[str]]:
    """One call, two artifacts: the hypothetical answer AND its canonical terms.

    Stage 2 Q1 of the next-stage guide.  The measured gap it targets: questions
    that describe a phenomenon without naming it ("读的时候不挡写、每个事务各看
    各的版本" for MVCC) leave both channels empty -- a single free-text HyDE
    sample often names the concept, but not reliably.  Asking the same call to
    also emit the canonical terms makes the term surface explicit and reusable
    as an extra dense/BM25 query representation.

    Contract: the terms feed *retrieval queries only* -- never evidence, never
    citations, never the gate's distance (which stays calibrated on the
    original question).  Failure degrades to plain HyDE text with no terms.
    """
    import json as _json
    import re as _re

    try:
        from rag_gate import _chat
        raw = _chat(
            [{"role": "user", "content":
              "针对下面的技术问题：\n"
              "1. 写一段简洁、专业的参考答案（120 字内，面向检索召回，直接给内容）；\n"
              "2. 列出这个问题实际在问的概念的规范技术术语（1-4 个，用行业标准叫法）。\n"
              "只输出一个 JSON 对象：\n"
              '{"hypothetical_answer": "...", "canonical_terms": ["..."]}\n\n'
              + question}],
            max_tokens=400, temperature=0.2,
            extra_payload={"enable_thinking": False})
        match = _re.search(r"\{.*\}", raw or "", _re.S)
        data = _json.loads(match.group(0)) if match else {}
        answer = str(data.get("hypothetical_answer") or "").strip()
        terms = [str(t).strip() for t in (data.get("canonical_terms") or [])
                 if str(t).strip()][:4]
        if answer:
            return question + "\n" + answer, terms
    except Exception:
        pass
    return question, []
