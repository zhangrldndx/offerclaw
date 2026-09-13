# -*- coding: utf-8 -*-
"""rag_gate.py — 知识库优先的「带门槛」RAG 问答（CLI 与 Web 共用）

把"命中才答、未命中坦白"的硬门槛逻辑收口到一处，供：
  - offerclaw_cli.py query（微信路径）
  - rag_api.py /api/query（Web UI 路径）
两条入口共用，保证行为完全一致。

门槛策略（基于 text-embedding-v4 标定）：
  - 强向量命中：最近邻距离 <= STRONG → in_kb
  - 词面救回：STRONG < dist <= RESCUE 且查询关键词在片段里字面出现 → in_kb
    （解决 "react" 与库内 "ReAct" 向量距离偏大但字面一致的语义鸿沟）
  - 否则 → in_kb=False，坦白"知识库暂无"，绝不用通用知识杜撰
"""

import os
import re

from rag_tools import get_collection_name, get_embeddings_batch, index_fingerprint
from rag_source_policy import NON_RAG_SOURCE_TYPES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 距离阈值是 **模型相关** 的（不同 embedding 的距离尺度不同）。
# 按 provider/model 给标定过的默认值；env(RAG_RELEVANCE_MAX_DIST 等)可覆盖。
# - 百炼 text-embedding-v3/v4：命中 0.5~0.8 / 假命中 ≥1.04 → strong 0.92, rescue 1.15
# - 本地 bge-base-zh-v1.5（Round 5 在 3299 块库上重标定）：正样本 ≤0.62 / 正经题 ≤0.686 / 边缘沾边假命中 ≥0.773
#   → strong 0.73, rescue 0.73（取间隙 0.686~0.773 中点；KB 从 1024 扩到 3299 后 0.85 太松、误纳 Vue3/SpringBoot 等沾边 query）, weak 1.10
_THRESHOLD_DEFAULTS = {
    "bailian":  {"strong": 0.92, "rescue": 1.15, "weak": 1.30},
    "zhipu":    {"strong": 0.92, "rescue": 1.15, "weak": 1.30},
    "local":    {"strong": 0.73, "rescue": 0.73, "weak": 1.10},
    "_default": {"strong": 0.92, "rescue": 1.15, "weak": 1.30},
}


def _thresholds() -> dict:
    """返回当前 embedding 模型的距离阈值（env 覆盖 > provider 标定默认）。"""
    try:
        from rag_tools import get_embedding_config
        provider = get_embedding_config().get("provider", "_default")
    except Exception:
        provider = "_default"
    base = _THRESHOLD_DEFAULTS.get(provider, _THRESHOLD_DEFAULTS["_default"])
    def ov(name, default):
        v = os.environ.get(name)
        try:
            return float(v) if v else default
        except ValueError:
            return default
    return {
        "strong": ov("RAG_RELEVANCE_MAX_DIST", base["strong"]),
        "rescue": ov("RAG_LEXICAL_RESCUE_DIST", base["rescue"]),
        "weak": ov("RAG_WEAK_CONTEXT_DIST", base["weak"]),
    }
# RAG 合成默认与项目主模型保持一致；可用 RAG_SYNTH_MODEL 单独覆盖。
# Bug 修复（2026-07-04）：原先在**模块 import 时**求值，而 .env.local 由 _chat 里的
# load_local_env() **之后**才加载——导致 .env.local 里 RAG_SYNTH_MODEL=qwen3.7-plus 从不
# 生效，合成永远用冻结的 qwen-turbo（其免费额度已耗尽 → 生产合成 403）。改为惰性读取。
def _synth_model() -> str:
    return os.environ.get(
        "RAG_SYNTH_MODEL", os.environ.get("LLM_MODEL", "gpt-5.6-terra")
    )


# 向后兼容：保留常量名（少数外部引用用它做展示），但每次访问都惰性求值。
RAG_SYNTH_MODEL = _synth_model()

FALLBACK_LABEL = "⚠️ 此问题知识库未直接覆盖，以下为通用知识回答（未经知识库验证，仅供参考）：\n\n"
STATE_LABEL = "📂 来源：实时 RAG 状态源（画像/留痕/计划/完整投递/亲历经验/缺口；实时读取，不依赖向量快照）：\n\n"
LIVE_STATE_SOURCES = ["实时 RAG 状态源（user_profile/daily_log/plans/applications/application_jds/experience_posts）"]

# 个人状态类问题的特征词：命中则在知识库未覆盖时改用"实时文件"作答（双源问答）。
# 解决"问答只看向量快照、答不了'我最近做了什么'"的断层——状态问题直接现读文件。
_STATE_HINTS = ("最近", "今天", "昨天", "本周", "上周", "进度", "留痕",
                "投递", "计划", "缺口", "目标", "复盘", "做了什么", "学了什么",
                "完成", "画像", "日志", "求职方向")
_SELF_STATE_HINTS = ("我的情况", "我的状态", "我的记录", "我目前", "我现在", "为我统计", "帮我统计")


def _is_state_question(question: str) -> bool:
    """兼容旧调用方的纯规则状态判断；实现真源已迁到 ``rule_plan_query``。"""
    try:
        from rag_query_plan import rule_plan_query
        return any(r.source in {"application_state", "application_experience", "application_jd", "profile_plan", "reflection_memory"}
                   for r in rule_plan_query(question).routes)
    except Exception:
        q = (question or "")
        return any(h in q for h in _STATE_HINTS) or any(h in q for h in _SELF_STATE_HINTS)


def _live_state_block(max_chars: int = 6000) -> str:
    """组装实时状态上下文（确定性现读文件，永不过期）。各段独立容错，整体限长。"""
    import datetime
    parts = [f"今天日期：{datetime.date.today().isoformat()}"]
    try:  # 当前计划：周期/本周/今日任务
        from plan_gen import summarize_plan_for_automation
        s = summarize_plan_for_automation()
        if s.get("has_plan"):
            cw = s.get("current_week") or {}
            seg = f"当前学习计划（{s.get('plan_file', '')}，周期 {s.get('period', '')}）"
            if cw:
                seg += f"：第{cw.get('n')}周 主题「{cw.get('theme', '')}」，本周交付：{cw.get('deliverable', '')}"
            if s.get("today_tasks"):
                seg += "；今日任务：" + "；".join(t[:50] for t in s["today_tasks"][:3])
            parts.append(seg)
    except Exception:
        pass
    try:  # 近 7 天留痕明细
        from summary_tool import extract_recent_blocks
        with open(os.path.join(BASE_DIR, "daily_log.md"), encoding="utf-8") as f:
            recent = extract_recent_blocks(f.read(), days=7)
        if recent:
            parts.append("近 7 天留痕：\n" + recent[:1400])
    except Exception:
        pass
    try:  # 投递池：完整字段，而不是只给公司/状态摘要
        from applications_store import list_applications, list_experiences
        rows = list_applications()
        if rows:
            def _g(r, k):
                return next((r[c] for c in r if k in c), "")
            app_lines = []
            for r in rows[:8]:
                line = (f"- {_g(r, '公司')}｜岗位：{_g(r, '岗位')}｜状态：{_g(r, '状态')}"
                        f"｜日期：{_g(r, '日期')}｜下一步：{_g(r, '下一步') or '—'}")
                for label, key in (("来源", "来源"), ("地点", "地点"),
                                   ("匹配结论", "匹配结论"), ("方向", "样本定位"),
                                   ("进展备注", "备注")):
                    value = _g(r, key)
                    if value and value != "—":
                        line += f"｜{label}：{value}"
                app_lines.append(line)
            parts.append("真实投递记录（applications.md，实时）：\n" + "\n".join(app_lines))
        else:
            parts.append("真实投递记录：暂无")

        experiences = list_experiences()
        if experiences:
            parts.append("亲历投递经验（experience_posts，实时）：\n" + "\n".join(
                f"- {e.get('company')}｜{e.get('position')}｜阶段：{e.get('stage')}｜"
                f"日期：{e.get('date')}｜总结：{(e.get('summary') or '')[:700]}"
                for e in experiences[:8]
            ))
    except Exception:
        pass
    try:  # 投递管理派生的活动 JD 学习目标
        from application_jd_store import plan_gaps_text, plan_targets
        target_data = plan_targets()
        if target_data.get("total"):
            parts.append(
                f"学习目标：投递管理中 {target_data['total']} 条活动 JD\n"
                + plan_gaps_text(max_chars=1200)
            )
    except Exception:
        pass
    try:  # 画像头部
        with open(os.path.join(BASE_DIR, "user_profile.md"), encoding="utf-8") as f:
            parts.append("画像摘录：\n" + f.read()[:700])
    except Exception:
        pass
    return "\n\n".join(parts)[:max_chars]


def _state_messages(question: str, live_block: str, weak_chunks: list) -> list:
    bg = ""
    if weak_chunks:
        bg = ("\n\n知识库弱相关片段（未必准确，仅参考）：\n"
              + "\n\n".join(f"[背景{i+1}]\n{c[:300]}" for i, c in enumerate(weak_chunks[:2])))
    return [
        {"role": "system", "content": (
            "你是 OfferClaw 的个人状态问答助手。用户在问 ta 自己的状态/进展/计划/投递，"
            "请**只依据下面的「实时状态」作答**（刚从用户状态文件现读，绝对最新）；"
            "实时状态里没有的信息就直说没有记录，不要编造。回答简洁、分点。\n\n"
            "========== 实时状态 ==========\n" + live_block + bg
        )},
        {"role": "user", "content": question},
    ]


def query_keywords(question: str) -> list:
    """抽取区分性英文/数字 token（len>=3）用于词面救回；中文交给向量。"""
    import re
    stop = {"什么", "怎么", "如何", "为什么", "the", "and", "what", "how", "why", "is", "are"}
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", question)
    return [t.lower() for t in tokens if t.lower() not in stop]


def lexical_hit(keywords: list, chunks: list) -> bool:
    if not keywords:
        return False
    blob = " ".join(chunks).lower()
    return any(kw in blob for kw in keywords)


def colloquial_gate_accept(rerank_top: float | None,
                           rerank_margin: float | None) -> bool:
    """口语证据门的判据(默认关,两个条件缺一不可)。

    单独拆成函数是为了让判据本身可单测:它埋在一个几百行的检索函数里时,
    只能靠源码断言钉,而源码断言钉不住"两个条件是与不是或"这种东西。

    标定见 docs/rag_eval/colloquial/GATE_20260827.md;τ 与 margin 都必须显式
    给出数值才生效,任何解析失败一律按"不放行"处理——证据门的默认方向是拒答。
    """
    raw = (os.environ.get("RAG_COLLOQUIAL_GATE_MIN", "") or "").strip()
    if not raw or rerank_top is None or rerank_margin is None:
        return False
    try:
        tau = float(raw)
        margin_min = float(
            (os.environ.get("RAG_COLLOQUIAL_GATE_MARGIN", "") or "0.10").strip()
        )
    except ValueError:
        return False
    return rerank_top >= tau and rerank_margin >= margin_min


def _answerability_enabled() -> bool:
    """惰性读 env:与本项目其它旋钮一致,避免 import 时冻结 .env.local。"""
    from rag_answerability import enabled
    return enabled()


def _answerability_early_exit() -> bool:
    """省延迟档:首名已是最高档就不再判后面几名,对 Top1 精确(双键最优),实测
    R@3/R@5 与满深度相同、调用减半。Final v3 默认翻转后**默认开**;显式 0 关闭。"""
    raw = os.getenv("RAG_ANSWERABILITY_EARLY_EXIT")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}




def _merge_bm25_channels(primary, secondary, *, rrf_k: int, limit: int):
    """Rank-fuse two BM25 hit lists; shared hits keep the primary tuple.

    Same shape as the dense merge but simpler: BM25 scores never reach the
    evidence gate, so only membership and order matter.  The fused list is cut
    back to ``limit`` -- the second query buys reach, not a bigger pool.
    """
    from rag_retrieval_trace import stable_chunk_id

    scores: dict[str, float] = {}
    payload: dict[str, tuple] = {}
    for hits in (primary, secondary):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = stable_chunk_id(hit[0], hit[1])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            payload.setdefault(chunk_id, hit)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    return [payload[chunk_id] for chunk_id in order]


def _merge_dense_channels(primary, secondary, *, collection, query_embedding,
                          rrf_k: int, limit: int):
    """Fuse two dense result lists by rank, but keep the *primary* query's距离.

    Rank fusion is what buys recall: a chunk both queries find rises, a chunk only
    the hypothetical answer finds still gets in but lower.  Distance is a separate
    question -- the evidence gate compares it against thresholds calibrated on the
    original question, so a secondary-query distance substituted here would be
    measured against the wrong distribution.  For chunks the primary query never
    returned, the true primary distance is computed from the stored vector rather
    than borrowed or guessed.
    """
    p_docs, p_metas, p_dists, p_ids = primary
    s_docs, s_metas, s_dists, s_ids = secondary

    scores: dict[str, float] = {}
    payload: dict[str, tuple] = {}
    for rank, (chunk_id, document, meta) in enumerate(zip(p_ids, p_docs, p_metas), start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        payload.setdefault(chunk_id, (document, meta))
    primary_distance = dict(zip(p_ids, p_dists))
    for rank, (chunk_id, document, meta) in enumerate(zip(s_ids, s_docs, s_metas), start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        payload.setdefault(chunk_id, (document, meta))

    order = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    missing = [cid for cid in order if cid not in primary_distance]
    if missing:
        try:
            stored = collection.get(ids=missing, include=["embeddings"])
            vectors = dict(zip(stored.get("ids") or [], stored.get("embeddings") or []))
        except Exception:
            vectors = {}
        for chunk_id in missing:
            vector = vectors.get(chunk_id)
            primary_distance[chunk_id] = (
                sum((a - b) ** 2 for a, b in zip(query_embedding, vector))
                if vector is not None else float("inf")
            )

    docs, metas, dists, ids = [], [], [], []
    for chunk_id in order:
        document, meta = payload[chunk_id]
        docs.append(document)
        metas.append(meta)
        dists.append(float(primary_distance.get(chunk_id, float("inf"))))
        ids.append(chunk_id)
    return docs, metas, dists, ids


def _answerability_tiebreak() -> bool:
    """平局裁决:4 档量表在"两块都含答案"处饱和,只在平局候选之间再问一次哪块最完整。"""
    return os.getenv("RAG_ANSWERABILITY_TIEBREAK", "").strip().lower() in {"1", "true", "yes", "on"}


def _answerability_gate() -> bool:
    """答案含量门:排序赢不等于回答赢,门本身是按精排分标定的,判据重排后会按构造误拒。
    Final v4 后默认开(三票共识+v5 提示词下盲集真误纳 1/40、纠正 12/12、有效作答翻倍);
    显式 0 关。"""
    from rag_mode import mode_env
    raw = mode_env("RAG_ANSWERABILITY_GATE")   # 显式 env 优先,fast 模式预设补缺省
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _answerability_gate_votes() -> int:
    """放行需要几票。单票版被实测否决:冷跑间判词翻转且两次都朝放行方向,
    所以默认 3 票取多数;拒答永远单票即定(宁拒不编,不对称是设计而非省钱)。"""
    raw = os.getenv("RAG_ANSWERABILITY_GATE_VOTES", "").strip()
    try:
        value = int(raw) if raw else 3
    except ValueError:
        return 3
    return value if value >= 1 else 3


def _answerability_depth() -> int:
    """判据看多深。默认沿用模块默认值;只有池子变宽时才有必要抬。"""
    from rag_answerability import ANSWERABILITY_DEPTH
    raw = os.getenv("RAG_ANSWERABILITY_DEPTH", "").strip()
    if not raw:
        return ANSWERABILITY_DEPTH
    try:
        value = int(raw)
    except ValueError:
        return ANSWERABILITY_DEPTH
    return value if value > 0 else ANSWERABILITY_DEPTH


def _evidence_gate(vector_in_kb: bool, rerank_top, best: float = None,
                   strong: float = None) -> bool:
    """Round 6 证据型门控 + Round 6.1 反向边缘救援：reranker 既是「否决权」也是「救援」。

    ① **否决**（解决正负 best 重叠）：距离/词法初判 in_kb 后，reranker 偏低（距离近但实际不相关，
       如 Docker：best 0.58 而 reranker 0.64）→ 否决拒答。阈值 RAG_RERANK_GATE_MIN（默认 0.85）。
    ② **救援**（解决正样本边缘误拒）：距离稍超 strong（如「LoRA 是什么」best 0.758）但 reranker
       极高（0.99=明显相关）→ 救回。窗口 strong<best≤RAG_RERANK_RESCUE_DIST(0.80)、
       reranker≥RAG_RERANK_RESCUE_MIN(0.95)，避免误救（Vue3 reranker 仅 0.77、今天天气 best 0.97 太远）。
    rerank_top=None（rerank 关）→ 退化纯距离门控，保持兼容。
    """
    if rerank_top is None:
        return vector_in_kb
    if vector_in_kb:
        return rerank_top >= float(os.environ.get("RAG_RERANK_GATE_MIN", "0.85"))
    # 反向边缘救援：距离稍超 strong 但 reranker 极高（明显相关）
    if best is not None and strong is not None:
        rescue_dist = float(os.environ.get("RAG_RERANK_RESCUE_DIST", "0.80"))
        rescue_min = float(os.environ.get("RAG_RERANK_RESCUE_MIN", "0.95"))
        if strong < best <= rescue_dist and rerank_top >= rescue_min:
            return True
    return False


def _chat(messages: list, max_tokens: int = 800, temperature: float = 0.2,
          model: str | None = None, extra_payload: dict | None = None,
          meta: dict | None = None, request_timeout: float | None = None,
          max_retries: int | None = None):
    """用 RAG 合成模型（默认继承项目 GPT 主模型）调一次 LLM；无 key 返回 None。

    **返回值是提取后的文本 str（或 None），不是原始响应 dict**——调用方别再解包 choices。
    ``temperature`` 默认 0.2（合成路径）；评审/裁判类调用应传 0.0 求近似确定性
    （否则同一答案多次评分会漂移，使 faithfulness 这类指标不可复现）。
    ``model`` 显式指定模型名（覆盖 _synth_model）；裁判类调用传不同厂家模型做
    交叉评审，量化"自评上浮偏差"。
    ``extra_payload`` 合并进请求体（如 doc2query 批量生成传
    ``{"enable_thinking": False}`` 关思考模式——短结构化输出任务的思考纯烧钱）。"""
    import requests
    from day1_api_starter import get_llm_config, build_zhipu_jwt, load_local_env
    load_local_env()
    cfg = get_llm_config()
    api_key = cfg["api_key"]
    if not api_key:
        return None
    bearer = build_zhipu_jwt(api_key) if cfg["is_zhipu"] else api_key
    # 模型优先级：显式 model 参数 > 智谱默认 > _synth_model（惰性读 .env.local）
    model = model or (cfg["model"] if cfg.get("is_zhipu") else _synth_model())
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]
    if extra_payload:
        payload.update(extra_payload)
    from day1_api_starter import chat_completion, extract_content  # A1 网关 + A2 防御解析
    from model_call_context import remaining_seconds
    data = chat_completion(
        f"{cfg['api_base']}/chat/completions",
        {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
        payload, timeout=remaining_seconds(request_timeout or 60),
        **({"max_retries": max_retries} if max_retries is not None else {}),
    )
    if meta is not None:
        # 回填**实际服务的模型**(2026-08-20 实测缺陷):A1 网关会在主模型失败时自动切兜底,
        # 于是"配置的模型"与"真正生成内容的模型"可以不同。别名等离线资产若按配置名记血缘,
        # 会写下与事实不符的出处——194 条别名曾全部标 gpt-5.6,实际全由 deepseek 兜底生成。
        meta["model"] = (data or {}).get("model") or model
        meta["finish_reason"] = ((data or {}).get("choices") or [{}])[0].get("finish_reason")
        meta["usage"] = (data or {}).get("usage")
    return extract_content(data)


def synthesize_grounded_answer(question: str, chunks: list,
                               answer_action: str = "answer"):
    """命中知识库：**仅基于检索片段**合成答案（不得用资料外知识）。无 key 返回 None。

    ``answer_action="correct_premise"`` 时附加纠偏合同（见 ``_CORRECT_PREMISE_RULES``）。
    """
    return _chat(_grounded_messages(question, chunks, answer_action))


def synthesize_fallback_answer(question: str, weak_chunks: list):
    """未命中知识库：用 LLM 通用知识 + 项目先验 + 弱相关片段（若有）生成答案。无 key 返回 None。

    外层会在答案前加 FALLBACK_LABEL 明确标注非知识库内容。
    """
    return _chat(_fallback_messages(question, weak_chunks))


def _chat_stream(messages: list, max_tokens: int = 800):
    """流式版 _chat：逐 token yield。无 key 时不产出任何 token。"""
    import json as _json
    import requests
    from day1_api_starter import get_llm_config, build_zhipu_jwt, load_local_env
    load_local_env()
    cfg = get_llm_config()
    api_key = cfg["api_key"]
    if not api_key:
        return
    bearer = build_zhipu_jwt(api_key) if cfg["is_zhipu"] else api_key
    model = cfg["model"] if cfg.get("is_zhipu") else _synth_model()
    payload = {"model": model, "messages": messages, "temperature": 0.2,
               "max_tokens": max_tokens, "stream": True}
    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]
    from day1_api_starter import chat_completion  # A1 统一 LLM 网关（重试+退避）
    from model_call_context import remaining_seconds
    resp = chat_completion(f"{cfg['api_base']}/chat/completions",
                           {"Authorization": f"Bearer {bearer}"},
                           payload, timeout=remaining_seconds(120), stream=True)
    with resp:
        # OpenAI 兼容 SSE 常省略 charset；requests 会按 latin-1 解码 text/*，
        # 中文 token 因而变成 ç®å 这类 mojibake。协议 JSON 明确按 UTF-8 解码。
        resp.encoding = "utf-8"
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            chunk = raw[6:]
            if chunk.strip() == "[DONE]":
                break
            try:
                delta = _json.loads(chunk)["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    yield delta
            except Exception:
                continue


# 粗召回规模：rerank/混合检索从更大的候选池里精排出 top_k（仅 reranker 可用时扩大）
RECALL_N = int(os.environ.get("RAG_RECALL_N", "20"))


def _rerank_pair_documents(
    docs: list[str], metas: list[dict], *, use_breadcrumb: bool,
    prefix_mode: str = "none",
) -> list[str] | None:
    """Build A3 scoring text while preserving original evidence documents."""

    mode = "full" if use_breadcrumb and prefix_mode == "none" else prefix_mode
    if mode == "none":
        return None
    output = []
    for document, meta in zip(docs, metas):
        meta = meta or {}
        source = os.path.basename(str(meta.get("source") or "")).strip()
        title = str(meta.get("title") or "").strip()
        heading = str(meta.get("heading_path") or meta.get("section_path") or "").strip()
        context_lines = []
        if mode == "compact32":
            heading_parts = [
                part.strip()
                for part in re.split(r"\s*(?:>|/|›)\s*", heading)
                if part.strip()
            ]
            compact = []
            if title:
                compact.append(f"标题：{title}")
            elif source:
                compact.append(f"文档：{source}")
            if heading_parts:
                compact.append("章节：" + " > ".join(heading_parts[-2:]))
            prefix = "\n".join(compact)[:96]
            context_lines = [prefix] if prefix else []
        else:
            if source:
                context_lines.append(f"来源文档：{source}")
            if title:
                context_lines.append(f"章节：{title}")
            if heading and heading != title:
                context_lines.append(f"章节路径：{heading}")
        output.append(
            "\n".join(context_lines + ["正文：", document])
            if context_lines else document
        )
    return output


def _retrieve_and_classify(question: str, top_k: int = 5, _crag_depth: int = 0,
                           source_types: set[str] | None = None,
                           exclude_source_types: set[str] | None = None,
                           metadata_filters: dict[str, set[str]] | None = None,
                           allow_paper_route: bool = True,
                           retrieval_profile=None,
                           _trace=None,
                           _shadow_internal: bool = False,
                           _answerability_shadow_route: str = "",
                           _answerability_shadow_question: str = "",
                           judgement_question: str | None = None) -> dict:
    """检索 + 门槛判定（命中/未命中），返回决策 + 片段，供流式/非流式共用。

    Round 1：若 rerank 启用，向量先粗召回 RECALL_N → bge-reranker 精排 → 取 top_k；
    门控 best 仍取**最小向量距离**（粗召回已含全局最近邻，rerank 不改变门控行为）。

    ``judgement_question``(2026-09-01):判据与共识 panel 的判定对象。multi 路
    检索用规划器改写的中性 subquery,但"含不含答案/是否与前提相矛盾"必须对
    **用户原问题**判——错误前提住在原问题里,subquery 已把它抹掉。缺省 =
    检索问题(直连路径与全部既有评测行为逐位不变)。
    """
    import time
    _jq = (judgement_question or "").strip() or question
    import chromadb
    from rag_retrieval_trace import (
        RetrievalTrace, bind_profile_fingerprint, candidate_traces, resolve_retrieval_profile,
        rrf_diagnostics, stable_chunk_id,
    )
    from rag_rerank import rerank, rerank_enabled

    total_started = time.perf_counter()
    requested_profile_name = (
        str(retrieval_profile).strip()
        if isinstance(retrieval_profile, str)
        else os.environ.get("RAG_RETRIEVAL_PROFILE", "baseline").strip()
        if retrieval_profile is None else ""
    )
    schedule_shadow = (
        not _shadow_internal and requested_profile_name == "candidate_shadow"
    )
    profile = resolve_retrieval_profile(retrieval_profile)
    trace = _trace if isinstance(_trace, RetrievalTrace) else RetrievalTrace(profile)

    def _finish(result: dict) -> dict:
        # 所有返回路径(主返回/CRAG 恢复/论文路)都从这里出门,动作章只在
        # 这里盖:新增返回路径自动继承,不靠人记得。派生只读已有判定,
        # 不改变 in_kb/门/排序的任何一位(检索基线红线)。
        result.setdefault("answer_action", answer_action_from_retrieval(result))
        if schedule_shadow:
            try:
                from rag_retrieval_shadow import enqueue_shadow_comparison
                enqueue_shadow_comparison(
                    question,
                    result,
                    top_k=top_k,
                    source_types=source_types,
                    exclude_source_types=exclude_source_types,
                    metadata_filters=metadata_filters,
                    allow_paper_route=allow_paper_route,
                )
            except Exception:
                # Shadow startup, queueing and logging are observability only.
                pass
        return result
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    col = client.get_collection(get_collection_name())
    fingerprint_started = time.perf_counter()
    fp = bind_profile_fingerprint(index_fingerprint(collection=col), profile)
    trace.latency_by_stage["fingerprint"] = round(
        (time.perf_counter() - fingerprint_started) * 1000, 3
    )
    trace.index_metadata = fp
    trace.index_fingerprint = str(fp.get("fingerprint_id") or "")
    # Round 3：HyDE 查询侧增强（默认关）——用「原问题+LLM假设答案」检索，拉近 query↔文档表述差。
    # 关键词救援/门控仍用原 question（见下），HyDE 只影响向量召回的语义匹配。
    from rag_hyde import hyde_expand, rewrite_query
    # 查询侧文本管线：HyDE(默认关) → 口语改写(Round9/P1.5,默认关)。两关全闭 = 原问题。
    # [多-agent ⑤ CRAG] 递归恢复时(depth>0)question 已是 CRAG 改写后的,不再二次套 rewrite/hyde（防双改写）。
    embedding_started = time.perf_counter()
    _q_for_emb = question
    if _crag_depth == 0:
        # 档位显式开启时把意图传下去:否则 profile 说开、函数里的 env 说关,
        # 整个档会静默按原问题跑(漏斗逐字节不变正是这个故障的指纹)。
        if profile.enable_hyde is not False:
            _q_for_emb = hyde_expand(
                _q_for_emb, enabled=True if profile.enable_hyde else None)
        if profile.enable_query_rewrite is not False:
            _q_for_emb = rewrite_query(
                _q_for_emb, enabled=True if profile.enable_query_rewrite else None)
    # 口语 query 侧 LoRA(默认关):把问题映射进**现有**文档空间,文档向量一个字节不动、
    # 无需重建索引。只在这一处生效——入库路径不 import 它,结构上够不到。
    from rag_query_adapter import embed_query as _adapter_embed_query
    _adapted = _adapter_embed_query(_q_for_emb)
    emb = [_adapted] if _adapted is not None else get_embeddings_batch([_q_for_emb])
    trace.latency_by_stage["embedding"] = round(
        (time.perf_counter() - embedding_started) * 1000, 3
    )
    # 惰性读 env（Round 9 修正）：原先用模块级 RECALL_N,import 时冻结——与 _synth_model
    # 曾犯的同款 bug:环境变量旋钮改了不生效,P4 延迟-精度扫描也无法在同进程内换挡。
    # 配额开启时召回深度**不随 rerank 开关变化**:多通道设计里"池子"与"最后怎么排"是两件事,
    # 若深度跟着 rerank 走,报告 §15 的 S0-S3 消融就同时动了两个变量(深度+排序层),测不出根因。
    from rag_quota import quota_enabled
    quota_active = (quota_enabled() if profile.enable_quota is None
                    else bool(profile.enable_quota))
    recall_n = (max(top_k, profile.pool_size) if (rerank_enabled() or quota_active)
                else top_k)
    source_types = {str(x) for x in (source_types or set()) if x}
    exclude_source_types = {str(x) for x in (exclude_source_types or set()) if x}
    source_types.difference_update(NON_RAG_SOURCE_TYPES)
    exclude_source_types.update(NON_RAG_SOURCE_TYPES)
    metadata_filters = {
        str(key): {str(value) for value in values if value}
        for key, values in (metadata_filters or {}).items() if values
    }
    trace.metadata_filters = {
        "source_types": sorted(source_types),
        "exclude_source_types": sorted(exclude_source_types),
        "metadata": {key: sorted(values) for key, values in metadata_filters.items()},
    }
    where_clauses = []
    if source_types:
        where_clauses.append({"source_type": {"$in": sorted(source_types)}})
    if exclude_source_types:
        where_clauses.append({"source_type": {"$nin": sorted(exclude_source_types)}})
    for key, values in metadata_filters.items():
        where_clauses.append({key: {"$in": sorted(values)}})
    where = (where_clauses[0] if len(where_clauses) == 1
             else {"$and": where_clauses} if where_clauses else None)
    query_kwargs = {
        "query_embeddings": emb,
        "n_results": recall_n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    dense_started = time.perf_counter()
    res = col.query(**query_kwargs)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    dense_ids = (res.get("ids") or [[]])[0]

    # [多路稠密] HyDE 作为**第二条查询**而不是替换第一条。默认关。
    # 动机来自 Final v2:池外 21 题里,原问题够得到 9 题、HyDE 够得到 15 题,
    # 而"两者取并"能到 17 题——它们是互补的,替换掉一边就等于丢掉另一边。
    # 更要紧的是替换会把整池距离换成「假设答案→文档」的距离,而证据门的阈值是按
    # 「问题→文档」标定的;Final v2 上这个错配代价是一条真误纳。
    # 所以这里合并**名次**(RRF,两个查询都找到的块自然靠前),但距离一律回到原问题:
    # 只被 HyDE 找到的块,用它在库里的向量与原问题向量现算 L2,而不是借用 HyDE 的距离。
    _hyde_channel_text = None
    _canonical_terms_text = None
    if _crag_depth == 0 and (profile.enable_hyde_channel
                             or profile.enable_hyde_bm25_channel):
        if profile.enable_canonical_terms:
            # 同一次调用产出假设答案+规范术语(Stage 2 Q1)。术语只作查询表示。
            from rag_hyde import hyde_with_terms
            _hyde_channel_text, _terms = hyde_with_terms(question)
            if _terms:
                _canonical_terms_text = " ".join(_terms)
        else:
            _hyde_channel_text = hyde_expand(question, enabled=True)
        if _hyde_channel_text == question:      # LLM 不可用 → 两条通道都静默退化
            _hyde_channel_text = None
            _canonical_terms_text = None
    if _hyde_channel_text and profile.enable_hyde_channel:
        _hyde_text = _hyde_channel_text
        if True:
            _extra_queries = [_hyde_text]
            if _canonical_terms_text:
                _extra_queries.append(_canonical_terms_text)
            for _q_extra in _extra_queries:
                _hk = dict(query_kwargs)
                _hk["query_embeddings"] = get_embeddings_batch([_q_extra])
                _hres = col.query(**_hk)
                docs, metas, dists, dense_ids = _merge_dense_channels(
                    (docs, metas, dists, dense_ids),
                    ((_hres.get("documents", [[]])[0], _hres.get("metadatas", [[]])[0],
                      _hres.get("distances", [[]])[0], (_hres.get("ids") or [[]])[0])),
                    collection=col, query_embedding=emb[0],
                    rrf_k=profile.rrf_k, limit=recall_n,
                )
            trace.latency_by_stage["dense_hyde_channel"] = round(
                (time.perf_counter() - dense_started) * 1000, 3)
    trace.latency_by_stage["dense"] = round(
        (time.perf_counter() - dense_started) * 1000, 3
    )
    dense_id_by_doc = {
        document: str(chunk_id)
        for document, chunk_id in zip(docs, dense_ids)
    }

    # Round 9 / P1：doc2query 文档侧增强（默认关，由 held-out A/B 决定去留）。
    # 离线合成的口语化问题命中后映射回父块原文合并进召回池——口语 query ↔ 合成问题
    # 的向量距离通常远小于 query ↔ 书面文档，故合并结果同样参与下方 best 门控距离；
    # 下游 BM25/rerank/路由拿到的都是父块原文，无感知。关闭（默认）时行为与改造前一致。
    from rag_doc2query import d2q_enabled, query_d2q_and_merge
    d2q_active = (d2q_enabled() if profile.enable_doc2query is None
                  else bool(profile.enable_doc2query))
    if d2q_active:
        if not source_types and not exclude_source_types and not metadata_filters:
            docs, metas, dists = query_d2q_and_merge(emb, docs, metas, dists, recall_n, client)
    trace.dense_candidates = candidate_traces(
        docs, metas,
        ids=[dense_id_by_doc.get(document, stable_chunk_id(document, meta))
             for document, meta in zip(docs, metas)],
        dense_distances=dists,
        channels={document: ["dense"] for document in docs},
    )

    # [配额 V1] 英文/混排保险池(方案报告 §9.3/§10.2 Q2)——**永不归零**,取代硬路由。
    # 用同一组 query 向量查英文分区,候选是"增加"而非"顶替",中文候选全程留在池中同台竞争,
    # 所以劫持在构造上不可能(硬路由两次实测栽在这:松门劫持 -7.7pp / 比分门净增益 0)。
    from rag_quota import (append_quota_pool, cap_per_source, pool_stats,
                           quota_candidates, quota_enabled)
    _quota_hits = (quota_candidates(emb, question=question)
                   if quota_active and not source_types and not exclude_source_types
                   and not metadata_filters else [])

    th = _thresholds()
    # 门控输入默认只看主库池(RAG_QUOTA_IN_GATE=0):V1 配额是**排序改造**,门控语义逐字节不变,
    # 中文拒答零风险;配额是否该进门控 = 报告 Phase 5 的 Gate 标定题,单独测(见 §23 profile 化)。
    _gate_docs, _gate_dists = docs, dists
    if _quota_hits and os.environ.get("RAG_QUOTA_IN_GATE", "0") == "1":
        _gate_docs = list(docs) + [h[0] for h in _quota_hits]
        _gate_dists = list(dists) + [h[2] for h in _quota_hits]
    # 结构块不作数(默认关,见 rag_structural_evidence):文档**标题块**与**页面结构目录**
    # 只有小节名、没有可作答的断言,却因为词面与问题高度重合而距离极近——v2aneg-015 的
    # best 0.576(全库最近)就来自一块 167 字的标题块。判据关闭时 _struct_tau=None,
    # 掩码恒为全 True,以下几行逐字节退化为原来的 best/rescue/vector_in_kb。
    # 池子里一块实心的都没有(真·"这份文档都讲了什么")→ 退回全池,不做无差别打压。
    from rag_structural_evidence import (evidence_anchor, is_structural,
                                         structural_fraction,
                                         structural_threshold, substantive_mask)
    _struct_tau = structural_threshold()
    _gate_solid = substantive_mask(_gate_docs, _struct_tau)
    if not any(_gate_solid):
        _gate_solid = [True] * len(_gate_docs)
    _gate_evidence = [(document, dist)
                      for document, dist, solid in zip(_gate_docs, _gate_dists, _gate_solid)
                      if solid]
    best = min((dist for _d, dist in _gate_evidence), default=99.0)  # 最小向量距离（门控判定用，不受 rerank 影响）
    keywords = query_keywords(question)
    # 门控判定基于「粗召回全集」：词法救援在更大候选池里更易命中；best=最小距离不变
    rescue_chunks = [document for document, dist in _gate_evidence
                     if dist <= th["rescue"]]
    lexical_rescued = best <= th["rescue"] and lexical_hit(keywords, rescue_chunks)
    vector_in_kb = bool(_gate_docs) and (best <= th["strong"] or lexical_rescued)  # 距离/词法初判（reranker 后再综合）

    # Round 2：BM25 稀疏召回 + RRF 融合，扩大 rerank 候选池（gate 已用上面的向量 best 判完，不受影响）。
    # 救「同主题多源 / 中文术语」：向量把近义子主题挤在一起，BM25 按关键词把目标 chunk 拉回候选池。
    from rag_bm25 import bm25_search, bm25_enabled, rrf_fuse
    _prov = None
    bm25_hits = []
    rrf_scores = {}
    fusion_channels = {document: ["dense"] for document in docs}
    dense_docs_for_fusion = list(docs)
    dense_metas_for_fusion = list(metas)
    dense_dists_for_fusion = list(dists)
    # 词法通道在配额模式下不再挂靠 rerank 开关(原链路把 BM25 定位成"rerank 候选喂料",
    # 于是 RAG_RERANK=0 时 BM25 一并失效,报告 §15 的 S1/S2 消融档根本测不出来)。
    bm25_started = time.perf_counter()
    if bm25_enabled() and (rerank_enabled() or _quota_hits):
        bm25_hits = bm25_search(
            question, recall_n,
            source_types=source_types,
            exclude_source_types=exclude_source_types,
            metadata_filters=metadata_filters,
        )
        def _bm25_allowed(hit) -> bool:
            meta = hit[1] or {}
            if source_types and meta.get("source_type") not in source_types:
                return False
            if exclude_source_types and meta.get("source_type") in exclude_source_types:
                return False
            return all(str(meta.get(key, "")) in values
                       for key, values in metadata_filters.items())
        if source_types or exclude_source_types or metadata_filters:
            # Defensive recheck preserves the evidence boundary if a custom or
            # monkeypatched BM25 implementation ignores the new arguments.
            bm25_hits = [h for h in bm25_hits if _bm25_allowed(h)][:recall_n]
        # [多路词法] 同一段 HyDE 文本再走一次 BM25,与原问题的词法结果按名次融合。
        # 词法侧没有距离错配问题(BM25 候选进池本就用占位距离),要防的只剩预算:
        # 融合后仍截回 recall_n,不放大池子。
        if _hyde_channel_text and profile.enable_hyde_bm25_channel:
            _lex_query = (_hyde_channel_text
                          + ((" " + _canonical_terms_text) if _canonical_terms_text else ""))
            _extra_hits = bm25_search(
                _lex_query, recall_n,
                source_types=source_types,
                exclude_source_types=exclude_source_types,
                metadata_filters=metadata_filters,
            )
            if source_types or exclude_source_types or metadata_filters:
                _extra_hits = [h for h in _extra_hits if _bm25_allowed(h)]
            bm25_hits = _merge_bm25_channels(
                bm25_hits, _extra_hits, rrf_k=profile.rrf_k, limit=recall_n)
        if bm25_hits:                        # 主池融合:与基线逐字节同构(同深度、同权重)
            fusion_started = time.perf_counter()
            rrf_scores, fusion_channels = rrf_diagnostics(
                dense_docs_for_fusion, bm25_hits,
                k=profile.rrf_k,
                dense_weight=profile.dense_rrf_weight,
                bm25_weight=profile.bm25_rrf_weight,
            )
            if profile.candidate_pool_strategy == "baseline_rrf20":
                # Keep the audited production baseline exactly as it was.
                docs, metas, dists = rrf_fuse(
                    docs, metas, dists, bm25_hits,
                    top_n=recall_n, bm25_only_dist=th["rescue"],
                    k=profile.rrf_k,
                    dense_weight=profile.dense_rrf_weight,
                    bm25_weight=profile.bm25_rrf_weight,
                )
            trace.latency_by_stage["fusion"] = round(
                (time.perf_counter() - fusion_started) * 1000, 3
            )

    if profile.candidate_pool_strategy != "baseline_rrf20":
        # Stage-C alternatives alter candidate *membership* only.  Apply the
        # selected strategy even if the BM25 channel is empty; otherwise the
        # same explicit profile would have query-dependent semantics.
        fusion_started = time.perf_counter()
        from rag_candidate_pool import ChannelCandidate, select_candidate_pool

        dense_channel = []
        evidence_by_id = {}
        for rank, (document, meta, distance) in enumerate(
            zip(
                dense_docs_for_fusion,
                dense_metas_for_fusion,
                dense_dists_for_fusion,
            ),
            start=1,
        ):
            # Use the same ingest-compatible identity exposed by the fusion
            # trace/qrels.  This also lets a dense hit and its BM25 copy merge
            # when Chroma does not echo its ID in metadata.
            chunk_id = stable_chunk_id(document, meta)
            dense_channel.append(ChannelCandidate(
                chunk_id=chunk_id,
                source=str((meta or {}).get("source") or ""),
                rank=rank,
                distance=float(distance),
            ))
            evidence_by_id[chunk_id] = (document, meta, float(distance))

        bm25_channel = []
        for rank, (document, meta, score) in enumerate(bm25_hits, start=1):
            chunk_id = stable_chunk_id(document, meta)
            bm25_channel.append(ChannelCandidate(
                chunk_id=chunk_id,
                source=str((meta or {}).get("source") or ""),
                rank=rank,
                score=float(score),
            ))
            evidence_by_id.setdefault(
                chunk_id, (document, meta, float(th["rescue"])),
            )

        selected_pool = select_candidate_pool(
            dense_channel,
            bm25_channel,
            strategy=profile.candidate_pool_strategy,
            pool_size=recall_n,
            max_per_source=4,
            exclusive_per_channel=profile.exclusive_per_channel,
            rrf_k=profile.rrf_k,
        )
        selected_evidence = [
            evidence_by_id[item.chunk_id] for item in selected_pool
            if item.chunk_id in evidence_by_id
        ]
        docs = [item[0] for item in selected_evidence]
        metas = [item[1] for item in selected_evidence]
        dists = [item[2] for item in selected_evidence]
        trace.latency_by_stage["fusion"] = round(
            (time.perf_counter() - fusion_started) * 1000, 3
        )
    trace.latency_by_stage["bm25"] = round(
        (time.perf_counter() - bm25_started) * 1000, 3
    )
    trace.bm25_candidates = candidate_traces(
        [item[0] for item in bm25_hits],
        [item[1] for item in bm25_hits],
        bm25_scores=[item[2] for item in bm25_hits],
        channels={item[0]: ["bm25"] for item in bm25_hits},
    )
    if "fusion" not in trace.latency_by_stage:
        trace.latency_by_stage["fusion"] = 0.0
    trace.fusion_candidates = candidate_traces(
        docs, metas, dense_distances=dists,
        rrf_scores=rrf_scores, channels=fusion_channels,
    )
    # [翻译通道] 中文 query → 英文 query(报告 §13.2),缓存命中零成本。默认关。
    # 只在配额有英文候选时才值得翻(纯中文问题的池子里没有英文块,翻了也没人用)。
    _en_q = None
    if _quota_hits:
        from rag_translate import english_query_for, translate_enabled
        # 省延迟旋钮 RAG_TRANSLATE_SKIP_IF_EN(**默认关**,实测后定的):对已含英文术语的
        # query 跳过翻译桥。看似白赚——xling_term 子域本来就 100%,跨语集实测开关后
        # 逐子域**一模一样**(68%/78%),p95 还降 3s(14.8→11.8)。但中文 held-out 从
        # 48.1 掉回 46.2(丢 1 题):那些"中文问句夹英文术语"的求职问题也在吃桥的红利。
        # 中文主业务优先是家法,且延迟本来就超预算、省这 3s 换不来上线资格 → 不跳过。
        _skip_en = (os.environ.get("RAG_TRANSLATE_SKIP_IF_EN", "0") == "1"
                    and bool(query_keywords(question)))
        if translate_enabled() and not _skip_en:
            _en_q = english_query_for(question)

    if _quota_hits:
        _lex_en = []
        if bm25_enabled() and os.environ.get("RAG_QUOTA_BM25", "0") == "1":  # 报告 §29 列为 V2
            from rag_quota import quota_collection
            # 词法通道优先用英文 query:中文口语概念对英文文档做 BM25 天然零覆盖
            # (报告 §13.1 举的正是这个例子),翻译后才有词面可匹配。
            _lex_en = [(d, m, None) for d, m, _s in bm25_search(
                _en_q or question, max(5, len(_quota_hits)),
                collection_name=quota_collection())]
        # 英文保险池**整段追加**在主池之后:主池的成员与顺序完全不动(零扰动),
        # 英文要抢名次必须在精排分上真赢过中文——竞争发生在精排,而不是发生在"谁能进池"。
        docs, metas, dists, _prov = append_quota_pool(
            docs, metas, dists, _quota_hits, _lex_en, fallback_dist=th["rescue"])
        docs, metas, dists = cap_per_source(docs, metas, dists)   # 报告 §12,默认关
    _pool = pool_stats(metas, dists, _prov, docs) if _quota_hits else None

    # 交叉编码器精排**整个粗召回池**（保留全序，不在此截断）；失败自动降级为截断。
    # Bug 修复（2026-07-04）：原先此处直接 rerank 到 top_k，使下方 audience 路由只能
    # 在已截断的 top_k 内重排——若目标 career 文件被 rerank 挤出 top_k，路由无法救回
    # （ca03「算法岗怎么转」即此症：目标在完整池 rank #1-2，却被提前截掉）。改为在完整
    # 精排池上先路由、最后统一截 top_k。Evidence Gate 在这些确定性
    # 排序全部完成后读取最终 top1 的原始 cross-encoder 分，避免证据身份错位。
    # Round 10 语体桥(RAG_RERANK_BRIDGE,默认关):给 reranker 递合成问题作同语体评分参照。
    # 注意它会影响 rerank_top → 间接影响证据门控,负样本回归是 A/B 硬门槛。
    _synth_map = None
    bridge_active = (os.environ.get("RAG_RERANK_BRIDGE", "0") == "1"
                     if profile.enable_rerank_bridge is None
                     else bool(profile.enable_rerank_bridge))
    if bridge_active:
        from rag_doc2query import build_synth_map
        _synth_map = build_synth_map(emb, recall_n, client)
    # 概念别名桥(默认关):给英文候选递一份**中文别名**作文档侧代表,让 cross-encoder
    # 打 (中文 query, 中文别名) 这一对并取 max —— 与语体桥/翻译桥同一机制,
    # 桥的对象换成"概念表达"。针对 xling_no_term(纯中文概念提问)这一翻译救不动的子域:
    # 缺的不是语言表层转换,是"中文说法 ↔ 英文学术术语"的映射(指导文档 §15/§16)。
    # 注意:别名只参与**排序**,喂给 LLM 的证据仍是英文原文块,引文也只引原文。
    from rag_alias import (alias_discount, alias_enabled, alias_lift_gate,
                           alias_map, alias_scope)
    _alias_map = None
    alias_active = (alias_enabled() if profile.enable_alias is None
                    else bool(profile.enable_alias))
    if alias_active:
        _am = alias_map()
        if _am:
            from rag_doc2query import doc_key
            # 作用范围 N(指导 §6):只取英文配额通道的**前 N 名**。配额结果按 dense 距离
            # 排序,前排是"这批英文里最像的",给尾部候选也算别名分只会增加无关竞争与对数。
            _scope = alias_scope()
            _allow = None
            if _scope > 0 and _quota_hits:
                _allow = {h[0] for h in _quota_hits[:_scope]}
            _alias_map = {}
            for _d, _m in zip(docs, metas):
                if _allow is not None and _d not in _allow:
                    continue
                _t = _am.get(doc_key(_d, _m))
                if _t:
                    _alias_map[doc_key(_d, _m)] = _t
            _alias_map = _alias_map or None
    # 跨语桥:只给**英文/混排候选**补一对 (英文 query, doc) 评分并取 max。
    # 为什么只补这些:中文候选用中文 query 打分本来就在同语条件下,补英文 query 纯属增噪+加钱;
    # 英文候选才是被跨语惩罚压住的那批。开销 = 英文候选数条 pair(并入同一次 predict 批)。
    _alt_idx = None
    if _en_q:
        from rag_lang import is_quota_lang
        _alt_idx = [i for i, m in enumerate(metas)
                    if is_quota_lang(((m or {}).get("document_language") or ""))
                    or (m or {}).get("source_type") == "paper"]
    _pre_docs = list(docs)     # 初排名次(送进精排前的顺序),供下方排名融合用
    _alias_diag: dict = {}
    _pre_metas = list(metas)   # 与 _alias_diag 的下标对齐(rerank 会重排),供 False Winner 审计
    rerank_started = time.perf_counter()
    rerank_pair_docs = _rerank_pair_documents(
        docs, metas, use_breadcrumb=profile.reranker_use_breadcrumb,
        prefix_mode=profile.reranker_prefix_mode,
    )
    rerank_input_count = len(docs)
    rerank_runtime = {
        "requested": profile.reranker_model,
        "actual": "",
        "status": "not_run",
        "scored_count": 0,
    }
    # 分工精排(默认关):英文候选下标与跨语桥同一识别逻辑;RAG_RERANK_EN_ONNX_DIR 设置时生效
    _en_bridge_idx = None
    if os.environ.get("RAG_RERANK_EN_ONNX_DIR", "").strip():
        from rag_lang import is_quota_lang as _iql
        _en_bridge_idx = [i for i, m in enumerate(metas)
                          if _iql(((m or {}).get("document_language") or ""))
                          or (m or {}).get("source_type") == "paper"]
    docs, metas, dists, rerank_scores = rerank(question, docs, metas, dists, recall_n,
                                               synth_map=_synth_map,
                                               alt_query=_en_q, alt_idx=_alt_idx,
                                               alias_map=_alias_map,
                                               alias_discount=alias_discount(),
                                               alias_lift_gate=alias_lift_gate(),
                                               diag=_alias_diag,
                                               model_name=profile.reranker_model,
                                               pair_docs=rerank_pair_docs,
                                               runtime_diag=rerank_runtime,
                                               en_bridge_idx=_en_bridge_idx)
    # 答案含有性重排(默认关):精排打的是话题相关度,在同源邻近小节之间没有分辨率。
    # 判据按"这一块含不含答案"给 0-3 档,同档内仍用精排分打破平局——没有可调阈值,
    # 因此没有可以过拟合到开发集的东西。裁判不可用时整段退化为空操作。
    _answerability_diag = {"applied": False, "reason": "disabled"}
    if _answerability_enabled():
        try:
            from rag_answerability import mode as _answerability_mode
            from rag_answerability import rerank_by_mode
            _before = list(docs)
            _stats: dict = {}
            docs, metas, dists, rerank_scores = rerank_by_mode(
                _jq, docs, metas, dists, rerank_scores,
                depth=_answerability_depth(),
                early_exit=_answerability_early_exit(), stats=_stats,
                tiebreak_ties=_answerability_tiebreak())
            _answerability_diag = {
                "applied": bool(_stats.get("applied", _stats.get("graded"))),
                "reason": _stats.get("reason", "ok"),
                "mode": _stats.get("mode", _answerability_mode()),
                "source": _stats.get("source"),
                "depth": _answerability_depth(),
                "calls": _stats.get("calls", 0),
                "graded": _stats.get("graded", 0),
                "grades": _stats.get("grades", {}),
                "tiebreak_candidates": _stats.get("tiebreak_candidates", 0),
                "tiebreak_moved": _stats.get("tiebreak_moved"),
                # relation 必须与 grade 一起带出:三态策略里 not_established 在任何
                # 档位都是拒答档,下游只拿 grade 就等于把三态偷偷退回两态。
                "relations": _stats.get("relations", {}),
                "confidences": _stats.get("confidences", {}),
                "student_model_sha256": _stats.get("model_sha256"),
                "fallback_used": bool(_stats.get("fallback_used")),
                "fallback_available": _stats.get("fallback_available"),
                "top1_changed": bool(_before and docs and _before[0] != docs[0]),
            }
        except Exception as _exc:
            _answerability_diag = {"applied": False,
                                   "reason": type(_exc).__name__}
    _consensus_diag = {"applied": False, "reason": "disabled"}
    if profile.consensus_guard_top_k:
        from rag_candidate_pool import protect_multichannel_consensus
        docs, metas, dists, rerank_scores, _consensus_diag = protect_multichannel_consensus(
            docs, metas, dists, rerank_scores,
            dense_chunk_ids=[item.chunk_id for item in trace.dense_candidates],
            bm25_chunk_ids=[item.chunk_id for item in trace.bm25_candidates],
            top_k=profile.consensus_guard_top_k,
            margin=profile.consensus_guard_margin,
        )
    # Third-party/test rerank callables from the compatibility period may not
    # populate ``runtime_diag``.  Infer success only when an aligned score list
    # is present; otherwise make the failed/disabled state explicit in trace.
    if rerank_runtime.get("status") == "not_run":
        if rerank_scores and len(rerank_scores) == len(docs):
            rerank_runtime.update({
                "actual": profile.reranker_model,
                "status": "ok",
                "scored_count": rerank_input_count,
            })
        elif not rerank_enabled():
            rerank_runtime["status"] = "disabled"
        elif rerank_input_count:
            rerank_runtime["status"] = "missing_scores"
    trace.reranker_requested = str(rerank_runtime.get("requested") or "")
    trace.reranker_actual = str(rerank_runtime.get("actual") or "")
    trace.reranker_status = str(rerank_runtime.get("status") or "not_run")
    trace.reranker_scored_count = int(rerank_runtime.get("scored_count") or 0)
    trace.latency_by_stage["rerank"] = round(
        (time.perf_counter() - rerank_started) * 1000, 3
    )
    trace.reranked_candidates = candidate_traces(
        docs, metas, dense_distances=dists, rerank_scores=rerank_scores,
        channels={document: list((_prov or {}).get(document, []))
                  or fusion_channels.get(document, [])
                  for document in docs},
    )
    rerank_score_by_chunk = {
        item.chunk_id: item.rerank_score for item in trace.reranked_candidates
    }
    # [排名融合,默认关] 初排 R@3=84% 而精排后掉到 66%——精排提 Top1 的同时踢掉了另一批
    # 正确候选。两侧名次融合意在同时保住(指导文档 §18 Experiment 2)。
    from rag_fusion import fuse_prerank_rerank, protect_top1_margin, reserve_slot
    docs, metas, dists, rerank_scores = fuse_prerank_rerank(
        _pre_docs, docs, metas, dists, rerank_scores)
    # English False Winner 保护:英文险胜中文时让中文上位(默认关,margin=0)。
    # 放在保底之前:先定 Top1 归属,再谈 Top3 的保底槽位,两者互不干扰。
    docs, metas, dists, rerank_scores = protect_top1_margin(
        docs, metas, dists, rerank_scores)
    if os.environ.get("RAG_RESERVE_SLOT", "").strip():
        # 保底槽位:配额通道的冠军(dense 最近的英文候选)进前 N 位。默认关。
        docs, metas, dists, rerank_scores = reserve_slot(
            docs, metas, dists, rerank_scores,
            champion=(_quota_hits[0][0] if _quota_hits else None))

    # Round 4：元数据/文件名感知路由——query 命中人群意图时，在同质 career 路径文件之间校正排序
    # （向量/BM25/rerank 都分不开「算法岗 vs 后端 vs 零基础」，靠文件名编码的人群精确区分）。
    # 在**完整精排池**上校正，再统一截 top_k（喂 LLM 合成 + 评测用顺序）。
    from rag_route import apply_audience_routing
    final_started = time.perf_counter()
    docs, metas, dists = apply_audience_routing(question, docs, metas, dists)
    # Evidence Gate must score the candidate that the answer path will actually
    # expose as Top-1.  Rank fusion, language protection and audience routing
    # can all move a lower cross-encoder candidate above the raw reranker
    # winner.  Gating before those deterministic rankers mixed two evidence
    # identities (observed on pp23/h9ca02): the raw winner's score decided
    # whether a different final chunk was allowed into the answer.
    final_pool_scores = [
        rerank_score_by_chunk.get(stable_chunk_id(document, meta))
        for document, meta in zip(docs, metas)
    ]
    final_pool_docs = list(docs)     # margin 的竞争者取自**截断前**的完整精排池
    docs, metas, dists = docs[:top_k], metas[:top_k], dists[:top_k]
    final_scores = final_pool_scores[:top_k]
    trace.final_candidates = candidate_traces(
        docs, metas, dense_distances=dists, rerank_scores=final_scores,
    )
    trace.final_top1_chunk_id = (
        trace.final_candidates[0].chunk_id if trace.final_candidates else ""
    )
    # 结构块不作数的第二半:门的判据换成**名次最靠前的实心块**,而不是排第一的块。
    # v2aneg-015 的排第一是标题块(精排 0.9712),换锚点后判据变成 0.8219 < 0.85 → 拒答;
    # 而 v2a-ca-016 / v2a-llm-001 这类"答案就在带小节标题的正文块里"的题,锚点仍是第 1 名
    # (它们本来就是实心块),逐字节不变。判据关闭时 _anchor 恒为 0,同样逐字节不变。
    _anchor = evidence_anchor(docs, _struct_tau)
    if _anchor is None:              # 全是结构块 → 退回原行为
        _anchor = 0 if docs else None
    trace.gate_candidate_chunk_id = (
        trace.final_candidates[_anchor].chunk_id
        if _anchor is not None and _anchor < len(trace.final_candidates) else ""
    )
    trace.gate_alignment = (_anchor == 0) if trace.gate_candidate_chunk_id else None
    trace.latency_by_stage["finalize"] = round(
        (time.perf_counter() - final_started) * 1000, 3
    )

    rerank_top = (
        float(final_scores[_anchor])
        if _anchor is not None and _anchor < len(final_scores)
        and final_scores[_anchor] is not None else None
    )
    # 竞争者同样排除结构块:margin 的语义是"它比次名明显更好",而一块目录不是
    # 有意义的次名——把它算进来会让 margin 变负,把口语第二通道整条打死。
    competing_scores = [
        float(score) for index, score in enumerate(final_pool_scores)
        if score is not None and index != _anchor
        and not is_structural(final_pool_docs[index], _struct_tau)
    ]
    rerank_margin = (
        rerank_top - max(competing_scores)
        if rerank_top is not None and competing_scores else None
    )
    bm25_best_rank = next(
        (item.rank for item in trace.bm25_candidates
         if item.chunk_id == trace.gate_candidate_chunk_id),
        None,
    )

    # Dense/lexical evidence remains a query-level support signal (``best`` is
    # deliberately the pool minimum used by the existing calibrated rule).
    # The candidate-specific signal is the final Top-1 cross-encoder score.
    # This is the minimal identity fix: aligned baseline queries retain their
    # previous decision, while post-rank candidates can no longer borrow the
    # raw reranker winner's score.
    in_kb = _evidence_gate(vector_in_kb, rerank_top, best, th["strong"])
    # 英文证据门(分工桥配套,默认关):主门按向量距离拒答,但最终 Top1 是**英文候选**且其
    # (含分工桥的)精排分 ≥ τ 时接受。τ=0.80 标定(第十二轮,50 正例+60 负样本+16 错排):
    # 正确命中簇 0.805~1.0 / 错误文件簇 ≤0.735 / 负样本英文 top1 上界 0.401——
    # 0.80 与两侧各留 ≥0.07/0.40 余量,新放行 19 正确 0 错误。宁拒不编:分数不够宁走 fallback。
    _en_accept = False
    _en_gate_raw = (os.environ.get("RAG_EN_GATE_MIN", "") or "").strip()
    if (not in_kb) and _en_gate_raw and metas and rerank_top is not None:
        try:
            _en_tau = float(_en_gate_raw)
        except ValueError:
            _en_tau = None
        if _en_tau is not None and rerank_top >= _en_tau:
            from rag_lang import is_quota_lang as _iql_gate
            # 判据块与语言判定必须是**同一块**:rerank_top 已换成锚点的分数,
            # 这里再读 metas[0] 就会拿 A 的分数配 B 的语言(正是本轮要修的身份混用)。
            _m0 = (metas[_anchor] if _anchor is not None and _anchor < len(metas)
                   else {}) or {}
            if (_iql_gate((_m0.get("document_language") or ""))
                    or _m0.get("source_type") == "paper"):
                in_kb = True
                _en_accept = True
    # 答案含量门(默认关):主门按向量距离拒答,但可答性裁判把门要判的**那一块**定为
    # grade=3(含答案)时接受。与英文门/口语门同形状,但证据更硬:那两条走的是精排分
    # 阈值(τ 是标定出来的数),这条走的是"这一块含不含这道题的答案"的定档,复用已冻结的
    # MIN_GRADE_TO_ACT=3,没有新阈值可调。按 _anchor 取档而不是按 Top1 取档:门评的
    # 是哪一块,就必须用哪一块的档,否则又是拿 A 的分配 B 的身份。
    _ans_accept = False
    if ((not in_kb) and _answerability_gate() and _anchor is not None
            and (_answerability_diag or {}).get("source") == "teacher"):
        from rag_answerability import action as _ans_action
        _grades = (_answerability_diag or {}).get("grades") or {}
        _relations = (_answerability_diag or {}).get("relations") or {}
        _grade = _grades.get(_anchor)
        _relation = _relations.get(_anchor)
        # 走既有的 action() 合同,不是 grade 阈值:not_established 在任何档位都是拒答档,
        # 只看 grade 等于把三态退回两态,正是本模块存在的理由。缺任一字段一律不放行。
        if (_grade is not None and _relation
                and _ans_action({"grade": _grade, "relation": _relation}) != "abstain"):
            _votes = _answerability_gate_votes()
            if _votes <= 1:
                in_kb = True
                _ans_accept = True
            else:
                # 单票要放行才追加确认票;拒答不升级。故障票计为拒答票。
                from rag_answerability import confirm_action
                try:
                    _panel = confirm_action(
                        _jq, docs[_anchor],
                        {"grade": _grade, "relation": _relation}, votes=_votes)
                except Exception:
                    _panel = {"action": "abstain", "votes": [], "unanimous": False}
                _answerability_diag["gate_panel"] = _panel
                if _panel["action"] != "abstain":
                    in_kb = True
                    _ans_accept = True
    # 口语证据门(默认关):主门按向量距离拒答,但精排对最终 Top1 既**高置信**又与次名
    # **拉开差距**时接受。口语问法与自己金标的距离天然比正式问法远(cos 0.56~0.62 vs
    # 0.66),所以用正式问法标定出来的 0.73 阈值会按构造误拒它们——Dev-New 实测:
    # 正确 Top1 的过门率 standard 82% / natural 40% / implicit_oral 30%。
    # τ=0.87 标定(80 正例 + 19 条逐条裁决的口语负例):当前被拒负例的精排分上界是
    # 0.863,取 0.87 留 0.007 余量,新增负例误纳 0。margin≥0.10 是第二个条件而不是
    # 装饰:逐条核对 τ 单独放行的 5 道错 Top1,其中 4 道都是"高分但与次名贴身"
    # (精排在一堆近重复里随手挑了一个),margin 门把它们全挡住,只留下一道经核对
    # 属于未标注等价证据的。宁拒不编:两个条件缺一不可,不够就走 fallback。
    _rr_gate_raw = (os.environ.get("RAG_COLLOQUIAL_GATE_MIN", "") or "").strip()
    _rr_accept = (not in_kb) and colloquial_gate_accept(rerank_top, rerank_margin)
    if _rr_accept:
        in_kb = True
    trace.gate_decision = in_kb
    trace.gate_features = {
        "best_dense_distance": best,
        "strong_threshold": th["strong"],
        "rescue_threshold": th["rescue"],
        "weak_threshold": th["weak"],
        "lexical_rescued": lexical_rescued,
        "vector_in_kb": vector_in_kb,
        "rerank_top": rerank_top,
        "bm25_best_rank": bm25_best_rank,
        "rerank_gate_min": float(os.environ.get("RAG_RERANK_GATE_MIN", "0.85")),
        "rerank_margin": rerank_margin,
        "colloquial_gate_min": _rr_gate_raw or None,
        "colloquial_gate_accepted": _rr_accept,
        "answerability_gate_accepted": _ans_accept,
        # 结构块判据的逐题留痕:换没换锚点、换到第几名、被跳过的那块结构度多少。
        # 没有这几项就只能从"分数怎么变了"倒推,而这正是本轮抓到 v2aneg-015 的方式。
        "structural_evidence_max": _struct_tau,
        "gate_anchor_rank": (_anchor + 1) if _anchor is not None else None,
        "gate_anchor_structural": (
            None if _struct_tau is None or _anchor is None
            else round(structural_fraction(docs[_anchor]), 4)
            if _anchor < len(docs) else None
        ),
        "skipped_structural_top1": (
            None if _struct_tau is None or not docs
            else round(structural_fraction(docs[0]), 4)
            if _anchor not in (None, 0) else None
        ),
        "consensus_guard": _consensus_diag,
        "answerability_rerank": _answerability_diag,
    }

    # [多-agent ⑤] CRAG recovery is evaluated only after the actual answer
    # candidate has failed the Gate.  A recovered query traverses this same
    # final-candidate Gate at depth=1, preserving the recursion guard.
    if (not in_kb) and _crag_depth == 0 and os.environ.get("RAG_CRAG", "0") == "1":
        _recovered = _crag_recover(
            question, top_k, source_types=source_types,
            exclude_source_types=exclude_source_types,
            metadata_filters=metadata_filters,
            allow_paper_route=allow_paper_route,
            retrieval_profile=profile,
            _shadow_internal=True,
        )
        if _recovered is not None and _recovered.get("in_kb"):
            _recovered["crag_recovered"] = True
            return _finish(_recovered)

    # The paper fallback also compares against the final main-library Top-1,
    # not a raw reranker winner that the answer path would never expose.
    if (not in_kb and allow_paper_route and not source_types
            and not (exclude_source_types - NON_RAG_SOURCE_TYPES)
            and not metadata_filters):
        try:
            from rag_paper_route import maybe_paper_route
            _paper = maybe_paper_route(question, main_in_kb=in_kb, top_k=top_k,
                                       main_rerank_top=rerank_top)
            if _paper is not None:
                return _finish(_paper)
        except Exception:
            pass

    if in_kb:
        if _en_accept or _rr_accept or _ans_accept:
            # 第二通道接受(英文证据 / 口语精排置信 / 答案含量):证据 = 最终排序前 3。
            # 这几条路进来的候选,距离**按构造**都超过 rescue 阈值——英文候选是跨语
            # 距离天然偏大,口语候选是问法导致距离偏大——所以再按距离 cutoff 过滤只会
            # 得到空 chunks,grounded 回答无料可用;按名次取才与"门为什么放行"一致。
            relevant = list(zip(docs, metas))[:3]
            chunks = [d for d, _m in relevant]
            sources = sorted({(m or {}).get("source", "?") for _d, m in relevant})
            matched_by = ("en_evidence" if _en_accept
                          else "answerability" if _ans_accept
                          else "rerank_confidence")
        else:
            cutoff = th["strong"] if best <= th["strong"] else th["rescue"]
            relevant = [(d, m) for d, m, dist in zip(docs, metas, dists) if dist <= cutoff]
            chunks = [d for d, _m in relevant]
            sources = sorted({(m or {}).get("source", "?") for _d, m in relevant})
            matched_by = "vector" if best <= th["strong"] else "lexical_rescue"
    else:
        chunks = [d for d, dist in zip(docs, dists) if dist <= th["weak"]]  # 弱相关背景
        sources = []
        matched_by = None
    trace.effective_hit = bool(in_kb and chunks and trace.final_candidates)
    # 只记录的 shadow(默认关):把不依赖裁判的触发特征落盘,并对触发请求 + 一小部分
    # 随机对照请求异步请裁判。它**结构上够不到响应**——返回 None、吞掉所有异常、
    # 只往有界队列里塞。worker 可能与后续回答生成重叠，所以资源竞争/延迟必须实测；
    # 这里保证的是判据结果不能回流改变本次回答，不是假装后台工作零成本。
    if _answerability_shadow_route == "reference_kb":
        try:
            from rag_shadow_answerability import observe as _shadow_observe
            from traffic_origin import current_traffic_origin
            _shadow_observe(
                _answerability_shadow_question or question,
                trace,
                route=_answerability_shadow_route,
                traffic_origin=current_traffic_origin(),
                retrieval_question=question,
            )
        except Exception:
            pass
    trace.latency_by_stage["total"] = round(
        (time.perf_counter() - total_started) * 1000, 3
    )
    return _finish({"in_kb": in_kb, "chunks": chunks, "sources": sources,
            "matched_by": matched_by, "best": best, "has_dists": bool(dists),
            "rerank_top": rerank_top,  # 最相关 doc 的 cross-encoder 分（证据型门控）
            # 检索后最终排序的原始结果（评测器/后续 rerank·混合检索改造在此重排）
            "docs": docs, "metas": metas, "dists": dists,
            "pool": _pool,   # 候选池指标(报告 §26.1);配额关 = None
            "source_filter": {"include": sorted(source_types),
                              "exclude": sorted(exclude_source_types)},
            "retrieval_profile": profile.name,
            "index_fingerprint": trace.index_fingerprint,
            "effective_hit": trace.effective_hit,
            "retrieval_trace": trace.to_dict(),
            # 逐题诊断(指导 §4.2):用于 English False Winner 审计——谁赢了、赢多少、
            # 别名给了多大抬升。评测器据此判断"英文抢位"是集中在固定几题还是弥散的,
            # 以及真命中与假抢位的 alias_lift 分布是否可分离(决定 D2 方向是否值得做)。
            "diag": _rank_diag(docs, metas, final_scores, _pre_metas, _alias_diag)})


def retrieve_with_trace(query: str, query_plan=None, retrieval_profile=None,
                        top_k: int = 5):
    """Run the production retrieval path and return its stage-level trace.

    ``query_plan`` may be omitted to exercise the real production planner.  A
    plan with one semantic-search source is translated to the exact source and
    owner filters used by :mod:`rag_multi_source`.  Structured-only plans
    correctly return an empty trace rather than silently querying the whole
    knowledge base.  Multi-source plans may contain several deterministic
    routes, but must contain at most one vector-search route for this singular
    trace API; full multi-source execution exposes one trace per vector route.
    """
    from rag_retrieval_trace import RetrievalTrace, resolve_retrieval_profile
    from rag_source_policy import PERSONAL_VECTOR_SOURCE_TYPES, INTERNAL_SOURCE_TYPES

    profile = resolve_retrieval_profile(retrieval_profile)
    trace = RetrievalTrace(profile)
    if query_plan is None:
        from rag_query_plan import plan_query
        query_plan = plan_query(query)
    decision = (query_plan.get("decision") if isinstance(query_plan, dict)
                else getattr(query_plan, "decision", "answer"))
    if decision != "answer":
        return trace
    raw_routes = (query_plan.get("routes", []) if isinstance(query_plan, dict)
                  else getattr(query_plan, "routes", []))
    routes = []
    for route in raw_routes:
        source = route.get("source") if isinstance(route, dict) else getattr(route, "source", "")
        operation = (route.get("operation") if isinstance(route, dict)
                     else getattr(route, "operation", ""))
        if operation in {"search", "lookup", "summarize", "explain", "advise"}:
            routes.append(source)
    vector_sources = [source for source in routes if source in {
        "application_experience", "project_memory", "resume_rules", "reference_kb",
    }]
    vector_sources = list(dict.fromkeys(vector_sources))
    if not vector_sources:
        return trace
    if len(vector_sources) > 1:
        raise ValueError(
            "retrieve_with_trace accepts at most one vector route; "
            "use execute_plan(...).retrieval_traces for multi-source plans"
        )
    source = vector_sources[0]
    kwargs = {"allow_paper_route": False}
    if source == "reference_kb":
        kwargs.update(
            exclude_source_types=set(PERSONAL_VECTOR_SOURCE_TYPES | INTERNAL_SOURCE_TYPES),
            metadata_filters={"owner_scope": {"curated"}},
        )
    elif source == "application_experience":
        kwargs.update(source_types={"experience"},
                      metadata_filters={"owner_scope": {"personal"}})
    elif source == "project_memory":
        kwargs.update(source_types={"project_context"},
                      metadata_filters={"owner_scope": {"personal"}})
    elif source == "resume_rules":
        kwargs.update(source_types={"resume_rule", "resume"},
                      metadata_filters={"owner_scope": {"personal"}})
    _retrieve_and_classify(
        query, top_k, retrieval_profile=retrieval_profile, _trace=trace,
        _answerability_shadow_route=(
            "reference_kb" if source == "reference_kb" else source
        ),
        **kwargs,
    )
    return trace


def _rank_diag(docs, metas, scores, pre_metas, alias_diag) -> dict:
    """汇总 Top1 的语言/来源/领先幅度,以及本次别名抬升的分布。"""
    from rag_lang import is_quota_lang

    def _is_en(m):
        return bool(is_quota_lang(((m or {}).get("document_language") or ""))
                    or (m or {}).get("source_type") == "paper")

    top1 = (metas[0] if metas else None)
    margin = None
    if (scores and len(scores) >= 2
            and scores[0] is not None and scores[1] is not None):
        margin = round(float(scores[0]) - float(scores[1]), 4)
    applied = [v for v in alias_diag.values() if v.get("applied")]
    # alias_diag 的键是**精排前**的下标 → 用 pre_metas 判断被抬升的是不是英文候选
    en_lifts = [v["lift"] for i, v in alias_diag.items()
                if v.get("applied") and i < len(pre_metas) and _is_en(pre_metas[i])]
    return {
        "top1_source": (top1 or {}).get("source", ""),
        "top1_is_english": _is_en(top1),
        "top1_margin": margin,
        "alias_pairs": len(alias_diag),
        "alias_applied": len(applied),
        "alias_lift_max": (round(max(en_lifts), 4) if en_lifts else None),
    }


def _crag_query_rewrites(question: str) -> list:
    """[⑤ CRAG] 为弱证据 query 产候选改写（纯代码,不依赖 LLM key）:书面改写(若开) + 关键词化。"""
    from rag_hyde import rewrite_query
    cands = []
    r = rewrite_query(question)
    if r and r != question:
        cands.append(r)
    kws = query_keywords(question)
    if kws:
        cands.append(" ".join(kws))
    seen, out = set(), []
    for q in cands:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:2]


def _crag_recover(question: str, top_k: int, *, source_types: set[str] | None = None,
                  exclude_source_types: set[str] | None = None,
                  metadata_filters: dict[str, set[str]] | None = None,
                  allow_paper_route: bool = True,
                  retrieval_profile=None,
                  _shadow_internal: bool = True):
    """[⑤ CRAG] 对弱证据 query 条件恢复:逐候选重检、重过门控,命中即返;depth=1 防双改写/无限递归。"""
    for q in _crag_query_rewrites(question):
        if (not source_types and not exclude_source_types and not metadata_filters
                and allow_paper_route):
            # 保留旧调用形状，兼容外部桩函数和历史扩展点。
            if retrieval_profile is None:
                res = _retrieve_and_classify(q, top_k, _crag_depth=1)
            else:
                res = _retrieve_and_classify(
                    q, top_k, _crag_depth=1,
                    retrieval_profile=retrieval_profile,
                    _shadow_internal=_shadow_internal,
                )
        else:
            kwargs = {
                "source_types": source_types,
                "exclude_source_types": exclude_source_types,
                "metadata_filters": metadata_filters,
                "allow_paper_route": allow_paper_route,
            }
            if retrieval_profile is not None:
                kwargs.update({
                    "retrieval_profile": retrieval_profile,
                    "_shadow_internal": _shadow_internal,
                })
            res = _retrieve_and_classify(q, top_k, _crag_depth=1, **kwargs)
        if res.get("in_kb"):
            return res
    return None


def _plan_meta(plan, *, planning_ms: float, retrieval_ms: float = 0.0, **extra) -> dict:
    """SSE/同步问答共用的可审计路由元数据。"""
    factors = getattr(plan, "confidence_factors", {}) or {}
    payload = {
        "service_mode": plan.intent_frame.service_mode,
        "interaction_kind": plan.intent_frame.interaction_kind,
        "turn_relation": plan.intent_frame.turn_relation,
        "relation_target_turn_id": plan.intent_frame.relation_target_turn_id,
        "action_request": plan.intent_frame.action_request,
        "requested_action": plan.intent_frame.requested_action,
        "routing_assurance": plan.intent_frame.routing_assurance,
        "decision_reasons": plan.intent_frame.decision_reasons,
        "context_resolution": plan.intent_frame.context_resolution,
        "planner_mode": plan.planner_mode,
        "resolver_mode": plan.resolver_mode,
        "decision": plan.decision,
        "clarification": plan.clarification,
        "intent_frame": plan.intent_frame.to_dict(),
        "routes": [{"source": r.source, "operation": r.operation,
                    "required": r.required, "filters": r.filters,
                    "depends_on": list(r.depends_on), "bindings": r.bindings,
                    "route_id": r.route_id, "source_role": r.source_role,
                    "selection_reason": r.selection_reason, "locked": r.locked}
                   for r in plan.routes],
        "candidate_routes": plan.candidate_routes,
        "confidence_factors": plan.confidence_factors,
        "route_reason": plan.reason,
        "planner_engine": getattr(plan, "planner_engine", "rules"),
        "planner_version": getattr(plan, "planner_version", "legacy-rule-v1"),
        "schema_version": getattr(plan, "schema_version", "query-plan-v1"),
        "route_model": getattr(plan, "route_model", ""),
        "repair_used": bool(getattr(plan, "repair_used", False)),
        "fallback_reason": getattr(plan, "fallback_reason", ""),
        # 规划器运行数据是可选扩展字段；旧路由没有这些值时
        # 保持空/0，不改变旧客户端对 meta 事件的解析。
        "planner_queue_ms": factors.get("planner_queue_ms", 0.0),
        "planner_provider_ms": factors.get("planner_provider_ms", 0.0),
        "planner_wall_ms": factors.get("planner_wall_ms", 0.0),
        "planner_deadline_ms": factors.get("planner_deadline_ms", 0.0),
        "planner_cache_hit": bool(factors.get("planner_cache_hit", False)),
        "planner_timeout_stage": factors.get("planner_timeout_stage", ""),
        "planner_late_response": bool(factors.get("planner_late_response", False)),
        "planner_circuit_state": factors.get("planner_circuit_state", ""),
        "prompt_tokens": factors.get("prompt_tokens"),
        "completion_tokens": factors.get("completion_tokens"),
        "structured_output_mode": factors.get("structured_output_mode", ""),
        "route_gateway": factors.get("route_gateway", ""),
        "route_reasoning_effort": factors.get("route_reasoning_effort", ""),
        "json_capability": factors.get("json_capability", ""),
        "latency_ms": {"planning": round(planning_ms, 1),
                       "retrieval": round(retrieval_ms, 1)},
    }
    payload.update(extra)
    return payload


def _multi_mode(plan, execution) -> str:
    """保留旧 wire-format mode，但只由统一服务模式派生，不再重复猜测意图。"""
    sources = {r.source for r in plan.routes}
    service_mode = plan.intent_frame.service_mode
    if service_mode == "guide":
        return "system_help"
    if service_mode == "diagnose":
        return "system_diagnostic"
    if service_mode == "recall":
        return "state_grounded" if sources <= {"application_state", "profile_plan"} else "personal_grounded"
    if service_mode == "advise":
        return "multi_source"
    if sources == {"paper_kb"}:
        return "paper_grounded" if execution.evidence else "general_fallback"
    return "kb_grounded" if execution.in_kb else "general_fallback"


def _allows_general_fallback(plan) -> bool:
    if plan is None:
        # Unknown routing state must fail closed. Otherwise a planner/import
        # exception on a personal question could be turned into an apparently
        # authoritative generic answer about the user.
        return False
    try:
        from rag_route_registry import SERVICE_REGISTRY
        return bool(SERVICE_REGISTRY[plan.intent_frame.service_mode]["allow_general_fallback"])
    except Exception:
        return False


def _execute_query_plan(plan, question: str, top_k: int):
    from rag_multi_source import execute_plan
    from rag_paper_route import retrieve_papers_explicit
    return execute_plan(plan, question, top_k, retrieve=_retrieve_and_classify,
                        retrieve_papers=retrieve_papers_explicit)


def _stage_event(stage: str, status: str, label: str, *, detail: str = "",
                 elapsed_ms: float | None = None, **extra) -> dict:
    """Build a small, user-safe progress event for the streaming UI.

    These events describe work that has actually started or finished.  They do
    not expose prompts, private evidence text or a made-up percentage.  Older
    clients can ignore the new ``stage`` event type and continue consuming the
    existing meta/delta/done sequence.
    """
    event = {
        "type": "stage",
        "stage": stage,
        "status": status,
        "label": label,
    }
    if detail:
        event["detail"] = detail
    if elapsed_ms is not None:
        event["elapsed_ms"] = round(float(elapsed_ms), 1)
    event.update({key: value for key, value in extra.items() if value is not None})
    return event


def gated_query_stream(question: str, top_k: int = 5, *,
                       context: list[str] | None = None,
                       conversation_id: str = ""):
    """多意图流式问答：规则/按需规划 → 分源检索 → 可审计合成。

    事件保持兼容，并在 meta 中追加 planner_mode/routes/source_status/coverage/freshness/latency_ms。
    事件形如：{"type":"meta", in_kb, mode, sources, matched_by, best_distance, ...}
              {"type":"delta", "text": token}
              {"type":"done"}
    """
    import time
    plan = None
    active_stage = "planning"
    request_started = time.perf_counter()
    try:
        from rag_query_plan import plan_is_reference_only, plan_query
        from rag_multi_source import (evidence_fallback, multi_source_messages,
                                      product_help_messages, REFERENCE_EXCLUDES)

        yield _stage_event(
            "planning", "active", "正在理解问题并规划检索路径",
            detail="识别回答对象、操作和所需数据源",
        )
        t0 = time.perf_counter()
        plan = plan_query(
            question, context=context, conversation_id=conversation_id,
        )
        planning_ms = (time.perf_counter() - t0) * 1000
        route_names = [f"{route.source}.{route.operation}" for route in plan.routes]
        yield _stage_event(
            "planning", "completed", "检索路径规划完成",
            detail=(f"已确定 {len(route_names)} 条只读路径" if route_names
                    else "需要先确认问题对象"),
            elapsed_ms=planning_ms, routes=route_names,
        )

        if plan.decision == "clarify":
            active_stage = "clarification"
            yield _stage_event(
                "clarification", "completed", "需要补充信息",
                detail="当前问题存在多个可能的回答对象，未读取个人数据",
            )
            meta = _plan_meta(
                plan, planning_ms=planning_ms, in_kb=False, mode="clarification",
                sources=[], matched_by="route_clarification", best_distance=None,
                source_status={}, coverage={}, freshness="",
            )
            yield {"type": "meta", **meta}
            yield {"type": "delta", "text": plan.clarification or "请补充你希望查询的主要对象。"}
            yield {"type": "done"}
            return

        from rag_tools import has_embedding_api_key

        # 单一普通知识问答完全复用既有混合检索、门控和合成路径，只排除易过期个人状态块。
        if plan_is_reference_only(plan):
            if not has_embedding_api_key():
                yield _stage_event(
                    "retrieval", "warning", "专业资料检索不可用",
                    detail="未配置本地向量检索能力，将明确降级为通用回答",
                )
                meta = _plan_meta(
                    plan, planning_ms=planning_ms, in_kb=False, mode="general_fallback",
                    sources=[], matched_by=None, best_distance=None,
                    source_status={"reference_kb": {"status": "unavailable", "count": 0}},
                    coverage={}, freshness="",
                )
                yield {"type": "meta", **meta}
                yield {"type": "delta", "text": FALLBACK_LABEL}
                active_stage = "synthesis"
                synthesis_started = time.perf_counter()
                yield _stage_event(
                    "synthesis", "active", "正在组织通用回答",
                    detail="回答不会冒充知识库证据",
                )
                for tok in _chat_stream(_fallback_messages(question, [])):
                    yield {"type": "delta", "text": tok}
                yield _stage_event(
                    "synthesis", "completed", "回答组织完成",
                    elapsed_ms=(time.perf_counter() - synthesis_started) * 1000,
                )
                yield {"type": "done"}
                return

            active_stage = "retrieval"
            yield _stage_event(
                "retrieval", "active", "正在检索专业资料",
                detail="仅检索已批准的策展知识库",
            )
            tr = time.perf_counter()
            g = _retrieve_and_classify(
                question, top_k, exclude_source_types=REFERENCE_EXCLUDES,
                metadata_filters={"owner_scope": {"curated"}},
                allow_paper_route=False,
                _answerability_shadow_route="reference_kb")
            retrieval_ms = (time.perf_counter() - tr) * 1000
            chunk_count = len(g.get("chunks") or [])
            yield _stage_event(
                "retrieval", "completed", "专业资料检索完成",
                detail=f"获得 {chunk_count} 条候选证据",
                elapsed_ms=retrieval_ms, sources=g.get("sources") or [],
                evidence_count=chunk_count,
            )
            active_stage = "evidence"
            yield _stage_event(
                "evidence", "completed" if g["in_kb"] else "warning",
                "证据校验通过" if g["in_kb"] else "内部证据不足",
                detail=("将严格依据通过门控的资料回答" if g["in_kb"]
                        else "将明确标注为通用知识回答"),
                evidence_count=chunk_count,
            )
            mode = "kb_grounded" if g["in_kb"] else "general_fallback"
            source_status = {"reference_kb": {
                "status": "ok" if g["in_kb"] else "no_evidence",
                "count": len(g.get("chunks") or []), "latency_ms": round(retrieval_ms, 1),
            }}
            meta = _plan_meta(
                plan, planning_ms=planning_ms, retrieval_ms=retrieval_ms,
                in_kb=g["in_kb"], mode=mode,
                sources=g["sources"] if g["in_kb"] else [],
                matched_by=g["matched_by"],
                best_distance=round(g["best"], 4) if g["has_dists"] else None,
                source_status=source_status,
                coverage={"reference_kb": len(g.get("chunks") or [])} if g["in_kb"] else {},
                freshness="",
                retrieval_profile=g.get("retrieval_profile", ""),
                index_fingerprint=g.get("index_fingerprint", ""),
                effective_hit=bool(g.get("effective_hit", False)),
                # 加性字段:生成用的是哪份动作合同,线上可观测(P0 修复配套)
                answer_action=g.get("answer_action", "answer"),
            )
            yield {"type": "meta", **meta}
            if g["in_kb"]:
                msgs = _grounded_messages(question, g["chunks"],
                                          g.get("answer_action", "answer"))
            else:
                yield {"type": "delta", "text": FALLBACK_LABEL}
                msgs = _fallback_messages(question, g["chunks"])
            active_stage = "synthesis"
            synthesis_started = time.perf_counter()
            yield _stage_event(
                "synthesis", "active", "正在组织最终回答",
                detail=("正在引用已校验证据" if g["in_kb"]
                        else "正在生成明确标注的通用回答"),
            )
            emitted = False
            for tok in _chat_stream(msgs):
                emitted = True
                yield {"type": "delta", "text": tok}
            if not emitted and g["in_kb"]:
                yield {"type": "delta", "text": "\n---\n".join(g["chunks"])}
            yield _stage_event(
                "synthesis", "completed", "回答组织完成",
                elapsed_ms=(time.perf_counter() - synthesis_started) * 1000,
            )
            yield {"type": "done"}
            return

        active_stage = "retrieval"
        planned_sources = list(dict.fromkeys(route.source for route in plan.routes))
        yield _stage_event(
            "retrieval", "active", "正在读取所需数据源",
            detail=("并行执行：" + "、".join(planned_sources)
                    if planned_sources else "执行已确认的只读查询"),
            sources=planned_sources,
        )
        tr = time.perf_counter()
        execution = _execute_query_plan(plan, question, top_k)
        retrieval_ms = (time.perf_counter() - tr) * 1000
        evidence_count = sum(
            int((status or {}).get("count") or 0)
            for status in (execution.source_status or {}).values()
        )
        failed_count = len(execution.failed_routes or [])
        yield _stage_event(
            "retrieval", "warning" if failed_count else "completed",
            "多来源读取完成" if not failed_count else "部分来源读取失败",
            detail=(f"获得 {evidence_count} 条证据"
                    + (f"，{failed_count} 条路径失败但其余结果保留" if failed_count else "")),
            elapsed_ms=retrieval_ms, sources=execution.sources,
            source_status=execution.source_status, evidence_count=evidence_count,
        )
        active_stage = "evidence"
        has_usable_answer = bool(execution.evidence or execution.deterministic_answer)
        yield _stage_event(
            "evidence", "completed" if has_usable_answer else "warning",
            "证据校验完成" if has_usable_answer else "未找到可用个人证据",
            detail=("结构化事实和证据将按来源优先级组织" if has_usable_answer
                    else "不会用通用知识替代缺失的个人事实"),
            evidence_count=evidence_count,
        )
        mode = _multi_mode(plan, execution)
        meta = _plan_meta(
            plan, planning_ms=planning_ms, retrieval_ms=retrieval_ms,
            in_kb=execution.in_kb, mode=mode, sources=execution.sources,
            matched_by="multi_route", best_distance=None,
            source_status=execution.source_status, coverage=execution.coverage,
            freshness=execution.freshness,
            resolved_entities=execution.resolved_entities,
            retrieval_profile=execution.retrieval_profile,
            index_fingerprint=execution.index_fingerprint,
            effective_hit=execution.effective_hit,
            answer_action=(execution.answer_action if execution.in_kb
                           else "abstain"),
        )
        yield {"type": "meta", **meta}

        active_stage = "synthesis"
        synthesis_started = time.perf_counter()
        yield _stage_event(
            "synthesis", "active", "正在组织最终回答",
            detail=("针对具体疑问整理功能事实"
                    if plan.intent_frame.service_mode == "guide"
                    else ("整合结构化事实与多来源证据"
                          if execution.evidence else "整理可审计的查询结果")),
        )
        focused_guide = (
            plan.intent_frame.service_mode == "guide"
            and any(route.source == "product_help" for route in plan.routes)
        )
        if focused_guide:
            yield {"type": "delta", "text": (
                execution.deterministic_answer or "当前没有可用的功能说明。"
            )}
            yield _stage_event(
                "synthesis", "completed", "已生成只读操作指引",
                detail="说明来自本地动作能力注册表；未执行任何写入",
                elapsed_ms=(time.perf_counter() - synthesis_started) * 1000,
            )
            yield {"type": "done"}
            return
        if execution.deterministic_answer and not focused_guide:
            yield {"type": "delta", "text": execution.deterministic_answer}

        needs_synthesis = bool(focused_guide or execution.evidence or execution.failed_routes)
        if needs_synthesis:
            if execution.deterministic_answer and not focused_guide:
                yield {"type": "delta", "text": "\n\n### 综合分析\n"}
            emitted = False
            stance = (_CORRECT_PREMISE_RULES
                      if execution.answer_action == "correct_premise" else "")
            messages = (
                product_help_messages(question, plan, execution)
                if focused_guide else multi_source_messages(question, plan, execution,
                                                            stance=stance)
            )
            for tok in _chat_stream(messages):
                emitted = True
                yield {"type": "delta", "text": tok}
            if not emitted:
                fallback = (execution.deterministic_answer if focused_guide
                            else evidence_fallback(execution))
                if not execution.deterministic_answer or fallback != execution.deterministic_answer:
                    yield {"type": "delta", "text": fallback}
                elif focused_guide:
                    yield {"type": "delta", "text": fallback}
        elif not execution.deterministic_answer:
            if _allows_general_fallback(plan):
                yield {"type": "delta", "text": FALLBACK_LABEL}
                emitted = False
                for tok in _chat_stream(_fallback_messages(question, [])):
                    emitted = True
                    yield {"type": "delta", "text": tok}
                if not emitted:
                    yield {"type": "delta", "text": "（当前未配置可用的通用回答模型。）"}
            else:
                yield {"type": "delta", "text": "没有找到可用的个人记录或知识库证据。"}
        yield _stage_event(
            "synthesis", "completed", "回答组织完成",
            elapsed_ms=(time.perf_counter() - synthesis_started) * 1000,
            total_elapsed_ms=(time.perf_counter() - request_started) * 1000,
        )
        yield {"type": "done"}
    except Exception as e:
        yield _stage_event(
            active_stage, "error", "当前阶段执行失败",
            detail="系统已停止该阶段，并按个人事实安全边界处理",
            total_elapsed_ms=(time.perf_counter() - request_started) * 1000,
        )
        can_fallback = _allows_general_fallback(plan)
        error_mode = "general_fallback" if can_fallback else "retrieval_error"
        yield {"type": "meta", "in_kb": False, "mode": error_mode,
               "sources": [], "matched_by": None, "best_distance": None,
               "planner_mode": "fallback", "routes": [], "source_status": {},
               "service_mode": (plan.intent_frame.service_mode if plan else "explain"),
               "coverage": {}, "freshness": "", "latency_ms": {}}
        if can_fallback:
            yield {"type": "delta", "text": FALLBACK_LABEL + f"（检索异常：{e}）"}
        else:
            yield {"type": "delta", "text": f"⚠️ 内部只读查询失败：{e}。未使用通用知识替代个人事实。"}
        yield {"type": "done"}


# 纠偏合同(2026-08-28 用户裁定):问题的前提可能为假,而资料足以反驳它。
# 那种情况的正确行为是**指出前提错误并纠正**,不是沉默——把它归到拒答会惩罚
# 系统利用证据纠正用户,也会把证据门训练成"只要前提不成立就闭嘴"。
def answer_action_from_retrieval(g: dict) -> str:
    """门动作的**唯一派生实现**(2026-08-31,answer-quality-v2 P0 修复)。

    病灶:门在 trace 里正确判出 correct_premise(负例 8/8),但动作没有随
    检索结果交给生成,最终答案只有 2/4 真正纠偏。这不是判据问题,是交接
    问题——所以修的是交接:每个检索结果在 `_finish` 收口处用本函数盖章
    `answer_action`,生成端只认这枚章。

    派生顺序与评测器 answer-quality-v2 使用的完全一致(该评测器现在直接
    import 本函数——同一件事只许有一把尺子):
      1. 不在库内 → abstain;
      2. 共识 panel 的 action(多票路径的最终决定);
      3. 锚点判据的 action() 合同(单票放行路径;grades/relations 容
         int/str 两种键型,gate_anchor_rank 是 1-based);
      4. 都没有 → "answer"(距离门等无判据通道的既有语义,一位不动)。
    """
    if not g.get("in_kb"):
        return "abstain"
    trace = g.get("retrieval_trace") or {}
    gate_features = trace.get("gate_features") or {}
    diag = (gate_features.get("answerability_rerank") or {})
    panel_action = (diag.get("gate_panel") or {}).get("action")
    if panel_action in {"answer", "correct_premise"}:
        return panel_action
    grades = diag.get("grades") or {}
    relations = diag.get("relations") or {}
    anchor_rank = gate_features.get("gate_anchor_rank")
    try:
        anchor_index = max(0, int(anchor_rank) - 1) if anchor_rank is not None else 0
    except (TypeError, ValueError):
        anchor_index = 0
    grade = grades.get(str(anchor_index), grades.get(anchor_index))
    relation = relations.get(str(anchor_index), relations.get(anchor_index))
    if grade is not None and relation:
        from rag_answerability import action
        observed = action({"grade": grade, "relation": relation})
        if observed in {"answer", "correct_premise"}:
            return observed
    return "answer"


_CORRECT_PREMISE_RULES = (
    "本题的前提与资料**相矛盾**。请按以下要求纠偏：\n"
    "1. 第一句明确指出前提不成立。\n"
    "2. 只根据所给资料说明正确的关系，不要引入资料之外的定义、原因、例子或解决方案。\n"
    "3. 复合问题若只有一部分能被资料纠正，对剩余部分明确说资料不足。\n"
    "4. 不要臆测用户想问什么；如需澄清，只提出一个问题。"
)


def _grounded_messages(question: str, chunks: list,
                       answer_action: str = "answer") -> list:
    """构造 grounded 合成消息。

    ``answer_action`` 是门给出的**结构化动作**（answer / correct_premise），
    不是自由文本提示。判据的 ``reason`` 属于审计元数据，**绝不传进来当答案**——
    生成器必须自己读原始证据，否则答案会退化成对裁判一句话的复述。
    """
    context = "\n\n".join(f"[资料{i+1}]\n{c}" for i, c in enumerate(chunks))
    stance = (_CORRECT_PREMISE_RULES + "\n\n"
              if answer_action == "correct_premise" else "")
    return [
        {"role": "system", "content": (
            "你是 OfferClaw 的知识库问答助手。**只能基于下面提供的「资料」回答**，"
            "严禁使用资料之外的知识或常识补充。若资料不足以回答，直接说明资料不足。"
            "回答简洁、分点，末尾用一行列出引用的资料编号。\n\n"
            + stance + context
        )},
        {"role": "user", "content": question},
    ]


def _fallback_messages(question: str, weak_chunks: list) -> list:
    bg = ""
    if weak_chunks:
        bg = ("\n\n以下是知识库里**可能相关但未必准确**的片段，可参考其中与项目相关的部分：\n"
              + "\n\n".join(f"[背景{i+1}]\n{c}" for i, c in enumerate(weak_chunks)))
    return [
        {"role": "system", "content": (
            "你是 OfferClaw 的求职/学习助手。用户的问题在策展知识库里没有直接覆盖。"
            "请结合你的通用知识、以及（若给出）下面的项目背景片段，给出有帮助、准确、简洁的回答；"
            "涉及大模型应用工程师方向时尽量贴合该语境。不要假装这是知识库的权威答案。" + bg
        )},
        {"role": "user", "content": question},
    ]


def gated_query(question: str, top_k: int = 5, *,
                context: list[str] | None = None,
                conversation_id: str = "") -> dict:
    """非流式多意图问答；与 ``gated_query_stream`` 共用查询规划和检索路径。

    - **命中**（强向量 / 词面救回）→ 仅基于 KB 片段合成答案 + 标出处，mode=kb_grounded。
    - **未命中** → 用 LLM 通用知识 + 项目先验 + 弱相关片段生成答案，开头加"非知识库"标注，
      mode=general_fallback。即始终给有用答案，但 KB-grounded 与否清晰区分。

    返回 dict：{query, in_kb, mode, answer, sources, matched_by, best_distance, retrieval_count}
    任何异常都安全降级为 general_fallback（不崩）。

    **门控/检索统一**（修测试报告 §7 残留）：本函数曾内联一套朴素检索 + 旧距离门控，导致
    `/api/query` 与微信 CLI 绕过了 Round 1/2/4 检索增强与 Round 6/8 证据门控（如「LoRA 是什么」漏判）。
    现统一委托 `_retrieve_and_classify`——与流式 `gated_query_stream`、评测 `eval_rag_bench` **同一条路径**，
    门控逻辑单一真源、production 与 eval 口径一致。
    """
    import time
    plan = None
    try:
        from rag_query_plan import plan_is_reference_only, plan_query
        from rag_multi_source import (REFERENCE_EXCLUDES, evidence_fallback,
                                      multi_source_messages, product_help_messages)
        t0 = time.perf_counter()
        plan = plan_query(
            question, context=context, conversation_id=conversation_id,
        )
        planning_ms = (time.perf_counter() - t0) * 1000

        if plan.decision == "clarify":
            return {
                "query": question, "in_kb": False, "mode": "clarification",
                "answer": plan.clarification or "请补充你希望查询的主要对象。",
                "sources": [], "matched_by": "route_clarification",
                "best_distance": None, "retrieval_count": 0,
                "source_status": {}, "coverage": {}, "freshness": "",
                **_plan_meta(plan, planning_ms=planning_ms),
            }

        from rag_tools import has_embedding_api_key
        if plan_is_reference_only(plan):
            if not has_embedding_api_key():
                ans = synthesize_fallback_answer(question, [])
                return {
                    "query": question, "in_kb": False, "mode": "general_fallback",
                    "answer": FALLBACK_LABEL + (ans or "（无 LLM key，无法作答）"),
                    "sources": [], "matched_by": None, "best_distance": None,
                    "retrieval_count": 0,
                    **_plan_meta(plan, planning_ms=planning_ms),
                }
            tr = time.perf_counter()
            g = _retrieve_and_classify(
                question, top_k, exclude_source_types=REFERENCE_EXCLUDES,
                metadata_filters={"owner_scope": {"curated"}},
                allow_paper_route=False,
                _answerability_shadow_route="reference_kb")
            retrieval_ms = (time.perf_counter() - tr) * 1000
            best_distance = round(g["best"], 4) if g["has_dists"] else None
            if g["in_kb"]:
                chunks = g["chunks"]
                answer = synthesize_grounded_answer(
                    question, chunks,
                    answer_action=g.get("answer_action", "answer"))
                if answer is None:
                    answer = "（无 LLM key，返回原始片段）\n\n" + "\n---\n".join(c[:300] for c in chunks)
                return {
                    "query": question, "in_kb": True, "mode": "kb_grounded",
                    "answer_action": g.get("answer_action", "answer"),
                    "matched_by": g["matched_by"], "answer": answer, "sources": g["sources"],
                    "best_distance": best_distance, "retrieval_count": len(chunks),
                    "source_status": {"reference_kb": {"status": "ok", "count": len(chunks),
                                                        "latency_ms": round(retrieval_ms, 1)}},
                    "coverage": {"reference_kb": len(chunks)}, "freshness": "",
                    **_plan_meta(
                        plan, planning_ms=planning_ms, retrieval_ms=retrieval_ms,
                        retrieval_profile=g.get("retrieval_profile", ""),
                        index_fingerprint=g.get("index_fingerprint", ""),
                        effective_hit=bool(g.get("effective_hit", False)),
                    ),
                }
            ans = synthesize_fallback_answer(question, g["chunks"])
            return {
                "query": question, "in_kb": False, "mode": "general_fallback",
                "answer_action": "abstain",
                "matched_by": None, "answer": FALLBACK_LABEL + (ans or "（无 LLM key，无法作答）"),
                "sources": [], "best_distance": best_distance, "retrieval_count": 0,
                "source_status": {"reference_kb": {"status": "no_evidence", "count": 0,
                                                    "latency_ms": round(retrieval_ms, 1)}},
                "coverage": {}, "freshness": "",
                **_plan_meta(
                    plan, planning_ms=planning_ms, retrieval_ms=retrieval_ms,
                    retrieval_profile=g.get("retrieval_profile", ""),
                    index_fingerprint=g.get("index_fingerprint", ""),
                    effective_hit=bool(g.get("effective_hit", False)),
                ),
            }

        tr = time.perf_counter()
        execution = _execute_query_plan(plan, question, top_k)
        retrieval_ms = (time.perf_counter() - tr) * 1000
        mode = _multi_mode(plan, execution)
        answer_parts = []
        focused_guide = (
            plan.intent_frame.service_mode == "guide"
            and any(route.source == "product_help" for route in plan.routes)
        )
        if focused_guide:
            answer_parts.append(
                execution.deterministic_answer or "当前没有可用的功能说明。"
            )
        elif execution.deterministic_answer:
            answer_parts.append(execution.deterministic_answer)
        if not focused_guide and (execution.evidence or execution.failed_routes):
            # 门动作随证据穿过 multi 执行层(ExecutionResult.answer_action);
            # 纠偏合同文本唯一源在本模块,以 stance 注入,不在 multi 侧复刻
            stance = (_CORRECT_PREMISE_RULES
                      if execution.answer_action == "correct_premise" else "")
            synthesized = _chat(multi_source_messages(question, plan, execution,
                                                      stance=stance))
            if synthesized:
                answer_parts.append("### 综合分析\n" + synthesized)
            elif not execution.deterministic_answer:
                answer_parts.append(evidence_fallback(execution))
        if not answer_parts:
            if _allows_general_fallback(plan):
                fallback = synthesize_fallback_answer(question, [])
                answer_parts.append(FALLBACK_LABEL + (fallback or "（当前未配置可用的通用回答模型。）"))
                mode = "general_fallback"
            else:
                answer_parts.append("没有找到可用的个人记录或知识库证据。")
        return {
            "query": question, "in_kb": execution.in_kb, "mode": mode,
            "answer_action": (execution.answer_action if execution.in_kb
                              else "abstain"),
            "matched_by": "multi_route", "answer": "\n\n".join(answer_parts),
            "sources": execution.sources, "best_distance": None,
            "retrieval_count": len(execution.evidence),
            "source_status": execution.source_status, "coverage": execution.coverage,
            "freshness": execution.freshness,
            "resolved_entities": execution.resolved_entities,
            **_plan_meta(
                plan, planning_ms=planning_ms, retrieval_ms=retrieval_ms,
                retrieval_profile=execution.retrieval_profile,
                index_fingerprint=execution.index_fingerprint,
                effective_hit=execution.effective_hit,
            ),
        }
    except Exception as e:
        if not _allows_general_fallback(plan):
            return {
                "query": question, "in_kb": False, "mode": "retrieval_error",
                "answer": f"⚠️ 内部只读查询失败：{e}。未使用通用知识替代个人事实。",
                "sources": [], "matched_by": None, "best_distance": None,
                "retrieval_count": 0, "planner_mode": "fallback", "routes": [],
                "service_mode": (plan.intent_frame.service_mode if plan else "recall"),
                "source_status": {}, "coverage": {}, "freshness": "", "latency_ms": {},
            }
        try:
            ans = synthesize_fallback_answer(question, [])
        except Exception:
            ans = None
        return {"query": question, "in_kb": False, "mode": "general_fallback",
                "answer": FALLBACK_LABEL + (ans or f"（检索异常：{e}）"),
                "sources": [], "matched_by": None, "best_distance": None, "retrieval_count": 0,
                "planner_mode": "fallback", "routes": [], "source_status": {},
                "coverage": {}, "freshness": "", "latency_ms": {}}
