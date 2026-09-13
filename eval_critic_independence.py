# -*- coding: utf-8 -*-
"""E1 — Critic 独立性:注入编造草稿上量化召回/误报（docs/MULTI_AGENT_UPGRADE.md §5）。

预登记判据(先登记后跑):Arm B(独立)recall 显著 > Arm A(自查)且 fpr 低 → 采纳 Critic 默认开;
否则 default-off + 记诚实负结果(保留代码腿黑名单/覆盖作轻量护栏)。

两部分:
- **确定性代码腿(无 key,现在就能跑)**:干净草稿 + 注入 K 条编造(stale 数字 118/37/15 + 黑名单)。
  代码腿"结构上独立"(它不是写作者),其召回是"独立审查"的下界;误报是它误伤合法数字的比率。
- **LLM 自查 vs 独立(需 key,跨模型,--llm)**:Arm A 写作模型审自己、Arm B 独立模型审——这才是
  "独立 > 自查"的正式对照(已知自评上浮:qwen 自评 9.92 vs deepseek 交叉 8.83)。
"""
import sys

import resume_critic

# 干净事实(与 metrics.json 当前真值一致 → 不该被判编造)
_CLEAN = [
    "构建 3340 chunks 知识库，同分布 R@1 达 86%。",
    "配 442 项 pytest 与 47 个故障注入测试。",
    "手写 MCP Server 暴露 46 路由。",
    "held-out 口语集真实口径 R@1 48%。",
]
# 注入编造(gold:每条都该被判 flag)
_PLANTED = [
    ("stale_number", "知识库 118 chunks，pytest 37/37 通过。"),   # 旧数字
    ("stale_number", "FastAPI 15 路由。"),                         # 旧路由数
    ("fabrication", "系统支持自动投递并对接 LinkedIn / Boss直聘。"),  # 黑名单
    ("fabrication", "用 React + Vue 写了前端，接了数据库和登录系统。"),
]


def run_code_leg() -> dict:
    """确定性代码腿:干净草稿 0 误报、注入编造全召回。"""
    fp = 0
    for c in _CLEAN:
        if resume_critic.fabrication_flags(c):
            fp += 1
    caught = 0
    for _kind, p in _PLANTED:
        if resume_critic.fabrication_flags(p):
            caught += 1
    recall = caught / len(_PLANTED)
    fpr = fp / len(_CLEAN)
    print(f"[代码腿·结构独立] 注入编造召回 {caught}/{len(_PLANTED)}={recall:.0%}  "
          f"干净草稿误报 {fp}/{len(_CLEAN)}={fpr:.0%}")
    return {"arm": "code", "recall": recall, "fpr": fpr,
            "caught": caught, "planted": len(_PLANTED)}


def run_llm_arms():
    """LLM 自查(Arm A) vs 独立(Arm B):正式独立性对照。需 key + 跨模型,重跑另配。"""
    print("[LLM 自查 vs 独立] 需 key + 跨模型（Arm A=写作模型审自己 / Arm B=独立模型审）。")
    print("  → 该臂重跑参照 OfferClaw 裁判团方法(答案只生成一次、同批、temp=0),"
          "结果与 go/no-go 判据写入 docs/rag_eval/。未测正前 Critic 默认关(测正才采纳)。")
    return {"arm": "llm", "status": "deferred(needs key + cross-model)"}


if __name__ == "__main__":
    run_code_leg()
    if "--llm" in sys.argv:
        run_llm_arms()
