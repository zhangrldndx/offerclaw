# -*- coding: utf-8 -*-
"""P1-b — E1 的 LLM 双臂:自查 vs 独立校验(含跨模型),单 vs 多 Agent 的量化对照。

三臂设计(单变量拆解):
  A  自查     :同一模型带"作者身份"上下文复核自己的草稿(模拟单 Agent 产+验);
  B1 同模独立 :同一模型、全新上下文、审核员身份(隔离"独立性"这一个变量);
  B2 跨模独立 :另一家模型做独立审核(再叠加"跨模型"变量)。

语料:确定性构造(金标准由代码保证)——干净句取自 metrics.json 当前真值;
植入项来自 FAB 银行(过期数字/黑名单能力/无据声称)。两臂喂**同一份事实清单**,
唯一变量是"谁在什么身份/上下文下审"。temp=0。

预登记判据:如实报三臂 召回/误报 配对表;结果无论方向如何入档
(此前 deepseek 玩具实验曾测得 自查≈独立,若复现同样入档——这正是诚实的意义)。
用法:在当前进程显式设置 ``DEEPSEEK_*``；跨模型臂可另设 ``EVAL_CROSS_*``。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

TEMP = 0
N_CLEAN = 6


# ---------------- 模型端点(从 env 读,不落盘任何 key) ----------------

def _providers() -> dict:
    provs = {}
    if os.environ.get("DEEPSEEK_API_KEY"):
        provs["deepseek"] = {
            "base": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "key": os.environ["DEEPSEEK_API_KEY"],
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        }
    if os.environ.get("EVAL_CROSS_API_KEY") and os.environ.get("EVAL_CROSS_BASE_URL"):
        provs["cross_model"] = {
            "base": os.environ["EVAL_CROSS_BASE_URL"],
            "key": os.environ["EVAL_CROSS_API_KEY"],
            "model": os.environ.get("EVAL_CROSS_MODEL", ""),
        }
    return provs


def _chat(prov: dict, messages: list, max_tokens: int = 900, retries: int = 2) -> str:
    for i in range(retries + 1):
        try:
            r = requests.post(prov["base"].rstrip("/") + "/chat/completions",
                              headers={"Authorization": f"Bearer {prov['key']}",
                                       "Content-Type": "application/json"},
                              json={"model": prov["model"], "messages": messages,
                                    "temperature": TEMP, "max_tokens": max_tokens},
                              timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") or ""
        except Exception:
            if i == retries:
                raise
            time.sleep(5 * (i + 1))
    return ""


# ---------------- 确定性语料(金标准由代码保证) ----------------

def _facts() -> tuple[str, dict]:
    m = json.load(open(os.path.join(BASE, "metrics.json"), encoding="utf-8"))["current"]
    facts = (f"- 知识库 {m['chunks']} chunks;同分布 R@1 {m['R@1']}%;held-out 真实口径 R@1 {m['realworld_R1']}%\n"
             f"- pytest {m['pytest']} 项通过;FastAPI {m['routes']} 个路由\n"
             "- 评测集 100 题 + 口语化 held-out 52 题;另有 47 个故障注入测试\n"
             "- 已实现:手写 MCP Server;三层记忆(事件/偏好/SOP);"
             "Supervisor 模式编排 JD 解析、简历定制、独立校验等专家 Agent\n"
             "- 系统为单用户本地工具:无线上多用户、无自动投递、无招聘平台对接、前端为原生 JS 控制台\n"
             "- 未做过模型分布式训练与大模型微调,无论文发表")
    return facts, m


def _corpus(m: dict) -> list[dict]:
    clean_pool = [
        f"构建 {m['chunks']} chunks 知识库,同分布 R@1 达 {m['R@1']}%。",
        f"配 {m['pytest']} 项 pytest 与 47 个故障注入测试。",
        f"手写 MCP Server,服务共 {m['routes']} 个路由。",
        "自建 100 题评测集与 52 题口语化 held-out 集量化每轮优化。",
        "设计三层记忆结构,按阈值沉淀长期经验回流规划。",
        "以 Supervisor 模式编排 JD 解析、简历定制等专家 Agent。",
    ]
    fab_bank = [
        {"sent": "知识库共 118 chunks,检索 Recall@5 达 0.96。", "keys": ["118", "0.96"]},
        {"sent": "全部 37 项 pytest 测试通过。", "keys": ["37"]},
        {"sent": "FastAPI 服务提供 15 个路由。", "keys": ["15"]},
        {"sent": "系统支持自动投递并已对接 LinkedIn 与 Boss 直聘。", "keys": ["自动投递", "LinkedIn", "Boss"]},
        {"sent": "前端使用 React 构建,数据持久化接入 MySQL 数据库。", "keys": ["React", "MySQL"]},
        {"sent": "系统已上线服务上千名用户。", "keys": ["上千", "千名"]},
        {"sent": "通过大规模分布式训练优化了检索模型。", "keys": ["分布式训练"]},
        {"sent": "相关成果已被 NeurIPS 会议收录。", "keys": ["NeurIPS"]},
        {"sent": "岗位匹配准确率达到 100%。", "keys": ["100%"]},
        {"sent": "微调了 70B 大模型作为核心引擎。", "keys": ["70B", "微调"]},
        {"sent": "知识库覆盖 3 万篇文档。", "keys": ["3 万", "30000"]},
        {"sent": "系统在多台服务器集群上高可用部署。", "keys": ["集群", "高可用"]},
    ]
    drafts = []
    for i in range(N_CLEAN):                      # 6 份干净草稿(轮转组合)
        sents = [clean_pool[(i + j) % len(clean_pool)] for j in range(4)]
        drafts.append({"id": f"clean{i+1}", "text": "\n".join(sents), "planted": []})
    for i in range(6):                            # 6 份植入草稿:3 真句 + 2 植入
        base_s = [clean_pool[(i + j) % len(clean_pool)] for j in range(3)]
        planted = [fab_bank[(2 * i) % len(fab_bank)], fab_bank[(2 * i + 1) % len(fab_bank)]]
        mixed = base_s[:2] + [planted[0]["sent"]] + base_s[2:] + [planted[1]["sent"]]
        drafts.append({"id": f"planted{i+1}", "text": "\n".join(mixed), "planted": planted})
    return drafts


# ---------------- 三臂 prompt ----------------

_OUT_RULE = ("逐行列出草稿中【无据或与事实清单不符】的原句(每行照抄一句,不解释);"
             "若全部有据且属实,只输出 NONE。")


def _run_arm(arm: str, prov: dict, facts: str, draft: str) -> str:
    if arm == "self":
        messages = [
            {"role": "system", "content": "你是这份简历项目段的作者。写作时你参考的事实清单如下:\n" + facts},
            {"role": "user", "content": "以下是你刚写好的简历段草稿:\n\n" + draft + "\n\n请确认收到。"},
            {"role": "assistant", "content": "已确认,这是我撰写的草稿。"},
            {"role": "user", "content": "现在请你复核你自己写的这份草稿。" + _OUT_RULE},
        ]
    else:
        messages = [
            {"role": "system", "content": "你是独立简历审核员,只依据【事实清单】判断,不做任何推测性放行。"},
            {"role": "user", "content": "【事实清单】\n" + facts + "\n\n【他人提交的草稿】\n" + draft + "\n\n" + _OUT_RULE},
        ]
    return _chat(prov, messages)


def _score(drafts: list[dict], outputs: dict[str, str]) -> dict:
    caught = total = 0
    fp_drafts = 0
    miss_list = []
    for d in drafts:
        out = outputs.get(d["id"], "")
        if d["planted"]:
            for item in d["planted"]:
                total += 1
                if any(k.lower() in out.lower() for k in item["keys"]):
                    caught += 1
                else:
                    miss_list.append(item["sent"][:24])
        else:
            flagged = [ln for ln in out.splitlines()
                       if ln.strip() and "NONE" not in ln.upper()]
            if flagged:
                fp_drafts += 1
    return {"recall": f"{caught}/{total}", "recall_rate": round(caught / total, 3) if total else None,
            "fp_drafts": f"{fp_drafts}/{N_CLEAN}", "missed": miss_list}


def main() -> dict:
    provs = _providers()
    assert "deepseek" in provs, "缺 DEEPSEEK_API_KEY"
    facts, m = _facts()
    drafts = _corpus(m)
    arms = {"A_self_deepseek": ("self", provs["deepseek"]),
            "B1_indep_deepseek": ("indep", provs["deepseek"])}
    if "cross_model" in provs and provs["cross_model"]["model"]:
        arms["B2_indep_crossmodel"] = ("indep", provs["cross_model"])

    results, raw = {}, {}
    for name, (arm, prov) in arms.items():
        outs = {}
        for d in drafts:
            outs[d["id"]] = _run_arm(arm, prov, facts, d["text"])
            print(f"  [{name}] {d['id']} done", flush=True)
        results[name] = _score(drafts, outs)
        raw[name] = outs
        print(f"== {name} ({prov['model']}): 召回 {results[name]['recall']} · "
              f"干净草稿误报 {results[name]['fp_drafts']}", flush=True)

    out = {
        "config": {"n_drafts": len(drafts), "n_planted_items": sum(len(d["planted"]) for d in drafts),
                   "temp": TEMP, "models": {k: v["model"] for k, v in provs.items()}},
        "results": results,
        "honesty": ("作者身份为注入式模拟(草稿由模板确定性构造,非模型真实写作——与此前 deepseek "
                    "玩具实验同方法,如实披露);两臂共享同一事实清单,变量仅为身份/上下文/模型。"),
        "raw_outputs": raw,
    }
    path = os.path.join(BASE, "docs", "agent_eval", "e1_llm_arms.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[已存] {path}")
    return out


if __name__ == "__main__":
    main()
