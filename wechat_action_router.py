# -*- coding: utf-8 -*-
"""Local, registry-backed matcher for WeChat action intents.

The matcher is deliberately model-free.  It first identifies a mutation verb
and business target, then ranks only compatible registered capabilities with a
sparse character n-gram vector.  Every capability must have examples, so a new
capability cannot silently become unreachable from WeChat.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from action_capabilities import ACTION_CAPABILITIES, ACTION_CAPABILITY_REGISTRY


_ACTION_WORDS = re.compile(
    r"新建|新增|创建|添加|录入|记录|上传|导入|保存|更新|修改|编辑|调整|更改|"
    r"改成|切换|补充|关联|绑定|链接|批准|审批|通过|接受|采纳|同意|拒绝|驳回|"
    r"放弃|删除|移除|清除|生成|制定|写一份|做一份|重排|重做|改写|归档|"
    r"停止使用|恢复|还原|重新启用|规划|写成"
)
_READ_ONLY_WORDS = re.compile(r"^(?:请|帮我)?(?:查看|查询|列出|显示|总结|告诉我|有哪些|最近|当前|目前)")
_RECORD_NOUN_QUERY = re.compile(
    r"(?:有哪些|有多少|多少|最近|历史|查看|查询|列出|显示|总结|告诉我|是什么|是否|有没有)"
    r".{0,24}记录|记录.{0,24}(?:有哪些|有多少|多少|是什么|是否|有没有|吗|呢|？|\?)"
)
_UNAMBIGUOUS_CREATE_WORDS = re.compile(r"新建|新增|创建|添加|录入|上传|导入|保存|补记")

_TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "application_jd": ("投递JD", "投递的JD", "岗位JD", "JD版本", "职位描述", "岗位描述"),
    "application": ("投递记录", "求职申请", "申请记录", "投递", "申请"),
    "daily_log": ("每日留痕", "执行记录", "学习记录", "今日记录", "今天记录", "日志", "留痕"),
    "reflection": ("每日复盘", "周复盘", "复盘", "反思"),
    "knowledge": ("知识库候选", "知识候选", "待审资料", "知识库", "资料库", "文档资料"),
    "memory": ("职业目标", "目标上下文", "系统记忆", "记忆"),
    "profile": ("画像建议", "用户画像", "个人画像", "画像"),
    "plan": ("求职计划", "学习计划", "今日计划", "计划草稿", "计划"),
    "resume": ("项目经历简历", "完整简历", "简历草稿", "简历", "CV", "resume"),
    "project": ("项目素材", "项目事实", "项目介绍", "GitHub项目", "项目"),
}

_VERB_ALIASES: dict[str, tuple[str, ...]] = {
    "create": ("新建", "新增", "创建", "添加", "录入", "记录", "上传", "导入", "保存"),
    "update": ("更新", "修改", "编辑", "调整", "更改", "改成", "切换", "补充"),
    "delete": ("删除", "移除", "清除", "彻底删除"),
    "link": ("关联", "绑定", "链接", "加入投递"),
    "approve": ("批准", "审批通过", "通过", "接受", "采纳", "同意"),
    "reject": ("拒绝", "驳回", "不接受", "放弃"),
    "generate": ("生成", "制定", "规划", "写一份", "写成", "做一份", "重排", "重做", "改写"),
    "archive": ("归档", "停止使用", "停用"),
    "restore": ("恢复", "还原", "重新启用"),
}

# These examples are intent vocabulary, not executable rules.  Target/verb
# compatibility still comes from ACTION_CAPABILITIES.
ACTION_EXAMPLES: dict[str, tuple[str, ...]] = {
    "application.create": ("新建一条投递", "添加求职申请", "录入投递记录"),
    "application.update_status": ("更新投递状态", "把申请改成面试中", "修改投递进度"),
    "application.update_details": ("修改投递备注", "更新投递日期和下一步", "编辑申请岗位"),
    "application.delete": ("删除一条投递", "移除申请记录"),
    "application_jd.link": ("给投递关联JD", "绑定岗位描述到申请"),
    "application_jd.update": ("更新投递JD", "修改职位描述版本"),
    "profile.update": ("修改我的画像", "更新用户画像"),
    "profile.approve_suggestion": ("批准画像建议", "接受画像建议"),
    "profile.reject_suggestion": ("拒绝画像建议", "驳回画像建议"),
    "plan.generate": ("生成新计划", "制定求职计划", "重新规划计划"),
    "plan.update": ("修改当前计划", "调整计划任务"),
    "plan.approve": ("批准计划草稿", "通过新计划"),
    "plan.reject": ("拒绝计划草稿", "放弃新计划"),
    "resume.generate_project": ("生成项目经历简历", "把项目写成简历经历"),
    "resume.generate_full": ("生成完整简历", "写一份新简历"),
    "resume.update": ("修改简历草稿", "提出简历修改要求"),
    "resume.approve": ("批准简历草稿", "通过这版简历"),
    "resume.reject": ("拒绝简历草稿", "放弃这版简历"),
    "daily_log.create": ("新增今日留痕", "记录今天的学习", "添加执行日志"),
    "reflection.generate": ("生成今日复盘", "做一份周复盘"),
    "knowledge.create": ("添加知识库资料", "上传文档到知识库"),
    "knowledge.approve": ("批准知识库候选", "通过待审资料"),
    "knowledge.reject": ("拒绝知识库候选", "驳回待审资料"),
    "knowledge.delete": ("删除知识库资料", "移除知识库文档"),
    "memory.update_goal": ("更新职业目标", "切换目标上下文"),
    "memory.archive": ("归档一条记忆", "停止使用这条记忆"),
    "memory.restore": ("恢复已归档记忆", "重新启用这条记忆"),
    "memory.delete": ("删除一条记忆", "彻底删除系统记忆"),
    "project.create": ("新增项目素材", "保存一个GitHub项目"),
    "project.update": ("更新项目素材", "修改项目介绍"),
}


@dataclass(frozen=True)
class ActionMatch:
    capability_id: str
    score: float
    target_type: str
    verb: str


def action_router_contract() -> dict:
    registered = set(ACTION_CAPABILITY_REGISTRY)
    covered = set(ACTION_EXAMPLES)
    missing = sorted(registered - covered)
    extra = sorted(covered - registered)
    return {
        "status": "ok" if not missing and not extra else "error",
        "registered_count": len(registered),
        "example_count": len(covered),
        "missing": missing,
        "extra": extra,
    }


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(text or "").lower())


def _vector(text: str) -> Counter[str]:
    value = _normalize(text)
    grams: Counter[str] = Counter()
    for width in (1, 2, 3):
        grams.update(value[index:index + width] for index in range(max(0, len(value) - width + 1)))
    return grams


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _find_targets(text: str) -> list[str]:
    lowered = str(text or "").lower()
    found = [
        target for target, aliases in _TARGET_ALIASES.items()
        if any(alias.lower() in lowered for alias in aliases)
    ]
    if re.search(r"(?:投递|申请).{0,8}(?:JD|职位描述|岗位描述)|(?:JD|职位描述|岗位描述).{0,8}(?:投递|申请)", text, re.I):
        found.append("application_jd")
    if re.search(r"(?:记录|添加|补记).{0,10}(?:今天|今日|学习|执行)", text) and "daily_log" not in found:
        found.append("daily_log")
    # A phrase such as "投递 JD" is more specific than its component nouns.
    if "application_jd" in found:
        found = [item for item in found if item != "application"]
    if "resume" in found and "project" in found and re.search(r"项目.{0,5}(?:经历)?简历|简历.{0,5}项目", text, re.I):
        found = [item for item in found if item != "project"]
    return found


def _find_verbs(text: str) -> list[str]:
    lowered = str(text or "").lower()
    found = [
        verb for verb, aliases in _VERB_ALIASES.items()
        if any(alias.lower() in lowered for alias in aliases)
    ]
    # Approval phrases contain "通过" but ordinary read questions may contain it too.
    if "approve" in found and not re.search(r"批准|审批|通过|接受|采纳|同意", text):
        found.remove("approve")
    # “学习记录/执行记录” uses 记录 as a noun.  Keep it as a create verb only
    # when the phrase explicitly asks to record/add something.
    if ("create" in found
            and re.search(r"(?:学习|执行|今日|今天)记录", text)
            and not re.search(r"新增|新建|创建|添加|录入|上传|导入|保存|记录.{0,8}(?:今天|今日|学习|执行)", text)):
        found.remove("create")
    # “记录” is a noun across domains in questions such as “有哪些投递记录” or
    # “知识库里有哪些记录”. Without a separate mutation verb, prefer a read.
    if ("create" in found and _RECORD_NOUN_QUERY.search(text)
            and not _UNAMBIGUOUS_CREATE_WORDS.search(text)):
        found.remove("create")
    return found


def _narrow_ambiguities(text: str, candidate_ids: list[str]) -> list[str]:
    ids = set(candidate_ids)
    lowered = text.lower()
    if {"application.update_status", "application.update_details"} <= ids:
        status_words = (
            "状态", "进度", "准备投递", "已评估", "已投递", "等待反馈", "面试中",
            "offer", "已拒绝", "主动放弃", "preparing", "applied", "interviewing",
        )
        keep = "application.update_status" if any(word in lowered for word in status_words) else "application.update_details"
        return [item for item in candidate_ids if item == keep]
    if {"resume.generate_project", "resume.generate_full"} <= ids:
        keep = "resume.generate_project" if re.search(r"项目.{0,5}(?:经历|简历)|简历.{0,5}项目", text, re.I) else "resume.generate_full"
        return [item for item in candidate_ids if item == keep]
    return candidate_ids


def match_action_capability(text: str) -> ActionMatch | None:
    """Return one registered action only when the text clearly requests mutation."""
    value = str(text or "").strip()
    if not value or not _ACTION_WORDS.search(value):
        return None
    if _READ_ONLY_WORDS.search(value) and not re.search(
        r"(?:并|然后|同时|顺便).{0,5}(?:" + _ACTION_WORDS.pattern + r")", value,
    ):
        return None
    targets = _find_targets(value)
    verbs = _find_verbs(value)
    if not targets or not verbs:
        return None

    candidate_ids = [
        item.capability_id for item in ACTION_CAPABILITIES
        if item.target_type in targets and item.verb in verbs
    ]
    candidate_ids = _narrow_ambiguities(value, candidate_ids)
    if not candidate_ids:
        return None

    query_vector = _vector(value)
    ranked: list[tuple[float, str]] = []
    for capability_id in candidate_ids:
        capability = ACTION_CAPABILITY_REGISTRY[capability_id]
        document = " ".join((
            capability.capability_id, capability.ui_entry, *capability.steps,
            *ACTION_EXAMPLES[capability_id],
        ))
        score = _cosine(query_vector, _vector(document))
        ranked.append((score, capability_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    score, capability_id = ranked[0]
    capability = ACTION_CAPABILITY_REGISTRY[capability_id]
    return ActionMatch(capability_id, round(score, 6), capability.target_type, capability.verb)


__all__ = ["ACTION_EXAMPLES", "ActionMatch", "action_router_contract", "match_action_capability"]
