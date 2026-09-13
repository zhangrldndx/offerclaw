# -*- coding: utf-8 -*-
"""Typed UI action capabilities used by the read-only top chat router.

The registry describes operations that the product can perform in its cards.
It does not grant the top chat permission to execute them.  Top chat always
renders guidance from these records and never calls a write service.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from domain_status import APPLICATION_STATUS_LABELS, ApplicationStatusCode


ACTION_CAPABILITY_REGISTRY_VERSION = "action-capability-v3"
TOP_CHAT_POLICY = "guide_only"


@dataclass(frozen=True)
class ActionCapability:
    capability_id: str
    target_type: str
    verb: str
    card_topic: str
    ui_entry: str
    allowed_fields: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    available: bool = True
    top_chat_policy: str = TOP_CHAT_POLICY
    steps: tuple[str, ...] = ()

    def to_prompt_row(self) -> list[Any]:
        return [
            self.capability_id,
            self.target_type,
            self.verb,
            ",".join(self.allowed_fields),
            1 if self.available else 0,
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _c(capability_id: str, target_type: str, verb: str, card_topic: str,
       ui_entry: str, *, fields: tuple[str, ...] = (),
       required: tuple[str, ...] = (), available: bool = True,
       steps: tuple[str, ...] = ()) -> ActionCapability:
    return ActionCapability(
        capability_id, target_type, verb, card_topic, ui_entry,
        fields, required, available, TOP_CHAT_POLICY, steps,
    )


ACTION_CAPABILITIES: tuple[ActionCapability, ...] = (
    _c("application.create", "application", "create", "application", "投递管理",
       fields=("company", "position", "status_code", "date", "next_action", "note"),
       required=("company", "position"),
       steps=("打开“投递管理”。", "填写企业、岗位和需要保存的状态等字段。", "检查预览后提交保存。")),
    _c("application.update_status", "application", "update", "application", "投递管理 → 对应投递卡片",
       fields=("status_code",), required=("application_id", "status_code"),
       steps=("打开“投递管理”，找到对应投递卡片。", "在状态控件中选择目标状态。", "确认后保存这条投递。")),
    _c("application.update_details", "application", "update", "application", "投递管理 → 对应投递卡片",
       fields=("company", "position", "date", "next_action", "note"), required=("application_id",),
       steps=("打开“投递管理”，找到对应投递卡片。", "修改下一步、备注、日期或岗位信息。", "确认后保存。")),
    _c("application.delete", "application", "delete", "application", "投递管理 → 对应投递卡片",
       required=("application_id",), available=False,
       steps=("当前投递卡片没有开放删除能力。", "可保留记录并把状态更新为合适的终态。")),
    _c("application_jd.link", "application_jd", "link", "application_jd", "投递管理 / JD 分析与匹配",
       fields=("jd_text", "jd_url", "jd_version_id"), required=("application_id", "jd_text"),
       steps=("在对应投递卡片选择“更新 JD”，或从 JD 分析区选择“加入投递管理”。", "核对投递与 JD 内容。", "确认关联或创建新版本。")),
    _c("application_jd.update", "application_jd", "update", "application_jd", "投递管理 → 更新 JD",
       fields=("jd_text", "jd_url", "active_version_id"), required=("application_id", "jd_text"),
       steps=("打开目标投递的“更新 JD”。", "提交新的 JD 内容并检查版本差异。", "确认后切换活动版本。")),
    _c("profile.update", "profile", "update", "profile", "用户画像 → 编辑画像",
       fields=("profile_markdown",), required=("profile_markdown",),
       steps=("打开“用户画像”，点击“编辑画像”。", "修改需要纠正的事实或约束。", "检查后保存新版本。")),
    _c("profile.approve_suggestion", "profile", "approve", "profile", "用户画像 → 画像建议",
       required=("suggestion_id",), steps=("打开“画像建议”。", "检查建议依据。", "接受、修改后接受或拒绝。")),
    _c("profile.reject_suggestion", "profile", "reject", "profile", "用户画像 → 画像建议",
       required=("suggestion_id",), steps=("打开“画像建议”。", "找到不接受的建议并检查依据。", "选择“拒绝”。")),
    _c("plan.generate", "plan", "generate", "plan", "求职组合 Plan Agent",
       required=("target_scope",), steps=("在 Plan Agent 卡片选择目标投递或 JD 范围。", "生成计划草稿。", "检查变化说明并确认后生效。")),
    _c("plan.update", "plan", "update", "plan", "求职组合 Plan Agent → 任务微调 / 修改今日",
       fields=("task", "date", "duration", "priority", "dependency"), required=("task_id",),
       steps=("打开 Plan Agent 卡片。", "小范围修改使用“任务微调”或“修改今日”。", "检查新计划版本后保存。")),
    _c("plan.approve", "plan", "approve", "plan", "Plan Agent 审批面板",
       required=("draft_id", "artifact_revision"), steps=("打开等待审批的计划草稿。", "核对评审和变化说明。", "确认后批准当前版本。")),
    _c("plan.reject", "plan", "reject", "plan", "Plan Agent 审批面板",
       required=("draft_id", "artifact_revision"), steps=("打开等待审批的计划草稿。", "确认当前版本不是要采用的方案。", "选择“放弃草稿”。")),
    _c("resume.generate_project", "resume", "generate", "resume_project", "简历工坊 → 产物范围=项目经历 → Resume Agent → Critic",
       required=("project_material",), steps=("在“简历工坊”选择已保存项目或提供项目素材。", "把产物范围设为“项目经历”，按需选择用于定制的 JD。", "点击唯一的 Resume Agent → Critic 按钮，检查评审结果后审批。")),
    _c("resume.generate_full", "resume", "generate", "resume_agent", "简历工坊 → 产物范围=完整简历 → Resume Agent → Critic",
       required=("application_id", "jd_version_id"), steps=("在“简历工坊”选择已关联活动 JD 的投递。", "把产物范围设为“完整简历”，点击唯一的 Resume Agent → Critic 按钮。", "检查评审结果后批准或提出修改要求。")),
    _c("resume.update", "resume", "update", "resume_agent", "简历工坊 → 审批面板 → 提出修改要求",
       fields=("change_request",), required=("draft_id", "artifact_revision", "change_request"),
       steps=("打开简历工坊中的待审草稿。", "在“提出修改要求”中写明建议。", "提交后由 Resume Agent 修改并由 Critic 按冻结标准复审。")),
    _c("resume.approve", "resume", "approve", "resume_agent", "简历工坊 → 审批面板",
       required=("draft_id", "artifact_revision"), steps=("打开待审简历草稿。", "核对证据与 Critic 结果。", "批准当前版本后再保存为正式草稿。")),
    _c("resume.reject", "resume", "reject", "resume_agent", "简历工坊 → 审批面板",
       required=("draft_id", "artifact_revision"), steps=("打开简历工坊中的待审草稿。", "确认当前版本不再继续修改。", "选择“放弃草稿”。")),
    _c("daily_log.create", "daily_log", "create", "reflection", "每日执行 & 复盘",
       fields=("business_date", "status_code", "minutes", "note", "task_id", "attachments"),
       required=("business_date", "note"), steps=("打开“每日执行 & 复盘”。", "填写执行状态、耗时和文字留痕，可关联计划任务。", "检查附件引用后保存。")),
    _c("reflection.generate", "reflection", "generate", "reflection", "每日执行 & 复盘 → 生成复盘",
       required=("business_date",), steps=("先保存当天真实执行记录。", "点击“生成复盘”。", "检查复盘内容并按需补充自己的判断。")),
    _c("knowledge.create", "knowledge", "create", "knowledge", "知识库维护",
       fields=("url", "file"), required=("url_or_file",), steps=("打开“知识库维护”。", "上传文件或填写资料 URL。", "预览抓取结果后进入待审区。")),
    _c("knowledge.approve", "knowledge", "approve", "knowledge", "知识库维护 → 待审候选",
       required=("candidate_id",), steps=("打开知识库待审候选。", "核对来源、类型和样例正文。", "明确批准后写入正式库并增量索引。")),
    _c("knowledge.reject", "knowledge", "reject", "knowledge", "知识库维护 → 待审候选",
       required=("candidate_id",), steps=("打开知识库待审候选。", "核对来源和正文。", "选择拒绝，使其不进入正式知识库。")),
    _c("knowledge.delete", "knowledge", "delete", "knowledge", "知识库维护",
       required=("document_id",), steps=("在知识库维护中找到目标资料。", "使用资料管理入口删除。", "确认派生索引已同步移除。")),
    _c("memory.update_goal", "memory", "update", "profile", "记忆 → 职业目标",
       fields=("goal_name", "active_context_id"), required=("goal_name",), steps=("打开“记忆”面板的职业目标区域。", "创建或选择目标上下文。", "确认切换后旧目标保留为历史。")),
    _c("memory.archive", "memory", "archive", "profile", "记忆 → 系统记住了什么",
       required=("memory_id",), steps=("在记忆面板找到目标记忆。", "查看依据和适用目标。", "选择“停止使用/归档”。")),
    _c("memory.restore", "memory", "restore", "profile", "记忆 → 已归档",
       required=("memory_id",), steps=("打开已归档记忆。", "检查当前目标是否仍适用。", "选择恢复。")),
    _c("memory.delete", "memory", "delete", "profile", "记忆 → 系统记住了什么",
       required=("memory_id",), steps=("在记忆面板找到目标内容。", "选择删除并核对派生影响。", "确认后删除内容及派生索引。")),
    _c("project.create", "project", "create", "project", "简历工坊 → 项目素材",
       fields=("repo_url", "folder", "project_text"), required=("repo_url_or_folder_or_text",),
       steps=("打开“简历工坊”的项目素材区域。", "填写 GitHub 仓库、文件夹或项目介绍。", "预览并确认保存项目事实。")),
    _c("project.update", "project", "update", "project", "简历工坊 → 已保存项目",
       fields=("repo_url", "folder", "project_text"), required=("project_id",),
       steps=("在简历工坊选择已保存项目。", "更新仓库地址、文件夹或介绍材料。", "确认后保存新版本。")),
)


ACTION_CAPABILITY_REGISTRY = {item.capability_id: item for item in ACTION_CAPABILITIES}

# One-release adapter for v2 planner output.  New prompts expose only the typed
# IDs above; these aliases exist solely for old checkpoints and test fixtures.
LEGACY_CAPABILITY_ALIASES: dict[str, tuple[str, ...]] = {
    "application": ("application.create",),
    "application_jd": ("application_jd.link",),
    "profile": ("profile.update",),
    "plan": ("plan.update",),
    "resume_agent": ("resume.generate_full",),
    "resume_project": ("resume.generate_project",),
    "reflection": ("daily_log.create",),
    "knowledge": ("knowledge.create",),
    "project": ("project.create",),
    "experience": ("application.update_details",),
    "resume_template": ("knowledge.create",),
}


def normalize_capability_ids(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        mapped = LEGACY_CAPABILITY_ALIASES.get(value, (value,))
        for item in mapped:
            if item and item not in normalized:
                normalized.append(item)
    return normalized


def capability_topics(values: Iterable[str]) -> list[str]:
    topics: list[str] = []
    for capability_id in normalize_capability_ids(values):
        capability = ACTION_CAPABILITY_REGISTRY.get(capability_id)
        if capability and capability.card_topic not in topics:
            topics.append(capability.card_topic)
    return topics


def action_capabilities_for_prompt() -> list[list[Any]]:
    return [item.to_prompt_row() for item in ACTION_CAPABILITIES]


def _field_changes(action_request: Any) -> list[dict[str, str]]:
    raw = getattr(action_request, "field_changes", None)
    if raw is None and isinstance(action_request, dict):
        raw = action_request.get("field_changes")
    return [
        (item if isinstance(item, dict) else item.model_dump())
        for item in (raw or [])
    ]


def infer_unique_capability_id(action_request: Any) -> str:
    """Infer an omitted capability only when the typed registry is unambiguous."""
    get = (lambda name, default="": action_request.get(name, default)) if isinstance(
        action_request, dict
    ) else (lambda name, default="": getattr(action_request, name, default))
    target_type = str(get("target_type") or "")
    verb = str(get("verb") or "")
    fields = {
        str(item.get("field") or "") for item in _field_changes(action_request)
        if item.get("field")
    }
    candidates = [
        item.capability_id for item in ACTION_CAPABILITIES
        if item.target_type == target_type
        and item.verb == verb
        and fields.issubset(set(item.allowed_fields))
    ]
    return candidates[0] if len(candidates) == 1 else ""


def validate_action_capabilities(action_request: Any,
                                 capability_ids: Iterable[str]) -> list[str]:
    """Validate model-selected IDs against the typed action request."""
    errors: list[str] = []
    ids = normalize_capability_ids(capability_ids)
    if len(ids) != 1:
        return ["action_requires_unique_capability"]
    capability = ACTION_CAPABILITY_REGISTRY.get(ids[0])
    if capability is None:
        return [f"unknown_action_capability:{ids[0]}"]
    get = (lambda name, default="": action_request.get(name, default)) if isinstance(
        action_request, dict
    ) else (lambda name, default="": getattr(action_request, name, default))
    target_type = str(get("target_type") or "")
    verb = str(get("verb") or "")
    if capability.target_type != target_type:
        errors.append("action_target_capability_mismatch")
    if capability.verb != verb:
        errors.append("action_verb_capability_mismatch")
    changes = _field_changes(action_request)
    seen_fields: dict[str, str] = {}
    for change in changes:
        field_name = str(change.get("field") or "")
        value_code = str(change.get("value_code") or "")
        if field_name in seen_fields and seen_fields[field_name] != value_code:
            errors.append(f"conflicting_action_field:{field_name}")
        seen_fields[field_name] = value_code
        if field_name not in capability.allowed_fields:
            errors.append(f"action_field_not_allowed:{field_name}")
            continue
        if field_name == "status_code" and target_type == "application":
            try:
                ApplicationStatusCode(value_code)
            except ValueError:
                errors.append(f"invalid_application_status_code:{value_code}")
    if capability.capability_id == "application.update_status" and "status_code" not in seen_fields:
        errors.append("action_missing_required_field:status_code")
    return list(dict.fromkeys(errors))


def action_guidance(capability_ids: Iterable[str], action_request: Any = None) -> str:
    """Render exact, read-only UI guidance from registered capability facts."""
    ids = normalize_capability_ids(capability_ids)
    blocks = [
        "### 已识别为操作请求\n\n"
        "顶部问答只提供操作指引，**没有修改任何数据**。"
    ]
    changes = _field_changes(action_request)
    if changes:
        display_changes = []
        for item in changes:
            field_name = str(item.get("field") or "")
            value_code = str(item.get("value_code") or "")
            suffix = ""
            if field_name == "status_code":
                try:
                    suffix = f"（{APPLICATION_STATUS_LABELS[ApplicationStatusCode(value_code)]}）"
                except (KeyError, ValueError):
                    pass
            display_changes.append(f"{field_name}={value_code}{suffix}")
        formatted = "、".join(display_changes)
        blocks.append(f"识别到的目标字段：`{formatted}`。")
    for capability_id in ids:
        capability = ACTION_CAPABILITY_REGISTRY.get(capability_id)
        if capability is None:
            continue
        steps = "\n".join(
            f"{index}. {step}" for index, step in enumerate(capability.steps, start=1)
        )
        availability = "当前可用" if capability.available else "当前未开放"
        blocks.append(
            f"### {capability.ui_entry}\n\n"
            f"能力：`{capability.capability_id}`（{availability}）\n\n{steps}\n\n"
            f"执行边界：`{capability.top_chat_policy}`。"
        )
    if len(blocks) == 1:
        blocks.append("当前没有匹配到唯一的产品能力，请补充要操作的对象和字段。")
    return "\n\n".join(blocks)


__all__ = [
    "ACTION_CAPABILITIES", "ACTION_CAPABILITY_REGISTRY",
    "ACTION_CAPABILITY_REGISTRY_VERSION", "ActionCapability",
    "LEGACY_CAPABILITY_ALIASES", "action_capabilities_for_prompt",
    "action_guidance", "capability_topics", "infer_unique_capability_id",
    "normalize_capability_ids",
    "validate_action_capabilities",
]
