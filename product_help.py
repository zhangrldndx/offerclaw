# -*- coding: utf-8 -*-
"""OfferClaw 产品能力与只读问答帮助注册表。

顶部问答框可以解释如何操作，但不能直接写入个人事实文件。这里集中维护可用入口、
审批边界和操作说明，避免在查询规划器里为每个 UI 说法增加一条特判。
"""

from __future__ import annotations

import re
from typing import Any

from action_capabilities import (
    ACTION_CAPABILITY_REGISTRY, action_guidance, capability_topics,
    normalize_capability_ids,
)


CARD_CAPABILITY_REGISTRY_VERSION = "card-capability-v6"


CARD_CAPABILITY_REGISTRY: dict[str, dict[str, Any]] = {
    "project": {
        "card_id": "resume_workshop",
        "ui_entry": "简历工坊（项目 → 简历段）",
        "available": True,
        "supported_inputs": ["GitHub 仓库地址", "项目介绍文件", "项目介绍文本"],
        "required_inputs": ["至少一种项目素材"],
        "limitations": ["问答框不能直接新增项目", "个人记忆必须预览确认后生效"],
        "title": "新增或更新个人项目材料",
        "summary": "项目材料先进入待确认区，用户确认后才持久化并进入个人项目检索。",
        "steps": [
            "打开“简历工坊（项目 → 简历段）”。",
            "通过仓库地址、上传项目介绍文件或直接粘贴文本提供项目素材。",
            "勾选“将项目素材放入个人记忆待确认区”，再执行项目分析。",
            "在待审候选中检查标题、来源和内容，明确批准后才正式入库。",
        ],
        "boundary": "顶部问答框只提供入口和流程说明，不会直接新增、覆盖或审批项目。",
    },
    "application_jd": {
        "card_id": "application_management",
        "ui_entry": "投递管理 / JD 分析与匹配",
        "available": True,
        "supported_inputs": ["JD 原文", "JD URL", "目标投递记录"],
        "required_inputs": ["企业", "岗位", "JD 正文"],
        "limitations": ["活动版本切换必须确认", "顶部问答不写入 JD"],
        "title": "为投递关联或更新 JD",
        "summary": "JD 必须通过投递记录建立版本关系，预览确认后才切换活动版本。",
        "steps": [
            "已有投递：在对应投递卡片点击“更新 JD”。",
            "尚无投递：在 JD 分析区完成临时分析后选择“加入投递管理”。",
            "核对公司、岗位、地点、来源和重复候选，再确认关联已有记录或新建投递。",
            "内容变化会创建新 JD 版本；旧版本不会被覆盖。",
        ],
        "boundary": "顶部问答不会写入 JD，也不会替用户确认版本切换。",
    },
    "application": {
        "card_id": "application_management",
        "ui_entry": "投递管理（真实投递追踪）",
        "available": True,
        "supported_inputs": ["企业", "岗位", "状态", "日期", "下一步", "备注"],
        "required_inputs": ["企业", "岗位"],
        "limitations": ["同公司同岗位多条记录必须按投递 ID 区分"],
        "title": "新增或更新投递记录",
        "summary": "投递事实通过投递管理表单或 JD 加入流程保存，问答保持只读。",
        "steps": [
            "在“投递管理”填写企业、岗位、状态和日期等必填信息。",
            "需要 JD 时可先创建投递，之后在投递卡片补充关联。",
            "若从 JD 分析区进入，使用“加入投递管理”预览并确认字段。",
            "状态、下一步、备注和经验总结由用户确认后保存。",
        ],
        "boundary": "问答不会自动改变投递状态、下一步动作或经验总结。",
    },
    "experience": {
        "card_id": "application_management",
        "ui_entry": "投递管理（经验总结）",
        "available": True,
        "supported_inputs": ["对应投递", "阶段", "经验总结", "是否加入知识库"],
        "required_inputs": ["企业", "岗位", "经验总结"],
        "limitations": ["个人亲历不能由模型代写成事实"],
        "title": "保存投递经验总结",
        "summary": "经验先绑定投递记录；是否额外进入个人知识库由用户选择。",
        "steps": [
            "在“投递管理”选择或填写对应企业、岗位和当前阶段。",
            "在“经验总结”中记录笔试、面试、流程或教训。",
            "需要用于 RAG、计划和每日建议时，勾选“把经验总结加入知识库”。",
            "提交后经验会保留投递 ID，并在存在 JD 时记录当前 JD 版本。",
        ],
        "boundary": "问答只能读取已经保存的经验，不能代表用户创建亲历内容。",
    },
    "knowledge": {
        "card_id": "knowledge_maintenance",
        "ui_entry": "知识库维护",
        "available": True,
        "supported_inputs": ["资料 URL", "本地文档"],
        "required_inputs": ["URL 或文件"],
        "limitations": ["必须先预览并人工批准", "论文与普通资料分源检索"],
        "title": "添加学习资料、博客、论文或文档",
        "summary": (
            "外部资料采用候选预览—用户确认—增量索引流程；这与每日执行附件的"
            "本地引用存储不是同一条链路。"
        ),
        "steps": [
            "打开“知识库维护”。",
            "粘贴资料 URL，或上传支持的文档文件。",
            "先查看抓取结果、评分、来源和样例内容。",
            "在“待审候选”中明确批准后再写入正式知识库和派生索引。",
        ],
        "boundary": (
            "顶部问答不会自动抓取、上传或批准资料；论文意图与普通学习资料仍分源检索。"
            "每日执行中附加的 PDF/图片不会因此自动进入正式知识库。"
        ),
    },
    "profile": {
        "card_id": "profile",
        "ui_entry": "用户画像",
        "available": True,
        "supported_inputs": ["直接编辑", "画像建议审批"],
        "required_inputs": [],
        "limitations": ["未批准建议不生效", "不会自动宣称掌握技能"],
        "title": "编辑个人画像或处理画像建议",
        "summary": "正式画像只接受用户编辑或用户批准的证据建议。",
        "steps": [
            "在“用户画像”卡片点击“编辑画像”直接修改当前事实。",
            "点击“画像建议”查看由复盘证据产生的待确认建议。",
            "逐项接受、修改或拒绝；未批准建议不会写入正式画像。",
            "画像变化只提示计划可能受影响，不会静默重写学习计划。",
        ],
        "boundary": "顶部问答不能自动宣称用户掌握技能，也不能直接覆盖画像。",
    },
    "plan": {
        "card_id": "portfolio_plan",
        "ui_entry": "求职组合 Plan Agent",
        "available": True,
        "supported_inputs": ["任务微调", "修改今日", "投递/JD 范围规划", "每日留痕关联计划任务"],
        "required_inputs": ["JD 驱动规划需要已关联 JD 的投递"],
        "limitations": ["顶部问答不启动 Agent", "任务微调不调用 LLM"],
        "title": "修改学习计划或计划任务",
        "summary": "小调整使用零 LLM 任务微调；范围或优先级重排才调用 Plan Agent。",
        "steps": [
            "在 Plan Agent 卡片点击“任务微调（0 LLM）”添加、移动、修改或删除任务。",
            "只修改当天内容时使用“修改今日”。",
            "目标投递或 JD 范围变化较大时，先“选择 JD 范围并规划”再确认重排。",
            "所有修改保存为新计划版本，不静默覆盖来源关系。",
            "每日执行里的“关联计划任务”用于说明本次留痕对应哪一条计划任务；选择后只把稳定 task_id 写入 daily_log.md，不会自动改写计划或把任务标成完成。",
            "该下拉框来自 GET /api/plan/today 返回的 task_items：后端从 plans/ 中最新计划文件解析当前展示日的核心任务；可选任务不进入下拉框。",
            "若今天不在计划范围或没有独立日块，/api/plan/today 会显示首日、末日或最近一天；关联前应核对“今日计划”上方实际显示的日期和任务。",
        ],
        "boundary": "顶部问答可以解释方案，但不会自动改变计划或启动 Agent。",
    },
    "reflection": {
        "card_id": "daily_reflection",
        "ui_entry": "每日执行 & 复盘",
        "available": True,
        "supported_inputs": ["执行状态", "耗时", "文字留痕", "PDF/图片附件引用", "生成复盘"],
        "required_inputs": ["实际执行内容"],
        "limitations": [
            "复盘证据不会自动修改画像",
            "每日附件不会自动进入正式知识库或通用 RAG 索引",
            "未抽取附件内容时，只能检索附件引用/文件名，不能搜索 PDF 或图片正文",
            "知识库维护当前可单独导入 PDF 等文档；图片不能从每日留痕自动转入",
        ],
        "title": "每日执行附件如何保存和检索",
        "summary": (
            "执行事实和附件均保存在本机，但分属两层：文字记录进入个人复盘检索，"
            "PDF/图片只作为本地附件引用，不自动成为正式知识库内容。"
        ),
        "steps": [
            "在“每日执行 & 复盘”添加 PDF/图片后，文件保存到本机 daily_attachments/日期/，daily_log.md 记录链接。",
            "查询某日做过什么、历史学习或复盘时，顶部问答走 reflection_memory：优先读取本地日志/复盘，主题问题可使用独立的个人复盘派生索引。",
            "这条检索可以找到文字留痕和附件引用，但当前不会解析每日附件的 PDF 正文、图片文字或图像含义。",
            "如果需要检索 PDF 内部知识，请另行从“知识库维护”上传 PDF，经待审确认后进入正式知识库和 reference_kb RAG；不要把同一份文件的每日附件上传误认为已经入库。",
        ],
        "boundary": (
            "顶部问答只读历史执行和复盘，不会替用户补写事实，也不会自动把私人附件"
            "复制、解析或批准进知识库。附件链接本身不能作为已掌握技能的证据。"
        ),
    },
    "resume_agent": {
        "card_id": "resume_workshop",
        "ui_entry": "简历工坊 → 产物范围=完整简历 → Resume Agent → Critic",
        "available": True,
        "supported_inputs": [
            "一条投递记录",
            "该投递的活动 JD",
            "已确认匹配快照",
            "正式个人画像和项目证据",
        ],
        "required_inputs": ["投递记录已关联活动 JD", "存在已确认匹配快照"],
        "limitations": [
            "一次只服务一条投递及其活动 JD",
            "Resume Agent 后固定进入独立 Critic Agent；可解析产物必须完成 LLM 语义评审",
            "一次自动评审周期最多四次 LLM 调用，最多自动修改一次",
            "用户批准前不写入本地简历草稿",
        ],
        "title": "为单条投递生成并审查完整定制简历",
        "summary": (
            "简历工坊的完整简历入口生成一份绑定 application_id + 活动 jd_version_id 的完整 Markdown "
            "简历草稿；Resume Agent 负责生成和修改，独立 Resume Critic Agent 负责语义评审，"
            "再由用户批准、提出修改要求、保存未评审草稿或放弃。"
        ),
        "steps": [
            "先在“投递管理”为目标投递关联活动 JD，并完成匹配确认；然后打开“简历工坊”。",
            "在“用于定制的投递 JD”中选择目标，把产物范围设为“完整简历”，点击唯一的“Resume Agent → Critic”按钮。",
            "Resume Agent 生成带 evidence_refs 的完整结构化简历，硬校验器检查引用与禁用主张，Critic Agent 独立评审真实性、岗位针对性和表达质量。",
            "在审批面板查看评审结果；可直接批准、用自然语言提出修改要求、进入高级编辑，或保存未评审草稿。",
            "批准后才保存到 resume_drafts/<application_id>/；顶部问答本身不会启动这条写入流程。",
        ],
        "boundary": (
            "项目经历和完整简历共用同一个生成按钮与同一组 Resume Agent / Critic Agent；"
            "产物范围只改变输出合同。校验器、路由和保存节点均不是 Agent。"
        ),
    },
    "resume_template": {
        "card_id": "resume_workshop",
        "ui_entry": "简历工坊（上传简历模板/写作指导）",
        "available": True,
        "supported_inputs": ["Markdown/TXT 模板", "写作指导", "真实简历范例"],
        "required_inputs": ["模板或指导文件"],
        "limitations": ["当前生成时会综合已学习材料，暂不支持在顶部问答指定单一模板"],
        "title": "上传简历模板、写作指导或范例",
        "summary": "模板和写作材料由简历工坊单独接收，用于控制输出格式。",
        "steps": [
            "打开“简历工坊（项目 → 简历段）”。",
            "点击“上传简历模板/写作指导”。",
            "上传模板、写作指导或真实范例，并核对材料类型。",
            "之后选择投递记录中已绑定的活动 JD 生成定制项目段。",
        ],
        "boundary": "顶部问答不会上传文件，也不会把临时 JD 分析结果当作正式定制依据。",
    },
    "resume_project": {
        "card_id": "resume_workshop",
        "ui_entry": "简历工坊 → 产物范围=项目经历 → Resume Agent → Critic",
        "available": True,
        "supported_inputs": ["项目仓库/文件/文本", "一条已绑定 JD 的投递", "项目名称"],
        "required_inputs": ["至少一种项目素材"],
        "limitations": [
            "当前可单选投递绑定的活动 JD",
            "当前综合使用已学习的简历材料，不能指定某一份模板",
            "当前不能精确指定输出职责条数",
            "顶部问答只指导，不执行生成",
        ],
        "title": "生成并审查单个项目经历",
        "summary": (
            "产物范围设为“项目经历”后，Resume Agent 依据项目素材生成单个项目经历；"
            "可选绑定真实投递的活动 JD。产物继续经过硬校验、独立 Critic、至多一次自动修订"
            "和用户审批，批准前不会保存。"
        ),
        "steps": [
            "打开“简历工坊”。",
            "通过仓库地址、项目介绍文件或粘贴文本提供项目素材。",
            "把产物范围设为“项目经历”。无需 JD 时直接点击“Resume Agent → Critic”；需要岗位定制时，先选择一条非终态投递并勾选“项目经历使用所选 JD 定制”。",
            "检查 Critic 结果与证据引用，可批准保存、提出修改要求或放弃。",
        ],
        "boundary": (
            "项目段和完整简历是同一 Review 工作流的两个产物范围，不增加 Project Agent。"
            "顶部问答不会直接生成或保存简历；当前也不能通过一句话指定某一模板或固定 4 条职责。"
        ),
    },
}


_MULTI_AGENT_RE = re.compile(
    r"多agent|multi-?agent|多智能体|双agent|agent协作|几个agent|哪些agent|哪个.{0,8}agent"
)

_TOPIC_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("resume_agent", ("writer agent", "critic agent", "写作agent", "审查agent",
                      "writer→critic", "writer+critic", "完整定制简历", "完整简历草稿",
                      "生成定制简历", "完整简历")),
    ("resume_project", ("简历工坊", "项目简历段", "项目经历段", "定制项目经历", "改写项目", "撰写项目", "按所选投递jd定制")),
    ("experience", ("经验总结", "投递经验", "面试经验", "笔试经验", "面经")),
    ("application_jd", ("关联jd", "绑定jd", "更新jd", "补充jd", "补jd", "jd版本", "职位描述")),
    ("resume_template", ("简历模板", "写作指导", "简历范例", "真实简历")),
    ("plan", ("学习计划", "计划任务", "任务微调", "修改计划", "调整计划")),
    ("reflection", ("学习记录", "执行记录", "今日记录", "每日执行", "学习留痕", "生成复盘", "补记", "复盘")),
    ("profile", ("个人画像", "用户画像", "画像建议", "技能状态", "能力状态", "修改画像", "编辑画像")),
    ("project", ("新项目", "新的项目", "这个项目", "项目材料", "项目素材", "项目记忆", "个人项目", "项目入库")),
    ("application", ("投递记录", "新增投递", "新投递", "投递管理", "记录投递", "投递状态")),
    ("knowledge", ("学习资料", "参考资料", "博客", "论文", "文献", "知识库", "上传文档", "导入文档")),
)

_MUTATION_RE = re.compile(
    r"添加|新增|加入|放入|放进|导入|录入|上传|粘贴|保存|补充|补记|关联|绑定|"
    r"更新|修改|改写|撰写|编辑|调整|微调|移动|删除|移除|清空|生成|重生成|写入|入库"
)
# “记录”同时可以是动词（记录一条投递）和名词（我的投递记录）。
# 只有带动作前缀和写入对象时才把它当作产品操作，避免历史查询被抢路由。
_RECORD_ACTION_RE = re.compile(
    r"(?:^记录|(?:想要|想|需要|请|帮我|怎么|如何|怎样).{0,4}记录)"
    r"(?:我的|一条|新的?|今天|本次|投递|经验|执行|学习|到)"
)
_SYSTEM_REQUEST_RE = re.compile(
    r"你(?:能|可以|可否)|能不能|可不可以|是否支持|系统(?:能|可以|支持)|"
    r"offerclaw|平台|帮我|帮助我|为我"
)
_DISCOVERY_RE = re.compile(r"怎么|怎样|如何|在哪|哪里|入口|按钮|流程|方法|能否|可以吗|支持吗|怎么办")
_PRODUCT_CORRECTION_RE = re.compile(
    r"不对|说错|答错|纠正|更正|修正|应该(?:在|是)|不是.{0,18}(?:而是|是在)|并非"
)
# “已注册卡片对象 + 明确操作问法”是 Guide 的高置信语义契约。
# 它不要求用户记住“投递管理”等 UI 全称，但也不会把“岗位怎么选”
# 这类业务建议误当产品帮助。
_OPERATION_HOWTO_RE = re.compile(
    r"(?:怎么|怎样|如何|该怎么|要怎么|在哪|哪里).{0,10}"
    r"(?:操作|使用|记录|新增|添加|加入|放入|放进|导入|录入|上传|关联|绑定|"
    r"更新|修改|编辑|调整|微调|删除|保存|补充|补记|生成)|"
    r"(?:记录|新增|添加|加入|放入|放进|导入|录入|上传|关联|绑定|更新|"
    r"修改|编辑|调整|微调|删除|保存|补充|补记|生成).{0,14}"
    r"(?:怎么操作|如何操作|怎么做|如何做|去哪里|在哪|哪里|哪个卡片|入口|流程)"
)
_PRODUCT_SURFACE_RE = re.compile(
    r"offerclaw|系统|平台|功能|卡片|入口|按钮|简历工坊|投递管理|jd分析|"
    r"知识库维护|用户画像|planagent|任务微调|每日执行"
)
_HISTORICAL_FACT_RE = re.compile(
    r"(?:已经|刚刚|刚才|之前|曾经|今天).{0,10}"
    r"(?:添加了|加入了|导入了|上传了|记录了|修改了|删除了).{0,10}"
    r"(?:哪些|什么|哪个|哪条|是否)"
)
_EXPLICIT_KB_CONTENT_LOOKUP_RE = re.compile(
    r"(?:从|在|根据|结合)(?:我的|个人)?(?:知识库|参考资料|学习资料)(?:里|中)?"
    r"(?:查|找|搜索|检索|回答|解释|说明)?|"
    r"(?:查|找|搜索|检索)(?:一下)?(?:我的|个人)?(?:知识库|参考资料|学习资料)"
)


def _normalise(text: str) -> str:
    return re.sub(r"[\s\u3000]+", "", str(text or "").strip().lower())


def is_explicit_knowledge_content_lookup(question: str) -> bool:
    """Return true when the user asks for content from their curated sources."""
    q = _normalise(question)
    return bool(
        q
        and not (_MUTATION_RE.search(q) or _RECORD_ACTION_RE.search(q))
        and _EXPLICIT_KB_CONTENT_LOOKUP_RE.search(q)
    )


def _topics(text: str) -> list[str]:
    found: list[str] = []
    if (_MULTI_AGENT_RE.search(text)
            and any(x in text for x in ("简历", "resume", "项目经历段", "项目简历段"))):
        found.append("resume_agent")
    if ("项目" in text and any(x in text for x in ("jd", "简历模板", "写作指导"))
            and any(x in text for x in ("生成", "改写", "撰写", "定制", "怎么写", "如何写"))
            and not _MULTI_AGENT_RE.search(text)):
        # 这里的 JD、模板名和项目名是简历生成的输入/限定，不是另外两个待执行
        # 的“关联 JD”或“维护知识库”请求。只返回一个卡片能力，避免用途词
        # 和模板标题触发多余指导。
        return ["resume_project"]
    if re.search(r"(?:关联|绑定|更新|补充|补).{0,5}jd|jd.{0,5}(?:关联|绑定|更新|补充)", text):
        found.append("application_jd")
    for key, patterns in _TOPIC_PATTERNS:
        if key not in found and any(pattern in text for pattern in patterns):
            found.append(key)
    if ("application" in found
            and any(topic in found for topic in ("resume_agent", "resume_project"))
            and not any(x in text for x in ("投递记录", "新增投递", "记录投递", "投递状态"))):
        found.remove("application")
    if not found and re.search(
        r"(?:添加|新增|加入|导入|录入|上传|保存).{0,6}(?:一个|新的?)?项目|"
        r"项目.{0,6}(?:添加|加入|导入|上传|保存|入库)", text,
    ):
        found.append("project")
    if not found and re.search(r"(?:添加|新增|记录|更新).{0,6}(?:一条|新的?)?投递", text):
        found.append("application")
    return found[:3]


def _action(text: str) -> str:
    if re.search(r"删除|移除|清空", text):
        return "delete"
    if re.search(r"关联|绑定", text):
        return "link"
    if re.search(r"生成|重生成|改写|撰写", text):
        return "generate"
    if re.search(r"更新|修改|编辑|调整|微调|移动", text):
        return "update"
    if re.search(r"上传|导入|粘贴", text):
        return "upload"
    if re.search(r"添加|新增|加入|放入|放进|录入|记录|保存|补充|补记|写入|入库", text):
        return "create"
    return "locate"


def detect_product_help_request(question: str, *, context: str = "") -> dict[str, Any]:
    """识别系统操作/能力询问，不读取任何个人数据。"""
    q = _normalise(question)
    if not q:
        return {"matched": False, "topics": [], "action": "", "confidence": 0.0}
    has_mutation = bool(_MUTATION_RE.search(q) or _RECORD_ACTION_RE.search(q))
    has_request = bool(_SYSTEM_REQUEST_RE.search(q))
    product_surface = bool(_PRODUCT_SURFACE_RE.search(q))
    context_topics = _topics(_normalise(context)) if context else []
    is_correction = bool(
        _PRODUCT_CORRECTION_RE.search(q) and (product_surface or context_topics)
    )
    if _HISTORICAL_FACT_RE.search(q) and not _DISCOVERY_RE.search(q):
        return {"matched": False, "topics": [], "action": "", "confidence": 0.0}
    if is_explicit_knowledge_content_lookup(q):
        # “从知识库查 X 怎么处理”中的“怎么”描述的是待检索内容，
        # 不是在询问知识库卡片怎么操作。显式内容检索优先走 reference_kb。
        return {"matched": False, "topics": [], "action": "", "confidence": 0.0}
    topics = _topics(q)
    if context_topics and (is_correction or not topics and has_mutation):
        topics.extend(topic for topic in context_topics if topic not in topics)
        topics = topics[:3]
    personal_content_query = bool(
        not has_mutation
        and any(x in q for x in ("我的", "根据我", "结合我", "按我"))
        and any(x in q for x in (
            "项目材料", "项目内容", "简历规则", "简历格式", "复盘记录", "投递经验",
        ))
        and not any(x in q for x in ("哪里", "在哪", "入口", "按钮", "卡片", "怎么用"))
    )
    if personal_content_query:
        # 用户在问“如何根据我的已存内容回答”，答案对象是个人材料；
        # OfferClaw/项目等词只是内容或用途，不应把问题升级成 Guide。
        return {"matched": False, "topics": [], "action": "", "confidence": 0.0}
    direct_command = bool(re.match(
        r"^(?:请)?(?:帮我)?(?:添加|新增|加入|放入|放进|导入|录入|记录|上传|补充|关联|更新|修改|改写|撰写|生成)", q,
    ))
    resume_command = bool("resume_project" in topics and has_mutation)
    operation_howto = bool(topics and _OPERATION_HOWTO_RE.search(q))
    discovery = bool(_DISCOVERY_RE.search(q) and product_surface)
    matched = bool(topics and (
        has_mutation and (has_request or direct_command or resume_command)
        or discovery
        or operation_howto
        or is_correction
    ))
    if not matched:
        return {"matched": False, "topics": [], "action": "", "confidence": 0.0}
    return {
        "matched": True,
        "topics": topics,
        "action": _action(q),
        "confidence": 0.97 if (_SYSTEM_REQUEST_RE.search(q) or operation_howto) else 0.94,
        "reason": (
            "识别到对 OfferClaw 功能说明的纠正"
            if is_correction else "识别到面向 OfferClaw 的数据管理或功能入口请求"
        ),
        "request_kind": "correction" if is_correction else "guide",
        "registry_version": CARD_CAPABILITY_REGISTRY_VERSION,
    }


def render_product_help(topics: list[str], action: str = "locate", *,
                        capability_ids: list[str] | None = None,
                        action_request: dict[str, Any] | None = None) -> str:
    """以确定性文本回答产品操作；不调用 LLM，也不产生写入。"""
    normalized_ids = normalize_capability_ids(capability_ids or [])
    if action_request and normalized_ids:
        return action_guidance(normalized_ids, action_request)
    if normalized_ids and not topics:
        topics = capability_topics(normalized_ids)
    valid = [topic for topic in topics if topic in CARD_CAPABILITY_REGISTRY]
    if not valid:
        return (
            "### OfferClaw 使用说明\n"
            "顶部问答框保持只读。请说明你想操作的是项目、投递/JD、画像、学习计划、"
            "每日复盘、简历模板还是知识库资料，我会指出对应入口和确认流程。"
        )
    blocks = []
    if action == "delete":
        blocks.append(
            "⚠️ 顶部问答框不执行删除。当前没有统一的问答删除入口；如果对应卡片没有明确的"
            "删除按钮，就表示该删除操作尚未开放，系统不会假装已经删除。"
        )
    for topic in valid:
        item = CARD_CAPABILITY_REGISTRY[topic]
        steps = "\n".join(f"{index}. {step}" for index, step in enumerate(item["steps"], start=1))
        supported = "、".join(item.get("supported_inputs") or []) or "无额外输入"
        limitations = "\n".join(f"- {value}" for value in (item.get("limitations") or []))
        block = (
            f"### {item['title']}\n\n"
            f"入口：**{item['ui_entry']}**\n\n"
            f"{item['summary']}\n\n"
            f"当前支持：{supported}\n\n"
            f"{steps}\n\n"
        )
        if limitations:
            block += f"当前限制：\n{limitations}\n\n"
        blocks.append(block + f"边界：{item['boundary']}")
    return "\n\n".join(blocks)


def card_guidance(topics: list[str], *,
                  capability_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """供 SSE/UI 展示真实卡片入口；只返回注册过的能力元数据。"""
    guidance = [{
        "topic": topic,
        "card_id": CARD_CAPABILITY_REGISTRY[topic]["card_id"],
        "ui_entry": CARD_CAPABILITY_REGISTRY[topic]["ui_entry"],
        "available": bool(CARD_CAPABILITY_REGISTRY[topic].get("available")),
        "required_inputs": list(CARD_CAPABILITY_REGISTRY[topic].get("required_inputs") or []),
        "limitations": list(CARD_CAPABILITY_REGISTRY[topic].get("limitations") or []),
    } for topic in topics if topic in CARD_CAPABILITY_REGISTRY]
    for capability_id in normalize_capability_ids(capability_ids or []):
        capability = ACTION_CAPABILITY_REGISTRY.get(capability_id)
        if capability is None:
            continue
        guidance.append({
            "capability_id": capability.capability_id,
            "topic": capability.card_topic,
            "card_id": CARD_CAPABILITY_REGISTRY.get(
                capability.card_topic, {}
            ).get("card_id", ""),
            "ui_entry": capability.ui_entry,
            "available": capability.available,
            "required_inputs": list(capability.required_inputs),
            "limitations": [f"顶部问答策略：{capability.top_chat_policy}"],
        })
    return list({
        (item.get("capability_id") or f"topic:{item.get('topic')}"): item
        for item in guidance
    }.values())
