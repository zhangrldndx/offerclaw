# -*- coding: utf-8 -*-
"""
OfferClaw · FastAPI 服务层

把 RAG 检索、岗位匹配、用户画像查询包装为 HTTP API。
覆盖常见 AI 应用 JD 中的系统原型、接口开发和部署职责

启动方式：
  uvicorn rag_api:app --host 127.0.0.1 --port 8000 --reload   # 需局域网共享实例时才显式改 host

测试方式：
  # 浏览器访问 http://localhost:8000/docs 打开 Swagger UI
  # 或用 curl:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/api/query -d '{"query": "我的求职方向"}'
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal, Optional
from contextlib import asynccontextmanager
import base64
import json
import os
import re
import sys
import datetime
import threading
import asyncio
import hashlib
import hmac
import ipaddress
from pathlib import Path

import requests as _requests
from memory_transactions import MemoryFileConflictError, write_text_with_memory

# 确保能找到 rag_tools
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
DAILY_ATTACHMENT_DIR = os.path.join(BASE_DIR, "daily_attachments")
DAILY_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
DAILY_ATTACHMENT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
HTML_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# 自动加载 .env.local（本地私密 Key，不进 git），补充到环境变量
_env_local = os.path.join(BASE_DIR, ".env.local")
if os.path.exists(_env_local):
    with open(_env_local, encoding="utf-8") as _f:
        for _ln in _f:
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#") and "=" in _ln:
                _k, _v = _ln.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from logging_utils import (
    get_logger,
    request_logging_middleware,
    current_request_id,
)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Recover durable writes and mark route prototypes as offline-only."""
    try:
        from memory_store import MemoryStore
        MemoryStore().recover_file_operations()
    except Exception:
        _log.warning("memory operation recovery failed", exc_info=True)
    _route_warmup_status.update({
        "status": "offline_only",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "detail": {"reason": "v4 structured planner owns online routing"},
    })
    if os.environ.get("OFFERCLAW_QUERY_SERVICE") == "1":
        from query_service import warm_query_runtime

        await asyncio.to_thread(warm_query_runtime)
    try:
        yield
    finally:
        # A graceful restart must not silently discard selected Window A/B
        # jobs.  This remains bounded; a stalled provider cannot hold shutdown
        # forever, and any remaining work is reported by the window audit.
        try:
            from rag_shadow_answerability import drain as _drain_shadow
            _drain_shadow(timeout=30.0)
        except Exception:
            pass


app = FastAPI(
    title="OfferClaw API",
    description="求职作战 Agent 的 HTTP API 接口层（RAG + 岗位匹配 + 用户画像 + SSE 流式）",
    version="1.1.0",
    lifespan=_lifespan,
)

app.middleware("http")(request_logging_middleware)

# Human-facing requests default to organic. Automated clients must declare
# their provenance so replay/synthetic traffic cannot enter an organic window
# merely because it traversed the real HTTP route.
from traffic_origin import traffic_origin_middleware  # noqa: E402

app.middleware("http")(traffic_origin_middleware)

# 多用户预留(tenancy seam):单用户零行为变化;请求头 X-OfferClaw-User
# 可切租户上下文,为 per-tenant collection / 配额 / 用量归属立好接缝。
# 详见 docs/MULTI_USER_ROADMAP.md。
from tenancy import current_tenant, tenant_middleware  # noqa: E402

app.middleware("http")(tenant_middleware)

_log = get_logger("offerclaw.api")

# 延迟加载 RAG Agent（避免启动时阻塞）
_rag_agent = None

# v3 的路由原型只供离线评测，线上不再预热或执行。
_route_warmup_lock = threading.Lock()
_route_warmup_status: dict = {
    "status": "offline_only", "started_at": "",
    "finished_at": "", "detail": {"reason": "v4 structured planner"},
}


def _active_llm_api_key() -> tuple[str, str]:
    """Return the active chat API key and env name from day1_api_starter config."""
    from day1_api_starter import get_llm_config

    cfg = get_llm_config()
    return cfg["api_key"], cfg["api_key_env"]


def get_rag_agent():
    """懒加载 RAG Agent"""
    global _rag_agent
    if _rag_agent is None:
        from rag_agent import RAGAgent
        _rag_agent = RAGAgent()
    return _rag_agent


def _safe_daily_attachment_name(name: str) -> str:
    """Return a local-safe file name while preserving the user's extension."""
    raw = os.path.basename(name or "").strip()
    stem, ext = os.path.splitext(raw)
    ext = ext.lower()
    stem = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", stem).strip("._-")
    if not stem:
        stem = "attachment"
    return f"{stem[:80]}{ext}"


def _unique_path(directory: str, filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    idx = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}-{idx}{ext}")
        idx += 1
    return candidate


def _decode_attachment_data(data_base64: str) -> bytes:
    payload = (data_base64 or "").strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="附件内容不是有效的 base64") from exc


def _html_file_response(filename: str) -> FileResponse:
    """Serve local UI HTML without browser cache during active iteration."""
    return FileResponse(
        os.path.join(BASE_DIR, "static", filename),
        headers=HTML_NO_CACHE_HEADERS,
    )


def _static_rev(filename: str) -> int:
    path = os.path.join(BASE_DIR, "static", filename)
    return int(os.path.getmtime(path))


# =====================================================
# 数据模型
# =====================================================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_retrieval: bool = True
    context: list[str] = Field(default_factory=list, max_length=3)
    conversation_id: str = ""
    message_id: str = ""
    operation_id: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieval_count: int
    timestamp: str
    in_kb: bool = True            # P2.5：是否命中知识库
    sources: list[str] = []       # 命中时的来源文件名
    matched_by: str = ""          # vector | lexical_rescue | ""
    # 门的结构化动作(answer/correct_premise/abstain):与 /api/stream 的
    # meta 同一枚章,生成用了哪份合同在两条路径上都可观测(P0 修复配套)
    answer_action: str = "answer"
    mode: str = ""
    planner_mode: str = ""
    resolver_mode: str = ""
    decision: str = "answer"
    interaction_kind: str = "query"
    turn_relation: str = "standalone"
    relation_target_turn_id: str = ""
    action_request: dict = Field(default_factory=dict)
    routing_assurance: str = "degraded"
    decision_reasons: list[str] = Field(default_factory=list)
    context_resolution: dict = Field(default_factory=dict)
    routes: list[dict] = Field(default_factory=list)
    intent_frame: dict = Field(default_factory=dict)
    planner_engine: str = ""
    planner_version: str = ""
    schema_version: str = ""
    route_model: str = ""
    repair_used: bool = False
    fallback_reason: str = ""
    planner_queue_ms: float = 0.0
    planner_provider_ms: float = 0.0
    planner_wall_ms: float = 0.0
    planner_deadline_ms: float = 0.0
    planner_cache_hit: bool = False
    planner_timeout_stage: str = ""
    planner_late_response: bool = False
    planner_circuit_state: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    structured_output_mode: str = ""
    route_gateway: str = ""
    route_reasoning_effort: str = ""
    json_capability: str = ""
    retrieval_profile: str = ""
    index_fingerprint: str = ""
    effective_hit: bool | None = None
    conversation_id: str = ""
    user_message_id: str = ""
    assistant_message_id: str = ""
    data_version: str = ""
    model_usage: dict = Field(default_factory=dict)
    trace_id: str = ""
    query_service_version: str = ""


class WeChatQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["offerclaw.wechat-query.request.v1"]
    question: str = Field(min_length=1, max_length=12_000)
    conversation_id: str = Field(min_length=16, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    message_id: str = Field(min_length=16, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    operation_id: str = Field(min_length=16, max_length=160, pattern=r"^[a-zA-Z0-9:_-]+$")
    top_k: int = Field(default=5, ge=1, le=20)


def _wechat_query_token_path() -> Path:
    configured = os.environ.get("OFFERCLAW_WECHAT_QUERY_TOKEN_FILE", "").strip()
    if configured:
        return Path(configured)
    return Path.home() / ".offerclaw-runtime" / "wechat-query.token"


def _require_wechat_query_auth(request: Request) -> None:
    host = str(request.client.host if request.client else "")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise HTTPException(status_code=403, detail="loopback source required")
    if request.headers.get("X-OfferClaw-Traffic-Origin", "").strip().lower() != "wechat_direct":
        raise HTTPException(status_code=403, detail="wechat_direct origin required")
    try:
        expected = _wechat_query_token_path().read_text(encoding="ascii").strip()
    except OSError:
        raise HTTPException(status_code=503, detail="internal query token unavailable")
    supplied = request.headers.get("X-OfferClaw-Internal-Token", "").strip()
    if len(expected) < 32 or not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=401, detail="invalid internal query token")


def _conversation_ids(req: QueryRequest) -> tuple[str, str, str, str]:
    from memory_store import new_id
    return (
        req.conversation_id.strip() or new_id("conv"),
        req.message_id.strip() or new_id("msg"),
        new_id("msg"),
        req.operation_id.strip() or current_request_id() or new_id("op"),
    )


def _record_conversation(role: str, content: str, state: str, *, conversation_id: str,
                         message_id: str, operation_id: str, source_refs=None) -> str:
    from memory_layers import record_business_event
    event = record_business_event(
        "conversation_message",
        {"role": role, "content": content or "(empty response)", "state": state,
         "message_id": message_id, "source_refs": source_refs or []},
        actor="user" if role == "user" else "assistant", source="top_chat",
        operation_id=f"{operation_id}:{role}", entity_type="conversation_message",
        entity_id=message_id, conversation_id=conversation_id,
        causation_id=None if role == "user" else operation_id,
        business_date=datetime.date.today().isoformat(),
    )
    return event["event_id"]


def _record_content_artifact(kind: str, content: str, *, action: str,
                             source: str, operation_id: str = "",
                             source_path: str = "", source_type: str = "",
                             application_id: str = "", saved_path: str = "",
                             result: dict | None = None,
                             actor: str = "user") -> dict:
    """Persist an artifact snapshot and its typed memory event."""
    from memory_layers import EpisodicMemory, record_business_event
    epi = EpisodicMemory()
    snapshot = epi.store.put_snapshot(content, media_type="text/markdown",
                                      source_path=source_path)
    if kind == "knowledge_material_changed":
        payload = {"action": action, "source_path": source_path,
                   "source_type": source_type, "snapshot_id": snapshot["snapshot_id"],
                   "content_hash": snapshot["content_hash"], "result": result or {}}
        entity_type = "knowledge_material"
        entity_id = source_path or snapshot["snapshot_id"]
    elif kind == "resume_artifact_changed":
        payload = {"action": action, "snapshot_id": snapshot["snapshot_id"],
                   "content_hash": snapshot["content_hash"],
                   "application_id": application_id, "saved_path": saved_path}
        entity_type = "resume_artifact"
        entity_id = saved_path or snapshot["snapshot_id"]
    else:
        raise ValueError(f"unsupported artifact kind: {kind}")
    return record_business_event(
        kind, payload, actor=actor, source=source,
        operation_id=operation_id or None, entity_type=entity_type, entity_id=entity_id,
    )


class MatchRequest(BaseModel):
    jd_text: str
    use_semantic: bool = False


class MatchResponse(BaseModel):
    status: str
    status_code: str = "unknown"
    summary: str
    direction: str = ""
    gap_list: dict = {}
    suggestions: list = []
    requirement_analysis: dict = {}
    matching_mode: str = "deterministic"
    semantic_status: str = "not_requested"


class FlowRunRequest(BaseModel):
    jd_text: str
    jd_title: str = "未命名 JD"
    skip_llm: bool = True


class FlowRunResponse(BaseModel):
    match_report: dict = {}
    gaps: dict = {}
    plan_outline: list = []
    today_advice: dict = {}
    resume_skeleton: dict = {}
    application_suggestion: dict = {}
    requires_confirmation: list = []
    trace: list = []
    errors: list = []
    route_history: list = []
    checkpoint_status: dict = {}
    memory_status: dict = {}
    fatal_error: bool = False
    fatal_reason: str = ""


class PlanRequest(BaseModel):
    gaps: str = ""
    revision_note: str = ""   # LLM 修改计划：用户一次性修改要求（无记忆，不落盘）
    start_date: str = ""      # 计划开始日期（用户定，空=今天）
    end_date: str = ""        # 结束日期（空=系统按任务量估算周期）


class PortfolioScopeRequest(BaseModel):
    instruction: str = ""
    application_ids: list[str] = []
    include_profile_goals: bool = True


class AgentFlowStartRequest(BaseModel):
    task: str
    scope_snapshot_id: str = ""
    application_id: str = ""
    jd_version_id: str = ""
    start_date: str = ""
    end_date: str = ""
    revision_note: str = ""
    resume_scope: str = "full_resume"
    project_repo_url: str = ""
    project_text: str = ""
    project_name: str = ""
    stage_project_memory: bool = False
    resume_source_text: str = ""


class AgentFlowResumeRequest(BaseModel):
    decision: str
    edited_content_md: str = ""
    change_request: str = Field(default="", max_length=2000)
    remember_preference: bool = False
    artifact_revision: Optional[int] = Field(default=None, ge=0)


class PlanResponse(BaseModel):
    plan_md: str
    saved_path: str = ""
    draft_id: str = ""
    requires_confirmation: bool = False
    daily_days: int = -1   # 生成后校验门:日计划层解析出的天数;0=格式打穿(前端警示),-1=未知


class CurrentPlanResponse(BaseModel):
    has_plan: bool = False
    content: str = ""
    filename: str = ""
    mtime: int = 0
    edited_by_user: bool = False
    target_status: dict = {}
    profile_status: dict = {}


class PlanSaveRequest(BaseModel):
    content: str
    note: str = ""        # 用户编辑说明（可选），记入记忆事件
    operation_id: str = ""


class PlanDraftDecisionRequest(BaseModel):
    decision: str


class PlanTodayResponse(BaseModel):
    """今日计划视图：整体计划 md 的"当日切片"（单一事实源，无副本可漂移）。"""
    has_plan: bool = False
    has_daily: bool = False
    status: str = ""            # in_range / in_range_nearest / before_start / after_end / no_daily / no_plan
    period_start: str = ""      # 当前计划周期（供前端把日期选择框预填成最新计划的日期）
    period_end: str = ""
    date: str = ""              # 实际展示的日期（可能与请求日不同，见 status）
    requested_date: str = ""
    label: str = ""             # 如 D5（08-08 周六）
    day_index: int | None = None
    total_days: int = 0
    week_n: int | None = None
    week_theme: str = ""
    tasks: list[str] = []
    optional_tasks: list[str] = []
    task_items: list[dict] = []
    hint: str = ""
    plan_file: str = ""
    plan_mtime: int = 0
    edited_by_user: bool = False


class PlanTodaySaveRequest(BaseModel):
    date: str                     # 要回写的日期（ISO，取自 GET 返回的 date）
    tasks: list[str] = []         # 新的今日任务列表（清空=当日休整）
    base_mtime: int | None = None  # 前端读到的计划 mtime；不匹配返回 409 防盲写
    note: str = ""


class PlanTaskOperation(BaseModel):
    op: str
    task_id: str = ""
    date: str = ""
    text: str = ""
    optional: bool | None = None
    estimated_hours: float | None = None
    priority: str = ""
    deliverable: str = ""


class PlanTaskPatchRequest(BaseModel):
    base_mtime: int | None = None
    operations: list[PlanTaskOperation] = []
    note: str = ""


class KBAddUrlRequest(BaseModel):
    url: str
    operation_id: str = ""


class KBPathRequest(BaseModel):
    rel: str                # knowledge_base 内相对路径（可带 knowledge_base/ 前缀）
    source_type: str = ""   # ingest_path 可选覆盖；默认按子目录推断
    operation_id: str = ""


class KBAddFileRequest(BaseModel):
    name: str
    content_base64: str = ""
    text: str = ""        # 也允许直接传纯文本（二选一）
    parser: str = ""        # ""/"text"=文字层(默认);"docling"=结构化(论文/表格,opt-in)
    operation_id: str = ""


class KBPromoteRequest(BaseModel):
    pending_file: str     # _score_and_save 返回的 saved（相对路径）
    to_subdir: str        # career_paths / experience_posts / learning_resources
    operation_id: str = ""


class GapTargetRequest(BaseModel):
    jd_text: str = ""
    gaps: dict = {}       # match 产出的 {分类: [条目...]}
    title: str = ""
    company: str = ""


class ApplicationUpsertRequest(BaseModel):
    company: str
    position: str
    status: str                  # applications_store.STATUSES 之一
    date: str = ""               # 默认今天
    source: str = ""
    location: str = ""
    next_action: str = ""
    note: str = ""
    experience: str = ""         # 经验总结（笔试/面试真题、流程、教训）
    experience_stage: str = ""   # 笔试 / 一面 / 二面 / HR面 / 终面 / 其他
    add_to_kb: bool = False      # 经验是否加入知识库（指导 RAG 与学习计划）
    application_id: str = ""     # 新客户端优先按稳定 ID 更新
    include_in_plan: Optional[bool] = None
    long_term_follow: Optional[bool] = None
    plan_priority: str = "medium"
    operation_id: str = ""


class ApplicationFromJDPreviewRequest(BaseModel):
    jd_text: str
    source_url: str = ""
    application_id: str = ""


class ApplicationFromJDCommitRequest(BaseModel):
    jd_text: str
    source_url: str = ""
    expected_content_hash: str
    mode: str = "create"         # create | link_existing
    application_id: str = ""
    company: str
    position: str
    location: str = ""
    source: str = ""
    status: str = "已评估"
    date: str = ""
    next_action: str = "决定投递/不投递"
    note: str = ""
    include_in_plan: bool = False
    long_term_follow: bool = False
    plan_priority: str = "medium"
    operation_id: str = ""
    preview_id: str = ""


class ApplicationPatchRequest(BaseModel):
    company: str = ""
    position: str = ""
    status: str = ""
    date: str = ""
    source: str = ""
    source_url: str = ""
    location: str = ""
    next_action: str = ""
    note: str = ""
    include_in_plan: Optional[bool] = None
    long_term_follow: Optional[bool] = None
    plan_priority: str = ""
    operation_id: str = ""


class ResumeProjectRequest(BaseModel):
    repo_url: str = ""           # 项目仓库地址（GitHub 公开仓库优先）
    text: str = ""               # 项目介绍文本（或上传文件解码后的内容）
    project_name: str = ""
    application_id: str = ""     # 非空→精确加载该投递绑定的活动 JD，不重新识别 JD
    jd_version_id: str = ""      # UI 看到的版本，用于防止生成期间活动版本被切换
    jd_text: str = ""            # 兼容旧调用；新 UI 的 JD 定制不再发送临时 JD 文本
    stage_memory: bool = False    # True→只放入 _pending，仍需用户预览确认后才正式入库


class ResumeTemplateUploadRequest(BaseModel):
    name: str                    # 文件名（.md/.txt）
    content_base64: str = ""
    text: str = ""
    stage_for_rag: bool = True  # 模板本地学习后，另放一份 RAG 候选等待人工确认


class DailyResponse(BaseModel):
    today_log: str = ""
    recent_summary: str = ""
    recent_days: int = 7


class DailyAppendRequest(BaseModel):
    text: str


class DailyLogStructuredRequest(BaseModel):
    tag: str = ""
    done: list[str] = []
    todo: list[str] = []
    notes: str = ""
    task_id: str = ""
    status: str = "done"
    minutes: int | None = None
    attachment_refs: list[str] = []
    operation_id: str = ""


class ReflectionRunRequest(BaseModel):
    date: str = ""
    mode: str = "daily"


class MemoryGoalSwitchRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class MemoryLifecycleRequest(BaseModel):
    lifecycle: str


class MemoryDeleteRequest(BaseModel):
    reason: str = "用户主动删除"


class DailyAttachmentPayload(BaseModel):
    name: str
    content_type: str = ""
    data_base64: str


class DailyAttachmentRequest(BaseModel):
    files: list[DailyAttachmentPayload] = []


class ResumeResponse(BaseModel):
    pitch: str = ""
    stories_preview: str = ""


class TodayResponse(BaseModel):
    today: str
    headline: str
    reason: str = ""
    source: str = ""
    next_actions: list[str] = []
    adjustments: list[str] = []  # P2：复盘沉淀的次日调整规则
    plan: dict = {}              # 当前学习计划的本周重点（自动化引用的最新计划）
    today_plan: list[str] = []   # 今日对照清单（每日执行卡对照 + 未完成自动判定基准）
    plan_drift: dict = {}        # 计划偏离评估（none/info/warn）：计划只在用户主动重排时改变
    stats: dict = {}


class DiscoverRequest(BaseModel):
    raw: str = ""
    url: str = ""


class DiscoverResponse(BaseModel):
    company: str = ""
    title: str = ""
    location: str = ""
    job_type: str = ""
    skills_detected: list[str] = []
    duties: str = ""
    requirements: str = ""
    career_domain: str | None = None
    role_family: str | None = None
    programming_language_requirement: dict = {}
    raw_chars: int = 0
    source_url: str = ""
    source_type: str = ""
    source_credibility: str = ""
    notice: str = ""
    jd_text: str = ""


class ResumeBuildRequest(BaseModel):
    jd_text: str
    company: str = ""
    title: str = ""
    application_id: str = ""
    operation_id: str = ""


class ResumeBuildResponse(BaseModel):
    resume_md: str
    jd_summary_chars: int = 0


class ResumeMarkdownRequest(BaseModel):
    jd_text: str = ""
    skip_llm: bool = True


class ResumeMarkdownResponse(BaseModel):
    resume_md: str
    sections: list[str] = []
    jd_chars: int = 0
    skip_llm: bool = True
    llm_used: bool = False
    llm_error: str = ""


class JDQueriesResponse(BaseModel):
    queries: list[str] = []
    profile_cities: list[str] = []
    profile_directions: list[str] = []


class JDCandidate(BaseModel):
    title: str
    jd_text: str


class JDRankRequest(BaseModel):
    candidates: list[JDCandidate]


class JDRankItem(BaseModel):
    title: str
    status: str = ""
    direction: str = ""
    gap_count: int = 0
    score: int = 0
    reason: str = ""


class JDRankResponse(BaseModel):
    ranked: list[JDRankItem]
    total: int = 0


class ProfileResponse(BaseModel):
    name: str
    direction: list[str]
    skills_summary: str
    updated_at: str
    data_version: str = ""


class ProfilePatchRequest(BaseModel):
    content_md: str
    base_hash: str
    reason: str = ""
    operation_id: str = ""


class ProfileEditPreviewRequest(BaseModel):
    content_md: str
    base_revision: int


class ProfileEditCommitRequest(BaseModel):
    preview_id: str
    reason: str = ""
    operation_id: str = ""


class ProfileSuggestionDecisionRequest(BaseModel):
    decision: Literal["accepted", "modified", "rejected"]
    modified_text: str = ""
    modified_value: Any = None
    base_revision: int | None = None
    reason: str = ""
    operation_id: str = ""


class AgentRequest(BaseModel):
    message: str
    mode: str = "deterministic"  # 'deterministic' | 'llm'
    max_steps: int = 3


class AgentResponse(BaseModel):
    answer: str
    tool_calls: list = []
    mode: str = "deterministic"
    steps: int = 0
    errors: list = []


def _parse_profile(content: str) -> tuple[str, list[str], str, str]:
    """从 user_profile.md 解析关键字段；缺字段时降级为占位值，不再硬编码。"""
    import re
    name = "未填写"
    updated_at = "未知"
    direction: list[str] = []
    skills_summary = "未提取"

    m = re.search(r"姓名[^：:]*[：:]\s*([^\n]+)", content)
    if m:
        name = m.group(1).strip().strip("【】 ")
    m = re.search(r"最近更新时间[：:]\s*([0-9\-/]+)", content)
    if m:
        updated_at = m.group(1).strip()

    block = re.search(r"目标方向[^\n]*\n((?:\s*\d+\.[^\n]+\n?)+)", content)
    if block:
        direction = [
            re.sub(r"^\s*\d+\.\s*", "", ln).strip()
            for ln in block.group(1).splitlines() if ln.strip()
        ]

    sk = re.search(r"##\s*3\.[^\n]*技能[\s\S]{0,800}", content)
    if sk:
        bullets = re.findall(r"-\s*([^\n]+)", sk.group(0))
        if bullets:
            skills_summary = "; ".join(b.strip() for b in bullets[:6])
    return name, direction, skills_summary, updated_at


# =====================================================
# API 路由
# =====================================================

@app.get("/")
async def root():
    """根路径：重定向到友好 UI。"""
    return RedirectResponse(url="/ui")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/ui")
async def ui(request: Request):
    """OfferClaw 控制台（零依赖单页应用）。"""
    if not request.url.query:
        return RedirectResponse(
            url=f"/ui?rev={_static_rev('index.html')}",
            status_code=307,
            headers=HTML_NO_CACHE_HEADERS,
        )
    return _html_file_response("index.html")


@app.get("/fonts/{filename}")
async def ui_font(filename: str) -> FileResponse:
    """展示字体：static/fonts/ 下的 woff2（index.html 的 @font-face 指向这里）。

    不用 StaticFiles 挂整个 static/：这里只需要暴露字体，挂目录会把以后放进
    该目录的任何东西一并变成可下载资源。防护与 ``get_daily_attachment`` 同构：
    basename 化 + 后缀白名单 + 解析后路径必须仍在 fonts/ 内。第三道当前是冗余的
    （basename 已消除分隔符），留着是因为前两道任何一道日后被放宽时它仍然成立。

    2026-09-01 补：此前 index.html 的 @font-face 一直指向 fonts/*.woff2，而服务端
    没有任何静态挂载——字体全程 404，页面靠 <head> 里的 Google Fonts 外链兜底才
    显示正常。把外链删掉（本地优先，不每次加载都打 googleapis）之后这个缺口才暴露。
    """
    name = os.path.basename(filename)
    if not name.endswith(".woff2"):
        raise HTTPException(status_code=404, detail="字体不存在")
    root = os.path.abspath(os.path.join(BASE_DIR, "static", "fonts"))
    path = os.path.abspath(os.path.join(root, name))
    if not path.startswith(root + os.sep) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="字体不存在")
    # 字体是稳定二进制资产，与 HTML 的 no-cache 不同档；但不用 immutable/一年，
    # 免得重新下载字体后浏览器长期抱着旧文件。
    return FileResponse(path, media_type="font/woff2",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/info")
async def info():
    """API 元信息（原 / 的内容）。"""
    return {
        "name": "OfferClaw API",
        "version": "1.1.0",
        "endpoints": {
            "GET /": "→ 重定向 /ui",
            "GET /ui": "友好控制台（推荐）",
            "GET /ui/console": "求职流程 Stepper 控制台（V3 Phase 3）",
            "GET /health": "健康检查",
            "GET /api/info": "本接口（元信息 + 路由清单）",
            "GET /api/profile": "用户画像摘要",
            "GET/PATCH /api/profile/editor|profile": "用户确认画像编辑（内容哈希防并发覆盖）",
            "GET/POST /api/profile/suggestions": "证据驱动的画像建议审批",
            "POST /api/query": "RAG 问答（一次性）",
            "POST /api/stream": "RAG 问答（SSE 流式）",
            "POST /api/search": "仅检索",
            "POST /api/match": "岗位匹配（三档结论 + 可选的画像证据语义对齐）",
            "POST /api/flow/run": "CareerFlow 主流程（profile→job_input→jd_analyze→match→gap→plan→today→resume→critic→application_suggest）",
            "POST /api/plan": "基于缺口生成 4 周路线规划",
            "POST /api/plan/stream": "4 周路线规划（SSE 流式）",
            "GET /api/daily": "今日 daily_log + 最近 7 天摘要",
            "POST /api/daily": "向 daily_log.md 追加今日条目",
            "POST /api/daily/attachments": "上传每日留痕附件（PDF / 图片）",
            "GET/POST /api/reflections": "长期执行/复盘检索与显式复盘生成",
            "GET/PATCH /api/plan/tasks": "稳定 task_id 的零 LLM 计划微调",
            "GET /api/resume": "简历素材聚合（pitch + 故事预览）",
            "GET /api/today": "今日建议（聚合投递池 + 日志 + 状态机）",
            "POST /api/discover": "JD 半自动抽取（粘贴或 URL → 结构化 JD）",
            "GET /api/jd/queries": "根据 profile 生成搜索关键词组合（半自动）",
            "POST /api/jd/rank": "对一组候选 JD 排序（调用 match_job）",
            "POST /api/resume/build": "JD 定制简历项目段生成（基于事实清单 + LLM）",
            "POST /api/resume/build/stream": "JD 定制简历项目段（SSE 流式）",
            "POST /api/resume/markdown": "完整 Markdown 简历草稿（默认无 LLM）",
            "POST /api/reset": "清空对话历史",
            "POST /mcp": "MCP Server（Streamable HTTP 传输）：REGISTRY 全部工具暴露给任意 MCP 客户端",
        },
    }


@app.post("/mcp", tags=["mcp"])
async def mcp_endpoint(request: Request):
    """MCP Server 端点（Streamable HTTP 传输，2025-03-26 规范）。

    单端点 JSON-RPC：initialize / ping / tools/list / tools/call。
    协议层实现见 mcp_server.py（手写、无第三方 MCP SDK 依赖）；
    工具与 ReAct Agent 共用同一 tools_registry.REGISTRY，零 schema 重复。
    """
    from fastapi.responses import JSONResponse, Response
    from mcp_server import handle_mcp_message, origin_allowed

    # 规范 MUST：Origin 校验防 DNS rebinding
    if not origin_allowed(request.headers.get("origin")):
        return JSONResponse(status_code=403, content={"error": "forbidden_origin"})

    status, payload = handle_mcp_message(await request.body())
    if payload is None:  # 通知类消息：202 Accepted 无响应体
        return Response(status_code=status)
    return JSONResponse(status_code=status, content=payload)


@app.get("/api/stats")
async def api_usage_stats():
    """运行数据看板(usage_report.py 的 HTTP 形态)。

    为多用户/开源阶段预留的运营接口:单用户阶段就开始记账,
    未来接入 per-tenant 维度时前端无需改动。只读,无副作用。
    """
    from usage_report import api_stats, kb_stats, llm_stats, loop_stats

    return {
        "tenant": current_tenant().user_id,
        "api": api_stats(),
        "llm_usage": llm_stats(),
        "knowledge_base": kb_stats(),
        "career_loop": loop_stats(),
        "disclosure": "single-user personal tool; counters include dev traffic",
    }


@app.get("/health")
async def health():
    """健康检查"""
    from career_multi_agent import agent_runtime_health
    from day1_api_starter import get_llm_config, _llm_fallback_config
    from rag_tools import describe_embedding_config, get_collection_name
    from structured_llm import structured_runtime_stats

    db_dir = os.path.join(BASE_DIR, "chroma_db")
    db_exists = os.path.exists(db_dir)
    collection_name = get_collection_name()
    
    # Chroma's Rust client can terminate the whole Windows process while another
    # process owns the same persistent database. Health checks only need metadata,
    # so read the SQLite catalog directly.
    collection_count = _kb_sqlite_stats(collection_name)[0] if db_exists else 0

    llm_cfg = get_llm_config()
    agent_workflows = agent_runtime_health()
    route_model = os.environ.get("RAG_ROUTE_MODEL", "").strip() or llm_cfg["model"]
    jd_model = os.environ.get("JD_ANALYSIS_MODEL", "").strip() or llm_cfg["model"]
    semantic_match_model = os.environ.get("MATCH_SEMANTIC_MODEL", "").strip() or llm_cfg["model"]
    with _route_warmup_lock:
        warmup = dict(_route_warmup_status)
        warmup["detail"] = dict(_route_warmup_status.get("detail") or {})
    return {
        "status": (
            "healthy"
            if collection_count > 0 and agent_workflows["status"] == "ready"
            else "degraded"
        ),
        "chroma_db": "connected" if db_exists else "not_found",
        "collection": collection_name,
        "collection_records": collection_count,
        "embedding": describe_embedding_config(),
        "llm": {
            "default_model": llm_cfg["model"],
            "fallback_enabled": _llm_fallback_config() is not None,
        },
        "agent_workflows": agent_workflows,
        "router": {
            "mode": "semantic_v4",
            "model": route_model,
            "reasoning_effort": os.environ.get("RAG_ROUTE_REASONING_EFFORT", "low"),
            "deadline_seconds": float(os.environ.get("RAG_ROUTE_TIMEOUT_SECONDS", "60") or 60),
            "prototype_routing": "offline_only",
            "prototypes_ready": False,
            "prewarm": warmup,
            "runtime": structured_runtime_stats(),
        },
        "jd_analyzer": {
            "mode": os.environ.get("JD_ANALYZER_MODE", "deterministic"),
            "model": jd_model,
            "reasoning_effort": os.environ.get("JD_ANALYSIS_REASONING_EFFORT", "low"),
            "deadline_seconds": float(os.environ.get("JD_ANALYSIS_TIMEOUT_SECONDS", "30") or 30),
        },
        "semantic_matcher": {
            "ui_enabled": True,
            "model": semantic_match_model,
            "reasoning_effort": os.environ.get("MATCH_SEMANTIC_REASONING_EFFORT", "low"),
            "deadline_seconds": float(os.environ.get("MATCH_SEMANTIC_TIMEOUT_SECONDS", "30") or 30),
            "authority": "evidence_alignment_only",
        },
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/profile", response_model=ProfileResponse)
async def get_profile():
    """获取正式画像摘要（与微信数据桥共用规范化读取服务）。"""
    from business_read_service import profile_snapshot
    snapshot = profile_snapshot()
    content = snapshot["content_md"]

    name, direction, skills_summary, updated_at = _parse_profile(content)
    return ProfileResponse(
        name=name,
        direction=direction,
        skills_summary=skills_summary,
        updated_at=snapshot.get("as_of") or updated_at,
        data_version=snapshot.get("revision") or "",
    )


@app.get("/api/profile/editor")
async def get_profile_editor():
    try:
        from profile_store import list_suggestions, read_profile
        return {"status": "ok", **read_profile(),
                "pending_suggestions": len(list_suggestions("pending"))}
    except Exception as exc:
        _log.exception("profile editor read failed")
        raise HTTPException(status_code=500, detail=f"读取画像失败: {exc}")


@app.patch("/api/profile")
async def patch_profile(req: ProfilePatchRequest):
    try:
        from profile_store import save_profile
        return save_profile(req.content_md, req.base_hash, reason=req.reason,
                            operation_id=req.operation_id or None)
    except Exception as exc:
        from profile_store import ProfileConflictError, ProfileValidationError
        if isinstance(exc, ProfileConflictError):
            raise HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, ProfileValidationError):
            raise HTTPException(status_code=400, detail=str(exc))
        _log.exception("profile save failed")
        raise HTTPException(status_code=500, detail=f"保存画像失败: {exc}")


@app.post("/api/profile/edit-preview")
async def preview_profile_edit(req: ProfileEditPreviewRequest):
    try:
        from profile_store import create_edit_preview
        return create_edit_preview(req.content_md, req.base_revision)
    except Exception as exc:
        from profile_store import ProfileConflictError, ProfileValidationError
        if isinstance(exc, ProfileConflictError):
            raise HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, (ProfileValidationError, ValueError)):
            raise HTTPException(status_code=400, detail=str(exc))
        _log.exception("profile edit preview failed")
        raise HTTPException(status_code=500, detail=f"画像预览失败: {exc}")


@app.post("/api/profile/edit-commit")
async def commit_profile_edit(req: ProfileEditCommitRequest):
    try:
        from profile_store import commit_edit_preview
        return commit_edit_preview(
            req.preview_id, reason=req.reason, operation_id=req.operation_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        from profile_store import ProfileConflictError, ProfileValidationError
        if isinstance(exc, ProfileConflictError):
            raise HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, (ProfileValidationError, ValueError)):
            raise HTTPException(status_code=400, detail=str(exc))
        _log.exception("profile edit commit failed")
        raise HTTPException(status_code=500, detail=f"画像提交失败: {exc}")


@app.get("/api/profile/suggestions")
async def get_profile_suggestions(status: str = ""):
    from business_read_service import suggestions_snapshot
    snapshot = suggestions_snapshot(status)
    return {"status": "ok", "suggestions": snapshot["suggestions"],
            "data_version": snapshot["revision"]}


@app.post("/api/profile/suggestions/audit")
async def audit_profile_suggestions_api():
    try:
        from profile_store import audit_profile_suggestions
        return await asyncio.to_thread(
            audit_profile_suggestions, max_items=5, trigger_kind="manual",
        )
    except Exception as exc:
        _log.exception("profile suggestion audit failed")
        raise HTTPException(status_code=500, detail=f"画像审计失败: {exc}")


@app.get("/api/profile/evidence/{evidence_id}")
async def get_profile_evidence_api(evidence_id: str):
    from profile_store import get_evidence
    item = get_evidence(evidence_id)
    if not item:
        raise HTTPException(status_code=404, detail="画像证据不存在")
    return {"status": "ok", "evidence": item}


@app.get("/api/profile/migration")
async def get_profile_migration_api():
    from profile_store import migration_report
    return migration_report()


@app.post("/api/profile/suggestions/{suggestion_id}/decision")
async def decide_profile_suggestion_api(suggestion_id: str,
                                        req: ProfileSuggestionDecisionRequest):
    try:
        from profile_store import decide_suggestion
        return decide_suggestion(suggestion_id, req.decision,
                                 modified_text=req.modified_text,
                                 modified_value=req.modified_value,
                                 base_revision=req.base_revision,
                                 reason=req.reason, operation_id=req.operation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        from profile_store import ProfileConflictError, ProfileValidationError
        if isinstance(exc, ProfileConflictError):
            raise HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, (ProfileValidationError, ValueError)):
            raise HTTPException(status_code=400, detail=str(exc))
        _log.exception("profile suggestion decision failed")
        raise HTTPException(status_code=500, detail=f"处理画像建议失败: {exc}")


@app.post("/api/query", response_model=QueryResponse)
async def rag_query(req: QueryRequest):
    """RAG 问答接口（知识库优先 · 带门槛）。

    与微信路径（offerclaw_cli query）共用 rag_gate.gated_query：
    命中知识库才基于 KB 合成答案并标注来源；未命中坦白"知识库暂无"，
    绝不退回通用知识杜撰。
    """
    import asyncio
    conversation_id, user_message_id, assistant_message_id, operation_id = _conversation_ids(req)
    _record_conversation("user", req.query, "sent", conversation_id=conversation_id,
                         message_id=user_message_id, operation_id=operation_id)
    try:
        from query_service import execute_query
        call = lambda: execute_query(
            req.query, req.top_k, context=req.context,
            conversation_id=conversation_id,
        ).to_dict()
        # asyncio.to_thread copies ContextVars; run_in_executor does not.  The
        # request's traffic provenance must reach the retrieval/shadow hook.
        d = await asyncio.to_thread(call)
        _record_conversation("assistant", d.get("answer", ""), "completed",
                             conversation_id=conversation_id,
                             message_id=assistant_message_id, operation_id=operation_id,
                             source_refs=d.get("sources", []))
        try:
            from conversation_context import record_successful_turn
            record_successful_turn(
                conversation_id, d, turn_id=assistant_message_id,
            )
        except Exception:
            _log.warning("conversation context write failed", exc_info=True)
        try:
            from memory_layers import distill_to_semantic, EpisodicMemory, SemanticMemory
            distill_to_semantic(EpisodicMemory(), SemanticMemory())
        except Exception:
            _log.warning("conversation distillation failed", exc_info=True)
        return QueryResponse(
            query=req.query,
            answer=d.get("answer", ""),
            retrieval_count=d.get("retrieval_count", 0),
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            in_kb=d.get("in_kb", False),
            sources=d.get("sources", []),
            matched_by=d.get("matched_by") or "",
            answer_action=d.get("answer_action") or "answer",
            mode=d.get("mode", ""), planner_mode=d.get("planner_mode", ""),
            resolver_mode=d.get("resolver_mode", ""), decision=d.get("decision", "answer"),
            interaction_kind=d.get("interaction_kind", "query"),
            turn_relation=d.get("turn_relation", "standalone"),
            relation_target_turn_id=d.get("relation_target_turn_id", ""),
            action_request=d.get("action_request", {}),
            routing_assurance=d.get("routing_assurance", "degraded"),
            decision_reasons=d.get("decision_reasons", []),
            context_resolution=d.get("context_resolution", {}),
            routes=d.get("routes", []), intent_frame=d.get("intent_frame", {}),
            planner_engine=d.get("planner_engine", ""),
            planner_version=d.get("planner_version", ""),
            schema_version=d.get("schema_version", ""),
            route_model=d.get("route_model", ""),
            repair_used=bool(d.get("repair_used", False)),
            fallback_reason=d.get("fallback_reason", ""),
            planner_queue_ms=float(d.get("planner_queue_ms", 0.0) or 0.0),
            planner_provider_ms=float(d.get("planner_provider_ms", 0.0) or 0.0),
            planner_wall_ms=float(d.get("planner_wall_ms", 0.0) or 0.0),
            planner_deadline_ms=float(d.get("planner_deadline_ms", 0.0) or 0.0),
            planner_cache_hit=bool(d.get("planner_cache_hit", False)),
            planner_timeout_stage=d.get("planner_timeout_stage", ""),
            planner_late_response=bool(d.get("planner_late_response", False)),
            planner_circuit_state=d.get("planner_circuit_state", ""),
            prompt_tokens=d.get("prompt_tokens"),
            completion_tokens=d.get("completion_tokens"),
            structured_output_mode=d.get("structured_output_mode", ""),
            route_gateway=d.get("route_gateway", ""),
            route_reasoning_effort=d.get("route_reasoning_effort", ""),
            json_capability=d.get("json_capability", ""),
            retrieval_profile=d.get("retrieval_profile", ""),
            index_fingerprint=d.get("index_fingerprint", ""),
            effective_hit=d.get("effective_hit"),
            conversation_id=conversation_id, user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            data_version=d.get("data_version", ""),
            model_usage=d.get("model_usage") or {},
            trace_id=d.get("trace_id", ""),
            query_service_version=d.get("query_service_version", ""),
        )
    except Exception as e:
        try:
            _record_conversation("assistant", str(e), "failed",
                                 conversation_id=conversation_id,
                                 message_id=assistant_message_id, operation_id=operation_id)
        except Exception:
            _log.warning("failed conversation memory write failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG 查询失败: {str(e)}")


@app.get("/api/internal/wechat-health")
async def wechat_query_health(request: Request):
    _require_wechat_query_auth(request)
    from query_service import (
        QUERY_SERVICE_VERSION, query_runtime_status, repository_fingerprint,
    )

    runtime = query_runtime_status()
    if os.environ.get("OFFERCLAW_QUERY_SERVICE") == "1" and runtime.get("status") != "ready":
        raise HTTPException(status_code=503, detail="OfferClaw query runtime is not ready")

    return {
        "status": "ok",
        "schema_version": "offerclaw.wechat-query.health.v1",
        "query_service_version": QUERY_SERVICE_VERSION,
        "repository_fingerprint": repository_fingerprint(),
        "authentication": "loopback_token",
        "runtime": runtime,
    }


@app.post("/api/internal/wechat-query")
async def wechat_query(req: WeChatQueryRequest, request: Request):
    """Run the canonical query core without storing raw WeChat text or replies."""
    _require_wechat_query_auth(request)
    from query_service import execute_query

    trace_id = "wxquery_" + hashlib.sha256(req.operation_id.encode("utf-8")).hexdigest()[:20]
    try:
        execution = await asyncio.wait_for(
            asyncio.to_thread(
                execute_query, req.question, req.top_k,
                conversation_id=req.conversation_id, timeout_seconds=40.0,
                trace_id=trace_id,
            ),
            timeout=41.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="OfferClaw query deadline exhausted")
    result = execution.to_dict()
    try:
        from conversation_context import record_successful_turn
        await asyncio.to_thread(
            record_successful_turn, req.conversation_id, result, turn_id=req.message_id,
        )
    except Exception:
        _log.warning("wechat compact context write failed trace=%s", trace_id)
    return execution.wechat_dict()


@app.post("/api/search")
async def rag_search(req: QueryRequest):
    """
    仅检索接口（不调 LLM）。
    返回检索到的原始文档片段。
    """
    try:
        agent = get_rag_agent()
        
        if agent.collection is None:
            raise HTTPException(status_code=503, detail="ChromaDB 未连接")

        docs = agent._retrieve(req.query, req.top_k)

        return {
            "query": req.query,
            "results": [
                {
                    "document": doc["document"][:300],
                    "source": doc.get("source", "unknown"),
                    "title": doc.get("title", ""),
                    "distance": doc.get("distance", 0),
                }
                for doc in (docs or [])
            ],
            "count": len(docs) if docs else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.post("/api/match", response_model=MatchResponse)
async def job_match(req: MatchRequest):
    """
    岗位匹配接口：调用 match_job.run_match 真实跑出三档结论。
    返回结构化 gap_list + suggestions，方便前端独立渲染缺口卡。
    """
    try:
        import asyncio
        from jd_parser import analyze_jd
        from match_job import run_match, format_report
        from profile_loader import load_profile
        from semantic_matcher import align_requirements
        profile = load_profile(include_evidence=req.use_semantic)

        def calculate():
            # The online match path has one LLM latency budget. Deterministic
            # parsing already grounds exact JD spans; the model budget is spent
            # on the part code cannot do reliably: cross-expression evidence
            # alignment. The standalone/full-flow JD analyzer may still run in
            # intelligent mode when explicitly requested there.
            analysis = analyze_jd(req.jd_text, mode="deterministic")
            semantic = align_requirements(
                profile, analysis.model_dump(mode="json"), enabled=req.use_semantic,
            )
            result = run_match(
                profile, req.jd_text, jd_title="API 请求",
                jd_analysis=analysis.model_dump(mode="json"),
                semantic_alignment=semantic.model_dump(mode="json"),
            )
            return result, semantic

        loop = asyncio.get_running_loop()
        report, semantic = await loop.run_in_executor(None, calculate)
        from domain_status import match_status_code
        code = match_status_code(report.conclusion).value
        try:
            from memory_layers import record_business_event, distill_to_semantic, EpisodicMemory, SemanticMemory
            record_business_event(
                "match_completed", {"status": report.conclusion, "status_code": code,
                                    "direction": report.direction, "jd_title": "API 请求",
                                    "gap_list": report.gap_list or {}},
                actor="user", source="match_api", entity_type="jd_match")
            distill_to_semantic(EpisodicMemory(), SemanticMemory())
        except Exception:
            _log.warning("match memory write failed", exc_info=True)
        return MatchResponse(
            status=report.conclusion,
            status_code=code,
            summary=format_report(report),
            direction=report.direction,
            gap_list=report.gap_list or {},
            suggestions=report.suggestions or [],
            requirement_analysis=report.requirement_analysis or {},
            matching_mode="semantic" if semantic.status == "completed" else "deterministic",
            semantic_status=semantic.status,
        )
    except Exception as e:
        _log.exception("match failed")
        raise HTTPException(status_code=500, detail=f"匹配失败: {str(e)}")


@app.post("/api/flow/run", response_model=FlowRunResponse)
async def flow_run(req: FlowRunRequest):
    """CareerFlow 主流程：profile → job_input → jd_analyze → match → gap → plan
    → today → resume → critic → application_suggest（10 个业务节点，另有 4 个
    router 记账节点）。UI 折叠成 8 个可见阶段，把 jd_analyze/critic 藏在内部，
    所以"8 步"只能指界面，不能用来描述这张图。

    返回完整 CareerState，前端可分段渲染（卡片 / Stepper）。
    任何写入意图都收在 ``requires_confirmation`` 中，**本接口不会写文件**。
    """
    import asyncio
    try:
        from career_flow import run_career_flow
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None,
            lambda: run_career_flow(
                req.jd_text, jd_title=req.jd_title, skip_llm=req.skip_llm,
            ),
        )
        return FlowRunResponse(
            match_report=out.get("match_report") or {},
            gaps=out.get("gaps") or {},
            plan_outline=out.get("plan_outline") or [],
            today_advice=out.get("today_advice") or {},
            resume_skeleton=out.get("resume_skeleton") or {},
            application_suggestion=out.get("application_suggestion") or {},
            requires_confirmation=out.get("requires_confirmation") or [],
            trace=out.get("trace") or [],
            errors=out.get("errors") or [],
            route_history=out.get("route_history") or [],
            checkpoint_status=out.get("checkpoint_status") or {},
            memory_status=out.get("memory_status") or {},
            fatal_error=bool(out.get("fatal_error")),
            fatal_reason=out.get("fatal_reason") or "",
        )
    except Exception as e:
        _log.exception("flow_run failed")
        raise HTTPException(status_code=500, detail=f"CareerFlow 失败: {str(e)}")


@app.post("/api/plan", response_model=PlanResponse)
async def gen_plan(req: PlanRequest):
    """基于缺口清单生成 4 周路线规划。无缺口时用 DATA_CONTRACT.md 风格的兜底输入。"""
    import asyncio
    try:
        from plan_gen import (
            prepare_plan_messages, call_llm_plain,
            append_resources_appendix, append_target_trace,
        )
        api_key, api_key_env = _active_llm_api_key()
        if not api_key:
            raise HTTPException(status_code=500, detail=f"{api_key_env} 未配置")
        gaps = _resolve_plan_gaps(req.gaps)
        # 统一入口：读依赖 + RAG 检索资源 + 组装 messages（与 CLI 一致）
        messages, resources = prepare_plan_messages(gaps, revision_note=req.revision_note,
                                                  start_date=req.start_date, end_date=req.end_date)
        loop = asyncio.get_event_loop()
        plan_md = await loop.run_in_executor(
            None, lambda: call_llm_plain(messages, api_key, max_tokens=7000)
        )
        # 退化产物（拒绝/无周结构）不落盘，避免污染"当前计划"
        from plan_gen import is_degenerate_plan, normalize_plan_dates
        if is_degenerate_plan(plan_md):
            return PlanResponse(plan_md=plan_md, saved_path="", daily_days=0)
        # 日期确定性归一（LLM 连排日期不可信，统一按 开始日期+序号 重写）
        plan_md = normalize_plan_dates(
            plan_md, req.start_date or datetime.date.today().isoformat())
        # 确定性追加参考资源附录，保证 API 路径也必含知识库引用
        plan_md = append_resources_appendix(plan_md, resources)
        plan_md = append_target_trace(plan_md)
        # 生成后校验门（fail-visible）：立即数日计划层，0=被模型笔迹打穿，前端据此警示
        try:
            from plan_daily import parse_plan_days
            daily_days = len(parse_plan_days(plan_md)["days"])
        except Exception:
            daily_days = -1
        from plan_gen import load_latest_plan, summarize_plan_changes
        from plan_drafts import create_plan_draft
        previous = load_latest_plan()
        changes = summarize_plan_changes(previous["content"] if previous else "", plan_md)
        draft = create_plan_draft(plan_md, changes=changes if previous else [],
                                  daily_days=daily_days, source="plan_api")
        return PlanResponse(plan_md=plan_md, draft_id=draft["draft_id"],
                            requires_confirmation=True, daily_days=daily_days)
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("plan failed")
        raise HTTPException(status_code=500, detail=f"规划失败: {str(e)}")


@app.get("/api/plan/current", response_model=CurrentPlanResponse)
async def current_plan():
    """读取当前（最新）学习计划，供页面打开时直接展示。无计划时 has_plan=False。"""
    try:
        from plan_gen import load_latest_plan
        latest = load_latest_plan()
        if not latest:
            return CurrentPlanResponse(has_plan=False)
        return CurrentPlanResponse(
            has_plan=True,
            content=latest["content"],
            filename=latest["filename"],
            mtime=latest["mtime"],
            edited_by_user=latest["edited_by_user"],
            target_status=latest.get("target_status") or {},
            profile_status=latest.get("profile_status") or {},
        )
    except Exception as e:
        _log.exception("current_plan failed")
        raise HTTPException(status_code=500, detail=f"读取计划失败: {str(e)}")


@app.post("/api/plan/drafts/{draft_id}/decision")
async def decide_plan_draft_api(draft_id: str, req: PlanDraftDecisionRequest):
    try:
        from plan_drafts import (PlanDraftNotFoundError, StalePlanDraftError,
                                 decide_plan_draft)
        return {"status": "ok", **decide_plan_draft(draft_id, req.decision)}
    except PlanDraftNotFoundError:
        raise HTTPException(status_code=404, detail="计划草稿不存在或已经处理")
    except StalePlanDraftError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/plan/save", response_model=PlanResponse)
async def save_edited_plan(req: PlanSaveRequest):
    """保存用户手动编辑后的计划：落盘 plans/（带 _user 标识）+ 记一条 episodic 记忆事件，
    让复盘 / 今日建议能感知『用户调整过计划』。"""
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="计划内容不能为空")
    try:
        from plan_gen import save_plan
        path = save_plan(content, edited_by_user=True,
                         operation_id=req.operation_id or None,
                         note=(req.note or "").strip())
        return PlanResponse(plan_md=content, saved_path=os.path.relpath(path, BASE_DIR))
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("save_edited_plan failed")
        raise HTTPException(status_code=500, detail=f"保存计划失败: {str(e)}")


@app.get("/api/plan/today", response_model=PlanTodayResponse)
async def plan_today(date: str = ""):
    """今日计划视图：按 UI 打开当天（或 ?date=）从整体计划切出当日安排。

    展示日期灵活兜底：计划未开始→首日；已结束→末日；范围内缺块→最近一天。
    """
    try:
        from plan_daily import get_today_view
        return PlanTodayResponse(**get_today_view(date or None))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"日期不合法: {str(e)}")
    except Exception as e:
        _log.exception("plan_today failed")
        raise HTTPException(status_code=500, detail=f"读取今日计划失败: {str(e)}")


@app.post("/api/plan/today/save", response_model=PlanTodayResponse)
async def plan_today_save(req: PlanTodaySaveRequest):
    """回写今日任务到整体计划：只替换当日 D 块的编号任务行，其余逐行保留，
    另存 _user 版——整体计划与今日视图共用同一文件，改哪边都自动一致。"""
    from plan_daily import (
        DayNotFoundError, NoDailyLayerError, NoPlanError, StalePlanError, save_today,
    )
    try:
        view = save_today(req.date, req.tasks, base_mtime=req.base_mtime)
    except (NoPlanError, DayNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NoDailyLayerError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except StalePlanError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("plan_today_save failed")
        raise HTTPException(status_code=500, detail=f"保存今日计划失败: {str(e)}")
    # 让 OfferClaw 知晓（复盘/今日建议可感知用户按天调整过）；失败不阻塞
    try:
        from memory_layers import EpisodicMemory
        EpisodicMemory().append({
            "kind": "plan_today_edited",
            "source": "web_ui",
            "date": req.date,
            "n_tasks": len([t for t in req.tasks if (t or "").strip()]),
            "note": (req.note or "").strip(),
        })
    except Exception:
        _log.warning("plan_today_edited memory append failed", exc_info=True)
    return PlanTodayResponse(**view)


@app.patch("/api/plan/tasks")
async def patch_plan_tasks_api(req: PlanTaskPatchRequest):
    """按 task_id 添加/修改/删除/移动任务；确定性执行，0 次 LLM。"""
    if not req.operations:
        raise HTTPException(status_code=400, detail="operations 不能为空")
    try:
        from plan_daily import patch_plan_tasks
        result = patch_plan_tasks([op.dict(exclude_none=True) for op in req.operations],
                                  base_mtime=req.base_mtime)
        try:
            from memory_layers import EpisodicMemory
            EpisodicMemory().append({
                "kind": "plan_task_patched", "source": "web_ui",
                "operations": [op.op for op in req.operations],
                "task_ids": result.get("changed_task_ids", []), "note": req.note[:500],
                "saved_path": os.path.relpath(result.get("saved_path", ""), BASE_DIR),
            })
        except Exception:
            _log.warning("plan task patch memory append failed", exc_info=True)
        plan = result.pop("plan", {}) or {}
        result["saved_path"] = os.path.relpath(result.get("saved_path", ""), BASE_DIR)
        result["plan"] = {
            "content": plan.get("content", ""), "filename": plan.get("filename", ""),
            "mtime": plan.get("mtime", 0), "target_status": plan.get("target_status", {}),
            "profile_status": plan.get("profile_status", {}),
        }
        return result
    except Exception as exc:
        from plan_daily import (DayNotFoundError, NoDailyLayerError, NoPlanError,
                                StalePlanError)
        if isinstance(exc, (StalePlanError, NoDailyLayerError)):
            raise HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, (NoPlanError, DayNotFoundError, KeyError)):
            raise HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc))
        _log.exception("plan task patch failed")
        raise HTTPException(status_code=500, detail=f"修改计划任务失败: {exc}")


@app.get("/api/plan/tasks")
async def get_plan_tasks_api():
    try:
        from plan_daily import ensure_task_ids, parse_plan_days
        from plan_gen import load_latest_plan
        latest = load_latest_plan()
        if not latest:
            return {"status": "ok", "has_plan": False, "days": []}
        parsed = parse_plan_days(ensure_task_ids(latest["content"]))
        return {
            "status": "ok", "has_plan": True, "filename": latest["filename"],
            "mtime": latest["mtime"], "profile_status": latest.get("profile_status", {}),
            "days": [{"date": d["date"].isoformat(), "label": d["label"],
                      "week_n": d["week_n"],
                      "tasks": [{"task_id": x["task_id"], "text": x["text"],
                                 "optional": x["optional"]} for x in d["task_items"]]
                     } for d in parsed["days"]],
        }
    except Exception as exc:
        _log.exception("plan task list failed")
        raise HTTPException(status_code=500, detail=f"读取计划任务失败: {exc}")


def _resolve_plan_gaps(req_gaps: str) -> str:
    """计划输入：显式入参 > 投递管理中的活动 JD 目标 > 画像通用方向。"""
    g = (req_gaps or "").strip()
    if g:
        return g
    try:
        from application_jd_store import plan_gaps_text
        stored = plan_gaps_text()
        if stored:
            return stored
    except Exception:
        _log.warning("read application plan targets failed", exc_info=True)
    return "（没有已纳入计划且已关联 JD 的投递，按用户画像通用方向规划）"


@app.post("/api/gaps/target")
async def set_gap_target(req: GapTargetRequest):
    """旧入口已停用：目标 JD 必须经投递管理确认，禁止分析区直接持久化。"""
    raise HTTPException(
        status_code=410,
        detail="目标 JD 已收口到投递管理，请使用 /api/applications/from-jd/preview 与 commit",
    )


@app.get("/api/gaps")
async def get_gaps():
    """兼容视图：由当前投递的活动 JD 派生目标与缺口，不读旧 gap_store。"""
    try:
        from application_jd_store import plan_gap_items, plan_gaps_text, plan_targets
        data = plan_targets()
        merged: dict[str, list[str]] = {}
        gap_items = plan_gap_items(data["included"])
        for item in gap_items:
            merged.setdefault(item["category"], []).append(item["text"])
        return {
            "status": "ok",
            "total_targets": data["total"],
            "merged_gap_count": len(gap_items),
            "targets": data["included"],
            "excluded": data["excluded"],
            "merged": merged,
            "gap_items": gap_items,
            "merged_text": plan_gaps_text(),
            "derived_from": "applications",
        }
    except Exception as e:
        _log.exception("get_gaps failed")
        raise HTTPException(status_code=500, detail=f"读取缺口库失败: {str(e)}")


# =====================================================
# 投递管理：用户上传真实投递情况 + 亲历经验入知识库
# =====================================================

@app.get("/api/applications")
async def get_applications():
    """投递清单 + JD/匹配/计划关联摘要 + 已保存经验。"""
    try:
        from applications_store import list_experiences, STATUSES
        from application_jd_store import snapshot_summary
        from business_read_service import applications_snapshot
        snapshot = applications_snapshot()
        rows = snapshot["applications"]
        for row in rows:
            row["jd"] = snapshot_summary(row.get("jd_id", ""),
                                         row.get("jd_version_id", ""),
                                         row.get("match_id", ""))
        return {"status": "ok", "rows": rows, "data_version": snapshot["revision"],
                "experiences": list_experiences(), "statuses": STATUSES}
    except Exception as e:
        _log.exception("get_applications failed")
        raise HTTPException(status_code=500, detail=f"读取投递清单失败: {str(e)}")


@app.post("/api/applications/upsert")
async def upsert_application_api(req: ApplicationUpsertRequest):
    """兼容入口：新客户端按 application_id，旧客户端仅在岗位唯一时更新。

    带经验总结时落盘 experience_posts/；勾选 add_to_kb 则**增量入向量库**
    （用户亲历的第一手经验，强指导 RAG 问答、学习计划与每日建议）。
    """
    import asyncio
    try:
        from applications_store import upsert_application, save_experience
        out = upsert_application(
            req.company, req.position, req.status,
            date=req.date, source=req.source, location=req.location,
            next_action=req.next_action, note=req.note,
            application_id=req.application_id,
            include_in_plan=req.include_in_plan,
            long_term_follow=req.long_term_follow,
            plan_priority=req.plan_priority,
            operation_id=req.operation_id or current_request_id(),
        )
        if out.get("status") != "ok":
            code = 409 if out.get("status") == "conflict" else 400
            raise HTTPException(status_code=code, detail=out)

        if (req.experience or "").strip():
            exp = save_experience(req.company, req.position,
                                  req.experience_stage or "投递过程", req.experience,
                                  application_id=out.get("application_id", ""),
                                  jd_version_id=out.get("jd_version_id", ""),
                                  operation_id=(req.operation_id or current_request_id()) + ":experience")
            if exp.get("status") != "ok":
                out["experience_error"] = exp.get("error", "经验保存失败")
            else:
                out["experience_saved"] = exp["saved"]
                if req.add_to_kb:
                    import subprocess
                    before = _kb_count()
                    loop = asyncio.get_event_loop()
                    proc = await loop.run_in_executor(None, lambda: subprocess.run(
                        [os.path.join(BASE_DIR, ".venv/bin/python"), "rag_ingest.py",
                         "--add", exp["saved"], "--source-type", "experience"],
                        cwd=BASE_DIR, capture_output=True, text=True, timeout=300,
                    ))
                    _kb_clear_cache()
                    after = _kb_count()
                    out["kb_ingest"] = "ok" if proc.returncode == 0 else "failed"
                    out["kb_chunks_added"] = max(0, after - before)
                    out["kb_chunks_total"] = after
                    if proc.returncode == 0:
                        try:
                            _record_content_artifact(
                                "knowledge_material_changed",
                                open(exp["saved_abs"], encoding="utf-8").read(),
                                action="indexed", source="application_review",
                                operation_id=(req.operation_id or current_request_id()) + ":experience:indexed",
                                source_path=exp["saved"], source_type="experience",
                                result={"chunks_added": out["kb_chunks_added"]},
                            )
                        except Exception:
                            _log.warning("application review index memory event failed", exc_info=True)
        return out
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("upsert_application failed")
        raise HTTPException(status_code=500, detail=f"投递记录失败: {str(e)}")


@app.post("/api/applications/from-jd/preview")
async def preview_application_from_jd(req: ApplicationFromJDPreviewRequest):
    """无副作用预览：抽取字段、匹配画像并列出现有候选，不持久化 JD。"""
    try:
        from application_jd_store import preview_from_jd
        # JD 抽取/匹配可能包含同步模型调用；放到线程里，避免占住 API 事件循环，
        # 让用户在等待预览时仍能刷新投递、计划等其他页面数据。
        out = await asyncio.to_thread(
            preview_from_jd, req.jd_text, source_url=req.source_url,
            application_id=req.application_id,
        )
        if out.get("status") != "ok":
            raise HTTPException(status_code=400, detail=out.get("error", "预览失败"))
        return out
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("preview_application_from_jd failed")
        raise HTTPException(status_code=500, detail=f"预览失败: {exc}")


def _existing_jd_commit_result(operation_id: str) -> dict:
    """Return the durable result of an already completed JD-confirm action."""
    if not operation_id:
        return {}
    from applications_store import get_application
    from memory_layers import EpisodicMemory

    event = EpisodicMemory().store.get_event_by_operation(operation_id)
    if not event or event.get("kind") != "application_changed":
        return {}
    application_id = str(event.get("application_id") or event.get("entity_id") or "")
    app = get_application(application_id)
    if not app:
        return {}
    return {
        "status": "ok",
        "action": "idempotent",
        "application_id": application_id,
        "jd_id": app.get("jd_id", ""),
        "jd_version_id": app.get("jd_version_id", ""),
        "match_id": app.get("match_id", ""),
    }


def _commit_application_from_jd_sync(req: ApplicationFromJDCommitRequest) -> dict:
    """Run one confirmed JD save outside the API event loop.

    The content-keyed lock makes retries and rapid double-clicks serialize. It
    also lets the second request discover the first durable row instead of
    creating another active application for the same JD.
    """
    from application_jd_store import (
        find_active_duplicate_application,
        jd_content_hash,
        new_application_id,
        prepare_approved_artifacts,
    )
    from applications_store import get_application, upsert_application
    from io_utils import file_lock

    operation_id = (req.operation_id or current_request_id() or "").strip()
    content_hash = jd_content_hash(req.jd_text)
    if req.expected_content_hash and req.expected_content_hash != content_hash:
        return {"status": "error", "error": "JD 内容已变化，请重新预览后确认"}

    lock_seed = "|".join((req.mode, req.company.strip(), req.position.strip(), content_hash))
    lock_name = hashlib.sha256(lock_seed.encode("utf-8")).hexdigest()
    lock_path = os.path.join(BASE_DIR, ".offerclaw", "locks", f"jd-commit-{lock_name}")
    with file_lock(lock_path):
        prior = _existing_jd_commit_result(operation_id)
        if prior:
            return prior

        current = get_application(req.application_id) if req.mode == "link_existing" else {}
        if req.mode == "link_existing" and not current:
            return {"status": "error", "error": "找不到要关联的 application_id", "http_status": 404}
        application_id = current.get("application_id") or new_application_id()
        company = current.get("company") or req.company.strip()
        position = current.get("position") or req.position.strip()
        status = current.get("status") or req.status

        if req.mode == "create":
            duplicate = find_active_duplicate_application(company, position, content_hash)
            if duplicate:
                return {
                    "status": "duplicate",
                    "error": "相同公司、岗位和 JD 内容已经在投递管理中。请关联已有记录，而不是再次新建。",
                    "existing_application": duplicate,
                }

        artifacts = prepare_approved_artifacts(
            application_id=application_id, jd_text=req.jd_text,
            company=company, position=position,
            location=current.get("location") or req.location,
            source_url=req.source_url,
            expected_hash=req.expected_content_hash,
            existing_jd_id=current.get("jd_id", ""),
            preview_id=req.preview_id,
        )
        if artifacts.get("status") != "ok":
            return artifacts
        match = artifacts["match"]
        out = upsert_application(
            company, position, status,
            application_id=application_id,
            force_new=req.mode == "create",
            date=current.get("date") or req.date,
            source=(current.get("source") if current.get("source") not in {"", "—"} else req.source),
            location=(current.get("location") if current.get("location") not in {"", "—"} else req.location),
            next_action=(current.get("next_action") if current.get("next_action") not in {"", "—"}
                         else req.next_action),
            note=req.note if req.mode == "create" else "",
            jd_id=artifacts["jd_id"], jd_version_id=artifacts["jd_version_id"],
            match_id=artifacts["match_id"], source_url=req.source_url,
            match_conclusion=match.get("status", ""), audience=match.get("direction", ""),
            include_in_plan=req.include_in_plan, plan_priority=req.plan_priority,
            long_term_follow=req.long_term_follow,
            operation_id=operation_id,
        )
        if out.get("status") != "ok":
            return out
        return {
            **out,
            "jd_id": artifacts["jd_id"],
            "jd_version_id": artifacts["jd_version_id"],
            "match_id": artifacts["match_id"],
            "version_action": artifacts["version_action"],
            "preview_reused": artifacts["preview_reused"],
        }


@app.post("/api/applications/from-jd/commit")
async def commit_application_from_jd(req: ApplicationFromJDCommitRequest):
    """确认 JD：先保存不可变证据，再把活动版本关联到投递记录。"""
    if req.mode not in {"create", "link_existing"}:
        raise HTTPException(status_code=400, detail="mode 必须是 create/link_existing")
    try:
        if req.include_in_plan and not req.jd_text.strip():
            raise HTTPException(status_code=400, detail="纳入计划前必须关联完整 JD")
        out = await asyncio.to_thread(_commit_application_from_jd_sync, req)
        if out.get("status") != "ok":
            status_code = out.pop("http_status", 409 if out.get("status") in {"conflict", "duplicate"} else 400)
            raise HTTPException(status_code=status_code, detail=out)
        return out
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("commit_application_from_jd failed")
        raise HTTPException(status_code=500, detail=f"加入投递管理失败: {exc}")


@app.patch("/api/applications/{application_id}")
async def patch_application_api(application_id: str, req: ApplicationPatchRequest):
    try:
        from applications_store import patch_application
        changes = req.dict()
        changes["operation_id"] = req.operation_id or current_request_id()
        out = patch_application(application_id, **changes)
        if out.get("status") != "ok":
            raise HTTPException(status_code=400, detail=out.get("error", "更新失败"))
        return out
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("patch_application failed")
        raise HTTPException(status_code=500, detail=f"更新失败: {exc}")


@app.get("/api/applications/{application_id}/jd")
async def get_application_jd(application_id: str):
    from applications_store import get_application
    from application_jd_store import compare_versions, list_versions, load_snapshot, snapshot_summary
    current = get_application(application_id)
    if not current:
        raise HTTPException(status_code=404, detail="找不到 application_id")
    if not current.get("jd_version_id"):
        return {"status": "ok", "linked": False, "application": current, "versions": []}
    summary = snapshot_summary(current.get("jd_id", ""), current["jd_version_id"],
                               current.get("match_id", ""))
    snapshot = load_snapshot(current["jd_version_id"], jd_id=current.get("jd_id", ""))
    return {"status": "ok", "linked": True, "application": current,
            "summary": summary, "snapshot": snapshot,
            "versions": list_versions(current.get("jd_id", "")),
            "latest_diff": compare_versions(current.get("jd_id", ""))}


@app.get("/api/plan/targets")
async def get_plan_targets():
    from application_jd_store import plan_targets, target_snapshot
    return {"status": "ok", **plan_targets(), "snapshot": target_snapshot()}


# =====================================================
# 选择性多 Agent：组合计划 + 单投递简历（显式触发、审批后写入）
# =====================================================

@app.post("/api/agent/plan/scopes/preview")
async def preview_agent_plan_scope(req: PortfolioScopeRequest):
    """Resolve a portfolio target set without writing files or checkpoints."""
    try:
        from career_multi_agent import preview_portfolio_scope

        return preview_portfolio_scope(
            instruction=req.instruction,
            application_ids=req.application_ids,
            include_profile_goals=req.include_profile_goals,
        )
    except Exception as exc:
        _log.exception("portfolio scope preview failed")
        raise HTTPException(status_code=500, detail=f"范围预览失败: {exc}")


@app.post("/api/agent/flows/start")
async def start_agent_flow_api(req: AgentFlowStartRequest):
    """Start one explicit Agent job and stream its draft/approval interrupt."""
    if req.task not in {"portfolio_plan", "resume"}:
        raise HTTPException(status_code=400, detail="task 必须是 portfolio_plan 或 resume")
    try:
        from career_multi_agent import agent_runtime_health, get_scope_snapshot, _resume_context

        runtime = agent_runtime_health()
        if runtime["status"] != "ready":
            raise HTTPException(
                status_code=503,
                detail=(
                    "Agent 持久化组件不可用；请在启动 OfferClaw 的同一 Python 环境安装 "
                    f"{runtime['dependency']}"
                ),
            )

        if req.task == "portfolio_plan":
            if not req.scope_snapshot_id:
                raise HTTPException(status_code=400, detail="请先预览并确认计划范围")
            get_scope_snapshot(req.scope_snapshot_id)
        else:
            _resume_context(
                req.application_id, req.jd_version_id,
                resume_scope=req.resume_scope,
                project_repo_url=req.project_repo_url,
                project_text=req.project_text,
                project_name=req.project_name,
                resume_source_text=req.resume_source_text,
            )
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    def generate():
        from career_multi_agent import start_agent_flow

        yield _sse_event({"type": "meta", "task": req.task,
                          "workflow_version": "v2", "max_llm_calls": 4,
                          "resume_scope": req.resume_scope if req.task == "resume" else ""})
        if req.task == "portfolio_plan":
            yield _sse_event({"type": "scope", "scope_snapshot_id": req.scope_snapshot_id})
        yield _sse_event({"type": "node", "node": "portfolio_plan_agent"
                          if req.task == "portfolio_plan" else "resume_author_agent",
                          "status": "running"})
        try:
            result = start_agent_flow(
                task=req.task, scope_snapshot_id=req.scope_snapshot_id,
                application_id=req.application_id, jd_version_id=req.jd_version_id,
                start_date=req.start_date, end_date=req.end_date,
                revision_note=req.revision_note,
                resume_scope=req.resume_scope,
                project_repo_url=req.project_repo_url,
                project_text=req.project_text,
                project_name=req.project_name,
                stage_project_memory=req.stage_project_memory,
                resume_source_text=req.resume_source_text,
            )
            for metric in result.get("metrics") or []:
                yield _sse_event({"type": "node", "status": "done", **metric})
            artifact = result.get("artifact") or {}
            if artifact:
                yield _sse_event({"type": "artifact", "artifact": artifact})
            if result.get("interrupt"):
                yield _sse_event({"type": "interrupt", "thread_id": result["thread_id"],
                                  "interrupt": result["interrupt"]})
            yield _sse_event({"type": "done", "thread_id": result["thread_id"],
                              "status": result["status"], "budget": result.get("budget") or {},
                              "saved_path": artifact.get("saved_path", "")})
        except Exception as exc:
            _log.exception("agent flow start failed")
            yield _sse_event({"type": "error", "error": str(exc)})

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/agent/flows/{thread_id}")
async def get_agent_flow_api(thread_id: str):
    try:
        from career_multi_agent import get_agent_flow

        return get_agent_flow(thread_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.exception("agent flow status failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/agent/flows/{thread_id}/resume")
async def resume_agent_flow_api(thread_id: str, req: AgentFlowResumeRequest):
    allowed = {"approve", "request_changes", "manual_edit",
               "save_unreviewed_draft", "reject", "edit"}
    if req.decision not in allowed:
        raise HTTPException(status_code=400, detail=(
            "decision 必须是 approve/request_changes/manual_edit/"
            "save_unreviewed_draft/reject"))
    try:
        from career_multi_agent import continue_agent_flow

        return continue_agent_flow(
            thread_id, decision=req.decision, edited_content_md=req.edited_content_md,
            change_request=req.change_request,
            remember_preference=req.remember_preference,
            artifact_revision=req.artifact_revision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.exception("agent flow resume failed")
        raise HTTPException(status_code=500, detail=str(exc))


# =====================================================
# 简历：项目 → 简历项目经历段（模板学习式生成）
# =====================================================

@app.get("/api/resume/templates")
async def list_resume_templates():
    """已学习的简历材料清单（写作指导 + 真实简历范例）。"""
    try:
        from resume_project import load_materials, extract_project_blocks
        m = load_materials()
        n_blocks = sum(len(extract_project_blocks(e["content"])) for e in m["examples"])
        return {
            "status": "ok",
            "guidance": [g["name"] for g in m["guidance"]],
            "examples": [e["name"] for e in m["examples"]],
            "project_blocks": n_blocks,
        }
    except Exception as e:
        _log.exception("list_resume_templates failed")
        raise HTTPException(status_code=500, detail=f"读取模板失败: {str(e)}")


@app.post("/api/resume/templates")
async def upload_resume_template(req: ResumeTemplateUploadRequest):
    """上传简历材料（.md/.txt）到 resume_templates/：
    文件名含 note/写法/指导 视为写作指导，否则视为真实简历范例（用于格式学习）。"""
    name = (req.name or "").strip()
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".md", ".markdown", ".txt"):
        raise HTTPException(status_code=400, detail=f"仅支持 .md/.txt，收到 {ext or name}")
    text = req.text or ""
    if not text and req.content_base64:
        try:
            text = base64.b64decode(req.content_base64).decode("utf-8", errors="replace")
        except Exception:
            raise HTTPException(status_code=400, detail="文件解码失败（需 UTF-8 文本）")
    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="内容太短（≥50 字）")
    try:
        from resume_project import TEMPLATES_DIR
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        safe = re.sub(r"[^\w.\-一-鿿]+", "_", name)[:80]
        if not safe.endswith(".md"):
            safe = os.path.splitext(safe)[0] + ".md"
        path = os.path.join(TEMPLATES_DIR, safe)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        from resume_project import load_materials
        m = load_materials()
        memory_candidate = None
        if req.stage_for_rag:
            from knowledge_crawler import stage_personal_memory
            memory_candidate = stage_personal_memory(
                text, title=os.path.splitext(safe)[0], kind="resume_rules",
                source_url=f"(用户上传简历材料:{safe})",
            )
        return {"status": "ok", "saved": safe,
                "guidance_count": len(m["guidance"]), "example_count": len(m["examples"]),
                "memory_candidate": memory_candidate}
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("upload_resume_template failed")
        raise HTTPException(status_code=500, detail=f"保存模板失败: {str(e)}")


@app.post("/api/resume/project/stream")
async def resume_project_stream(req: ResumeProjectRequest):
    """Compatibility endpoint backed by the reviewed Resume workflow."""

    def generate():
        from career_multi_agent import start_agent_flow

        yield _sse_event({
            "type": "meta", "task": "resume", "resume_scope": "project_section",
            "workflow_version": "v2", "max_llm_calls": 4,
            "deprecated_endpoint": True,
        })
        try:
            result = start_agent_flow(
                task="resume", resume_scope="project_section",
                application_id=req.application_id, jd_version_id=req.jd_version_id,
                project_repo_url=req.repo_url, project_text=req.text,
                project_name=req.project_name,
                stage_project_memory=req.stage_memory,
            )
            artifact = result.get("artifact") or {}
            yield _sse_event({"type": "artifact", "artifact": artifact})
            if result.get("interrupt"):
                yield _sse_event({"type": "interrupt", "thread_id": result["thread_id"],
                                  "interrupt": result["interrupt"]})
            yield _sse_event({"type": "done", "thread_id": result["thread_id"],
                              "status": result["status"]})
        except Exception as exc:
            _log.exception("reviewed resume project flow failed")
            yield _sse_event({"type": "error", "error": str(exc)})

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/daily/log")
async def append_daily_structured(req: DailyLogStructuredRequest):
    """结构化留痕（Web 表单专用）：主线/已完成/未完成/笔记 → daily_log.md。

    与 CLI cmd_log、晚间复盘 _parse_log_block 走同一写入器，格式统一可解析。
    """
    try:
        from summary_tool import append_structured_daily_log, analyze_incomplete
        if not (req.done or req.todo or req.notes.strip() or req.tag.strip()):
            raise HTTPException(status_code=400, detail="留痕内容不能全空")
        # 未完成不由用户填写：对照 OfferClaw 今日计划自动判定（系统分析生成）
        planned: list = []
        try:
            from career_agent import get_today_advice
            planned = get_today_advice().get("today_plan", [])
        except Exception:
            _log.warning("load today_plan failed", exc_info=True)
        auto_todo = analyze_incomplete(req.done, planned)
        # 兼容外部调用方显式传入的 todo（如微信留痕），合并去重
        final_todo = auto_todo + [t for t in (req.todo or []) if t and t not in auto_todo]
        plan_file = plan_hash = ""
        try:
            from plan_gen import load_latest_plan
            latest = load_latest_plan() or {}
            plan_file = latest.get("filename", "")
            import hashlib as _hashlib
            plan_hash = _hashlib.sha256(latest.get("content", "").encode("utf-8")).hexdigest()
        except Exception:
            pass
        result = append_structured_daily_log(
            tag=req.tag.strip(), done=req.done, todo=final_todo, notes=req.notes,
            task_id=req.task_id, status=req.status, minutes=req.minutes,
            plan_file=plan_file, plan_hash=plan_hash,
            attachment_refs=req.attachment_refs,
            operation_id=req.operation_id or current_request_id(),
        )
        result["today_plan"] = planned
        result["auto_incomplete"] = auto_todo
        return result
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("structured daily log failed")
        raise HTTPException(status_code=500, detail=f"留痕失败: {str(e)}")


@app.post("/api/daily/attachments")
async def upload_daily_attachments(req: DailyAttachmentRequest):
    """保存学习留痕附件，返回可写入 daily_log.md 的本地链接。"""
    if not req.files:
        raise HTTPException(status_code=400, detail="files 不能为空")
    if len(req.files) > 8:
        raise HTTPException(status_code=400, detail="单次最多上传 8 个附件")

    today = datetime.date.today().isoformat()
    target_dir = os.path.join(DAILY_ATTACHMENT_DIR, today)
    os.makedirs(target_dir, exist_ok=True)

    saved = []
    for item in req.files:
        filename = _safe_daily_attachment_name(item.name)
        ext = os.path.splitext(filename)[1].lower()
        content_type = (item.content_type or "").lower()
        if ext not in DAILY_ATTACHMENT_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的附件类型: {ext or item.name}")
        if content_type and not (content_type == "application/pdf" or content_type.startswith("image/")):
            raise HTTPException(status_code=400, detail=f"不支持的附件 MIME: {content_type}")

        data = _decode_attachment_data(item.data_base64)
        if not data:
            raise HTTPException(status_code=400, detail=f"附件为空: {item.name}")
        if len(data) > DAILY_ATTACHMENT_MAX_BYTES:
            raise HTTPException(status_code=400, detail=f"附件过大: {item.name}")

        path = _unique_path(target_dir, filename)
        with open(path, "wb") as f:
            f.write(data)
        public_name = os.path.basename(path)
        url = f"/daily_attachments/{today}/{public_name}"
        saved.append({
            "name": public_name,
            "url": url,
            "markdown": f"[{public_name}]({url})",
            "size": len(data),
            "content_type": content_type,
        })
    return {"status": "ok", "date": today, "count": len(saved), "files": saved}


@app.get("/daily_attachments/{date_str}/{filename}")
async def get_daily_attachment(date_str: str, filename: str):
    """读取学习留痕附件。"""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        raise HTTPException(status_code=404, detail="附件不存在")
    filename = _safe_daily_attachment_name(filename)
    path = os.path.abspath(os.path.join(DAILY_ATTACHMENT_DIR, date_str, filename))
    root = os.path.abspath(DAILY_ATTACHMENT_DIR)
    if not path.startswith(root + os.sep) or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="附件不存在")
    return FileResponse(path)


@app.get("/api/daily", response_model=DailyResponse)
async def get_daily():
    """读 daily_log.md：返回今日块 + 最近 7 天聚合。"""
    try:
        from summary_tool import read_text, extract_date_block, extract_recent_blocks, DAILY_LOG_PATH
        log = read_text(DAILY_LOG_PATH)
        today = datetime.date.today().isoformat()
        return DailyResponse(
            today_log=extract_date_block(log, today),
            recent_summary=extract_recent_blocks(log, days=7),
            recent_days=7,
        )
    except FileNotFoundError:
        # Daily logs are private runtime data and are absent in a public clone.
        return DailyResponse()
    except Exception as e:
        _log.exception("daily get failed")
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@app.get("/api/reflections")
async def get_reflections(q: str = "", operation: str = "get_recent", limit: int = 20,
                          include_orphaned: bool = False):
    if operation not in {"get_recent", "get_by_date", "search_topic", "get_profile_evidence",
                         "get_history_overview"}:
        raise HTTPException(status_code=400, detail="不支持的 reflection operation")
    try:
        from reflection_memory import build_inventory, search_memory
        rows = search_memory(q, operation=operation, limit=max(1, min(limit, 50)),
                             include_orphaned=include_orphaned)
        return {"status": "ok", "operation": operation, "items": rows,
                "stats": build_inventory()["stats"]}
    except Exception as exc:
        _log.exception("reflection list failed")
        raise HTTPException(status_code=500, detail=f"读取复盘失败: {exc}")


@app.get("/api/memory/timeline")
async def memory_timeline(kind: str = "", limit: int = 50,
                          include_archived: bool = False):
    from memory_layers import memory_health
    from memory_store import MemoryStore
    store = MemoryStore()
    return {
        "status": "ok",
        "items": store.list_events(kind=kind, limit=max(1, min(limit, 200)),
                                   include_archived=include_archived),
        "goals": store.list_goals(),
        "health": memory_health(),
    }


@app.get("/api/memory/semantic")
async def memory_semantic(include_inactive: bool = False):
    from memory_layers import SemanticMemory, active_semantic_memories
    semantic = SemanticMemory()
    if include_inactive:
        items = semantic.store.list_semantic(include_inactive=True)
    else:
        items = active_semantic_memories(semantic)
    return {"status": "ok", "items": items,
            "target_context_id": semantic.store.active_goal_id()}


@app.get("/api/memory/semantic/{memory_id}/evidence")
async def memory_semantic_evidence(memory_id: str):
    from memory_store import MemoryStore
    items = MemoryStore().semantic_evidence_events(memory_id)
    return {"status": "ok", "items": items,
            "support": sum(item["stance"] == "support" for item in items),
            "oppose": sum(item["stance"] == "oppose" for item in items)}


@app.get("/api/memory/sops")
async def memory_sops(include_inactive: bool = True):
    from memory_store import MemoryStore
    store = MemoryStore()
    return {"status": "ok", "items": store.list_sops(include_inactive=include_inactive),
            "target_context_id": store.active_goal_id()}


@app.get("/api/memory/search")
async def memory_search(q: str, purpose: str = "recall", limit: int = 5,
                        target_context_id: str = ""):
    if purpose not in {"recall", "advice"}:
        raise HTTPException(status_code=400, detail="purpose 必须是 recall 或 advice")
    if not q.strip():
        raise HTTPException(status_code=400, detail="q 不能为空")
    from memory_search import search_personal_memory
    return search_personal_memory(q.strip(), purpose=purpose,
                                  limit=max(1, min(limit, 20)),
                                  target_context_id=target_context_id or None)


@app.post("/api/memory/distill")
async def memory_distill(model_assisted: bool = False):
    import asyncio
    from memory_layers import (EpisodicMemory, ProceduralMemory, SemanticMemory,
                               distill_free_text_with_model,
                               distill_reflections_to_semantic,
                               distill_to_semantic)
    episodic, semantic = EpisodicMemory(), SemanticMemory()
    result = {
        "status": "ok",
        "semantic": distill_to_semantic(episodic, semantic),
        "reflection": distill_reflections_to_semantic(episodic, semantic),
        "sops": ProceduralMemory().list(),
    }
    if model_assisted:
        result["model_assisted"] = await asyncio.to_thread(
            distill_free_text_with_model, episodic, semantic, force=True)
    return result


@app.post("/api/memory/goals/switch")
async def memory_switch_goal(req: MemoryGoalSwitchRequest):
    from memory_layers import switch_goal_context
    try:
        return {"status": "ok", "goal": switch_goal_context(req.name)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/memory/goals/{context_id}/activate")
async def memory_activate_goal(context_id: str):
    from memory_layers import activate_goal_context
    try:
        return {"status": "ok", "goal": activate_goal_context(context_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="目标上下文不存在")


@app.patch("/api/memory/{object_type}/{object_id}")
async def memory_set_lifecycle(object_type: str, object_id: str,
                               req: MemoryLifecycleRequest):
    from memory_layers import EpisodicMemory, ProceduralMemory, SemanticMemory
    from memory_store import MemoryStore
    try:
        changed = MemoryStore().set_lifecycle(object_type, object_id, req.lifecycle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not changed:
        raise HTTPException(status_code=404, detail="记忆不存在或已删除")
    if object_type == "event":
        EpisodicMemory()._export()
    elif object_type == "semantic":
        SemanticMemory()._export()
    else:
        ProceduralMemory()._export()
    return {"status": "ok", "object_type": object_type,
            "object_id": object_id, "lifecycle": req.lifecycle}


@app.delete("/api/memory/{object_type}/{object_id}")
async def memory_delete(object_type: str, object_id: str,
                        req: MemoryDeleteRequest | None = None):
    from memory_layers import EpisodicMemory, ProceduralMemory, SemanticMemory
    from memory_search import remove_from_index
    from memory_store import MemoryStore
    if object_type not in {"event", "semantic", "sop"}:
        raise HTTPException(status_code=400, detail="object_type 非法")
    changed = MemoryStore().delete_object(
        object_type, object_id, (req.reason if req else "用户主动删除"))
    if not changed:
        raise HTTPException(status_code=404, detail="记忆不存在或已删除")
    if object_type == "event":
        remove_from_index(object_id)
        EpisodicMemory()._export()
        SemanticMemory()._export()
        ProceduralMemory()._export()
    elif object_type == "semantic":
        SemanticMemory()._export()
    else:
        ProceduralMemory()._export()
    return {"status": "ok", "deleted": True, "object_type": object_type,
            "object_id": object_id}


@app.post("/api/memory/index/rebuild")
async def memory_rebuild_index():
    import asyncio
    from memory_search import rebuild_index
    result = await asyncio.to_thread(rebuild_index)
    return result


@app.post("/api/reflections/run")
async def run_reflection_api(req: ReflectionRunRequest):
    """显式生成日/周复盘；缺日志时 fail-visible，不拿其他日期内容兜底。"""
    if req.mode not in {"daily", "weekly"}:
        raise HTTPException(status_code=400, detail="mode 必须是 daily 或 weekly")
    date = req.date or datetime.date.today().isoformat()
    try:
        datetime.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="date 必须是 YYYY-MM-DD")
    import asyncio
    import subprocess

    python_bin = os.path.join(BASE_DIR, ".venv", "bin", "python")
    cmd = [python_bin if os.path.exists(python_bin) else sys.executable,
           os.path.join(BASE_DIR, "summary_tool.py")]
    if req.mode == "weekly":
        cmd.extend(["--weekly", "--date", date])
    else:
        cmd.extend(["--date", date])
    proc = await asyncio.to_thread(
        subprocess.run, cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=180)
    if proc.returncode == 2:
        raise HTTPException(status_code=409, detail=(proc.stdout or proc.stderr).strip())
    if proc.returncode != 0:
        raise HTTPException(status_code=502, detail=(proc.stderr or proc.stdout).strip()[-1000:])
    from reflection_memory import search_memory, write_derived_index
    write_derived_index()
    return {"status": "ok", "date": date, "mode": req.mode,
            "items": search_memory(date, "get_by_date", 10), "output": proc.stdout[-1000:]}


@app.post("/api/daily", response_model=DailyResponse)
async def append_daily(req: DailyAppendRequest):
    """向 daily_log.md 追加今日条目（如果今日块不存在则新建标题）。"""
    try:
        from summary_tool import read_text, extract_date_block, extract_recent_blocks, DAILY_LOG_PATH
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text 不能为空")
        today = datetime.date.today().isoformat()
        log_path = DAILY_LOG_PATH
        existing = read_text(log_path) if os.path.exists(log_path) else ""
        block = extract_date_block(existing, today)
        ts = datetime.datetime.now().strftime("%H:%M")
        if block:
            new_log = existing.rstrip() + f"\n- {ts} {text}\n"
        else:
            sep = "\n\n" if existing else ""
            new_log = existing + f"{sep}## {today}\n- {ts} {text}\n"
        import hashlib as _hashlib
        from memory_store import new_id
        log_id = new_id("log")
        write_text_with_memory(
            log_path, new_log, event_kind="daily_log_recorded",
            event_payload={"log_id": log_id, "date": today, "status": "done",
                           "notes": text, "done": [], "incomplete": [], "task_id": ""},
            event_options={"actor": "user", "source": "daily_api",
                           "entity_type": "daily_log", "entity_id": log_id,
                           "business_date": today},
            operation_id=current_request_id() or None,
            expected_before_hash=_hashlib.sha256(existing.encode("utf-8")).hexdigest() if existing else "",
        )
        return DailyResponse(
            today_log=extract_date_block(new_log, today),
            recent_summary=extract_recent_blocks(new_log, days=7),
            recent_days=7,
        )
    except MemoryFileConflictError:
        raise HTTPException(status_code=409, detail="每日记录已被其他操作更新，请重试")
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("daily append failed")
        raise HTTPException(status_code=500, detail=f"追加失败: {str(e)}")


@app.get("/api/resume", response_model=ResumeResponse)
async def get_resume():
    """聚合简历素材：docs/archive/resume_pitch.md 全文 + interview_story_bank.md 标题列表。"""
    try:
        pitch_path = os.path.join(BASE_DIR, "docs", "archive", "resume_pitch.md")
        story_path = os.path.join(BASE_DIR, "interview_story_bank.md")
        pitch = open(pitch_path, encoding="utf-8").read() if os.path.exists(pitch_path) else "（缺 docs/archive/resume_pitch.md）"
        stories_preview = ""
        if os.path.exists(story_path):
            import re as _re
            content = open(story_path, encoding="utf-8").read()
            titles = _re.findall(r"^## (Story.*)$", content, _re.MULTILINE)
            stories_preview = "\n".join(f"- {t}" for t in titles) or "（未识别到 Story 标题）"
        return ResumeResponse(pitch=pitch, stories_preview=stories_preview)
    except Exception as e:
        _log.exception("resume failed")
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@app.get("/api/today", response_model=TodayResponse)
async def get_today():
    """V2 阶段三：聚合 applications + daily_log + profile，给一句"今天最该做什么"。"""
    try:
        from career_agent import get_today_advice
        return TodayResponse(**get_today_advice())
    except Exception as e:
        _log.exception("today failed")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/api/discover", response_model=DiscoverResponse)
async def discover(req: DiscoverRequest):
    """V2 阶段四：JD 半自动抽取。raw 文本或 URL 都可，返回结构化 JD。
    URL 模式：先 requests 快速抓，若 SPA 则自动启动 Playwright 无头浏览器渲染。
    """
    import asyncio
    try:
        from job_discovery import discover as _disc
        loop = asyncio.get_event_loop()
        # Playwright 是同步阻塞调用，放线程池避免卡事件循环
        out = await loop.run_in_executor(None, lambda: _disc(raw=req.raw, url=req.url))
        return DiscoverResponse(**out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("discover failed")
        raise HTTPException(status_code=500, detail=f"抽取失败: {str(e)}")


# =====================================================
# 知识库维护：采集/上传 → 评分预览 → 人工确认 → 增量入库
# =====================================================

def _kb_count() -> int:
    """当前 collection 的块数（count 读 SQLite 元数据，跨进程即时准确）。"""
    from rag_tools import get_collection_name
    return _kb_sqlite_stats(get_collection_name())[0]


def _kb_sqlite_stats(collection_name: str) -> tuple[int, dict[str, int]]:
    """Read Chroma metadata without loading the native vector extension."""
    from rag_tools import get_collection_sqlite_stats
    return get_collection_sqlite_stats(os.path.join(BASE_DIR, "chroma_db"), collection_name)


def _indexed_sources() -> set:
    """当前 collection 里全部 chunk 的来源文件名集合（供「未入库文件」扫描做差集）。"""
    import chromadb
    from rag_tools import get_collection_name
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    col = client.get_collection(get_collection_name())
    seen, offset, page = set(), 0, 2000
    while True:
        got = col.get(include=["metadatas"], limit=page, offset=offset)
        metas = got.get("metadatas") or []
        for m in metas:
            src = (m or {}).get("source")
            if src:
                seen.add(src)
        if len(metas) < page:
            break
        offset += page
    return seen


def _kb_clear_cache():
    """清 ChromaDB 进程内缓存：让长驻 API 的后续向量查询能立刻看到新入库内容
    （count 本就即时，但 HNSW 段缓存对跨进程新写入会滞后）。"""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        _log.warning("clear_system_cache failed", exc_info=True)


@app.get("/api/kb/status")
async def kb_status():
    """知识库概览：collection 名、块数、按 source_type 的来源分布。"""
    from rag_tools import get_collection_name
    try:
        name = get_collection_name()
        chunks, sources = _kb_sqlite_stats(name)
        return {"status": "ok" if chunks else "degraded", "collection": name,
                "chunks": chunks, "sources": sources}
    except Exception as e:
        _log.exception("kb_status failed")
        raise HTTPException(status_code=500, detail=f"读取知识库状态失败: {str(e)}")


@app.get("/api/kb/pending")
async def kb_pending():
    """待人工确认的候选清单（已落 _pending、尚未入库）。"""
    try:
        from knowledge_crawler import cmd_list_pending, flag_existing_in_kb, BASE_DIR as KC_BASE
        out = cmd_list_pending()
        for it in out.get("items", []):
            if it.get("saved_abs"):
                it["saved"] = os.path.relpath(it["saved_abs"], KC_BASE)
        # 「疑似已在库」标记：正式库存在同名/同标题文件 → 大概率是历史遗留的暂存副本
        flag_existing_in_kb(out.get("items", []))
        return out
    except Exception as e:
        _log.exception("kb_pending failed")
        raise HTTPException(status_code=500, detail=f"读取待审列表失败: {str(e)}")


@app.post("/api/kb/add_url")
async def kb_add_url(req: KBAddUrlRequest):
    """① 抓取 URL → 打分 → 落 _pending（待确认），不直接入库。
    命中登录/安全验证墙时返回 400 + 明确提示（同 /api/discover）。"""
    import asyncio
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url 不能为空")
    try:
        from knowledge_crawler import cmd_crawl
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(None, lambda: cmd_crawl(url))
        if out.get("status") == "ok":
            try:
                source_path = str(out.get("saved") or out.get("saved_abs") or "")
                content = json.dumps(out, ensure_ascii=False, indent=2)
                candidate = source_path if os.path.isabs(source_path) else os.path.join(BASE_DIR, source_path)
                if os.path.isfile(candidate):
                    content = open(candidate, encoding="utf-8").read()
                _record_content_artifact(
                    "knowledge_material_changed", content, action="submitted",
                    source="kb_add_url", operation_id=req.operation_id,
                    source_path=source_path, source_type="url_candidate", result=out,
                )
            except Exception:
                _log.warning("knowledge URL memory event failed", exc_info=True)
        return out  # status: ok / rejected / error，前端据此展示
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("kb_add_url failed")
        raise HTTPException(status_code=500, detail=f"抓取失败: {str(e)}")


@app.post("/api/kb/add_file")
async def kb_add_file(req: KBAddFileRequest):
    """① 接收本地文件（.md/.txt 文本；.pdf/.docx 服务端抽取文字）→ 打分 → 落 _pending（待确认）。
    用户主动上传，故即便相关性偏低也保留（降级 C 供确认）。
    PDF 只抽文字层（扫描版明确拒绝，不做 OCR——避免入库空壳的静默劣化）。"""
    import asyncio
    name = (req.name or "upload.md").strip()
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".md", ".txt", ".markdown", ".pdf", ".docx"):
        raise HTTPException(status_code=400,
                            detail=f"仅支持 .md/.txt/.markdown/.pdf/.docx，收到 {ext or name}")
    parser = (req.parser or "").strip().lower()
    if parser not in ("", "text", "docling"):
        raise HTTPException(status_code=400, detail=f"parser 仅支持 text/docling,收到 {parser}")
    text = req.text or ""
    origin = "本地上传"
    if ext in (".pdf", ".docx"):
        if not req.content_base64:
            raise HTTPException(status_code=400, detail="PDF/Word 需以 content_base64 上传")
        try:
            raw = base64.b64decode(req.content_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="文件内容解码失败")
        try:
            # 自动路由(2026-08-09,用户产品决策:不让用户选引擎):
            #   .pdf  → 默认 Docling 结构化(论文/表格保真),失败自动回退文字层;
            #   .docx → 默认原生解析(python-docx,指导文档 §7:不绕道 PDF);
            #   parser 显式指定时尊重指定,且显式 docling 失败不静默换引擎(fail-visible)。
            from knowledge_crawler import extract_text_for_kb, extract_text_structured
            use_structured = (parser == "docling") or (parser == "" and ext == ".pdf")
            if use_structured:
                try:
                    text = extract_text_structured(name, raw)
                    origin = f"本地上传({ext[1:]} Docling 结构化)"
                except ValueError as de:
                    if parser == "docling":
                        raise
                    text = extract_text_for_kb(name, raw)
                    origin = f"本地上传({ext[1:]} 文字层·结构化回退:{str(de)[:40]})"
            else:
                text = extract_text_for_kb(name, raw)
                origin = f"本地上传({ext[1:]} 抽取文本)"
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif not text and req.content_base64:
        try:
            text = base64.b64decode(req.content_base64).decode("utf-8", errors="replace")
        except Exception:
            raise HTTPException(status_code=400, detail="文件内容解码失败（需 UTF-8 文本）")
    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")
    try:
        from knowledge_crawler import _score_and_save
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None,
            lambda: _score_and_save(text, url=f"(本地上传:{name})", origin=origin, force_keep=True),
        )
        if out.get("status") == "ok":
            try:
                _record_content_artifact(
                    "knowledge_material_changed", text, action="submitted",
                    source="kb_add_file", operation_id=req.operation_id,
                    source_path=str(out.get("saved") or name), source_type="file_candidate",
                    result=out,
                )
            except Exception:
                _log.warning("knowledge file memory event failed", exc_info=True)
        return out
    except Exception as e:
        _log.exception("kb_add_file failed")
        raise HTTPException(status_code=500, detail=f"处理上传失败: {str(e)}")


@app.post("/api/kb/promote")
async def kb_promote(req: KBPromoteRequest):
    """② 人工确认后：把 _pending 文件提升到正式子目录并**增量入库**（不重建）。
    返回入库前后块数；清进程缓存让 Web 查询即时可见。"""
    import asyncio
    pending = (req.pending_file or "").strip()
    subdir = (req.to_subdir or "").strip()
    if not pending or not subdir:
        raise HTTPException(status_code=400, detail="pending_file 与 to_subdir 必填")
    try:
        from knowledge_crawler import cmd_promote, VALID_SUBDIRS
        if subdir not in VALID_SUBDIRS:
            raise HTTPException(status_code=400, detail=f"to_subdir 必须是 {sorted(VALID_SUBDIRS)} 之一")
        before = _kb_count()
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None, lambda: cmd_promote(pending, subdir, ingest=True)
        )
        if out.get("status") != "ok":
            raise HTTPException(status_code=400, detail=out.get("error", "提升失败"))
        _kb_clear_cache()  # 让本进程后续向量查询能看到新入库内容
        after = _kb_count()
        out["chunks_before"] = before
        out["chunks_after"] = after
        out["chunks_added"] = max(0, after - before)
        try:
            promoted = str(out.get("promoted_to") or "")
            absolute = promoted if os.path.isabs(promoted) else os.path.join(BASE_DIR, promoted)
            content = open(absolute, encoding="utf-8").read() if os.path.isfile(absolute) else json.dumps(out, ensure_ascii=False)
            _record_content_artifact(
                "knowledge_material_changed", content, action="promoted",
                source="kb_promote", operation_id=req.operation_id,
                source_path=promoted, source_type=subdir, result=out,
            )
        except Exception:
            _log.warning("knowledge promote memory event failed", exc_info=True)
        return out
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("kb_promote failed")
        raise HTTPException(status_code=500, detail=f"入库失败: {str(e)}")


@app.get("/api/kb/preview")
async def kb_preview(rel: str):
    """审核用本地预览：只读 knowledge_base 子树内 .md/.txt（候选与正式文件通用）。"""
    try:
        from knowledge_crawler import read_kb_preview
        return read_kb_preview(rel)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("kb_preview failed")
        raise HTTPException(status_code=500, detail=f"预览失败: {str(e)}")


@app.post("/api/kb/reveal")
async def kb_reveal(req: KBPathRequest):
    """在系统文件管理器中定位该文件（本地单用户工具；macOS `open -R`）。"""
    import subprocess
    import sys as _sys
    try:
        from knowledge_crawler import safe_kb_file
        abs_p = safe_kb_file(req.rel)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if _sys.platform != "darwin":
        raise HTTPException(status_code=400, detail="「在访达中显示」仅支持 macOS 本机")
    subprocess.Popen(["open", "-R", abs_p])
    return {"status": "ok", "path": abs_p}


@app.get("/api/kb/unindexed")
async def kb_unindexed():
    """文件夹直投轨道：正式库里「磁盘上有、索引里没有」的文件（待审核后增量入库）。"""
    try:
        from knowledge_crawler import list_unindexed
        sources = _indexed_sources()
        items = list_unindexed(sources)
        return {"count": len(items), "items": items, "indexed_sources": len(sources)}
    except Exception as e:
        _log.exception("kb_unindexed failed")
        raise HTTPException(status_code=500, detail=f"扫描未入库文件失败: {str(e)}")


@app.post("/api/kb/ingest_path")
async def kb_ingest_path(req: KBPathRequest):
    """文件夹直投文件审核通过后的增量入库（镜像经验贴的 --add 子进程路径，不重建）。"""
    import asyncio
    import subprocess
    try:
        from knowledge_crawler import safe_kb_file
        abs_p = safe_kb_file(req.rel)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if f"{os.sep}_pending{os.sep}" in abs_p:
        raise HTTPException(status_code=400, detail="候选文件请走「选它 → 确认入库」(promote) 流程")
    rel_repo = os.path.relpath(abs_p, BASE_DIR)
    try:
        from rag_ingest import _infer_source_type
        st = (req.source_type or "").strip() or _infer_source_type(rel_repo)
    except Exception:
        st = (req.source_type or "").strip() or "doc"
    before = _kb_count()
    loop = asyncio.get_event_loop()
    proc = await loop.run_in_executor(None, lambda: subprocess.run(
        [os.path.join(BASE_DIR, ".venv/bin/python"), "rag_ingest.py",
         "--add", rel_repo, "--source-type", st],
        cwd=BASE_DIR, capture_output=True, text=True, timeout=600,
    ))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-300:]
        raise HTTPException(status_code=500, detail=f"增量入库失败: {tail}")
    _kb_clear_cache()
    after = _kb_count()
    out = {"status": "ok", "file": rel_repo, "source_type": st,
           "chunks_before": before, "chunks_after": after,
           "chunks_added": max(0, after - before)}
    try:
        content = open(abs_p, encoding="utf-8").read()
        _record_content_artifact(
            "knowledge_material_changed", content, action="indexed",
            source="kb_ingest_path", operation_id=req.operation_id,
            source_path=rel_repo, source_type=st, result=out,
        )
    except Exception:
        _log.warning("knowledge ingest memory event failed", exc_info=True)
    return out


@app.post("/api/resume/build", response_model=ResumeBuildResponse)
async def build_resume(req: ResumeBuildRequest):
    """V2 阶段五：针对一份 JD 生成 OfferClaw 项目段（bullet + 段落 + 命中分析）。"""
    import asyncio
    try:
        from resume_builder import build_resume_for_jd
        meta = []
        if req.company: meta.append(f"公司：{req.company}")
        if req.title: meta.append(f"岗位：{req.title}")
        meta.append("JD 原文：\n" + req.jd_text[:4000])
        jd_summary = "\n".join(meta)
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(None, lambda: build_resume_for_jd(jd_summary))
        try:
            event = _record_content_artifact(
                "resume_artifact_changed", str(out.get("resume_md") or ""),
                action="generated", source="resume_build",
                operation_id=req.operation_id, application_id=req.application_id,
                actor="assistant",
            )
            out["memory_event_id"] = event["event_id"]
        except Exception:
            _log.warning("resume generation memory event failed", exc_info=True)
        return ResumeBuildResponse(**out)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        _log.exception("resume build failed")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/api/resume/build/stream")
async def build_resume_stream(req: ResumeBuildRequest):
    """流式版简历生成：SSE 逐 token 返回，首 token 约 1s 内到达。"""
    import asyncio, json as _json
    api_key, api_key_env = _active_llm_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail=f"{api_key_env} 未配置")
    from resume_builder import build_messages as _bm, _read
    from plan_gen import call_llm_stream
    meta = []
    if req.company: meta.append(f"公司：{req.company}")
    if req.title: meta.append(f"岗位：{req.title}")
    meta.append("JD 原文：\n" + req.jd_text[:4000])
    messages = _bm(jd_summary="\n".join(meta),
                   profile=_read(os.path.join(BASE_DIR, "user_profile.md")))

    async def generate():
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()
        full_text: list[str] = []

        def _producer():
            try:
                for tok in call_llm_stream(messages, api_key, max_tokens=2000):
                    loop.call_soon_threadsafe(q.put_nowait, tok)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, ("__stream_error__", str(exc)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        loop.run_in_executor(None, _producer)
        while True:
            tok = await q.get()
            if tok is None:
                if full_text:
                    try:
                        _record_content_artifact(
                            "resume_artifact_changed", "".join(full_text), action="generated",
                            source="resume_build_stream", operation_id=req.operation_id,
                            application_id=req.application_id, actor="assistant",
                        )
                    except Exception:
                        _log.warning("resume stream memory event failed", exc_info=True)
                yield _sse_event({"type": "done"})
                break
            if isinstance(tok, tuple) and tok and tok[0] == "__stream_error__":
                yield _sse_event({"type": "error", "error": tok[1]})
                break
            full_text.append(tok)
            yield _sse_event({"type": "delta", "text": tok})

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/plan/stream")
async def gen_plan_stream(req: PlanRequest):
    """流式版规划生成：SSE 逐 token 返回。"""
    import asyncio, json as _json
    api_key, api_key_env = _active_llm_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail=f"{api_key_env} 未配置")
    from plan_gen import (
        prepare_plan_messages, call_llm_stream,
        append_resources_appendix, append_target_trace,
    )
    gaps = _resolve_plan_gaps(req.gaps)
    # 统一入口：读依赖 + RAG 检索资源 + 组装 messages（与 CLI / 非流式一致）
    messages, resources = prepare_plan_messages(gaps, revision_note=req.revision_note,
                                                  start_date=req.start_date, end_date=req.end_date)

    async def generate():
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()
        full_text = []

        def _producer():
            try:
                from plan_gen import (is_degenerate_plan, normalize_plan_dates,
                                      load_latest_plan, summarize_plan_changes)
                prev = load_latest_plan()   # 保存前取上一版，用于变化摘要
                for tok in call_llm_stream(messages, api_key, max_tokens=7000):
                    full_text.append(tok)
                    loop.call_soon_threadsafe(q.put_nowait, ("tok", tok))
                raw = "".join(full_text)
                # 退化产物（拒绝/无周结构）不落盘，避免污染"当前计划"被后续注入自我复制
                if is_degenerate_plan(raw):
                    loop.call_soon_threadsafe(
                        q.put_nowait, ("done", {"path": "", "changes": [], "daily_days": 0}))
                    return
                # 日期确定性归一（LLM 连排日期不可信）
                raw = normalize_plan_dates(
                    raw, req.start_date or datetime.date.today().isoformat())
                # 流结束后确定性追加参考资源附录，并保存为待批准草稿。
                plan_md = append_resources_appendix(raw, resources)
                plan_md = append_target_trace(plan_md)
                # 把附录部分也作为最后一段 token 推给前端
                appendix = plan_md[len(raw):]
                if appendix:
                    loop.call_soon_threadsafe(q.put_nowait, ("tok", appendix))
                changes = summarize_plan_changes(prev["content"] if prev else "", plan_md)
                # 生成后校验门（fail-visible）：日计划层解析天数随 done 事件推给前端
                try:
                    from plan_daily import parse_plan_days
                    daily_days = len(parse_plan_days(plan_md)["days"])
                except Exception:
                    daily_days = -1
                from plan_drafts import create_plan_draft
                draft = create_plan_draft(
                    plan_md, changes=changes if prev else [], daily_days=daily_days,
                    source="plan_stream",
                )
                loop.call_soon_threadsafe(q.put_nowait, ("done", {
                    "path": "", "draft_id": draft["draft_id"],
                    "changes": changes if prev else [],
                    "daily_days": daily_days}))
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, ("err", str(exc)))

        loop.run_in_executor(None, _producer)
        while True:
            kind, val = await q.get()
            if kind == "tok":
                yield _sse_event({"type": "delta", "text": val})
            elif kind == "done":
                payload = val if isinstance(val, dict) else {"path": val, "changes": []}
                yield _sse_event({"type": "done", "saved_path": payload["path"],
                                  "draft_id": payload.get("draft_id", ""),
                                  "requires_confirmation": bool(payload.get("draft_id")),
                                  "changes": payload["changes"],
                                  "daily_days": payload.get("daily_days", -1)})
                break
            else:
                yield _sse_event({"type": "error", "error": val})
                break

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stream")
async def rag_stream(req: QueryRequest):
    """SSE 流式问答（知识库优先 · 带门槛 · 与 /api/query 同一套 rag_gate）。

    先推 meta（in_kb / mode / sources），再逐 token 推 delta：
    - 命中知识库 → 基于 KB 合成 + 标出处；
    - 未命中 → 通用知识 + 项目先验兜底，开头带"非知识库"标注。
    合成模型默认继承项目主模型（当前为 GPT）。
    """
    from query_service import iter_query_events
    from traffic_origin import bind_traffic_origin_iterable, current_traffic_origin
    rid = current_request_id()
    conversation_id, user_message_id, assistant_message_id, operation_id = _conversation_ids(req)
    _record_conversation("user", req.query, "sent", conversation_id=conversation_id,
                         message_id=user_message_id, operation_id=operation_id)
    request_traffic_origin = current_traffic_origin()
    _log.info(f"stream start q={req.query[:60]!r}")

    def gen():
        answer_parts: list[str] = []
        source_refs: list[str] = []
        last_meta: dict = {}
        persisted = False
        try:
            stream = iter_query_events(
                req.query, req.top_k, context=req.context,
                conversation_id=conversation_id,
            )
            for ev in stream:
                t = ev.get("type")
                if t == "stage":
                    stage = {k: ev[k] for k in
                             ("stage", "status", "label", "detail", "elapsed_ms",
                              "total_elapsed_ms", "routes", "sources",
                              "source_status", "evidence_count") if k in ev}
                    stage["request_id"] = rid
                    yield _sse_event({"type": "stage", **stage})
                elif t == "meta":
                    meta = {k: ev[k] for k in
                            ("in_kb", "mode", "sources", "matched_by", "best_distance",
                             "planner_mode", "routes", "route_reason", "source_status",
                             "coverage", "freshness", "latency_ms",
                             "resolved_entities", "intent_frame", "resolver_mode",
                             "decision", "clarification", "candidate_routes",
                             "confidence_factors", "service_mode", "interaction_kind",
                             "turn_relation", "relation_target_turn_id",
                             "action_request", "routing_assurance", "decision_reasons",
                             "context_resolution", "requested_action", "planner_engine",
                             "planner_version", "schema_version", "route_model",
                             "repair_used", "fallback_reason", "planner_queue_ms",
                             "planner_provider_ms", "planner_wall_ms",
                             "planner_deadline_ms", "planner_cache_hit",
                             "planner_timeout_stage", "planner_late_response",
                             "planner_circuit_state", "prompt_tokens",
                             "completion_tokens", "structured_output_mode",
                             "route_gateway", "route_reasoning_effort",
                             "json_capability", "retrieval_profile",
                             "index_fingerprint", "effective_hit",
                             "answer_action", "data_version",
                             "query_service_version") if k in ev}
                    meta["request_id"] = rid
                    meta["conversation_id"] = conversation_id
                    meta["user_message_id"] = user_message_id
                    meta["assistant_message_id"] = assistant_message_id
                    source_refs[:] = list(ev.get("sources") or [])
                    last_meta.clear()
                    last_meta.update(meta)
                    yield _sse_event({"type": "meta", **meta})
                elif t == "delta":
                    chunk = ev.get("text", "")
                    answer_parts.append(chunk)
                    yield _sse_event({"type": "delta", "text": chunk})
                elif t == "done":
                    _record_conversation("assistant", "".join(answer_parts), "completed",
                                         conversation_id=conversation_id,
                                         message_id=assistant_message_id,
                                         operation_id=operation_id, source_refs=source_refs)
                    try:
                        from conversation_context import record_successful_turn
                        record_successful_turn(
                            conversation_id, last_meta,
                            turn_id=assistant_message_id,
                        )
                    except Exception:
                        _log.warning("stream conversation context write failed", exc_info=True)
                    persisted = True
                    try:
                        from memory_layers import distill_to_semantic, EpisodicMemory, SemanticMemory
                        distill_to_semantic(EpisodicMemory(), SemanticMemory())
                    except Exception:
                        _log.warning("stream conversation distillation failed", exc_info=True)
                    yield _sse_event({"type": "done", "conversation_id": conversation_id,
                                      "assistant_message_id": assistant_message_id})
            _log.info("stream done")
        except Exception as e:
            try:
                _record_conversation("assistant", "".join(answer_parts) or str(e), "failed",
                                     conversation_id=conversation_id,
                                     message_id=assistant_message_id,
                                     operation_id=operation_id, source_refs=source_refs)
                persisted = True
            except Exception:
                _log.warning("stream failure memory write failed", exc_info=True)
            _log.exception("stream failed")
            yield _sse_event({"type": "error", "error": str(e)})
        finally:
            if not persisted:
                try:
                    _record_conversation("assistant", "".join(answer_parts) or "stream interrupted",
                                         "interrupted", conversation_id=conversation_id,
                                         message_id=assistant_message_id,
                                         operation_id=operation_id, source_refs=source_refs)
                except Exception:
                    _log.warning("stream interrupt memory write failed", exc_info=True)

    return StreamingResponse(
        bind_traffic_origin_iterable(gen(), request_traffic_origin),
        media_type="text/event-stream",
    )


def _sse_event(payload: dict) -> str:
    """统一 SSE wire 格式（前端 P0 还债）：每条消息一行 ``data: <json>``，
    payload 必带 ``type``；基础流使用 meta/delta/done/error，Agent 流扩展
    stage/scope/node/artifact/interrupt，delta 的文本统一放 ``text`` 字段。所有流式端点
    都必须经过本函数（tests/test_sse_wire_format.py
    用源码级 lint 强制）；改这里必须同步 static/index.html 的 streamSSE()。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/reset")
async def reset_conversation(conversation_id: str = ""):
    """清空对话历史"""
    try:
        if _rag_agent is not None:
            _rag_agent.reset()
        cleared = 0
        if conversation_id:
            from conversation_context import clear_conversation_context
            cleared = clear_conversation_context(conversation_id)
        return {"status": "ok", "message": "对话历史已清空",
                "cleared_context_turns": cleared}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置失败: {str(e)}")


# =====================================================
# Phase 3 / 4 / 5：UI Console + JD Discovery 增强 + Resume Markdown
# =====================================================

@app.get("/ui/console")
async def ui_console(request: Request):
    """求职流程 Stepper 控制台（Phase 3）。"""
    if not request.url.query:
        return RedirectResponse(
            url=f"/ui/console?rev={_static_rev('console.html')}",
            status_code=307,
            headers=HTML_NO_CACHE_HEADERS,
        )
    return _html_file_response("console.html")


@app.get("/api/jd/queries", response_model=JDQueriesResponse)
async def jd_queries():
    """根据 user_profile.md 生成搜索关键词组合（不爬，仅推荐）。"""
    try:
        from job_discovery import build_search_queries
        from profile_loader import load_profile
        prof = load_profile()
        return JDQueriesResponse(
            queries=build_search_queries(prof),
            profile_cities=prof.get("可接受地域") or [],
            profile_directions=prof.get("方向优先级") or [],
        )
    except Exception as e:
        _log.exception("jd_queries failed")
        raise HTTPException(status_code=500, detail=f"生成搜索词失败: {str(e)}")


@app.post("/api/jd/rank", response_model=JDRankResponse)
async def jd_rank(req: JDRankRequest):
    """对一组候选 JD 调 match_job 排序，输出推荐顺序。"""
    try:
        from job_discovery import rank_candidates
        ranked = rank_candidates([c.model_dump() for c in req.candidates])
        items = [JDRankItem(
            title=r.get("title", ""), status=r.get("status", ""),
            direction=r.get("direction", ""), gap_count=r.get("gap_count", 0),
            score=r.get("score", 0), reason=r.get("reason", ""),
        ) for r in ranked]
        return JDRankResponse(ranked=items, total=len(items))
    except Exception as e:
        _log.exception("jd_rank failed")
        raise HTTPException(status_code=500, detail=f"JD 排序失败: {str(e)}")


@app.post("/api/resume/markdown", response_model=ResumeMarkdownResponse)
async def resume_markdown(req: ResumeMarkdownRequest):
    """生成完整 Markdown 简历草稿（默认无 LLM）。"""
    try:
        from resume_builder import build_resume_markdown
        out = build_resume_markdown(jd_text=req.jd_text, skip_llm=req.skip_llm)
        return ResumeMarkdownResponse(
            resume_md=out["resume_md"], sections=out["sections"],
            jd_chars=out["jd_chars"], skip_llm=out["skip_llm"],
            llm_used=out.get("llm_used", False), llm_error=out.get("llm_error", ""),
        )
    except Exception as e:
        _log.exception("resume_markdown failed")
        raise HTTPException(status_code=500, detail=f"生成简历草稿失败: {str(e)}")


@app.post("/api/agent", response_model=AgentResponse)
async def agent_run(req: AgentRequest):
    """ReAct Agent：一句自然语言 → tool 调用 → 结论（V4 §3）。

    mode='deterministic'（默认）：纯关键词路由，无 KEY 也能用，秒级返回。
    mode='llm'：function calling 工具循环，无 KEY 自动降级到 deterministic。
    """
    try:
        from react_agent import run as agent_run_fn
        out = agent_run_fn(req.message, mode=req.mode, max_steps=req.max_steps)
        return AgentResponse(**out)
    except Exception as e:
        _log.exception("agent_run failed")
        raise HTTPException(status_code=500, detail=f"agent 执行失败: {str(e)}")


# =====================================================
# Observability：trace + 重放（V4 §4）
# =====================================================

@app.get("/api/trace")
async def trace_list(limit: int = 20):
    """列出最近 N 条 CareerFlow trace（按 mtime 倒序）。"""
    from observability import list_traces
    return {"items": list_traces(limit=limit)}


@app.get("/api/trace/{trace_id}")
async def trace_detail(trace_id: str):
    """读回单条 trace 的全部 JSONL 事件。"""
    from observability import read_trace
    try:
        return read_trace(trace_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"trace 不存在: {trace_id}")


@app.post("/api/flow/run_traced", response_model=FlowRunResponse)
async def flow_run_traced(req: FlowRunRequest):
    """跑 routed CareerFlow 并落 trace 文件，返回 state（含 _trace_id）。"""
    from observability import trace_career_flow
    try:
        out = trace_career_flow(
            req.jd_text, jd_title=req.jd_title, skip_llm=req.skip_llm,
            routed=True,
        )
        return FlowRunResponse(
            match_report=out.get("match_report", {}),
            gaps=out.get("gaps", {}),
            plan_outline=out.get("plan_outline", []),
            today_advice=out.get("today_advice", {}),
            resume_skeleton=out.get("resume_skeleton", {}),
            application_suggestion=out.get("application_suggestion", {}),
            requires_confirmation=out.get("requires_confirmation", []),
            trace=out.get("trace", []) + [{"_trace_id": out["_trace_id"]}],
            errors=out.get("errors", []),
            route_history=out.get("route_history", []),
            checkpoint_status=out.get("checkpoint_status", {}),
            memory_status=out.get("memory_status", {}),
            fatal_error=bool(out.get("fatal_error")),
            fatal_reason=out.get("fatal_reason", ""),
        )
    except Exception as e:
        _log.exception("flow_run_traced failed")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# 启动入口
# =====================================================

if __name__ == "__main__":
    import uvicorn
    # 数据主权默认值(2026-07-09 验收):只绑本机回环地址——绑 0.0.0.0 会把
    # 无鉴权的知识库/画像/运营接口暴露给整个局域网,与「隐私即架构」矛盾。
    # 共享实例场景(见 docs/MULTI_USER_ROADMAP.md)由用户显式设置
    # OFFERCLAW_HOST=0.0.0.0 自担边界。
    uvicorn.run(
        app,
        host=os.getenv("OFFERCLAW_HOST", "127.0.0.1"),
        port=int(os.getenv("OFFERCLAW_PORT", "8000")),
        reload=True,
    )
