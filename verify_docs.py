"""verify_docs.py — 文档指标与仓库隐私门禁。

单一事实源 = `metrics.json`（current 当前真值 + stale_blacklist 已知旧值 + scan_files 名单）。
改任何对外指标：只改 metrics.json 的 current，再跑本脚本。

巡检逻辑（v2）：
  1. 从每个被扫描文档里剔除 <!--HIST--> ... <!--/HIST--> 围栏区（历史/坑故事合法保留）；
  2. 再剔除含「历史标记词」的整行（V1/V2/V3/旧/此前/基线/→ 等——按上下文软围栏）；
  3. 对剩余文本，用各指标的上下文正则抓取数值；
  4. 凡抓到的值命中该指标的 stale_blacklist（如 pytest 37/354、chunks 160/118、routes 19/24）→ 判为**漂移**（未围栏的旧口径裸露）；
  5. 额外交叉核对：从 rag_api.py 数路由装饰器，必须等于 current.routes（防文档与实现脱钩）；
  6. 扫描 Git 跟踪文本，拒绝非本机字面 IP 服务地址、私密运行文件，以及与
     本机 `.env.local` 完全相同且未公开在 `.env.example` 的密钥或端点值；
  7. 使用 `--privacy-revision REV` 时，再扫描该 revision 的全部可达历史文本对象，
     防止隐私只存在于中间提交、在当前工作树删除后仍随 push 公开。

退出码：0 = 干净；1 = 发现漂移或路由脱钩；2 = 找不到文件/metrics.json。

用法：
    python verify_docs.py
    python verify_docs.py --json
    python verify_docs.py --privacy-revision HEAD
"""
from __future__ import annotations

import argparse
import io
import ipaddress
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

def _force_utf8_stdio() -> None:
    """Windows 控制台默认 GBK，中文报告会直接抛 UnicodeEncodeError。

    只在**当脚本跑**时改全局流：以前这段在模块顶层，import 一次就把
    ``sys.stdout`` 换掉，pytest 的输出捕获随之失效（``I/O operation on closed
    file``）——本模块因此长期无法被测试，它的正则盲区也就一直没人发现。
    """
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "metrics.json"

# 每个指标的上下文正则（捕获组 = 数值）。刻意收紧，靠"路由/chunks/题/passed"等语境词定位，
# 避免误伤日期/行号/维度等无关数字。
# 2–6 位整数，允许千分位写法（1,430）。捕获组含逗号，比对前统一去掉。
_NUM = r"(\d{1,3}(?:,\d{3})+|\d{2,6})"
# 空格或 URL 编码空格——shields.io 徽章链接里写作 `tests-1430%20passed`。
_SP = r"(?:\s|%20)*"

METRIC_PATTERNS = {
    # v3 补盲区①（2026-07-04）：README「项目数据」表格是 值在前、标签在后 的
    # <strong>N</strong><br><sub>标签</sub> 结构——语境词正则抓不到。镜像
    # langgraph_nodes 已有的单元格 pattern，给 routes/pytest/chunks/R@1 补齐。
    "routes": [r"(\d{1,3})\s*(?:个\s*)?(?:FastAPI\s*)?(?:路由|routes|接口)",
               r"(\d{1,3})</strong>\s*<br>\s*<sub>\s*FastAPI"],
    # v4 补盲区④（2026-09-02）：位数上限写死成 \d{2,3}/\d{2,4}，测试数破千后
    # 门禁**再也看不见 pytest 漂移**——15 处 `1,430` 静静躺在 4 份 canonical 文档里
    # 而扫描全绿。数字还常写成千分位（`1,430`），逗号让任何纯 \d 正则都匹配不上。
    # 修法两条：位数放宽到 6 位，并接受千分位写法（比对前在 scan_doc 里去逗号）。
    "pytest": [r"pytest[^。\n]{0,14}?" + _NUM + r"\b",
               # `1,430 tests passed` / badge 里的 `tests-1430%20passed`
               _NUM + _SP + r"(?:tests?" + _SP + r")?passed",
               # `1,430 全量测试通过` / `1430 个用例`
               _NUM + r"\s*个?\s*(?:全量\s*)?(?:测试|用例)",
               # README「项目数据」单元格实际标签是「测试通过」而不是「pytest」，
               # 只写 pytest 的话这一格常年扫不到。
               _NUM + r"</strong>\s*<br>\s*<sub>\s*(?:pytest|测试)"],
    "chunks": [_NUM + r"\s*(?:个\s*)?chunks?", r"chunks?[^。\n0-9]{0,8}" + _NUM,
               r"collection_records[\"'：:\s]{0,4}" + _NUM,
               _NUM + r"</strong>\s*<br>\s*<sub>\s*RAG\s*chunks"],
    "R@1": [r"R@1[^0-9\n]{0,8}(\d{2,3})",
            r"(\d{2,3})%?</strong>\s*<br>\s*<sub>\s*RAG\s*R@1"],
    "Recall@5": [r"Recall@5[^0-9\n]{0,10}([01]\.\d{2,3})", r"R@5[^0-9\n]{0,10}([01]\.\d{2,3})"],
    "MRR": [r"MRR[^0-9\n]{0,8}([01]\.\d{2,3})"],
    "doctor_ok": [r"doctor[^0-9\n]{0,14}(\d{1,2})\s*OK", r"(\d{1,2})\s*OK\s*[·/、]\s*\d\s*WARN"],
    "eval_set": [r"(\d{1,3})\s*题"],
    # 论文域 R@1(2026-08-18 新增):"论文向 56%" 是主库还含论文那个时期的数字,
    # 架构改成"纯化主库+路由默认关"后论文域已不可达(0%),旧值必须被门禁抓住。
    # 语境词收紧到"论文向/论文域",避免误伤正文里的其它百分数。
    "paper_domain_r1": [r"论文(?:向|域)[^0-9\n]{0,10}(\d{1,3})\s*%",
                        r"paper[_\s]?domain[^0-9\n]{0,10}(\d{1,3})\s*%"],
    "langgraph_nodes": [r"(\d{1,2})[^0-9\n]{0,10}LangGraph\s*节点",
                        r"LangGraph[^0-9\n]{0,10}(\d{1,2})\s*节点",
                        r"(\d{1,2})</strong>\s*<br>\s*<sub>\s*LangGraph"],
}

# 含这些词的整行 → 视为历史语境（软围栏），不参与漂移判定。
# v3 补盲区②（2026-07-04）：裸箭头改为「数字→」转变语境才算历史——原先任何
# 箭头都杀整行，ASCII 架构图（`─→ ChromaDB (N chunks)`）因此常年漏扫。
HISTORY_MARKER = re.compile(
    r"V1\.5|V1|V2|V3|V4|V5|Round|轮次|旧|历史|此前|曾|原为|原本|已降|里程碑|快照|"
    r"当时|当年|之前|早期|打满|从\s*\d|\d\s*%?\s*(?:→|->|—>)|基线|提升[到至]|提到|一路优化|升级为|收口审计"
)
HIST_FENCE = re.compile(r"<!--\s*/?HIST\s*-->", re.IGNORECASE)
HIST_REGION = re.compile(r"<!--\s*HIST\s*-->.*?<!--\s*/HIST\s*-->", re.IGNORECASE | re.DOTALL)

_TEXT_SUFFIXES = {
    ".cfg", ".command", ".css", ".csv", ".html", ".ini", ".js", ".json",
    ".md", ".mjs", ".ps1", ".py", ".sh", ".toml", ".ts", ".tsv", ".txt",
    ".xml", ".yaml", ".yml", ".tmpl",
}
_TEXT_FILENAMES = {".env.example", ".gitignore", ".gitattributes"}
_PRIVATE_PATHS = {
    "applications.md",
    "daily_log.md",
    "gap_store.json",
    "growth_journal.md",
    "interview_story_bank.md",
    "jd_candidates.md",
    "memory.json",
    "target_rules.local.md",
    "user_profile.md",
    "deployment.md",
    "integrations/openclaw/lab-fixtures/env.local",
    "docs/rag_eval/judge_panel_artifacts.json",
}
_PRIVATE_PATH_PREFIXES = (
    ".offerclaw/",
    "_gpt_exports/",
    "application_jds/",
    "chroma_db/",
    "chroma_db_test/",
    "daily_attachments/",
    "data/archive/",
    "docs/rag_eval/",
    "docs/screenshots/",
    "knowledge_base/",
    "logs/",
    "memory/",
    "plans/",
    "profiles/private_",
    "resume_drafts/",
    "summaries/",
)
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|BASE_URL|API_BASE|ENDPOINT|GATEWAY)", re.IGNORECASE
)
_IP_ENDPOINT = re.compile(
    r"(?<![\w.])(?P<scheme>https?://)?"
    r"(?P<host>(?:\d{1,3}\.){3}\d{1,3})"
    r"(?::(?P<port>\d{2,5}))?(?![\d.])",
    re.IGNORECASE,
)
_LOCAL_IDENTITY = re.compile(
    r"(?i)(?:[A-Z]:\\Users\\(?!<user>\\)[A-Za-z0-9._ -]+\\|"
    r"/mnt/[a-z]/Users/(?!<user>/)[A-Za-z0-9._-]+/|"
    r"/Users/(?!<user>/)[A-Za-z0-9._-]+/|"
    r"/home/(?!user/|ubuntu/|runner/)[A-Za-z0-9._-]+/|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.local\b)"
)
_PRIVATE_WORKSPACE_URL = re.compile(
    r"https?://(?:my|[a-z0-9]{8,})\.feishu\.cn/wiki/[A-Za-z0-9]+",
    re.IGNORECASE,
)
_CONFIG_ENDPOINT = re.compile(
    r"(?i)\b(?:[A-Z0-9_]*(?:BASE_URL|API_BASE|ENDPOINT|GATEWAY_URL)|"
    r"base_url|api_base|endpoint_url)\b[^\n]{0,80}?"
    r"(?P<url>https?://[^\s\"'`\\]+)"
)
_PUBLIC_ENDPOINT_HOSTS = {
    "api.deepseek.com",
    "api.openai.com",
    "dashscope.aliyuncs.com",
    "hf-mirror.com",
    "localhost",
    "open.bigmodel.cn",
}
_PRIVATE_OPERATIONAL_MARKERS = (
    "offerclaw-" + "proxy",
    "sk-" + "***",
    "当前本机 OpenAI 兼容" + "代理可用",
    "已完成扫码和" + "本人私聊绑定",
    "Key 已迁移到 Windows 用户级 " + "DPAPI",
)
_SECRET_LENGTH_METADATA = re.compile(
    r"(?i)(?:api[_ -]?key|access[_ -]?token|secret|密钥|令牌)"
    r"[^\n]{0,64}\blen\s*=\s*(?:\d+|N)\b"
)
_PRIVATE_RUNTIME_TELEMETRY = re.compile(
    "(?:"
    + "投递" + r"共\s*\d+\s*条"
    + "|(?:真实" + r"索引|密钥扫描|本地直返|转发开销)"
    + r"[^\n]{0,120}(?:\d+\s*(?:个|条|ms|秒|chunks)|P95|成功)"
    + ")",
    re.IGNORECASE,
)
_PROFILE_PRIVATE_FILES = (
    "user_profile.md",
    "profiles/private_user.json",
)
_PROFILE_PRIVATE_LABEL = re.compile(
    r"(?i)[\"']?(?P<label>persona_id|desc|姓名|学校|院校|专业|所在地|"
    r"可接受工作地域|可接受地域|方向优先级|明确不做|工作性质偏好|"
    r"毕业时间|期望薪资|熟练技能|会用技能|项目数量|实习数量|英语自评|"
    r"邮箱|电话|手机号)[\"']?\s*[:：]"
)
_PROFILE_FACT_MIN_LENGTH = {
    "persona_id": 6,
    "desc": 8,
    "姓名": 2,
    "学校": 4,
    "院校": 4,
    "专业": 4,
    "所在地": 4,
    "毕业时间": 4,
    "期望薪资": 4,
    "邮箱": 5,
    "电话": 7,
    "手机号": 7,
    "可接受工作地域": 12,
    "可接受地域": 12,
    "方向优先级": 12,
    "明确不做": 12,
    "熟练技能": 12,
    "会用技能": 12,
}


def _is_text_path(rel: str) -> bool:
    path = Path(rel)
    return path.suffix.lower() in _TEXT_SUFFIXES or path.name in _TEXT_FILENAMES


def _tracked_text_paths() -> list[tuple[str, Path]]:
    """Return Git-tracked text files from the working tree."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    paths = []
    for raw in proc.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if not raw:
            continue
        path = ROOT / raw
        if path.is_file() and _is_text_path(raw):
            paths.append((raw.replace("\\", "/"), path))
    return paths


def _is_private_path(rel: str) -> bool:
    """Reject local runtime/profile material even when it is force-added to Git."""
    normalized = rel.replace("\\", "/")
    name = Path(normalized).name
    if normalized in {"docs/rag_eval/README.md", "knowledge_base/README.md"}:
        return False
    if normalized in _PRIVATE_PATHS or normalized.startswith(_PRIVATE_PATH_PREFIXES):
        return True
    if name == ".env.example":
        return False
    if name == ".env" or name.startswith(".env.") or name.endswith((".key", ".pem")):
        return True
    return normalized.startswith("profiles/") and normalized.endswith("_local.json")


def _literal_ip_endpoint_lines(text: str) -> list[int]:
    """Find non-loopback literal IP endpoints without returning the address itself."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in _IP_ENDPOINT.finditer(line):
            if not match.group("scheme") and not match.group("port"):
                continue
            try:
                address = ipaddress.ip_address(match.group("host"))
            except ValueError:
                continue
            if address.is_loopback or address.is_unspecified:
                continue
            hits.append(lineno)
            break
    return hits


def _unapproved_endpoint_lines(text: str) -> list[int]:
    """Find configured service endpoints that are neither examples nor known public APIs."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in _CONFIG_ENDPOINT.finditer(line):
            try:
                host = (urlsplit(match.group("url")).hostname or "").lower()
            except ValueError:
                host = ""
            if (
                host in _PUBLIC_ENDPOINT_HOSTS
                or host in {"127.0.0.1", "0.0.0.0"}
                or host.endswith((".example", ".example.com", ".invalid"))
            ):
                continue
            hits.append(lineno)
            break
    return hits


def _private_operational_metadata_lines(text: str) -> list[int]:
    """Find publication-only metadata that fingerprints a private deployment."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if (
            any(marker in line for marker in _PRIVATE_OPERATIONAL_MARKERS)
            or _SECRET_LENGTH_METADATA.search(line)
            or _PRIVATE_RUNTIME_TELEMETRY.search(line)
        ):
            hits.append(lineno)
    return hits


def _local_profile_lines() -> list[tuple[str, str]]:
    """Load identifying profile payloads without exposing their values."""
    values = []
    for rel in _PROFILE_PRIVATE_FILES:
        path = ROOT / rel
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            value = line.strip().strip(",")
            match = _PROFILE_PRIVATE_LABEL.search(value)
            if not match:
                continue
            payload = value[match.end():].strip().strip(",").strip("\"'")
            label = match.group("label").lower()
            minimum = _PROFILE_FACT_MIN_LENGTH.get(label)
            if minimum is None:
                continue
            if len(_normalize_profile_fact(payload)) >= minimum:
                values.append((f"{rel}:{label}", payload))
    return values


def _normalize_profile_fact(value: str) -> str:
    """Ignore presentation differences while retaining the complete fact."""
    return re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "", value).lower()


def _local_sensitive_values() -> list[tuple[str, str]]:
    """Load private local values while excluding endpoints intentionally in the public template."""
    env_path = ROOT / ".env.local"
    if not env_path.is_file():
        return []
    template = (ROOT / ".env.example").read_text(encoding="utf-8", errors="ignore")
    values = []
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if not match or not _SENSITIVE_ENV_NAME.search(match.group(1)):
            continue
        name, value = match.group(1), match.group(2).strip().strip("\"'")
        if len(value) < 8 or value in template:
            continue
        values.append((name, value))
    return values


def _scan_privacy_text(
    rel: str,
    text: str,
    local_values: list[tuple[str, str]],
    profile_lines: list[tuple[str, str]],
    *,
    source: str | None = None,
) -> list[dict]:
    """Scan one text payload without echoing any detected private value."""
    violations = []
    common = {"path": rel}
    if source:
        common["source"] = source
    for lineno in _literal_ip_endpoint_lines(text):
        violations.append({"kind": "literal_ip_endpoint", "line": lineno, **common})
    for lineno in _unapproved_endpoint_lines(text):
        violations.append({"kind": "unapproved_service_endpoint", "line": lineno, **common})
    for lineno in _private_operational_metadata_lines(text):
        violations.append({"kind": "private_operational_metadata", "line": lineno, **common})
    for lineno, line in enumerate(text.splitlines(), 1):
        if _LOCAL_IDENTITY.search(line):
            violations.append({"kind": "local_identity", "line": lineno, **common})
        if _PRIVATE_WORKSPACE_URL.search(line):
            violations.append({"kind": "private_workspace_url", "line": lineno, **common})
    for name, value in local_values:
        for lineno, line in enumerate(text.splitlines(), 1):
            if value in line:
                violations.append({
                    "kind": "local_env_value",
                    "line": lineno,
                    "variable": name,
                    **common,
                })
    for profile_source, value in profile_lines:
        normalized_value = _normalize_profile_fact(value)
        if len(normalized_value) < 8:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if normalized_value in _normalize_profile_fact(line):
                violations.append({
                    "kind": "local_profile_value",
                    "line": lineno,
                    "variable": profile_source,
                    **common,
                })
    return violations


def _dedupe_privacy_violations(violations: list[dict]) -> list[dict]:
    """Prefer working-tree locations when the same finding is also present at HEAD."""
    seen = set()
    unique = []
    for hit in violations:
        key = (hit["kind"], hit["path"], hit.get("line"), hit.get("variable"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def scan_repository_privacy() -> list[dict]:
    """Scan publishable working-tree text without exposing detected values."""
    violations = []
    local_values = _local_sensitive_values()
    profile_lines = _local_profile_lines()
    for rel, path in _tracked_text_paths():
        if _is_private_path(rel):
            violations.append({"kind": "private_path", "path": rel, "line": None})
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(_scan_privacy_text(rel, text, local_values, profile_lines))
    return _dedupe_privacy_violations(violations)


def scan_git_history_privacy(revisions: list[str]) -> list[dict]:
    """Scan unique text blobs reachable from revisions, including deleted files."""
    if not revisions:
        return []
    failure = [{"kind": "git_history_scan_error", "path": "<git-history>", "line": None}]
    try:
        proc = subprocess.run(
            ["git", "-c", "core.quotePath=false", "rev-list", "--objects", *revisions],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return failure

    objects: dict[str, str] = {}
    historical_paths = set()
    for raw_line in proc.stdout.decode("utf-8", errors="surrogateescape").splitlines():
        oid, separator, rel = raw_line.partition(" ")
        if not separator or not re.fullmatch(r"[0-9a-f]{40,64}", oid):
            continue
        rel = rel.replace("\\", "/")
        historical_paths.add(rel)
        if _is_text_path(rel):
            objects.setdefault(oid, rel)

    violations = [
        {"kind": "private_path", "path": rel, "line": None, "source": "git-history"}
        for rel in sorted(historical_paths)
        if _is_private_path(rel)
    ]
    if not objects:
        return _dedupe_privacy_violations(violations)

    local_values = _local_sensitive_values()
    profile_lines = _local_profile_lines()
    child = None
    try:
        child = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if child.stdin is None or child.stdout is None:
            raise OSError("git cat-file pipes are unavailable")
        for oid, rel in objects.items():
            child.stdin.write(f"{oid}\n".encode("ascii"))
            child.stdin.flush()
            header = child.stdout.readline().decode("ascii", errors="replace").strip().split()
            if len(header) != 3 or header[1] != "blob" or not header[2].isdigit():
                raise ValueError("unexpected git cat-file header")
            payload = child.stdout.read(int(header[2]))
            if child.stdout.read(1) != b"\n":
                raise ValueError("truncated git cat-file payload")
            if b"\0" in payload:
                continue
            text = payload.decode("utf-8", errors="ignore")
            violations.extend(_scan_privacy_text(
                rel,
                text,
                local_values,
                profile_lines,
                source=f"git:{oid[:12]}",
            ))
        child.stdin.close()
        if child.wait(timeout=30) != 0:
            return failure
    except (OSError, BrokenPipeError, subprocess.SubprocessError, ValueError):
        if child is not None:
            child.kill()
            child.wait()
        return failure
    return _dedupe_privacy_violations(violations)


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _strip_history(text: str) -> str:
    """剔除 HIST 围栏区，再把含历史标记词的整行替换为空行。

    v3 修正（2026-07-04）：
    ① 围栏区替换保留换行数——原先多行围栏被折叠成一格，其后所有行号漂移；
    ② 盲区③：显式打过 HIST 围栏的行视为作者已人工分界（如「当前口径 X
       （旧索引曾测得 <!--HIST-->Y<!--/HIST-->）」）——只剔除围栏段，行的
       其余部分照常参与扫描；原先这类行会被"旧/曾"等标记词整行误杀，
       行内的当前口径数值失去门禁保护。"""
    fenced_lines = {
        i for i, ln in enumerate(text.splitlines()) if HIST_FENCE.search(ln)
    }
    text = HIST_REGION.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    out = []
    for i, ln in enumerate(text.splitlines()):
        if i in fenced_lines:
            out.append(ln)
        else:
            out.append("" if HISTORY_MARKER.search(ln) else ln)
    return "\n".join(out)


def scan_file(path: Path, blacklist: dict) -> dict:
    """返回 {metric: [(value, lineno, line), ...]} 只含命中黑名单的漂移。"""
    if not path.exists():
        return {"_error": f"missing: {path}"}
    raw = path.read_text(encoding="utf-8", errors="ignore")
    stripped = _strip_history(raw)
    lines = stripped.splitlines()
    drifts: dict = {}
    for metric, pats in METRIC_PATTERNS.items():
        bad = blacklist.get(metric, [])
        if not bad:
            continue
        hits = []
        for i, ln in enumerate(lines, 1):
            for pat in pats:
                for m in re.finditer(pat, ln, re.IGNORECASE):
                    val = m.group(1).replace(",", "")   # 1,430 与 1430 同一个数
                    if val in bad:
                        hits.append((val, i, ln.strip()[:120]))
        if hits:
            drifts[metric] = hits
    return drifts


def count_routes() -> int | None:
    p = ROOT / "rag_api.py"
    if not p.exists():
        return None
    return len(re.findall(r"^@(?:app|router)\.", p.read_text(encoding="utf-8"), re.MULTILINE))


def count_chunks() -> int | None:
    """当前激活 collection 的真实 chunks 数（与 doctor.py 同口径：
    rag_tools.get_collection_name + chroma_db/）。

    背景：2026-07-04 检验发现 metrics.json 写 3299 而真实索引是 2530——
    路由有代码交叉核对、chunks 却没有，单一事实源对"索引重建"这类漂移
    是盲的。本函数补上这道核对。chroma 不可用/索引缺失时返回 None
    （跳过，不误伤离线或未建库环境——与 count_routes 的 fail-soft 一致）。"""
    db = ROOT / "chroma_db"
    if not db.is_dir():
        return None
    try:
        from rag_tools import get_collection_name, get_collection_sqlite_stats

        name = get_collection_name()
        count, _ = get_collection_sqlite_stats(str(db), name)
        return count
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="docs metric drift gate (v2)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--privacy-revision",
        action="append",
        default=[],
        metavar="REV",
        help="also scan all Git text objects reachable from REV (repeatable)",
    )
    args = ap.parse_args()

    if not METRICS_PATH.exists():
        print(f"missing metrics.json at {METRICS_PATH}", file=sys.stderr)
        return 2

    m = load_metrics()
    current = m["current"]
    blacklist = m["stale_blacklist"]
    files = m["scan_files"]["canonical"] + m["scan_files"]["extended"]

    report: dict = {"drifts": {}, "route_crosscheck": None,
                    "chunks_crosscheck": None, "missing": [],
                    "privacy_violations": []}
    exit_code = 0

    for rel in files:
        d = scan_file(ROOT / rel, blacklist)
        if "_error" in d:
            report["missing"].append(rel)
            continue
        if d:
            report["drifts"][rel] = d
            exit_code = 1

    # 路由数与代码交叉核对
    actual = count_routes()
    if actual is not None:
        expected = int(current["routes"])
        report["route_crosscheck"] = {"code": actual, "metrics_json": expected,
                                      "ok": actual == expected}
        if actual != expected:
            exit_code = 1

    # chunks 与真实索引交叉核对（激活 collection 实测 vs metrics.json）
    actual_chunks = count_chunks()
    if actual_chunks is not None:
        expected_chunks = int(current["chunks"])
        report["chunks_crosscheck"] = {"index": actual_chunks,
                                       "metrics_json": expected_chunks,
                                       "ok": actual_chunks == expected_chunks}
        if actual_chunks != expected_chunks:
            exit_code = 1

    privacy_violations = scan_repository_privacy()
    privacy_violations.extend(scan_git_history_privacy(args.privacy_revision))
    report["privacy_violations"] = _dedupe_privacy_violations(privacy_violations)
    if report["privacy_violations"]:
        exit_code = 1

    if args.json:
        print(json.dumps({"exit": exit_code, "report": report, "current": current},
                         ensure_ascii=False, indent=2))
        return exit_code

    # 文本报告
    print("# verify_docs v2 — 指标漂移门禁\n")
    print(f"单一事实源 metrics.json · current: routes={current['routes']} pytest={current['pytest']} "
          f"chunks={current['chunks']} R@1={current['R@1']} Recall@5={current['Recall@5']} "
          f"MRR={current['MRR']} doctor={current['doctor_ok']} eval={current['eval_set']}题\n")
    cc = report["route_crosscheck"]
    if cc:
        mark = "✅" if cc["ok"] else "❌"
        print(f"{mark} 路由交叉核对：rag_api.py 装饰器 {cc['code']} vs metrics.json {cc['metrics_json']}")
    kc = report["chunks_crosscheck"]
    if kc:
        mark = "✅" if kc["ok"] else "❌"
        print(f"{mark} chunks 交叉核对：激活 collection 实测 {kc['index']} vs metrics.json {kc['metrics_json']}")
    else:
        print("⚠️  chunks 交叉核对跳过（chroma 不可用或索引缺失）")
    if report["missing"]:
        print(f"⚠️  未找到（跳过）：{report['missing']}")
    if report["privacy_violations"]:
        print("\n## ❌ 仓库隐私门禁发现待处理项\n")
        for hit in report["privacy_violations"]:
            location = hit["path"]
            if hit.get("line"):
                location += f":{hit['line']}"
            if hit.get("source"):
                location += f" ({hit['source']})"
            detail = hit["kind"]
            if hit.get("variable"):
                detail += f" ({hit['variable']})"
            print(f"  - {location} [{detail}]")
    else:
        print("\n## ✅ 仓库隐私门禁：未发现私密路径、字面 IP 服务地址或本机凭据值。")
    if not report["drifts"]:
        print("\n## ✅ 所有被扫描文档：0 处未围栏的旧口径裸露。")
    else:
        n = sum(len(v) for f in report["drifts"].values() for v in f.values())
        print(f"\n## ❌ 发现 {n} 处指标漂移（未被 HIST 围栏/历史语境覆盖的旧值）\n")
        for rel, metrics in report["drifts"].items():
            print(f"### {rel}")
            for metric, hits in metrics.items():
                exp = current.get(metric, "?")
                for val, lineno, line in hits:
                    print(f"  - L{lineno} [{metric}] 旧值 `{val}`（当前应为 `{exp}`）: {line}")
            print()
    return exit_code


if __name__ == "__main__":
    _force_utf8_stdio()
    sys.exit(main())
