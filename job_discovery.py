# -*- coding: utf-8 -*-
"""
OfferClaw · 主动求职 / JD 抽取助手 (job_discovery.py)

V2 阶段四职责：
    把"用户给的 JD 原文（粘贴或 URL 抓取）"快速整理成
    结构化 JD（公司 / 岗位 / 地点 / 职责 / 要求 / 工作性质），
    供 /api/discover 直接消费。

设计取舍：
    - V1：用户给 JD 原文（最快、最稳）
    - V2：用户给 URL，本脚本用 requests + 简单 readability 抓正文（不在本阶段实现，留接口）
    - 不做爬虫遍历招聘网站（违反 V2 报告 §七 任务四"半自动 不自动"）
    - 抽取用规则 + 关键字匹配优先；若用户传入 use_llm=True 才走 LLM（v1 默认关闭，秒级）
"""

import re
from typing import Dict, List, Optional


_VALUE_SEP = r"\s*(?:[：:]\s*|\n+\s*)"
_LOC_HINT = re.compile(
    rf"(?:工作地点|工作城市|办公地点|办公地址|职位地点|Location|城市|地点){_VALUE_SEP}([^\n，,；;|｜]+)",
    re.IGNORECASE,
)
_TYPE_HINT = re.compile(
    rf"(?:岗位性质|工作性质|招聘类型|用工类型|类型){_VALUE_SEP}([^\n，,；;|｜]+)",
    re.IGNORECASE,
)
_TITLE_HINT = re.compile(
    rf"(?:岗位名称|职位名称|职位|岗位){_VALUE_SEP}([^\n，,；;]+)",
    re.IGNORECASE,
)
_COMPANY_HINT = re.compile(
    rf"(?:公司名称|招聘公司|Company|公司){_VALUE_SEP}([^\n，,；;]+)",
    re.IGNORECASE,
)
_TITLE_DETAIL_HINT = re.compile(r"(?:职位详情|岗位详情)\s*\n+\s*([^\n]{4,100})")
_COMPANY_SECTION_HINT = re.compile(r"公司信息\s*\n+\s*([^\n]{2,80})")

# Feishu Jobs / BOSS / 拉勾 等段落式 JD 的兜底：直接扫描城市名 / 工作性质
_CITIES_FALLBACK = ["北京","上海","深圳","广州","杭州","成都","南京","武汉","西安","苏州","天津","重庆","厦门","合肥","宁波","青岛"]
_JOB_TYPE_FALLBACK = ["实习","全职","校招","社招","兼职","劳务"]


def _scan_first_city(text: str) -> str:
    """Feishu Jobs 页眉常是「城市｜实习｜部门」或「上海社招实习」，纯文字段无键值。"""
    head = text[:300]
    for c in _CITIES_FALLBACK:
        if c in head:
            return c
    return ""


def _scan_first_job_type(text: str) -> str:
    head = text[:300]
    for t in _JOB_TYPE_FALLBACK:
        if t in head:
            return t
    return ""


def _guess_title_first_line(text: str) -> str:
    """没有「岗位名称：」键值时，取第一非空行作为标题候选（限长 60，去括号噪声）。"""
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if not s or len(s) < 4:
            continue
        if s in {"AI人才专项", "校招职位", "实习生职位", "招聘动态", "关于我们", "关于搜狐",
                 "职位列表", "职位详情", "公司信息", "职位描述", "岗位职责", "任职要求"}:
            continue
        if s.startswith("首页/") or s.startswith("首页 / "):
            continue
        # 跳过明显非标题行（纯城市横线分隔头）
        if all(p.strip() in _CITIES_FALLBACK + _JOB_TYPE_FALLBACK
               for p in re.split(r"[｜|·\-\s]+", s) if p.strip()):
            continue
        return s[:60]
    return ""


def _first_match(pat: re.Pattern, text: str) -> str:
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def extract_jd(raw: str) -> Dict[str, object]:
    """从 JDAnalysis 生成旧 API 所需视图；不再维护独立技能白名单。"""
    raw = raw.strip()
    from jd_parser import analyze_jd
    analysis = analyze_jd(raw)
    hits = analysis.keyword_names()
    duties = "\n".join(analysis.responsibilities)
    requirements = "\n".join(
        item.text for item in analysis.requirements if item.kind != "responsibility"
    )

    career_domain = None
    role_family = None
    language_requirement: Dict[str, object] = {}
    try:
        from career_domains import detect_role_family, resolve_domain
        from job_requirements import analyze_programming_languages
        career_domain = resolve_domain(raw)
        role_family = detect_role_family(raw)
        language_requirement = analyze_programming_languages(raw).as_dict()
    except Exception:
        pass

    return {
        "company": analysis.company,
        "title": analysis.title,
        "location": analysis.location,
        "job_type": analysis.job_type,
        "skills_detected": hits,
        "duties": duties,
        "requirements": requirements,
        "career_domain": career_domain,
        "role_family": role_family,
        "programming_language_requirement": language_requirement,
        "jd_analysis": analysis.model_dump(mode="json"),
        "raw_chars": len(raw),
    }


def _strip_html(html: str) -> str:
    """去掉 script/style/标签，保留可读正文。"""
    import re as _re
    clean = _re.sub(r"<script.*?</script>", "", html, flags=_re.S | _re.I)
    clean = _re.sub(r"<style.*?</style>", "", clean, flags=_re.S | _re.I)
    clean = _re.sub(r"<[^>]+>", "\n", clean)
    clean = _re.sub(r"[ \t]+", " ", clean)
    clean = _re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean


# ---- 登录 / 安全验证墙识别 --------------------------------------------------
# 像 BOSS 直聘（zhipin）这类站点，对未登录 / 无头客户端会把 job_detail 302 跳到
# 安全验证页（/web/passport/zp/security.html，Geetest 行为验证）。这类墙：
#   1) 必须靠用户已登录的浏览器会话或人机验证才能过——OfferClaw 不做验证码绕过；
#   2) 抓回来的「正文」其实是验证页 HTML，绝不能当 JD 喂给匹配。
# 故命中即明确告知用户，并引导手动粘贴（最稳路径）。
_WALL_URL_MARKERS = ("/security.html", "/passport", "/login", "/sec-", "captcha", "verify.")
_WALL_TEXT_MARKERS = ("安全验证", "完成验证", "滑动验证", "拖动滑块", "人机验证",
                      "访问异常", "请先登录", "登录后查看", "captcha", "robot check")

_WALL_HINT = (
    "该招聘站点需要登录或安全验证（如 BOSS 直聘 zhipin），无法自动抓取正文。\n"
    "请在已登录的浏览器里打开该 JD，全选复制正文后粘贴到左侧文本框，再点「开始匹配」。"
)


def _wall_reason(final_url: str, text: str) -> str:
    """返回非空字符串表示命中登录/安全验证墙；否则空串。"""
    u = (final_url or "").lower()
    hit_url = next((m for m in _WALL_URL_MARKERS if m in u), "")
    if hit_url:
        return f"页面跳转到登录/安全验证（{final_url}）"
    t = (text or "").lower()
    hit_txt = next((m for m in _WALL_TEXT_MARKERS if m.lower() in t), "")
    if hit_txt:
        return f"页面要求登录/安全验证（检出『{hit_txt}』）"
    return ""


def _fetch_with_playwright(url: str, timeout_ms: int = 30000) -> tuple[str, str]:
    """用 Playwright 渲染页面，返回 (正文, 最终 URL)。优先本机 Chrome，失败再用 bundled。"""
    from playwright.sync_api import sync_playwright
    # 通用 headless 兼容：去掉 AutomationControlled、隐藏 webdriver 标志。
    # 这是面向「正常 SPA 招聘页」的兼容性处理，并不能绕过 zhipin 这类行为验证墙。
    launch_args = ["--disable-blink-features=AutomationControlled"]
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome", headless=True, args=launch_args)
        except Exception:
            browser = pw.chromium.launch(headless=True, args=launch_args)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="zh-CN",
            viewport={"width": 1440, "height": 900},
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
        page = ctx.new_page()
        # domcontentloaded 比 networkidle 更稳（反爬页的常驻轮询会让 networkidle 永不静止/超时）。
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception:
            pass  # 即便超时也尝试读取已加载内容
        page.wait_for_timeout(2500)
        # 尝试等待岗位正文容器出现（命中其一即可）
        for selector in ["[class*='job']", "[class*='position']", "[class*='jd']",
                         "main", "article", "#content", ".detail"]:
            try:
                page.wait_for_selector(selector, timeout=2500)
                break
            except Exception:
                continue
        final_url = page.url
        text = page.inner_text("body")
        browser.close()
    return text.strip(), final_url


def fetch_url(url: str, timeout: int = 20) -> str:
    """从 URL 抓取 JD 正文。
    策略：
    1. 先用轻量 requests 快速抓取（静态页直接返回）
    2. 任一阶段命中「登录/安全验证墙」→ 立即给出明确提示（不把验证页当 JD，也不空跑 playwright）
    3. 若正文 < 300 字（典型 SPA）→ 启动 Playwright 无头浏览器渲染
    4. 仍失败 → 抛出清晰、分类的错误
    """
    import requests

    final_url = url
    clean = ""
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        r.raise_for_status()
        final_url = str(r.url)
        # 只检查用户可见正文。招聘 SPA 的脚本依赖中经常出现 captcha/login
        # 字符串，但这并不代表页面真的展示了验证墙；直接扫描原始 HTML 会误判。
        clean = _strip_html(r.text)
        reason = _wall_reason(final_url, clean)
        if reason:
            raise ValueError(f"{_WALL_HINT}\n（原因：{reason}）")
    except ValueError:
        raise
    except Exception:
        clean = ""  # requests 失败（非墙）也走 playwright

    # 静态页内容充足，直接返回
    if len(clean) >= 300:
        return clean

    # SPA / 动态页：启动无头浏览器
    try:
        text, final_url = _fetch_with_playwright(url, timeout_ms=timeout * 1500)
    except Exception as pw_err:
        raise ValueError(
            "无法自动抓取该页面内容（动态渲染失败、超时或网络问题）。\n"
            f"错误详情：{pw_err}\n\n"
            "建议：在浏览器手动打开页面，全选复制 JD 文字后粘贴到左侧文本框。"
        )

    # 渲染后再判墙（zhipin 会把 job_detail 跳到 security.html）
    reason = _wall_reason(final_url, text)
    if reason:
        raise ValueError(f"{_WALL_HINT}\n（原因：{reason}）")
    if len(text) < 50:
        raise ValueError(
            "页面正文为空——多半被站点反爬拦截或需要登录。\n" + _WALL_HINT)
    return text


def _classify_source(url: str, raw: str) -> tuple[str, str]:
    """根据 URL 域名判断来源类型与可信度（A/B/C，与 source_policy.md 对齐）。

    Returns:
        (source_type, source_credibility)
        source_type ∈ {"official", "platform", "manual_paste", "unknown"}
        source_credibility ∈ {"A", "B", "C"}
    """
    if not url and raw:
        return "manual_paste", "B"
    u = (url or "").lower()
    # A 级：公司官网 / 官方招聘门户
    official_domains = ("jobs.bytedance.com", "talent.alibaba.com", "careers.tencent.com",
                        "campus.baidu.com", "join.jd.com",
                        "career.huawei.com", "jobs.netease.com", "jobs.didiglobal.com",
                        "campus.meituan.com", "talent.kuaishou.com", "careers.bilibili.com",
                        "app.mokahr.com/", ".feishu.cn/", "/careers", "/jobs/")
    for d in official_domains:
        if d in u:
            return "official", "A"
    # B 级：第三方招聘平台
    platform_domains = ("zhipin.com", "lagou.com", "linkedin.com", "liepin.com",
                        "51job.com", "zhaopin.com", "maimai.cn", "greenhouse.io",
                        "lever.co", "ashbyhq.com", "workable.com")
    for d in platform_domains:
        if d in u:
            return "platform", "B"
    return "unknown", "C"


def discover(raw: str = "", url: str = "") -> Dict[str, object]:
    """统一入口：raw 优先，否则 fetch_url(url)。返回结构化 JD。

    输出新增字段：
        source_type: official / platform / manual_paste / unknown
        source_credibility: A (官网) / B (招聘平台) / C (未知/匿名)
        notice: 不登录、不批量抓取、不自动投递 — 见 docs/ethical_use.md
    """
    if not raw and not url:
        raise ValueError("raw 与 url 至少给一个")
    text = raw or fetch_url(url)
    parsed = extract_jd(text)
    parsed["source_url"] = url
    parsed["jd_text"] = text  # 透传给前端，方便直接喂给 /api/match
    src_type, src_cred = _classify_source(url, raw)
    parsed["source_type"] = src_type
    parsed["source_credibility"] = src_cred
    parsed["notice"] = "OfferClaw 仅做半自动抽取：不登录、不批量、不自动投递。最终录入 applications.md 需用户确认。"
    return parsed


if __name__ == "__main__":
    import json, sys
    sample = sys.stdin.read() if not sys.stdin.isatty() else "岗位名称：LLM 实习生\n公司：测试\n地点：上海\n要求：Python / RAG / FastAPI"
    print(json.dumps(discover(raw=sample), ensure_ascii=False, indent=2))


# =====================================================================
# Phase 4：JD Discovery 增强 —— query_builder + ranker
# =====================================================================

def build_search_queries(profile: dict | None = None, max_queries: int = 6) -> List[str]:
    """根据 profile 生成搜索关键词组合（不去爬，仅供用户复制到搜索引擎）。"""
    if profile is None:
        from profile_loader import load_profile
        profile = load_profile()
    cities = profile.get("可接受地域") or ["上海", "南京"]
    directions = profile.get("方向优先级") or ["AI 应用开发"]
    skills = (profile.get("熟练技能") or [])[:3]
    work_type = "实习" if "实习" in (profile.get("求职性质") or "实习") else "校招"

    queries: List[str] = []
    for d in directions[:3]:
        for c in cities[:2]:
            base = f"{d} {work_type} {c}"
            if skills:
                base += " " + " ".join(skills)
            queries.append(base.strip())
            if len(queries) >= max_queries:
                return queries
    return queries


def rank_candidates(candidates: List[Dict], profile: dict | None = None) -> List[Dict]:
    """对候选 JD 列表调用 match_job，返回带 status/direction/score 的排序结果。

    candidates: [{"title": str, "jd_text": str}, ...]
    返回排序后的 [{...原字段, "status", "direction", "score", "reason"}]
    """
    from jd_parser import analyze_jd
    from match_job import run_match, format_report
    if profile is None:
        from profile_loader import load_profile
        profile = load_profile()

    from domain_status import MatchStatusCode, match_status_code
    _ORDER = {MatchStatusCode.SUITABLE: 3, MatchStatusCode.STRETCH: 2,
              MatchStatusCode.UNKNOWN: 1, MatchStatusCode.NOT_RECOMMENDED: 0}

    out = []
    for c in candidates:
        jd_text = c.get("jd_text", "")
        title = c.get("title", "未命名")
        try:
            jd_analysis = c.get("jd_analysis")
            if not isinstance(jd_analysis, dict) or not jd_analysis:
                jd_analysis = analyze_jd(jd_text).model_dump(mode="json")
            r = run_match(
                profile, jd_text, jd_title=title, jd_analysis=jd_analysis)
            status = r.conclusion or ""
            direction = r.direction or ""
            gaps = sum(len(v or []) for v in (r.gap_list or {}).values())
            reason = (r.conclusion_reason or format_report(r) or "")[:140]
        except Exception as e:
            status, direction, gaps, reason = "error", "", 99, str(e)
        status_code = match_status_code(status)
        out.append({
            **c,
            "status": status,
            "status_code": status_code.value,
            "direction": direction,
            "gap_count": gaps,
            "score": _ORDER.get(status_code, 0),
            "reason": reason,
        })
    out.sort(key=lambda x: (-x["score"], x.get("gap_count", 99)))
    return out
