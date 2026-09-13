"""normalize_applications.py — applications.md 校验与去重

规则：
1. 必填字段：日期 / 公司 / 岗位 / 来源 / 地点 / 匹配结论 / 当前状态 / 下一步动作
2. 状态必须在合法枚举中
3. 同一 (公司, 岗位) 不能重复
4. 日期需要 YYYY-MM-DD 格式

退出码：0 = 通过；1 = 有 violations；2 = 找不到文件。

用法：
    python normalize_applications.py
    python normalize_applications.py --json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
APP_FILE = ROOT / "applications.md"

VALID_STATES = {
    "已评估", "准备投递", "不投递", "已投递",
    "等待反馈", "面试中", "已 Offer", "已拒绝", "主动放弃",
}

REQUIRED_FIELDS = ["日期", "公司", "岗位", "来源", "地点",
                   "匹配结论", "当前状态", "下一步动作"]


def parse_table(text: str) -> list[dict]:
    """提取"投递清单"段下的 Markdown 表格行"""
    # 找到清单段落
    m = re.search(r"##\s*投递清单\s*\n(.*?)(?=\n##\s|\Z)", text, re.S)
    if not m:
        return []
    block = m.group(1)
    lines = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []
    # 第一行 header，第二行分隔
    header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(header_cells):
            continue
        rows.append(dict(zip(header_cells, cells)))
    return rows


def normalize_state(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def validate(rows: list[dict]) -> list[dict]:
    issues: list[dict] = []
    seen: dict[tuple, int] = {}
    for idx, row in enumerate(rows, start=1):
        company = row.get("公司", "") or ""
        # [DEMO] 行 → 标 INFO 跳过其他校验（仅证明字段约束可演示，不计 warn）
        if company.lstrip().startswith("[DEMO]"):
            issues.append({"row": idx, "level": "info",
                           "msg": f"DEMO row (示例数据 · 非真实投递): {company}"})
            continue
        # 跳过仍是 demo 占位行（旧式 *（待填...)*）
        if any("（待填" in (row.get(f) or "") for f in row):
            issues.append({"row": idx, "level": "warn",
                           "msg": "row contains placeholder '（待填...)' — fill or remove"})
            continue
        # 必填
        for f in REQUIRED_FIELDS:
            if not row.get(f):
                issues.append({"row": idx, "level": "error",
                               "msg": f"missing field: {f}"})
        # 状态
        st = normalize_state(row.get("当前状态", ""))
        if st and st not in VALID_STATES:
            issues.append({"row": idx, "level": "error",
                           "msg": f"invalid 当前状态: '{st}' (allowed: {sorted(VALID_STATES)})"})
        # 日期
        d = row.get("日期", "")
        if d and not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            issues.append({"row": idx, "level": "error",
                           "msg": f"日期 must be YYYY-MM-DD, got '{d}'"})
        # 去重
        key = (row.get("公司", ""), row.get("岗位", ""))
        if key[0] and key[1]:
            if key in seen:
                issues.append({"row": idx, "level": "error",
                               "msg": f"duplicate (公司, 岗位): {key} also at row {seen[key]}"})
            else:
                seen[key] = idx
    return issues


def render_text(rows: list[dict], issues: list[dict]) -> str:
    out = ["# normalize_applications report", ""]
    out.append(f"- rows scanned: **{len(rows)}**")
    errs = [i for i in issues if i["level"] == "error"]
    warns = [i for i in issues if i["level"] == "warn"]
    infos = [i for i in issues if i["level"] == "info"]
    out.append(f"- errors: **{len(errs)}** · warnings: **{len(warns)}** · info(DEMO): **{len(infos)}**")
    out.append("")
    if not issues:
        out.append("## ✅ 0 violations — applications tracker is clean.")
        return "\n".join(out)
    out.append("## Findings")
    for it in issues:
        marker = {"error": "❌", "warn": "⚠️", "info": "ℹ️"}.get(it["level"], "·")
        out.append(f"- {marker} row {it['row']}: {it['msg']}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not APP_FILE.exists():
        print(f"missing: {APP_FILE}", file=sys.stderr)
        return 2
    text = APP_FILE.read_text(encoding="utf-8")
    rows = parse_table(text)
    issues = validate(rows)

    if args.json:
        print(json.dumps({"rows": len(rows), "issues": issues},
                         ensure_ascii=False, indent=2))
    else:
        print(render_text(rows, issues))

    has_error = any(i["level"] == "error" for i in issues)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
