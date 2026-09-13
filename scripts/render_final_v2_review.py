#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the Final v2 draft as something a human can actually adjudicate.

The set is only blind if its labels were confirmed by a person, so this exists
to make that confirmation cheap: for each anchor it shows the two questions, the
requirement each answer must meet, and the passage the gold label points at, so
the reviewer can check "is this passage really the answer" without opening the
index.

It also surfaces the near-twin screen.  A twin in the *same* file is usually an
adjacent section and rarely a competing answer; a twin in a *different* file is
the case worth a second look, because if it answers the question too, then a
retrieval that returns it is not actually wrong and the qrels need a second
target rather than the run needing a fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CROSS_FILE_ALERT = 0.45


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", default="docs/rag_eval/final_v2/final_v2_draft.json")
    parser.add_argument("--output", default="docs/rag_eval/final_v2/REVIEW_SHEET.md")
    args = parser.parse_args()

    data = json.loads((ROOT / args.draft).read_text(encoding="utf-8"))
    twins = {row["anchor_id"]: row for row in data["near_twin_screen"]}
    by_anchor: dict[str, list] = {}
    negatives = []
    for item in data["items"]:
        (negatives if item["case_kind"] == "negative"
         else by_anchor.setdefault(item["anchor_id"], [])).append(item)

    lines = [
        "# Final v2 候选集 — 人工确认单",
        "",
        f"状态：`{data['status']}`。{data['design']['positives']} 正例 / "
        f"{data['design']['negatives']} 负例 / {data['design']['anchors']} 锚点。",
        "",
        "**金标不是判据打的**：每道题都是照着下面那段原文写出来的，所以金标就是那一段。",
        "判据（answerability teacher）全程没有参与出题和标注。",
        "",
        "确认时只需要回答两件事：",
        "",
        "1. 这段原文是不是真的回答了这道题？（不是 → 记下题号）",
        "2. 带 ⚠️ 的那些，另一篇文档里的近邻是不是**也**算正确答案？"
        "（是 → 该题应加第二个金标，否则模型答对了会被判成错）",
        "",
        "---",
        "",
        "## 一、正例（40 锚点 × 2 问法）",
        "",
    ]

    for anchor_id in sorted(by_anchor):
        items = by_anchor[anchor_id]
        target = items[0]["relevant_targets"][0]
        twin = twins.get(anchor_id, {})
        nearest = (twin.get("nearest") or [{}])[0]
        cross = (nearest.get("source") and nearest["source"] != target["source"]
                 and nearest.get("distance", 9) < CROSS_FILE_ALERT)
        flag = " ⚠️ 跨文件近邻" if cross else ""
        lines += [
            f"### {anchor_id}{flag} — `{target['source']}`",
            "",
            f"**金标原文**（{target['heading_path'][0] or '无标题'}）：",
            "",
            "> " + target["evidence_excerpt"][:420].replace("\n", " ").strip(),
            "",
        ]
        for item in items:
            lines += [
                f"- **[{item['query_style']}]** {item['question']}",
                f"  - 答案须含：{'；'.join(item['answer_requirements'])}",
            ]
        if cross:
            lines += [
                "",
                f"  ⚠️ 另一篇文档里有近邻（距离 {nearest['distance']}）："
                f"`{nearest['source']}` — {nearest['preview'][:80].strip()}…",
                "  请判断它是否也能回答上面的题。",
            ]
        lines.append("")

    lines += ["---", "", "## 二、负例（40 条）", "",
              "`拒答` = 库里没有依据，应当拒答；"
              "`纠正` = 前提是错的但库里有反证，应当回答并指出前提错误。", ""]
    for item in negatives:
        kind = "拒答" if item["expected_behavior"] == "abstain_from_kb" else "**纠正**"
        lines += [f"- [{kind}] {item['question']}",
                  f"  - 理由：{item['negative_rationale']}"]

    out = ROOT / args.output
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    flagged = sum(1 for a in by_anchor
                  if (twins.get(a, {}).get("nearest") or [{}])[0].get("source")
                  not in (None, by_anchor[a][0]["relevant_targets"][0]["source"])
                  and (twins.get(a, {}).get("nearest") or [{}])[0].get("distance", 9) < CROSS_FILE_ALERT)
    print(f"[review] wrote {out.relative_to(ROOT)} — {len(by_anchor)} 锚点，"
          f"{flagged} 个带跨文件近邻警告，{len(negatives)} 条负例")


if __name__ == "__main__":
    main()
