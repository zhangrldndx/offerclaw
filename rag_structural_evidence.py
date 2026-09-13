# -*- coding: utf-8 -*-
"""rag_structural_evidence.py — 证据门的「结构块不作数」判据(默认关)。

**要修的实测缺陷**(docs/rag_eval/colloquial/GATE_20260827.md §1.1):口语负例
`v2aneg-015`(最左前缀是不是也决定 RAG 检索候选顺序)被**主门**放行——
dense 距离 0.576(阈值 0.73)、精排 0.9712,两项都远超门槛。取回的却是
`backend_basic_03_08_mysql_leftmost_prefix.md` 的**文档标题块**与**页面结构目录**:
一块只有 `# 8.MySQL数据库——索引潜规则（最左前缀原则）` 加两行入库说明,
另一块是十几个小节名。目录块与问题词面高度重合,所以向量距离近、精排分高,
但它**不含任何可作答的断言**——门是被"词面回声"骗的,不是被证据说服的。

**不能用"目录块一律扣分"来修**。同一批评测里有确认为正确的题,答案就在
带小节标题的正文块里(v2a-ca-016 / v2a-llm-001,标题下每条都有正文)。
把目录块无差别打压会连它们一起打死。所以这里判的不是"像不像目录",
而是**这一块自己有没有正文**:

    structural_fraction = (裸标题 + 入库说明引用)字符数 / 非空字符数

裸标题 = 底下没有正文就直接接下一个标题的标题行——这正是目录的定义
(小节名没有小节内容)。标题下面**有**正文时不计入,所以"带小节标题的正文块"
天然是低分。实测(3348 块冻结索引,两套守卫集合共 37 个金标块):

| | structural_fraction |
|---|---|
| 金标块最大值(`backend_basic_03_12_mysql_replication`) | **0.250** |
| v2aneg-015 目录块 | **0.391** |
| v2aneg-015 标题块 / v2a-be-001 标题块 | **1.000** |

τ 取间隙中点 **0.32**(两侧各留 0.07 / 0.07),与项目给距离阈值定档的做法一致。
全库只有 92/3348 块(2.7%)≥0.32,其中 77 块 ≥0.70——是个又小又分得开的尾巴。

**用法是"换锚点"而不是"扣分"**:门不再拿排第一的块当判据,而是拿**名次最靠前
的实心块**当判据(距离与精排分都取它的)。排序、召回、返回给上层的数值一律不动。
若整个候选池**全是**结构块(真·"这份文档都讲了什么"),则退回原行为——
宁可少修一个边角,也不做无差别打压。

判据单独成模块是为了可单测:它埋在几百行的检索函数里时,只能靠源码断言钉,
而源码断言钉不住"τ 是上界还是下界""全结构时是放行还是拒答"这种东西。
"""
from __future__ import annotations

import os
import re

# 结构线索。`^#{1,6}\s` 只认标准 ATX 标题;**围栏内的同形行不算标题**——
# 全库 5208 行标题形状里有 1016 行在 ``` 围栏内,是 Python/Shell 注释(`# 执行两种检索`),
# 不跟踪围栏会把大量代码块误判成目录。
_HEADING = re.compile(r"^#{1,6}\s")
_FENCE = re.compile(r"^```")
# 入库期写进正文的溯源横幅(本仓 ingest 自己写的),不是知识。
# 只认这几种固定抬头:`> <strong>提示</strong>`、`> [完整代码](...)` 这类引用是**内容**,
# 把所有 blockquote 一律当样板会误伤它们。
_PROVENANCE = re.compile(
    r"^(来源|入库说明|所属路径|建议归类|采集时间|原文链接|source|url)\s*[:：]",
    re.IGNORECASE,
)

# 标定见模块 docstring。显式给出才生效,任何解析失败一律按"关"处理——
# 证据门的改动只在被明确打开时才改变行为。
ENV_THRESHOLD = "RAG_STRUCTURAL_EVIDENCE_MAX"
CALIBRATED_THRESHOLD = 0.32

_HEADING_LINE, _PROVENANCE_LINE, _BODY_LINE = "h", "p", "b"


def _classify(text: str, in_fence: bool = False) -> list[tuple[str, str]]:
    """按行分成 标题 / 溯源样板 / 正文 三类(丢空行,围栏内一律算正文)。

    ``in_fence`` 是**起始**围栏状态。切块会把一个代码块拦腰截断,后半块以"围栏内"
    开始却看不到那个 ``` ——全库 453/3348 块的围栏标记数是奇数,状态天然存疑。
    """
    lines: list[tuple[str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not line:
            continue
        if in_fence:
            lines.append((_BODY_LINE, line))
        elif _HEADING.match(line):
            lines.append((_HEADING_LINE, line))
        elif line.startswith(">"):
            inner = line.lstrip(">").strip()
            lines.append(
                (_PROVENANCE_LINE, line)
                if (not inner or _PROVENANCE.match(inner))
                else (_BODY_LINE, line)
            )
        else:
            lines.append((_BODY_LINE, line))
    return lines


def structural_fraction(text: str) -> float:
    """这一块有多大比例的字符是「没有正文的结构」。空块记 0.0(交给别的判据)。

    裸标题的判定跳过溯源行:标题与它的正文之间夹一行 `> 来源：...` 时,
    那仍是一个有正文的标题,不该算成目录条目。

    围栏标记数为奇数时起始状态存疑,两种读法都算一遍**取小**——存疑时宁可少判一块
    结构块。方向是单边的:这只会让判据更保守,不会新造出误拒。
    """
    if (text or "").count("```") % 2:
        return min(_structural_fraction(text, False), _structural_fraction(text, True))
    return _structural_fraction(text, False)


def _structural_fraction(text: str, in_fence: bool) -> float:
    lines = _classify(text, in_fence)
    total = sum(len(line) for _kind, line in lines)
    if not total:
        return 0.0
    kinds = [kind for kind, _line in lines]
    structural = 0
    for index, (kind, line) in enumerate(lines):
        if kind == _PROVENANCE_LINE:
            structural += len(line)
            continue
        if kind != _HEADING_LINE:
            continue
        following = next(
            (kinds[j] for j in range(index + 1, len(kinds))
             if kinds[j] != _PROVENANCE_LINE),
            None,
        )
        if following != _BODY_LINE:      # 底下没有正文 = 目录条目
            structural += len(line)
    return structural / total


def structural_threshold() -> float | None:
    """读旋钮:未设置/解析失败 → None(判据整体关闭,行为与改造前逐字节相同)。"""
    raw = (os.environ.get(ENV_THRESHOLD, "") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0.0 < value <= 1.0 else None


def is_structural(text: str, threshold: float | None) -> bool:
    """threshold=None(关)时恒为 False——判据关闭时没有任何块是"结构块"。"""
    if threshold is None:
        return False
    return structural_fraction(text) >= threshold


def substantive_mask(docs: list, threshold: float | None) -> list[bool]:
    """逐块给出"是否实心"。判据关闭时全 True,调用方无需再分支。"""
    if threshold is None:
        return [True] * len(docs)
    return [not is_structural(doc, threshold) for doc in docs]


def evidence_anchor(docs: list, threshold: float | None) -> int | None:
    """门该拿哪一块当判据:名次最靠前的实心块。

    返回 None 有两种含义,调用方都应**退回原行为**:候选为空,或候选**全是**
    结构块。后者是刻意的放行——真要问"这份文档都讲了什么",目录就是答案,
    此时再拒答就成了无差别打压(这正是本判据不做"目录块一律扣分"的原因)。
    """
    if not docs:
        return None
    if threshold is None:
        return 0
    for index, doc in enumerate(docs):
        if not is_structural(doc, threshold):
            return index
    return None


__all__ = [
    "CALIBRATED_THRESHOLD",
    "ENV_THRESHOLD",
    "evidence_anchor",
    "is_structural",
    "structural_fraction",
    "structural_threshold",
    "substantive_mask",
]
