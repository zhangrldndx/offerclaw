# -*- coding: utf-8 -*-
"""rag_lang.py — 文档语言判定(方案报告 §9.2 Metadata 分区)。

为什么不只用 source_type=paper:英文资料未来还包括 blog / manual / api_doc / tutorial,
按 source_type 枚举会越加越长且漏项;语言才是候选池失衡的真正维度——
报告 §6 实测的瓶颈是"英文块在中文占多数的池子里进不了前排",与它是不是论文无关。
故配额优先按 language 分区,再叠加 source_type。

判定为纯词频规则(零依赖、可复现、入库期一次性):CJK 与拉丁字母的相对占比。
阈值取宽松带:中英混排的技术文档(英文正文 + 中文注释)判 mixed,同样吃配额。
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[一-鿿㐀-䶿]")
_LATIN = re.compile(r"[A-Za-z]")

ZH, EN, MIXED, UNKNOWN = "zh", "en", "mixed", "unknown"

# 配额通道服务的分区:英文与中英混排都要保底进池(mixed 里的英文术语同样是跨语盲区)
QUOTA_LANGS = (EN, MIXED)


def detect_language(text: str) -> str:
    """返回 zh / en / mixed / unknown。按 CJK 字符占「有效字符」的比例分档。

    有效字符 = CJK + 拉丁字母(不含数字/标点/空白——代码块与公式里它们占比极高,
    会把判定拉向噪声)。样本过小(<20 有效字符)判 unknown,不参与语言配额。
    """
    if not text:
        return UNKNOWN
    n_cjk = len(_CJK.findall(text))
    n_lat = len(_LATIN.findall(text))
    total = n_cjk + n_lat
    if total < 20:
        return UNKNOWN
    ratio = n_cjk / total
    # 阈值说明:英文技术文档常夹少量中文注释;中文文档常夹大量英文术语/代码。
    # 0.05 / 0.30 的宽带把"英文正文+零星中文"判 en、"中文正文+大量术语"判 zh,
    # 中间地带判 mixed(两边配额都不亏待它)。
    if ratio < 0.05:
        return EN
    if ratio > 0.30:
        return ZH
    return MIXED


def is_quota_lang(lang: str) -> bool:
    """该语言是否属于「英文/混排保险池」——配额通道的取数范围。"""
    return lang in QUOTA_LANGS
