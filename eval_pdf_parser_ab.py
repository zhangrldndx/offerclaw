# -*- coding: utf-8 -*-
"""eval_pdf_parser_ab.py — 第 2 层 A/B:P0 文字层(pypdf) vs Docling 结构化解析。

语料 = 入库 Golden Set(合成,金标准由代码保证)+ 3 篇真实论文(真值探针取自
人工阅读原始 PDF 首页的原文句子):
  - ReAct(ICLR,单栏)· MemGPT(ICML,双栏)· Memory Sandbox(ACM,双栏)

指标(全部代码计算,零 LLM):
  presence  探针句召回(空白归一子串)
  within    栏内相邻句顺序(a 在 b 前且间距 < 1500 字符)
  column    跨栏完整性——同一物理高度的左/右栏句子,正确阅读顺序下相距应
            很远(>1000 字符);文字层按 y 坐标串读时会挨在一起(<1000 = 串读)
  headings  Markdown 标题数(结构恢复的直接证据;文字层天然为 0)
  golden    复用 eval_ingestion 的 Golden Set 指标(表格 row_order 是预登记靶点)

公平性:Docling 关闭 OCR(do_ocr=False)——与 P0 同样只处理文字层,不借 OCR 加分。
预登记判据(采纳 Docling 为 PDF 默认解析的条件,三条同时满足):
  1) golden 表格 row_order 显著改善(0.5 → ≥0.75);
  2) 双栏论文跨栏完整性不低于 P0,栏内顺序与探针召回不回退;
  3) 单篇解析耗时可接受(≤120s/篇,本机 CPU)。
结果无论方向如何入档 docs/rag_eval/ingestion/docling_ab.json。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DL = os.path.expanduser("~/Downloads")
REAL_PDFS = {
    "react_single_col": {
        "path": os.path.join(DL, "REAC T - SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS.pdf"),
        "presence": [
            "While large language models (LLMs) have demonstrated impressive performance",
            "reasoning traces help the model induce, track, and update action plans",
            "on two interactive decision making benchmarks (ALFWorld and WebShop)",
            "A unique feature of human intelligence is the ability to seamlessly combine",
        ],
        "within_pairs": [
            ("While large language models (LLMs) have demonstrated impressive performance",
             "their abilities for reasoning"),
            ("A unique feature of human intelligence",
             "which has been theorized to play an important role in human cognition"),
        ],
        "cross_pairs": [],
    },
    "memgpt_two_col": {
        "path": os.path.join(DL, "MemGPT- Towards LLMs as Operating Systems.pdf"),
        "presence": [
            "Large language models (LLMs) have revolutionized AI",
            "we propose virtual context management",
            "Directly extending the context length of transformers",
            "we treat context windows as a constrained memory resource",
        ],
        "within_pairs": [
            ("Large language models (LLMs) have revolutionized AI",
             "hindering their utility in tasks like extended conversations"),
            ("Directly extending the context length of transformers",
             "due to the transformer architecture"),
        ],
        # 左栏摘要中段 vs 右栏同高度句——正确顺序=远,y 轴串读=近
        "cross_pairs": [
            ("hindering their utility in tasks like extended conversations",
             "curs a quadratic increase in computational time"),
            ("technique drawing inspiration from hierarchical memory systems",
             "recent research shows that long"),
        ],
    },
    "sandbox_two_col": {
        "path": os.path.join(DL, "Memory Sandbox- Transparent and Interactive Memory Management for Conversational Agents.pdf"),
        "presence": [
            "The recent advent of large language models (LLM) has resulted",
            "we present Memory Sandbox, an interactive system and design",
            "Multiple strategies have been introduced to manage agents",
            "Explainable AI research seeks to help people form mental models",
        ],
        "within_pairs": [
            ("The recent advent of large language models (LLM) has resulted",
             "in high-performing conversational agents"),
            ("Large Language Models (LLMs) are currently capable of generating",
             "human-like responses in open-domain tasks"),
        ],
        "cross_pairs": [
            ("these agents have limited memory and can be distracted",
             "memory management strategies are hidden behind the interface"),
            ("users currently lack affordances",
             "difficult for users to repair conversational breakdowns"),
        ],
    },
}

# 归一化剥 空白+连字符:双栏 PDF 的换行连字(如 revolution-⏎ized)会打断子串匹配,
# 这是 2026-08-08 首跑时被当成"内容缺失"的评测器 bug——探针没丢,是匹配太严。
_NORM = re.compile(r"[\s\-‐‑–—]+")


def _n(s: str) -> str:
    return _NORM.sub("", s or "")


def _find(text_norm: str, probe: str) -> int:
    return text_norm.find(_n(probe))


def score_text(text: str, spec: dict) -> dict:
    tn = _n(text)
    presence = sum(1 for p in spec["presence"] if _find(tn, p) >= 0)
    within_ok = 0
    for a, b in spec["within_pairs"]:
        ia, ib = _find(tn, a), _find(tn, b)
        if ia >= 0 and ib >= 0 and ia < ib and (ib - ia) < 1500:
            within_ok += 1
    col_ok, col_detail = 0, []
    for left, right in spec["cross_pairs"]:
        il, ir = _find(tn, left), _find(tn, right)
        if il < 0 or ir < 0:
            col_detail.append(None)          # 探针缺失,无法判定
            continue
        sep = abs(ir - il)
        col_detail.append(sep)
        if sep > 1000:
            col_ok += 1
    return {
        "chars": len(text),
        "presence": f"{presence}/{len(spec['presence'])}",
        "within_order": f"{within_ok}/{len(spec['within_pairs'])}",
        "column_integrity": (f"{col_ok}/{len(spec['cross_pairs'])}"
                             if spec["cross_pairs"] else "n/a"),
        "column_separations": col_detail,
        "headings_md": len(re.findall(r"^#{1,4} ", text, re.M)),
    }


# ---------------- 两个引擎 ----------------

def parse_pypdf(path: str) -> str:
    from knowledge_crawler import extract_text_for_kb
    with open(path, "rb") as f:
        return extract_text_for_kb(os.path.basename(path), f.read())


_DOCLING_CONVERTER = None


def _docling():
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        arts = os.path.expanduser("~/.cache/docling/models")   # docling-tools 预取目录(弱网离线)
        if os.path.isdir(arts):
            opts.artifacts_path = arts
        _DOCLING_CONVERTER = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _DOCLING_CONVERTER


def parse_docling(path: str) -> tuple[str, int]:
    res = _docling().convert(path)
    doc = res.document
    tables = len(getattr(doc, "tables", []) or [])
    return doc.export_to_markdown(), tables


# ---------------- 主流程 ----------------

def run_ab() -> dict:
    from eval_ingestion import (TABLE_ROWS, build_scanned_pdf, build_table_pdf,
                                build_text_pdf, run_ingestion_eval)

    out: dict = {"arms": {}, "real_pdfs": {}, "golden": {}}

    # -- Golden Set:P0 直接复用既有评测;Docling 跑同一套合成 PDF --
    out["golden"]["p0"] = run_ingestion_eval()["cases"]

    import tempfile
    gcases = {}
    with tempfile.TemporaryDirectory() as td:
        for name, data in (("text.pdf", build_text_pdf()),
                           ("table.pdf", build_table_pdf()),
                           ("scan.pdf", build_scanned_pdf())):
            p = os.path.join(td, name)
            with open(p, "wb") as f:
                f.write(data)
            t0 = time.time()
            try:
                md, ntab = parse_docling(p)
                case = {"parsed": True, "chars": len(md), "tables_found": ntab,
                        "secs": round(time.time() - t0, 1)}
                if name == "table.pdf":
                    norm = _n(md)
                    cells = [v for row in TABLE_ROWS for v in row]
                    case["cell_recall"] = round(
                        sum(1 for c in cells if _n(c) in norm) / len(cells), 3)
                    ordered = 0
                    for row in TABLE_ROWS:
                        idxs = [norm.find(_n(v)) for v in row]
                        ok = all(i >= 0 for i in idxs) and idxs == sorted(idxs)
                        near = ok and (idxs[-1] - idxs[0]) <= sum(len(v) for v in row) + 20
                        ordered += 1 if (ok and near) else 0
                    case["row_order"] = round(ordered / len(TABLE_ROWS), 3)
                if name == "scan.pdf":
                    case["empty_as_expected"] = len(_n(md)) < 30   # 关 OCR 应抽不出字
            except Exception as e:
                case = {"parsed": False, "error": str(e)[:150]}
            gcases[name] = case
    out["golden"]["docling"] = gcases

    # -- 真实论文双引擎 --
    for key, spec in REAL_PDFS.items():
        if not os.path.exists(spec["path"]):
            out["real_pdfs"][key] = {"error": "file not found"}
            continue
        entry = {}
        t0 = time.time()
        try:
            txt = parse_pypdf(spec["path"])
            entry["p0_pypdf"] = {**score_text(txt, spec),
                                 "secs": round(time.time() - t0, 1)}
        except Exception as e:
            entry["p0_pypdf"] = {"error": str(e)[:150]}
        t0 = time.time()
        try:
            md, ntab = parse_docling(spec["path"])
            entry["docling"] = {**score_text(md, spec), "tables_found": ntab,
                                "secs": round(time.time() - t0, 1)}
        except Exception as e:
            entry["docling"] = {"error": str(e)[:150]}
        out["real_pdfs"][key] = entry
    return out


def main() -> int:
    out = run_ab()
    path = os.path.join(BASE, "docs", "rag_eval", "ingestion", "docling_ab.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[written] {os.path.relpath(path, BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
