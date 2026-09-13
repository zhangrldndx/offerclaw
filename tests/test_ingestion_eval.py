# -*- coding: utf-8 -*-
"""入库 Golden Set 基线快照(入库指导 §19 · 第 1 层):数字漂移必须是有意识的。

语料由代码确定性生成(reportlab/python-docx),零二进制、零网络、零 LLM。
P0 已知弱点被诚实钉死:表格 PDF 单元格全召回但行序仅 0.5——
这是第 2 层(Docling 结构化解析)A/B 对比的预登记靶点,不许静默挪动。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_ingestion import run_ingestion_eval  # noqa: E402


def test_golden_set_baseline_snapshot():
    r = run_ingestion_eval()
    s = r["summary"]
    assert s["parse_success"] == 3 and s["refusal_correct"] == 2
    c = r["cases"]
    # 文字型 PDF:中英文已知句全召回(CID 中文字体抽取干净)
    assert c["text_pdf"]["coverage_cn"] == 1.0
    assert c["text_pdf"]["coverage_en"] == 1.0
    # 拒收路径:扫描版/损坏文件 → 明确拒绝且报错可读
    assert c["scanned_pdf"]["refused"] and c["scanned_pdf"]["message_readable"]
    assert c["corrupt"]["refused"] and c["corrupt"]["message_readable"]
    # docx:段落/表格全召回 + 行级自含表头格式生效
    assert c["resume_docx"]["coverage"] == 1.0
    assert c["resume_docx"]["table_row_selfcontain"] is True
    # P0 结构性弱点(预登记基线):字符满分、行序 0.5——升级解析器应打这里
    assert c["table_pdf"]["table_cell_recall"] == 1.0
    assert c["table_pdf"]["table_row_order"] == 0.5
