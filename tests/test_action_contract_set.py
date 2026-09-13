# -*- coding: utf-8 -*-
"""动作合同专项集的**离线钉**(集合合同 + 判定器单元,零 LLM 零 Chroma)。

live 运行走 scripts/run_action_contract_set.py(硬门 24/24、≥11/12、12/12);
这里钉住的是"集合本身不许悄悄变形"与"首句判定器行为"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_action_contract_set import (
    ActionContractError,
    DEFAULT_SET,
    first_sentence,
    load_set,
    sentence_corrects_premise,
)


def test_checked_in_set_satisfies_the_contract():
    payload = load_set(DEFAULT_SET)
    kinds = [c["kind"] for c in payload["cases"]]
    assert kinds.count("correct_premise") == 12
    assert kinds.count("abstain") == 12


def test_set_rejects_paraphrase_padding(tmp_path: Path):
    payload = json.loads(DEFAULT_SET.read_text(encoding="utf-8"))
    payload["cases"][1]["question"] = payload["cases"][0]["question"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ActionContractError, match="unique"):
        load_set(bad)


def test_correct_premise_cases_carry_evidence_hint(tmp_path: Path):
    payload = json.loads(DEFAULT_SET.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["kind"] == "correct_premise":
            case.pop("evidence_hint", None)
            break
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ActionContractError, match="evidence_hint"):
        load_set(bad)


def test_first_sentence_judge_matches_the_contract_wording():
    # 合同措辞(_CORRECT_PREMISE_RULES 教出来的开头)必须判过
    assert sentence_corrects_premise("前提不成立。RRF 并不相加分数……")
    assert sentence_corrects_premise("这个说法有误!RRF 用的是倒数排名。")
    # 正常作答开头不许误判成纠偏
    assert not sentence_corrects_premise("RRF 的原理如下:按排名取倒数再累加。")
    assert not sentence_corrects_premise("资料显示,混合检索结合稀疏与密集两路。")
    # 只看首句:纠偏出现在第二句不算履约
    assert not sentence_corrects_premise("先说结论。前提不成立这一点后面再讲。")
    assert first_sentence("甲。乙。") == "甲。"
