# -*- coding: utf-8 -*-
"""P1-a 回归:记忆/执行追踪消融的确定性结果钉死(机制变了必须有意识地更新)。"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import eval_memory_ablation as ema


def test_tracking_off_repeats_forever():
    off = ema._simulate_days(tracking_on=False)
    assert off["repeat_rate"] == 1.0 and off["escalations"] == 0


def test_tracking_on_reduces_repeats_and_escalates():
    on = ema._simulate_days(tracking_on=True)
    off = ema._simulate_days(tracking_on=False)
    assert on["repeat_rate"] < off["repeat_rate"]
    assert on["escalations"] > 0


def test_memory_injection_reflows_with_bounded_cost():
    mem_on = ema._memory_injection(mem_on=True)
    mem_off = ema._memory_injection(mem_on=False)
    assert mem_on["sops"] >= 1 and mem_on["lessons"] == 3       # 回流成立
    assert 0 < mem_on["injected_chars"] < 2000                   # 成本有界
    assert mem_off["sops"] == 0 and mem_off["injected_chars"] == 0


def test_deterministic():
    assert ema._simulate_days(True) == ema._simulate_days(True)
