"""A5 记忆持久化原子化 + 损坏不静默 + 写锁 + 事件校验测试。"""
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_layers import (
    EpisodicMemory, SemanticMemory,
    _atomic_write_json, _safe_load_json, _validate_event,
)


def test_atomic_write_complete_no_tmp_leak(tmp_path):
    p = str(tmp_path / "x.json")
    _atomic_write_json(p, {"a": 1, "b": [1, 2]})
    assert json.load(open(p)) == {"a": 1, "b": [1, 2]}
    assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))   # 临时文件已清理


def test_safe_load_corrupt_backs_up_not_silent(tmp_path):
    """损坏 JSON → 备份 .corrupt.<ts> + 返回 default（不静默丢失）。"""
    p = str(tmp_path / "sem.json")
    open(p, "w").write("{bad json not closed")
    out = _safe_load_json(p, {"default": 1})
    assert out == {"default": 1}
    assert any(".corrupt." in f for f in os.listdir(tmp_path))         # 坏文件被备份保全


def test_validate_event_requires_kind():
    _validate_event({"kind": "reflection"})                            # 合法
    for bad in ({"no_kind": 1}, {"kind": ""}, "not_a_dict", None):
        with pytest.raises(ValueError):
            _validate_event(bad)


def test_semantic_corrupt_preserves_sediment(tmp_path):
    """损坏 semantic.json 后再 set：坏文件被备份（数据可恢复）而非静默覆写。"""
    sem = SemanticMemory(base_dir=str(tmp_path))
    sem.set("preferred_direction", "大模型应用工程")
    with open(sem.path, "a") as f:                                     # 追加垃圾→JSON 不可解析，但原内容仍在
        f.write("\n{{corrupt garbage")
    sem.set("other", "x")                                              # _load 走 safe_load 备份坏文件
    backups = [f for f in os.listdir(tmp_path) if ".corrupt." in f]
    assert backups                                                     # 坏文件被备份保全（非静默覆写）
    assert "大模型应用工程" in open(os.path.join(tmp_path, backups[0])).read()  # 原沉淀在备份里可恢复
    assert sem.get("other") == "x"                                     # 新值正常写入


def test_episodic_concurrent_append_no_interleave(tmp_path):
    """50 线程并发 append（flock 串行化）→ 行数完整、每行可 json.loads（无交错损坏）。"""
    epi = EpisodicMemory(base_dir=str(tmp_path))

    def w(i):
        epi.append({"kind": "test", "i": i})

    ts = [threading.Thread(target=w, args=(i,)) for i in range(50)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = open(epi.path).read().strip().split("\n")
    assert len(lines) == 50
    assert all(json.loads(ln)["kind"] == "test" for ln in lines)      # 每行合法、无半行交错


def test_episodic_rejects_event_without_kind(tmp_path):
    epi = EpisodicMemory(base_dir=str(tmp_path))
    with pytest.raises(ValueError):
        epi.append({"no_kind": 1})                                    # schema 校验拦截脏事件
