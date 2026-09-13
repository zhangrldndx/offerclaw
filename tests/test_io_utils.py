"""B1/B2 状态文件原子写 + 文件锁测试。"""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io_utils import atomic_write_json, atomic_write_text, file_lock


def test_atomic_write_json_complete_no_tmp(tmp_path):
    p = str(tmp_path / "x.json")
    atomic_write_json(p, {"a": 1, "b": [1, 2]})
    assert json.load(open(p)) == {"a": 1, "b": [1, 2]}
    assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))


def test_atomic_write_text(tmp_path):
    p = str(tmp_path / "x.md")
    atomic_write_text(p, "| 公司 | 状态 |\n| A | 已投 |\n")
    assert open(p).read().endswith("已投 |\n")


def test_file_lock_releases(tmp_path):
    p = str(tmp_path / "s")
    with file_lock(p):
        pass
    with file_lock(p):        # 释放后能再次获取（不死锁）
        pass
    assert os.path.exists(p + ".lock")


def test_concurrent_atomic_writes_never_half(tmp_path):
    """30 线程并发原子写同一文件 → 结果永远是完整可解析 JSON（无半截）。"""
    p = str(tmp_path / "x.json")

    def w(i):
        atomic_write_json(p, {"v": i, "pad": "x" * 500})

    ts = [threading.Thread(target=w, args=(i,)) for i in range(30)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert isinstance(json.load(open(p)), dict)         # 完整可解析，非半截
    assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))
