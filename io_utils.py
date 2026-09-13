"""[B1/B2] 状态文件的原子写 + 进程级文件锁。

把 A5 给记忆层做的「原子写 + 损坏可见」哲学，推广到所有共享状态文件
（gap_store.json / applications.md / growth_metrics.json / growth_journal.md）。

- ``atomic_write_*``：写同目录 .tmp → flush+fsync → os.replace（POSIX rename 原子），
  杜绝「写到一半 kill」留半截文件；最后一个 rename 赢，永远是完整旧版或新版。
- ``file_lock``：基于独立 .lock 文件的 fcntl.flock 排他锁，给 read-modify-write 串行化
  （CLI 每次新进程 / Web 长驻 / cron 三方并发写同一文件时防 lost-update）。
  Windows 使用 ``msvcrt.locking``，POSIX 使用 ``fcntl.flock``。
"""
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager


_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: str) -> threading.RLock:
    key = os.path.abspath(path)
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def atomic_write_text(path: str, text: str) -> None:
    """原子写文本：tmp + flush+fsync + os.replace。"""
    with _thread_lock(path):
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(6):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if os.name != "nt" or attempt == 5:
                        raise
                    time.sleep(0.01 * (2 ** attempt))
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise


def atomic_write_json(path: str, data) -> None:
    """原子写 JSON（indent=2, 非 ASCII 保留）。"""
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


@contextmanager
def file_lock(path: str):
    """Cross-platform exclusive lock that remains stable across ``os.replace``."""
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)) or ".", exist_ok=True)
    thread_lock = _thread_lock(lock_path)
    with thread_lock:
        handle = open(lock_path, "a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        windows = os.name == "nt"
        try:
            handle.seek(0)
            if windows:
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                handle.seek(0)
                if windows:
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
