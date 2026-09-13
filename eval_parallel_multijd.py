# -*- coding: utf-8 -*-
"""E3 — 并行多 JD:墙钟提速 + correctness（结构腿，docs/MULTI_AGENT_UPGRADE.md §5）。

预登记判据(先登记后跑,防挪门柱):
  go = speedup 显著 >1 且 correctness 全等（并行终态==串行终态，逐字段相等）。
  no-go/部分采纳 = 若投机开销吃掉收益 → 保留串行,记诚实结果。

诚实纪律:
- ``skip_llm=True`` 隔离**纯编排开销**(token=0);此档下单 JD 流程极快,并行提速本就有限——
  真正的墙钟收益在 ``skip_llm=False``(LLM 时延主导),需 key,另测。
- 并行只是"单条 decode 流结构上做不到"的提速,**不是变聪明**;不夸大。

用法:``python eval_parallel_multijd.py [N] [--llm]``
"""
import sys
import time


def _make_jds(n: int):
    return [{"jd_text": (f"岗位名称：大模型应用开发实习生（样本{i}）\n"
                         f"技术要求：Python、LangGraph、RAG、FastAPI、Embedding\n"
                         f"学历：硕士\n工作性质：实习"),
             "jd_title": f"JD#{i}"} for i in range(n)]


def _summ(out):
    """逐字段可比摘要（correctness 判据用）:标题→(结论,路径)。"""
    return [(r["jd_title"], r["status"], r["route_taken"]) for r in out["runs"]]


def run(n: int = 6, skip_llm: bool = True) -> dict:
    from supervisor import run_supervisor
    jds = _make_jds(n)
    run_supervisor(jds, skip_llm=skip_llm)          # 预热:吸收首次导入/模型加载/文件读,避免污染计时
    t0 = time.monotonic()
    seq = run_supervisor(jds, skip_llm=skip_llm)
    wall_seq = time.monotonic() - t0
    t0 = time.monotonic()
    par = run_supervisor(jds, skip_llm=skip_llm, parallel=True, max_workers=4)
    wall_par = time.monotonic() - t0

    correctness = _summ(seq) == _summ(par)          # 并行终态 == 串行终态
    speedup = (wall_seq / wall_par) if wall_par else 0.0
    res = {"n": n, "skip_llm": skip_llm, "wall_seq_s": round(wall_seq, 3),
           "wall_par_s": round(wall_par, 3), "speedup": round(speedup, 2),
           "correctness_equal": correctness}
    print(f"N={n} skip_llm={skip_llm}  串行 {wall_seq:.2f}s  并行 {wall_par:.2f}s  "
          f"speedup={speedup:.2f}x  correctness={'PASS' if correctness else 'FAIL'}")
    if skip_llm:
        print("  [诚实] skip_llm=True 流程 CPU 密集,受 Python GIL 限制,线程并行提速≈1x 属正常且预期。"
              "线程并行的真收益在 skip_llm=False——LLM 网络等待释放 GIL,I/O 密集才真正并行。需 key 另测。")
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
    run(n, skip_llm="--llm" not in sys.argv)
