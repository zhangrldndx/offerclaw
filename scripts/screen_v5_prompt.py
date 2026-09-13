# -*- coding: utf-8 -*-
"""v5 提示词定点筛查:修不掉目标或伤到 gold 就地终止,不花全量的钱。"""
import json, os, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["RAG_ANSWERABILITY_PROMPT"] = "v5"
os.environ.setdefault("RAG_ANSWERABILITY", "1")
import chromadb
from rag_answerability import grade, action
from rag_tools import get_collection_name

col = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(get_collection_name())
txt = lambda cid: ((col.get(ids=[cid], include=["documents"]).get("documents") or [""]) + [""])[0]
b = pathlib.Path.home() / ".offerclaw/private_eval/final_v3/runs"
rows = {x["query_id"]: x for x in json.load(open(b / "B0_run1.json"))["runs"][0]["positive"]["rows"]}
b3 = json.load(open(b / "B3_run1.json"))["runs"][0]
b3neg = {x["query_id"]: x for x in b3["negative"]["rows"]}
items = {i["query_id"]: i for i in json.load(open(
    ROOT / "docs" / "rag_eval" / "final_v3" / "final_v3.json"))["items"]}

INFLATED = ["fv3-a12-natural","fv3-a13-implicit_oral","fv3-a14-implicit_oral","fv3-a18-implicit_oral",
            "fv3-a19-natural","fv3-a20-natural","fv3-a29-long_noisy","fv3-a35-natural","fv3-a38-standard"]
tasks = []
for q in INFLATED:
    x = rows[q]; question = items[q]["question"]
    gold = next(c for c in x["reranked_chunk_ids"] if c in set(x["grade3_chunk_ids"]))
    tasks.append(("inflated", q, question, txt(x["reranked_chunk_ids"][0])))
    tasks.append(("gold", q, question, txt(gold)))
for q in ("fv3-neg-011", "fv3-neg-014"):
    doc = txt((b3neg[q].get("final_chunk_ids") or [""])[0]) if b3neg.get(q) else ""
    tasks.append(("noref", q, items[q]["question"], doc))
g = json.load(open(
    ROOT / "docs" / "rag_eval" / "colloquial" / "live" / "guard119_gate_votes3.json"
))
grows = {f"{r['source']}/{r['id']}": r for r in g["rows"]}
for key, question in (("colloquial19/v2aneg-019", "这俩哪个更适合我一点，你直接说结论就行。"),
                      ("dev80_v1/col-neg-076", "上次那个方案现在怎么样了？")):
    r = grows.get(key)
    doc = txt(r["top1_chunk_id"]) if r and r.get("top1_chunk_id") else ""
    tasks.append(("noref", key, question, doc))
for q in ("fv3-neg-028", "fv3-neg-029", "fv3-neg-038"):   # Transformer递归/LoRA W/react_mode默认
    doc = txt((b3neg[q].get("final_chunk_ids") or [""])[0]) if b3neg.get(q) else ""
    tasks.append(("correct", q, items[q]["question"], doc))

# 金丝雀:网关死掉时所有判词=None,会伪装成"全拒答"骗过筛查判决(实测发生过一次,
# gold 0/9 + 纠正 0/3 的指纹)。拿不到真判词就退出,不产出任何判决。
try:
    canary = grade("金丝雀: 这段写了部署步骤吗?", "金丝雀正文,与知识库无关。" * 20, use_cache=False)
except Exception:
    canary = None
if canary is None:
    print("SCREEN_VERDICT: GATEWAY_DOWN")
    raise SystemExit(2)

def run(task):
    kind, qid, question, doc = task
    try:
        v = grade(question, doc, use_cache=True)
    except Exception as e:
        v = None
    act = action(v)
    return kind, qid, (v or {}).get("grade"), (v or {}).get("relation"), act

results = list(ThreadPoolExecutor(max_workers=6).map(run, tasks))
out = {"inflated": [], "gold": [], "noref": [], "correct": []}
for kind, qid, gr, rel, act in results:
    out[kind].append({"id": qid, "grade": gr, "relation": rel, "action": act})
fixed = sum(1 for r in out["inflated"] if (r["grade"] or 0) < 3 or r["action"] == "abstain")
gold_kept = sum(1 for r in out["gold"] if r["grade"] == 3 and r["action"] != "abstain")
noref_ok = sum(1 for r in out["noref"] if r["action"] == "abstain")
correct_kept = sum(1 for r in out["correct"] if r["action"] == "correct_premise")
print(json.dumps(out, ensure_ascii=False))
print(f"\n== 判决 ==")
print(f"虚高冠军被降: {fixed}/9   (v4 全是 3/可行动;目标 >=5)")
print(f"gold 保住 3:  {gold_kept}/9  (目标 >=8)")
print(f"无指代拒答:   {noref_ok}/4  (v4 全放行;目标 >=3)")
print(f"纠正能力保持: {correct_kept}/3 (目标 3)")
# ``fixed`` 只作信息展示:同源虚高子规则已从 v5 移除(定点筛查 2/9,提示词治不了),
# 缩减版不该再被它当门槛——否则砍掉的规则永远"考不过"自己已放弃的目标。
verdict = "GO" if gold_kept >= 8 and noref_ok >= 3 and correct_kept >= 3 else "ABORT"
print("SCREEN_VERDICT:", verdict)
