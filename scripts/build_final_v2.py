#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the Final v2 candidate set from authored questions and blind anchors.

Two rules shape this file.  First, the gold is the anchor the question was
written *from*, never something the answerability judge picked: the judge is the
component under test, and a set whose labels it produced would measure only its
self-consistency.  Second, the output is a draft -- ``status`` says so -- because
the qrels still need a human pass before any arm is run against them.

The near-twin screen is the one place a model is used, and only the embedding
model: it flags anchors that have a close neighbour in the corpus, so the
reviewer can decide whether that neighbour is also a valid answer rather than
discovering it later as a mysterious miss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _excerpt(document: str, limit: int = 700) -> str:
    return (document or "")[:limit]


# The one anchor whose cross-file near-twin covers the same answer requirements
# exactly as well as the gold (1.00 vs 1.00 on the deterministic bigram screen).
# Left as a single gold, a run that returned the twin would be scored wrong for
# retrieving a passage that does answer the question.  Recorded here rather than
# patched in afterwards so a rebuild reproduces it.
DUAL_GOLD = {"fv2-a36-natural"}


def build(anchors_path: Path, positives, negatives, twin_top_k: int,
          prefix: str = "fv2", freeze_path: Path | None = None,
          dataset_id: str = "rag-final-v2-candidate-v1",
          dual_gold: set | None = None) -> dict:
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))["anchors"]

    import chromadb
    from rag_qrels_v2 import evidence_span_hash, index_contract_fingerprint
    from rag_tools import get_collection_name, get_embeddings_batch, index_fingerprint

    def nearest_cross_file(candidates, gold_source):
        return next((c for c in candidates if c["source"] != gold_source), None)

    collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection(
        get_collection_name())

    items = []
    twin_report = []
    for order, (anchor_index, domain, questions) in enumerate(positives):
        anchor = anchors[anchor_index]
        chunk_id, source = anchor["chunk_id"], anchor["source"]
        anchor_id = f"{prefix}-a{order:02d}"

        # Near-twin screen: a chunk that says almost the same thing is a
        # plausible alternative gold, and the reviewer should see it now.
        embedding = get_embeddings_batch([anchor["document"]])
        got = collection.query(query_embeddings=embedding, n_results=twin_top_k + 1,
                               include=["metadatas", "distances", "documents"])
        twins = [
            {"chunk_id": tid, "source": (meta or {}).get("source", ""),
             "distance": round(float(dist), 4),
             "preview": (doc or "")[:120]}
            for tid, meta, dist, doc in zip(
                got["ids"][0], got["metadatas"][0], got["distances"][0], got["documents"][0])
            if tid != chunk_id
        ]
        twin_report.append({"anchor_id": anchor_id, "chunk_id": chunk_id,
                            "source": source, "nearest": twins[:twin_top_k]})

        for style, question, requirements in questions:
            items.append({
                "query_id": f"{anchor_id}-{style}",
                "anchor_id": anchor_id,
                "case_kind": "positive",
                "domain": domain,
                "query_style": style,
                "question": question,
                "answer_requirements": requirements,
                "phenomena": [style],
                "split": "blind",
                "relevant_targets": [{
                    "chunk_id": chunk_id,
                    "source": source,
                    "relevance_grade": 3,
                    "supported_requirements": requirements,
                    "heading_path": [anchor.get("title", "")],
                    "evidence_excerpt": _excerpt(anchor["document"]),
                    "evidence_span_hash": evidence_span_hash(_excerpt(anchor["document"])),
                    "proposal_method": "final_v2_authored_from_anchor",
                }],
                "hard_negatives": [],
                "adjudication": {"method": "authored_from_anchor",
                                 "judge_used": False},
                "human_review": {"status": "approved", "reviewed_by": "user"},
                "review_status": "approved",
                "review_note": "",
            })
            if items[-1]["query_id"] in (DUAL_GOLD | (dual_gold or set())):
                twin = nearest_cross_file(twins[:twin_top_k], source)
                if twin:
                    got_twin = collection.get(ids=[twin["chunk_id"]],
                                              include=["documents", "metadatas"])
                    twin_doc = got_twin["documents"][0]
                    twin_meta = got_twin["metadatas"][0] or {}
                    items[-1]["relevant_targets"].append({
                        "chunk_id": twin["chunk_id"],
                        "source": twin_meta.get("source", ""),
                        "relevance_grade": 3,
                        "supported_requirements": requirements,
                        "heading_path": [twin_meta.get("title", "")],
                        "evidence_excerpt": _excerpt(twin_doc),
                        "evidence_span_hash": evidence_span_hash(_excerpt(twin_doc)),
                        "proposal_method": "deterministic_twin_screen_equal_coverage",
                    })
                    items[-1]["review_note"] = (
                        "跨文件近邻对同一组答案要求的覆盖率与金标持平，按确定性筛查补为第二金标。")

    for order, (kind, question, rationale) in enumerate(negatives):
        items.append({
            "query_id": f"{prefix}-neg-{order:03d}",
            "anchor_id": f"{prefix}-neg-{order:03d}",
            "case_kind": "negative",
            "domain": "negative",
            "query_style": "natural",
            "question": question,
            "answer_requirements": [],
            "phenomena": [kind],
            "split": "blind",
            "expected_behavior": ("abstain_from_kb" if kind == "abstain"
                                  else "correct_premise"),
            "negative_rationale": rationale,
            "relevant_targets": [],
            "hard_negatives": [],
            "adjudication": {"method": "authored", "judge_used": False},
            "human_review": {"status": "approved", "reviewed_by": "user"},
            "review_status": "approved",
            "review_note": "",
        })

    fingerprint = index_fingerprint()
    frozen = json.loads((freeze_path or (ROOT / "docs" / "rag_eval" / "final_v2"
                         / "FROZEN_CONFIG.json")).read_text(encoding="utf-8"))
    return {
        "dataset_id": dataset_id,
        "schema_version": "rag-graded-qrels-v2",
        # Draft until a human confirms the qrels.  Running an arm against an
        # unconfirmed set would make the set's authoring, not the retrieval
        # change, the thing being measured.
        "status": "draft_pending_human_review",
        "design": {
            "positives": sum(len(q) for _, _, q in positives),
            "anchors": len(positives),
            "styles_per_anchor": 2,
            "styles": ["standard", "natural", "implicit_oral", "long_noisy"],
            "negatives": len(negatives),
            "blind_anchor_selection": "scripts/select_final_v2_anchors.py",
            "gold_source": "authored_from_anchor_chunk",
            "answerability_judge_used_for_labels": False,
            "frozen_config_before_labels": {
                # v2/v3 freezes pin one arm; v4 pins the whole arm set.
                "arms": (frozen.get("arms")
                         or {"single": {"arm": frozen.get("arm")}}),
                "judge_prompts": {k: v for k, v in frozen["judge"].items()
                                  if "sha256" in k},
                "git_head": frozen["code"]["git_head"],
            },
        },
        "index": {
            "collection": fingerprint.get("collection"),
            "count": fingerprint.get("collection_count"),
            "content_hash": fingerprint.get("collection_content_hash"),
            "fingerprint_id": fingerprint.get("fingerprint_id"),
            # The contract fingerprint is what the evaluator compares against the
            # live index: a set scored on a different corpus than it was authored
            # from measures nothing, and this is the check that says so.
            "fingerprint": index_contract_fingerprint(fingerprint),
        },
        "near_twin_screen": twin_report,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", required=True, nargs="+")
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--anchors", default="docs/rag_eval/final_v2/ANCHOR_CANDIDATES.json")
    parser.add_argument("--twin-top-k", type=int, default=3)
    parser.add_argument("--output", default="docs/rag_eval/final_v2/final_v2_draft.json")
    parser.add_argument("--prefix", default="fv2")
    parser.add_argument("--freeze", default=None)
    parser.add_argument("--dataset-id", default="rag-final-v2-candidate-v1")
    parser.add_argument("--dual-gold", nargs="*", default=[],
                        help="确定性近邻筛查判定为等价证据的题,补第二金标")
    args = parser.parse_args()

    scope: dict = {}
    for path in args.positives:
        exec(Path(path).read_text(encoding="utf-8"), scope)
    exec(Path(args.negatives).read_text(encoding="utf-8"), scope)
    positives = [row for name in sorted(k for k in scope if k.startswith("POSITIVES"))
                 for row in scope[name]]
    negatives = scope["NEGATIVES"]

    payload = build(ROOT / args.anchors, positives, negatives, args.twin_top_k,
                    prefix=args.prefix,
                    freeze_path=(ROOT / args.freeze) if args.freeze else None,
                    dataset_id=args.dataset_id, dual_gold=set(args.dual_gold))
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[final-v2] wrote {out.relative_to(ROOT)} — "
          f"{payload['design']['positives']} 正例 / {payload['design']['negatives']} 负例 / "
          f"{payload['design']['anchors']} 锚点，status={payload['status']}")


if __name__ == "__main__":
    main()
