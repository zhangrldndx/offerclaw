#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feasibility probe for a query-side adapter over frozen document vectors.

The question this answers is narrow and cheap: *is there a query-side
transform, learned only from Train, that moves held-out colloquial questions
closer to their gold chunk without touching a single document vector?*

Design constraints that make the answer trustworthy:

* Document vectors are read from the frozen production collection and never
  recomputed or rewritten.  A query-side map needs no re-indexing at all,
  which is the whole reason to try it before anything doc-side.
* Train and Dev anchors are disjoint, so a Dev gain cannot be memorisation of
  a specific gold chunk.
* Every configuration is scored against two null controls.  With d=768 and
  only a few hundred training rows, a regression can "improve" a metric purely
  by shrinking or rotating the space, so an uncontrolled number here would be
  worthless:

  ``shuffled``  the same fit on permuted (query, gold) pairs -- destroys the
                pairing but keeps the marginals.  Any gain it shows is an
                artefact of the fit, not of learned alignment.
  ``identity``  no adapter.  The baseline the adapter must beat.

Ridge regression is used rather than gradient training because at this sample
size the closed form is exact, instant, and has one interpretable knob.  A
negative result here is informative; a positive result is a licence to build
the real contrastive version, not a model to ship.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20260827
REFERENCE_STYLE = "standard"
LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
# Blend the adapted query back toward the original.  alpha=0 is the identity;
# a full replacement (alpha=1) throws away a query encoder that is already
# right about most questions, so the useful region is usually interior.
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def _positives(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in payload["items"]
            if item.get("case_kind", "positive") == "positive"
            and (split is None or item.get("split") == split)]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def fit_ridge(queries: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge:  W = (Q'Q + lam I)^-1 Q'G."""
    dim = queries.shape[1]
    gram = queries.T @ queries + lam * np.eye(dim)
    return np.linalg.solve(gram, queries.T @ targets)


def apply_adapter(queries: np.ndarray, matrix: np.ndarray | None,
                  alpha: float) -> np.ndarray:
    if matrix is None or alpha == 0.0:
        return _normalize(queries)
    adapted = _normalize(queries @ matrix)
    return _normalize((1.0 - alpha) * queries + alpha * adapted)


def rank_metrics(queries: np.ndarray, corpus: np.ndarray,
                 gold_indices: list[set[int]],
                 styles: list[str]) -> dict[str, Any]:
    scores = queries @ corpus.T
    ranks: list[int] = []
    for row, gold in zip(scores, gold_indices):
        # rank of the best-placed gold: how many chunks outscore it, +1
        best = max(row[i] for i in gold)
        ranks.append(int((row > best).sum()) + 1)
    out: dict[str, Any] = {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for style, rank in zip(styles, ranks):
        grouped[style].append(rank)
        grouped["__all__"].append(rank)
    for style, group in grouped.items():
        key = "overall" if style == "__all__" else style
        out[key] = {
            "n": len(group),
            "r1": sum(1 for r in group if r == 1),
            "r3": sum(1 for r in group if r <= 3),
            "r20": sum(1 for r in group if r <= 20),
            "mrr": round(sum(1.0 / r for r in group) / len(group), 6),
            "median_rank": int(np.median(group)),
        }
    return {"metrics": out, "ranks": ranks}


def displacement_coherence(vectors: np.ndarray, anchors: list[str],
                           styles: list[str]) -> dict[str, Any]:
    """How much do different anchors need the *same* correction?

    For every anchor that carries both a colloquial and a formal phrasing,
    ``standard_vec - colloquial_vec`` is the displacement a perfect style
    normaliser would have to apply.  A single matrix can only apply one
    direction-consistent transform, so if these displacements point in
    unrelated directions across anchors, no linear map can express the fix and
    the failure of the ridge probe is structural rather than a tuning miss.

    ``mean_pairwise_cosine`` near 0 means mutually orthogonal (content
    dependent); near 1 means one shared "register shift" direction that a
    constant offset would capture.
    """
    reference = {anchor: index for index, (anchor, style)
                 in enumerate(zip(anchors, styles)) if style == REFERENCE_STYLE}
    out: dict[str, Any] = {}
    for style in sorted(set(styles) - {REFERENCE_STYLE}):
        rows = [index for index, (anchor, item_style)
                in enumerate(zip(anchors, styles))
                if item_style == style and anchor in reference]
        if len(rows) < 2:
            continue
        deltas = np.asarray(
            [vectors[reference[anchors[i]]] - vectors[i] for i in rows])
        unit = _normalize(deltas)
        gram = unit @ unit.T
        upper = gram[np.triu_indices(len(rows), k=1)]
        mean_delta = deltas.mean(axis=0)
        # How much of a typical displacement the shared component explains.
        shared = float(np.linalg.norm(mean_delta)
                       / np.linalg.norm(deltas, axis=1).mean())
        out[style] = {
            "pairs": len(rows),
            "mean_pairwise_cosine": round(float(upper.mean()), 6),
            "p90_pairwise_cosine": round(float(np.quantile(upper, 0.9)), 6),
            "shared_component_fraction": round(shared, 6),
            "mean_displacement_norm": round(
                float(np.linalg.norm(deltas, axis=1).mean()), 6),
        }
    return out


def build(train_paths: list[Path], dev_paths: list[Path], *,
          train_split: str | None = None,
          dev_split: str | None = None,
          target: str = "gold") -> dict[str, Any]:
    import chromadb
    from rag_tools import get_collection_name, get_embeddings_batch, index_fingerprint

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    stored = collection.get(include=["embeddings"])
    corpus_ids: list[str] = list(stored["ids"])
    corpus = _normalize(np.asarray(stored["embeddings"], dtype=np.float64))
    position = {chunk_id: i for i, chunk_id in enumerate(corpus_ids)}
    print(f"corpus: {corpus.shape[0]} chunks x {corpus.shape[1]} dims", file=sys.stderr)

    def load(paths: list[Path], split: str | None) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for path in paths:
            items.extend(_positives(path, split))
        if not items:
            raise SystemExit(
                f"no positive rows in {[p.name for p in paths]} for split={split!r}"
            )
        usable, dropped = [], 0
        for item in items:
            gold = [position[t["chunk_id"]] for t in item.get("relevant_targets", [])
                    if t["chunk_id"] in position]
            if not gold:
                dropped += 1
                continue
            usable.append({"item": item, "gold": set(gold)})
        if dropped:
            print(f"dropped {dropped} rows whose gold is absent from the index",
                  file=sys.stderr)
        texts = [row["item"]["question"] for row in usable]
        print(f"embedding {len(texts)} questions ...", file=sys.stderr)
        vectors = _normalize(np.asarray(get_embeddings_batch(texts), dtype=np.float64))
        return {
            "vectors": vectors,
            "gold": [row["gold"] for row in usable],
            "styles": [row["item"].get("query_style", "standard") for row in usable],
            "anchors": [row["item"]["anchor_id"] for row in usable],
            "query_ids": [row["item"]["query_id"] for row in usable],
        }

    train, dev = load(train_paths, train_split), load(dev_paths, dev_split)
    overlap = set(train["anchors"]) & set(dev["anchors"])
    if overlap:
        raise SystemExit(
            f"train/dev anchor overlap would invalidate the probe: {sorted(overlap)[:5]}"
        )

    if target == "gold":
        # Predict the document from the query.  This asks the map to carry
        # topic knowledge, which a single global matrix cannot hold.
        targets = _normalize(np.asarray(
            [corpus[sorted(gold)].mean(axis=0) for gold in train["gold"]],
            dtype=np.float64,
        ))
        rows = list(range(len(train["vectors"])))
    else:
        # Predict the *same anchor's* formal phrasing.  The paired design makes
        # this available for free, and it is a far smaller displacement: the
        # map only has to undo a register shift, not know the answer.  Rows
        # whose anchor has no standard variant, and the standard rows
        # themselves, are excluded -- training the map to fix what is already
        # correct would just teach it the identity.
        reference = {}
        for index, (anchor, style) in enumerate(zip(train["anchors"], train["styles"])):
            if style == REFERENCE_STYLE:
                reference[anchor] = index
        rows = [index for index, (anchor, style)
                in enumerate(zip(train["anchors"], train["styles"]))
                if style != REFERENCE_STYLE and anchor in reference]
        if not rows:
            raise SystemExit(
                f"target=standard_query needs anchors carrying a "
                f"'{REFERENCE_STYLE}' variant alongside other styles"
            )
        targets = train["vectors"][[reference[train["anchors"][i]] for i in rows]]

    # Upper bound for any perfect style normaliser: hand each Dev colloquial
    # query the embedding of its own standard phrasing.  Unreachable in
    # production (it needs the answer's own wording), but it prices the prize.
    dev_reference = {}
    for index, (anchor, style) in enumerate(zip(dev["anchors"], dev["styles"])):
        if style == REFERENCE_STYLE:
            dev_reference[anchor] = index
    oracle = dev["vectors"].copy()
    oracle_rows = 0
    for index, anchor in enumerate(dev["anchors"]):
        if anchor in dev_reference:
            oracle[index] = dev["vectors"][dev_reference[anchor]]
            oracle_rows += 1

    return {"collection": collection, "corpus": corpus, "train": train, "dev": dev,
            "targets": targets, "target_rows": rows, "target": target,
            "oracle_vectors": oracle, "oracle_rows": oracle_rows,
            "index": index_fingerprint(collection=collection)}


def run(train_paths: list[Path], dev_paths: list[Path], *,
        train_split: str | None = None,
        dev_split: str | None = None,
        target: str = "gold") -> dict[str, Any]:
    data = build(train_paths, dev_paths, train_split=train_split,
                 dev_split=dev_split, target=target)
    corpus, train, dev, targets = (data["corpus"], data["train"], data["dev"],
                                   data["targets"])
    fit_inputs = train["vectors"][data["target_rows"]]
    rng = np.random.default_rng(SEED)
    shuffled_targets = targets[rng.permutation(len(targets))]

    baseline_dev = rank_metrics(apply_adapter(dev["vectors"], None, 0.0), corpus,
                                dev["gold"], dev["styles"])
    baseline_train = rank_metrics(apply_adapter(train["vectors"], None, 0.0), corpus,
                                  train["gold"], train["styles"])
    oracle_dev = rank_metrics(data["oracle_vectors"], corpus, dev["gold"],
                              dev["styles"])
    results = [{
        "arm": "identity", "lambda": None, "alpha": 0.0,
        "train": baseline_train["metrics"], "dev": baseline_dev["metrics"],
    }, {
        "arm": "oracle_standard_query", "lambda": None, "alpha": None,
        "train": None, "dev": oracle_dev["metrics"],
        "note": (f"{data['oracle_rows']}/{len(dev['vectors'])} Dev rows replaced "
                 "by their own standard phrasing; upper bound, not reachable"),
    }]

    for label, goal in (("adapter", targets), ("shuffled_control", shuffled_targets)):
        for lam in LAMBDAS:
            matrix = fit_ridge(fit_inputs, goal, lam)
            for alpha in ALPHAS:
                dev_scored = rank_metrics(
                    apply_adapter(dev["vectors"], matrix, alpha), corpus,
                    dev["gold"], dev["styles"])
                train_scored = rank_metrics(
                    apply_adapter(train["vectors"], matrix, alpha), corpus,
                    train["gold"], train["styles"])
                results.append({
                    "arm": label, "lambda": lam, "alpha": alpha,
                    "train": train_scored["metrics"], "dev": dev_scored["metrics"],
                    "dev_ranks": dev_scored["ranks"] if label == "adapter" else None,
                })
                print(f"{label:17s} lam={lam:7.1f} alpha={alpha:.2f}  "
                      f"train R@1={train_scored['metrics']['overall']['r1']:3d} "
                      f"dev R@1={dev_scored['metrics']['overall']['r1']:3d} "
                      f"dev R@20={dev_scored['metrics']['overall']['r20']:3d} "
                      f"dev MRR={dev_scored['metrics']['overall']['mrr']:.4f}",
                      file=sys.stderr)

    best = max((r for r in results if r["arm"] == "adapter"),
               key=lambda r: (r["dev"]["overall"]["r1"], r["dev"]["overall"]["mrr"]))
    best_control = max((r for r in results if r["arm"] == "shuffled_control"),
                       key=lambda r: (r["dev"]["overall"]["r1"],
                                      r["dev"]["overall"]["mrr"]))
    verdict = {
        "target": target,
        "identity_dev_r1": baseline_dev["metrics"]["overall"]["r1"],
        "oracle_standard_query_dev_r1": oracle_dev["metrics"]["overall"]["r1"],
        "best_adapter_dev_r1": best["dev"]["overall"]["r1"],
        "best_shuffled_control_dev_r1": best_control["dev"]["overall"]["r1"],
        "beats_identity": best["dev"]["overall"]["r1"] > baseline_dev["metrics"]["overall"]["r1"],
        # The control is selected on Dev too, so it enjoys the same selection
        # advantage; requiring the adapter to clear it is the honest bar.
        "beats_shuffled_control": (best["dev"]["overall"]["r1"]
                                   > best_control["dev"]["overall"]["r1"]),
        "best_config": {"lambda": best["lambda"], "alpha": best["alpha"]},
    }
    verdict["gate"] = ("go_build_contrastive_version"
                       if verdict["beats_identity"] and verdict["beats_shuffled_control"]
                       else "no_go_linear_query_map_insufficient")

    return {
        "schema_version": "colloquial-query-adapter-probe-v1",
        "seed": SEED,
        "target": target,
        "index": data["index"],
        "train_rows": len(train["vectors"]),
        "dev_rows": len(dev["vectors"]),
        "train_anchors": len(set(train["anchors"])),
        "dev_anchors": len(set(dev["anchors"])),
        "dev_query_ids": dev["query_ids"],
        "dev_styles": dev["styles"],
        "displacement_coherence": {
            "train": displacement_coherence(train["vectors"], train["anchors"],
                                            train["styles"]),
            "dev": displacement_coherence(dev["vectors"], dev["anchors"],
                                          dev["styles"]),
        },
        "results": results,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, action="append")
    parser.add_argument("--dev", required=True, action="append")
    parser.add_argument("--train-split", default="train",
                        help="split label to keep from --train files ('' = all)")
    parser.add_argument("--dev-split", default="dev",
                        help="split label to keep from --dev files ('' = all)")
    parser.add_argument("--target", choices=("gold", "standard_query"),
                        default="gold",
                        help="regress query -> gold chunk, or query -> the same "
                             "anchor's formal phrasing (style normalisation)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = run([Path(p).expanduser().resolve() for p in args.train],
                 [Path(p).expanduser().resolve() for p in args.dev],
                 train_split=args.train_split or None,
                 dev_split=args.dev_split or None,
                 target=args.target)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n" + json.dumps(report["verdict"], ensure_ascii=False, indent=2),
          file=sys.stderr)
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
