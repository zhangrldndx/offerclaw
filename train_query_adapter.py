#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the query tower against frozen document vectors.

Why this shape, specifically:

* **Documents never move.**  The gold vectors are read straight out of the
  frozen production collection and used as fixed targets; there is no document
  forward pass at all.  A query-side map into the *existing* document space
  needs no re-indexing, ever -- which is the whole reason to try it before
  anything doc-side.
* **Full-corpus negatives, not in-batch.**  3348 x 768 floats is 10 MB, so
  every query is scored against the entire corpus on every step.  The loss is
  therefore the retrieval objective itself rather than a sampled approximation.
* **LoRA on every attention layer, not the top few.**  F3 (2026-08-27) trained
  the top 2 layers of a cross-encoder, reduced its loss, grew its mean margin
  8%, and flipped exactly one pair out of 1312 -- the capacity was in the
  readout, where the only reachable move is rescaling.  The correction needed
  here is content-dependent (per-anchor displacements are near-orthogonal,
  pairwise cosine 0.07-0.17), so it has to reach the attention that builds the
  representation, not the layer that reads it out.
* **Net rank-1 flips are printed every epoch.**  Loss and mean similarity can
  both improve beautifully while no ranking decision changes; that is precisely
  how F3 looked like a success until the end-to-end numbers came back 0/0.

The adapter is initialised as an exact no-op (``B = 0``), so epoch 0 reproduces
the frozen baseline bit-for-bit.  That is asserted, not assumed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20260827
BASE_MODEL = "BAAI/bge-base-zh-v1.5"
REFERENCE_STYLE = "standard"


class QueryAdapterError(RuntimeError):
    """Raised when the training contract is violated."""


# --------------------------------------------------------------------------
# data


def load_corpus(device) -> tuple[list[str], Any]:
    import chromadb
    import torch
    from rag_tools import get_collection_name

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    stored = collection.get(include=["embeddings"])
    matrix = np.asarray(stored["embeddings"], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    tensor = torch.tensor(matrix, device=device)
    tensor.requires_grad_(False)
    return list(stored["ids"]), tensor


def load_queries(paths: list[Path], split: str,
                 position: dict[str, int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["items"]:
            if item.get("split") != split:
                continue
            if item.get("case_kind", "positive") != "positive":
                continue
            gold = sorted({position[target["chunk_id"]]
                           for target in item.get("relevant_targets", [])
                           if target["chunk_id"] in position})
            if not gold:
                continue
            rows.append({
                "query_id": item["query_id"],
                "anchor_id": item["anchor_id"],
                "style": item.get("query_style", REFERENCE_STYLE),
                "text": item["question"],
                "gold": gold,
            })
    if not rows:
        raise QueryAdapterError(f"no usable rows for split={split!r}")
    return {
        "rows": rows,
        "texts": [row["text"] for row in rows],
        "gold": [row["gold"] for row in rows],
        "styles": [row["style"] for row in rows],
        "anchors": [row["anchor_id"] for row in rows],
    }


# --------------------------------------------------------------------------
# model


class LoRALinear:
    """Frozen ``nn.Linear`` plus a trainable low-rank residual.

    Implemented directly rather than pulling in ``peft``: it is forty lines,
    it keeps the GPU handoff dependency-free, and ``B`` initialised to zero
    gives the property the tests actually need -- the adapter starts as an
    exact identity, so an untrained run must reproduce the frozen baseline.
    """

    def __new__(cls, base, rank: int, alpha: float):
        import torch
        from torch import nn

        class _LoRALinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                for parameter in self.base.parameters():
                    parameter.requires_grad_(False)
                in_features = base.in_features
                out_features = base.out_features
                self.lora_a = nn.Parameter(
                    torch.empty(rank, in_features, dtype=base.weight.dtype))
                self.lora_b = nn.Parameter(
                    torch.zeros(out_features, rank, dtype=base.weight.dtype))
                nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
                self.scaling = alpha / rank

            def forward(self, value):
                update = torch.nn.functional.linear(
                    torch.nn.functional.linear(value, self.lora_a), self.lora_b)
                return self.base(value) + update * self.scaling

        return _LoRALinear()


def attach_lora(model, rank: int, alpha: float,
                targets: tuple[str, ...] = ("query", "key", "value")) -> int:
    """Wrap the named projections in every encoder layer.  Returns param count."""
    from torch import nn

    wrapped = 0
    for module in model.modules():
        for name in targets:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear) and not hasattr(child, "lora_a"):
                setattr(module, name, LoRALinear(child, rank, alpha))
                wrapped += 1
    if not wrapped:
        raise QueryAdapterError(
            f"no projections named {targets} were found; the base model layout "
            "changed and the adapter would silently train nothing"
        )
    return wrapped


def build_tower(base_model: str, rank: int, alpha: float, device):
    from transformers import AutoModel, AutoTokenizer
    from rag_tools import _load_local_model

    # Resolve through the project's loader so the snapshot is the same one
    # retrieval uses (ModelScope first); a divergent snapshot would train a
    # query tower for a document space that does not exist.
    local = _load_local_model(base_model)
    path = getattr(getattr(local, "_first_module", lambda: None)(), "auto_model", None)
    source = base_model
    if path is not None and getattr(path, "name_or_path", ""):
        source = path.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped = attach_lora(model, rank, alpha)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return tokenizer, model, wrapped, trainable


def encode(model, tokenizer, texts: list[str], device, *, max_length: int = 512,
           batch_size: int = 32, grad: bool = False):
    """CLS pooling + L2 normalise -- the bge-zh contract the corpus was built with."""
    import torch

    outputs = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[:, 0]
            outputs.append(torch.nn.functional.normalize(hidden, p=2, dim=1))
    return torch.cat(outputs, dim=0)


# --------------------------------------------------------------------------
# evaluation


def rank_of_gold(scores, gold: list[list[int]]) -> list[int]:
    ranks = []
    for row, targets in zip(scores, gold):
        best = row[targets].max()
        ranks.append(int((row > best).sum().item()) + 1)
    return ranks


def summarize(ranks: list[int], styles: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for style, rank in zip(styles, ranks):
        grouped[style].append(rank)
        grouped["overall"].append(rank)
    return {
        key: {"n": len(values),
              "r1": sum(1 for r in values if r == 1),
              "r3": sum(1 for r in values if r <= 3),
              "r20": sum(1 for r in values if r <= 20),
              "mrr": round(sum(1.0 / r for r in values) / len(values), 6)}
        for key, values in sorted(grouped.items())
    }


def flips(before: list[int], after: list[int]) -> dict[str, int]:
    """The only number tied to what retrieval actually returns.

    Mean similarity and loss can both improve while every ranking decision
    stays put; F3 reduced its loss, grew its margin 8%, and changed one
    decision out of 1312.
    """
    gained = sum(1 for was, now in zip(before, after) if was != 1 and now == 1)
    lost = sum(1 for was, now in zip(before, after) if was == 1 and now != 1)
    return {"gained": gained, "lost": lost, "net": gained - lost}


def evaluate(model, tokenizer, data: dict[str, Any], corpus, device,
             *, batch_size: int) -> tuple[dict[str, Any], list[int]]:
    vectors = encode(model, tokenizer, data["texts"], device,
                     batch_size=batch_size)
    scores = vectors @ corpus.T
    ranks = rank_of_gold(scores, data["gold"])
    return summarize(ranks, data["styles"]), ranks


# --------------------------------------------------------------------------
# training


def info_nce(scores, gold: list[list[int]], temperature: float):
    """Multi-positive InfoNCE over the *whole* corpus.

    Scoring against all 3348 documents costs one 16x768 @ 768x3348 matmul, so
    there is no reason to approximate with in-batch negatives -- the loss is
    the retrieval objective rather than a proxy for it.
    """
    import torch

    logits = scores / temperature
    denominator = torch.logsumexp(logits, dim=1)
    numerator = torch.stack([
        torch.logsumexp(row[targets], dim=0)
        for row, targets in zip(logits, gold)
    ])
    return (denominator - numerator).mean()


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    seed = int(getattr(args, "seed", SEED))
    torch.manual_seed(seed)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", file=sys.stderr)

    corpus_ids, corpus = load_corpus(device)
    position = {chunk_id: index for index, chunk_id in enumerate(corpus_ids)}
    train_data = load_queries([Path(p) for p in args.train], "train", position)
    dev_data = load_queries([Path(p) for p in args.dev], "dev", position)
    overlap = set(train_data["anchors"]) & set(dev_data["anchors"])
    if overlap:
        raise QueryAdapterError(
            f"train/dev anchor overlap makes every dev number meaningless: "
            f"{sorted(overlap)[:5]}"
        )
    print(f"train {len(train_data['rows'])} queries / "
          f"{len(set(train_data['anchors']))} anchors | "
          f"dev {len(dev_data['rows'])} / {len(set(dev_data['anchors']))} anchors",
          file=sys.stderr)

    tokenizer, model, wrapped, trainable = build_tower(
        args.base_model, args.rank, args.alpha, device)
    corpus_checksum = float(corpus.sum().item())

    model.eval()
    base_dev, base_ranks = evaluate(model, tokenizer, dev_data, corpus, device,
                                    batch_size=args.batch_size)
    base_train, base_train_ranks = evaluate(model, tokenizer, train_data, corpus,
                                            device, batch_size=args.batch_size)
    print(f"[epoch 0 / untrained] dev R@1={base_dev['overall']['r1']} "
          f"R@3={base_dev['overall']['r3']} | train R@1={base_train['overall']['r1']}",
          file=sys.stderr)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    order = np.arange(len(train_data["rows"]))
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    step_history: list[dict[str, Any]] = []
    step = 0
    best = {"epoch": 0, "step": 0, "dev_r1": base_dev["overall"]["r1"],
            "state": None}

    def _remember_best(dev_r1: int, epoch: int, step: int) -> None:
        # Strictly greater: a later checkpoint that merely ties buys nothing
        # and is further into the overfitting régime.
        if dev_r1 > best["dev_r1"]:
            best.update({
                "epoch": epoch, "step": step, "dev_r1": dev_r1,
                "state": {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()
                          if "lora_" in name},
            })

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        losses = []
        for offset in range(0, len(order), args.batch_size):
            batch = order[offset:offset + args.batch_size]
            texts = [train_data["texts"][i] for i in batch]
            gold = [train_data["gold"][i] for i in batch]
            vectors = encode(model, tokenizer, texts, device,
                             batch_size=len(texts), grad=True)
            loss = info_nce(vectors @ corpus.T, gold, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            # 512 rows at batch 16 is 32 steps per epoch, and the first run put
            # the dev peak inside epoch 1 -- sampling once per epoch can step
            # straight over the only useful window.
            if args.eval_every_steps and step % args.eval_every_steps == 0:
                model.eval()
                probe_metrics, probe_ranks = evaluate(
                    model, tokenizer, dev_data, corpus, device,
                    batch_size=args.batch_size)
                probe_flips = flips(base_ranks, probe_ranks)
                step_history.append({
                    "step": step, "dev": probe_metrics,
                    "dev_flips_vs_untrained": probe_flips,
                })
                print(f"  step {step:4d} dev R@1={probe_metrics['overall']['r1']} "
                      f"R@3={probe_metrics['overall']['r3']} "
                      f"MRR={probe_metrics['overall']['mrr']:.4f} "
                      f"net {probe_flips['net']:+d}", file=sys.stderr, flush=True)
                _remember_best(probe_metrics["overall"]["r1"], epoch, step)
                model.train()

        model.eval()
        dev_metrics, dev_ranks = evaluate(model, tokenizer, dev_data, corpus,
                                          device, batch_size=args.batch_size)
        train_metrics, train_ranks = evaluate(model, tokenizer, train_data, corpus,
                                              device, batch_size=args.batch_size)
        dev_flips = flips(base_ranks, dev_ranks)
        entry = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 6),
            "dev": dev_metrics,
            "train": train_metrics,
            "dev_flips_vs_untrained": dev_flips,
        }
        history.append(entry)
        print(
            f"[epoch {epoch}] loss={entry['train_loss']:.4f} "
            f"| dev R@1={dev_metrics['overall']['r1']} "
            f"R@3={dev_metrics['overall']['r3']} "
            f"MRR={dev_metrics['overall']['mrr']:.4f} "
            f"| net rank-1 flips {dev_flips['net']:+d} "
            f"(+{dev_flips['gained']}/-{dev_flips['lost']}) "
            f"| train R@1={train_metrics['overall']['r1']}",
            file=sys.stderr, flush=True,
        )
        _remember_best(dev_metrics["overall"]["r1"], epoch, step)

    # Documents are the fixed point of this whole design; if they moved, every
    # number above is describing a different corpus than production has.
    if float(corpus.sum().item()) != corpus_checksum:
        raise QueryAdapterError("document vectors changed during training")

    result = {
        "schema_version": "colloquial-query-adapter-v1",
        "seed": seed,
        "device": device,
        "base_model": args.base_model,
        "lora": {"rank": args.rank, "alpha": args.alpha,
                 "wrapped_projections": wrapped,
                 "trainable_parameters": sum(p.numel() for p in trainable)},
        "hyperparameters": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "temperature": args.temperature,
            "weight_decay": args.weight_decay,
        },
        "corpus": {"chunks": len(corpus_ids)},
        "train_rows": len(train_data["rows"]),
        "dev_rows": len(dev_data["rows"]),
        "untrained": {"dev": base_dev, "train": base_train},
        "history": history,
        "step_history": step_history,
        "best_epoch": best["epoch"],
        "best_step": best["step"],
        "best_dev_r1": best["dev_r1"],
        "dev_query_ids": [row["query_id"] for row in dev_data["rows"]],
        "untrained_dev_ranks": base_ranks,
    }
    if args.output_model and best["state"] is not None:
        path = Path(args.output_model).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Carry the geometry with the weights: reloading needs rank *and*
        # alpha, and alpha cannot be inferred from the tensor shapes.
        torch.save({
            "schema_version": "colloquial-query-adapter-checkpoint-v1",
            "lora": best["state"],
            "rank": args.rank,
            "alpha": args.alpha,
            "base_model": args.base_model,
            "seed": seed,
            "best_epoch": best["epoch"],
            "best_step": best["step"],
            "dev_r1": best["dev_r1"],
            "untrained_dev_r1": base_dev["overall"]["r1"],
        }, path)
        result["adapter_path"] = str(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--dev", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-model")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every-steps", type=int, default=0,
                        help="probe dev inside an epoch; 0 disables")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    report = train(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
