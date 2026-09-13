#!/usr/bin/env python3
"""Train the frozen MiniLM four-grade answerability student.

The command is dry-run by default.  It refuses blind rows, runs the pilot data
gate before loading a model, freezes token embeddings, trains every transformer
block, and selects checkpoints by validation ranking/flip metrics rather than
loss or mean margin.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from rag_answerability_student import (
    DEFAULT_MODEL_NAME, MODEL_MANIFEST_SCHEMA, checkpoint_tree_sha256,
)
from rag_answerability_student_data import (
    StudentDataError, audit_pilot, build_strong_pairs, pair_flip_report,
    read_jsonl, training_rows,
)


SEEDS = (17, 29, 43)
MAX_LENGTH = 384
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_EPOCHS = 4
EARLY_STOP_PATIENCE = 2
PAIRWISE_WEIGHT = 0.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise StudentDataError("output directory must not exist or must be empty")
    path.mkdir(parents=True, exist_ok=True)


def _resolve_base_model(value: str) -> str:
    expanded = Path(value).expanduser()
    if expanded.is_dir():
        return str(expanded.resolve())
    cached = Path.home() / ".cache" / "modelscope" / "hub" / "models" / value
    if cached.is_dir():
        return str(cached.resolve())
    return value


def _freeze_token_embeddings(model) -> dict[str, int]:
    embedding = model.get_input_embeddings()
    for parameter in embedding.parameters():
        parameter.requires_grad = False
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def _class_weights(rows: list[dict[str, Any]], torch_module):
    counts = Counter(int(row["teacher_grade"]) for row in rows)
    total = sum(counts.values())
    weights = [total / (4 * counts[grade]) if counts[grade] else 0.0 for grade in range(4)]
    return torch_module.tensor(weights, dtype=torch_module.float32), dict(counts)


def _sample_epoch_pairs(pairs: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    if not pairs:
        return []
    weights = [float(pair["sampling_weight"]) for pair in pairs]
    return rng.choices(pairs, weights=weights, k=len(pairs))


def _batch_logits(tokenizer, model, device: str, rows: list[dict[str, Any]], *, torch_module):
    encoded = tokenizer(
        [row["question"] for row in rows],
        [row["chunk_text"] for row in rows],
        padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded).logits


def _score_rows(tokenizer, model, device: str, rows: list[dict[str, Any]], *, torch_module):
    scores: dict[tuple[str, str], float] = {}
    grades: dict[tuple[str, str], int] = {}
    model.eval()
    with torch_module.inference_mode():
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            logits = _batch_logits(tokenizer, model, device, batch, torch_module=torch_module)
            probs = torch_module.softmax(logits.float(), dim=-1).cpu()
            for row, values in zip(batch, probs.tolist()):
                key = (row["query_id"], row["chunk_id"])
                scores[key] = sum(index * float(value) for index, value in enumerate(values))
                grades[key] = max(range(4), key=lambda index: values[index])
    return scores, grades


def _ranking_metrics(rows: list[dict[str, Any]], scores: dict[tuple[str, str], float]):
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    hits = {1: 0, 3: 0, 5: 0}
    ndcg_total = 0.0
    eligible = 0
    top_chunks: dict[str, str] = {}
    for query_id, group in by_query.items():
        if not any(row["teacher_grade"] == 3 for row in group):
            continue
        ranked = sorted(
            group,
            key=lambda row: (
                -scores.get((query_id, row["chunk_id"]), float("-inf")),
                row["current_bge_rank"],
            ),
        )
        eligible += 1
        top_chunks[query_id] = ranked[0]["chunk_id"]
        for cutoff in hits:
            hits[cutoff] += int(any(row["teacher_grade"] == 3 for row in ranked[:cutoff]))
        gains = [(2 ** int(row["teacher_grade"]) - 1) for row in ranked[:5]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sorted((2 ** int(row["teacher_grade"]) - 1 for row in group), reverse=True)[:5]
        idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
        ndcg_total += dcg / idcg if idcg else 0.0
    return {
        "n": eligible,
        "r1": hits[1] / eligible if eligible else 0.0,
        "r3": hits[3] / eligible if eligible else 0.0,
        "r5": hits[5] / eligible if eligible else 0.0,
        "r1_hits": hits[1], "r3_hits": hits[3], "r5_hits": hits[5],
        "ndcg5": ndcg_total / eligible if eligible else 0.0,
        "top_chunks": top_chunks,
    }


def _base_scores(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    return {
        (row["query_id"], row["chunk_id"]): float(row["base_student_score"])
        for row in rows if row.get("base_student_score") is not None
    }


def evaluate(tokenizer, model, device: str, rows: list[dict[str, Any]], pairs, *, torch_module):
    post_scores, predicted_grades = _score_rows(
        tokenizer, model, device, rows, torch_module=torch_module,
    )
    base = _ranking_metrics(rows, _base_scores(rows))
    post = _ranking_metrics(rows, post_scores)
    wins = losses = 0
    for query_id in set(base["top_chunks"]) & set(post["top_chunks"]):
        group = [row for row in rows if row["query_id"] == query_id]
        grade_by_chunk = {row["chunk_id"]: row["teacher_grade"] for row in group}
        before = grade_by_chunk[base["top_chunks"][query_id]] == 3
        after = grade_by_chunk[post["top_chunks"][query_id]] == 3
        wins += int(after and not before)
        losses += int(before and not after)
    flips = pair_flip_report(pairs, post_scores)
    corrective_n = sum(1 for pair in pairs if pair["corrective"])
    stable_n = sum(1 for pair in pairs if not pair["corrective"])
    return {
        "base": {key: value for key, value in base.items() if key != "top_chunks"},
        "post": {key: value for key, value in post.items() if key != "top_chunks"},
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "pair_flips": {key: value for key, value in flips.items() if key != "rows"},
        "corrective_pair_count": corrective_n,
        "corrective_flip_rate": flips["corrected"] / corrective_n if corrective_n else 0.0,
        "stability_regression_rate": flips["regressed"] / stable_n if stable_n else 0.0,
        "predicted_grade_counts": dict(Counter(predicted_grades.values())),
    }


def _selection_tuple(metrics: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(metrics["post"]["r1"]),
        float(metrics["post"]["ndcg5"]),
        int(metrics["pair_flips"]["net_flips"]),
    )


def train(*, dataset_path: Path, output: Path, base_model: str, seed: int,
          device_requested: str | None = None) -> dict[str, Any]:
    if seed not in SEEDS:
        raise StudentDataError(f"seed must be one of {SEEDS}")
    rows = training_rows(read_jsonl(dataset_path))
    pilot = audit_pilot(rows)
    if pilot["status"] != "go_train_pilot":
        raise StudentDataError("pilot data gate failed; refusing to load/train a model")
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    train_pairs = [pair for pair in build_strong_pairs(train_rows) if pair["split"] == "train"]
    validation_pairs = [
        pair for pair in build_strong_pairs(validation_rows)
        if pair["split"] == "validation"
    ]
    if not train_pairs or not validation_pairs:
        raise StudentDataError("train and validation both require strong pairs")

    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved = _resolve_base_model(base_model)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForSequenceClassification.from_pretrained(
        resolved, num_labels=4, ignore_mismatched_sizes=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    parameter_stats = _freeze_token_embeddings(model)
    if device_requested:
        device = device_requested
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    class_weights, class_counts = _class_weights(train_rows, torch)
    class_weights = class_weights.to(device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    grade2_rows = [row for row in train_rows if row["teacher_grade"] == 2]
    micro_steps = len(train_pairs) + len(grade2_rows)
    optimizer_steps = max(1, math.ceil(micro_steps / GRADIENT_ACCUMULATION) * MAX_EPOCHS)
    warmup_steps = int(round(optimizer_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = random.Random(seed)
    history = []
    best_tuple: tuple[float, float, int] | None = None
    best_state = None
    patience = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_items: list[tuple[str, Any]] = [
            ("pair", pair) for pair in _sample_epoch_pairs(train_pairs, rng)
        ] + [("point", row) for row in grade2_rows]
        rng.shuffle(epoch_items)
        running_loss = 0.0
        for micro_index, (kind, item) in enumerate(epoch_items, 1):
            amp_context = torch.autocast(
                device_type="cuda", dtype=torch.float16,
            ) if use_amp else nullcontext()
            with amp_context:
                if kind == "pair":
                    pair = item
                    batch = [pair["positive"], pair["negative"]]
                    logits = _batch_logits(tokenizer, model, device, batch, torch_module=torch)
                    labels = torch.tensor(
                        [row["teacher_grade"] for row in batch],
                        dtype=torch.long, device=device,
                    )
                    point_loss = ce_loss(logits, labels)
                    probs = torch.softmax(logits.float(), dim=-1)
                    grade_axis = torch.arange(4, device=device, dtype=probs.dtype)
                    expected = (probs * grade_axis).sum(dim=-1)
                    pair_loss = torch.nn.functional.softplus(-(expected[0] - expected[1]))
                    loss = point_loss + PAIRWISE_WEIGHT * pair_loss
                else:
                    logits = _batch_logits(
                        tokenizer, model, device, [item], torch_module=torch,
                    )
                    label = torch.tensor([2], dtype=torch.long, device=device)
                    loss = ce_loss(logits, label)
                scaled_loss = loss / GRADIENT_ACCUMULATION
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())
            if micro_index % GRADIENT_ACCUMULATION == 0 or micro_index == len(epoch_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        train_metrics = evaluate(
            tokenizer, model, device, train_rows, train_pairs, torch_module=torch,
        )
        validation_metrics = evaluate(
            tokenizer, model, device, validation_rows, validation_pairs,
            torch_module=torch,
        )
        candidate_tuple = _selection_tuple(validation_metrics)
        improved = best_tuple is None or candidate_tuple > best_tuple
        history.append({
            "epoch": epoch,
            "mean_training_loss_diagnostic_only": running_loss / max(1, len(epoch_items)),
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_tuple": list(candidate_tuple),
            "best_so_far": improved,
        })
        print(json.dumps({
            "epoch": epoch,
            "train_net_pair_flips": train_metrics["pair_flips"]["net_flips"],
            "validation_net_pair_flips": validation_metrics["pair_flips"]["net_flips"],
            "validation_r1": validation_metrics["post"]["r1"],
            "validation_wins": validation_metrics["wins"],
            "validation_losses": validation_metrics["losses"],
        }, ensure_ascii=False), flush=True)
        if improved:
            best_tuple = candidate_tuple
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    if best_state is None:
        raise StudentDataError("training produced no checkpoint")
    model.load_state_dict(best_state)
    final_train = evaluate(tokenizer, model, device, train_rows, train_pairs, torch_module=torch)
    final_validation = evaluate(
        tokenizer, model, device, validation_rows, validation_pairs, torch_module=torch,
    )
    _prepare_output(output)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    checkpoint_hash = checkpoint_tree_sha256(output)
    train_pair_acc = (
        final_train["pair_flips"]["stable_correct"] + final_train["pair_flips"]["corrected"]
    ) / max(1, final_train["pair_flips"]["n"])
    validation_pair_acc = (
        final_validation["pair_flips"]["stable_correct"]
        + final_validation["pair_flips"]["corrected"]
    ) / max(1, final_validation["pair_flips"]["n"])
    pilot_checks = {
        "validation_r1_gain_gte_0_05": (
            final_validation["post"]["r1"] - final_validation["base"]["r1"] >= 0.05
        ),
        "validation_net_wins_gte_3": final_validation["net_wins"] >= 3,
        "corrective_flip_rate_gte_0_20": final_validation["corrective_flip_rate"] >= 0.20,
        "stability_regression_rate_lte_0_05": final_validation["stability_regression_rate"] <= 0.05,
        "r3_regression_lte_1": (
            final_validation["post"]["r3_hits"]
            >= final_validation["base"]["r3_hits"] - 1
        ),
        "r5_regression_lte_1": (
            final_validation["post"]["r5_hits"]
            >= final_validation["base"]["r5_hits"] - 1
        ),
        "train_validation_pair_accuracy_gap_lte_0_10": (
            train_pair_acc - validation_pair_acc <= 0.10
        ),
    }
    manifest = {
        "schema_version": MODEL_MANIFEST_SCHEMA,
        "release_status": "development",
        "checkpoint_sha256": checkpoint_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "base_model_resolved": resolved,
        "dataset": {
            "name": dataset_path.name,
            "bytes": dataset_path.stat().st_size,
            "sha256": _sha256(dataset_path),
            "blind_rows_consumed": 0,
        },
        "seed": seed,
        "frozen_hyperparameters": {
            "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO, "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "pairwise_weight": PAIRWISE_WEIGHT,
            "token_embeddings_frozen": True,
            "all_transformer_blocks_trainable": True,
        },
        "device": device,
        "class_counts": class_counts,
        "parameters": parameter_stats,
        "pilot_data_gate": pilot,
        "history": history,
        "final_train": final_train,
        "final_validation": final_validation,
        "pilot_model_checks": pilot_checks,
        "pilot_model_status": (
            "seed_pass" if all(pilot_checks.values()) else "seed_no_go"
        ),
        "selection_uses": ["validation_r1", "validation_ndcg5", "net_pair_flips"],
        "selection_forbids": ["blind_a", "blind_b", "loss_only", "mean_margin_only"],
    }
    (output / "student_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    train_only = training_rows(rows)
    pilot = audit_pilot(train_only)
    preview = {
        "status": "validated_not_trained",
        "dataset": str(args.dataset),
        "seed": args.seed,
        "base_model": args.base_model,
        "pilot": pilot,
        "frozen_hyperparameters": {
            "max_length": MAX_LENGTH,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
        },
    }
    if not args.run_training:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    manifest = train(
        dataset_path=args.dataset, output=args.output,
        base_model=args.base_model, seed=args.seed,
        device_requested=args.device,
    )
    print(json.dumps({
        "status": "trained",
        "output": str(args.output),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "pilot_model_status": manifest["pilot_model_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
