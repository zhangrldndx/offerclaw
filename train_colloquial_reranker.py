#!/usr/bin/env python3
"""Train a development-only CrossEncoder on the frozen colloquial triples.

This entry point never invents a validation split: the 48 Train anchors and
16 Dev anchors were separated before wording review and remain disjoint.  The
private Blind80 set is rejected by schema and provenance checks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_colloquial_reranker_training import SCHEMA_VERSION
from build_f3_training_package import SCHEMA_VERSION as F3_SCHEMA_VERSION

# F3 merges two waves and carries per-triple weights, but the row contract the
# trainer consumes is unchanged, so both schemas load through the same path.
SUPPORTED_SCHEMAS = (SCHEMA_VERSION, F3_SCHEMA_VERSION)
from train_reranker_hard_negatives import (
    TrainingInputError,
    assert_output_path_allowed,
    labeled_pairs,
    train_pairwise_cross_encoder,
    triple_kind_counts,
)


ROOT = Path(__file__).resolve().parent


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_colloquial_training_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    lowered_path = str(source).lower()
    if "private_eval" in lowered_path or "blind" in source.name.lower():
        raise TrainingInputError("private Blind data is forbidden")
    artifact = json.loads(source.read_text(encoding="utf-8"))
    if artifact.get("schema_version") not in SUPPORTED_SCHEMAS:
        raise TrainingInputError(
            f"unsupported colloquial training schema: "
            f"{artifact.get('schema_version')!r}"
        )
    if artifact.get("development_only") is not True:
        raise TrainingInputError("training artifact must be development_only")
    if artifact.get("contains_text") is not True:
        raise TrainingInputError("training artifact must contain scoring text")
    if artifact.get("private_blind_set") is not False:
        raise TrainingInputError("private Blind-set training is forbidden")
    provenance = json.dumps(artifact.get("provenance") or {}, ensure_ascii=False).lower()
    if "private_eval" in provenance or "blind_v" in provenance:
        raise TrainingInputError("private Blind provenance is forbidden")
    triples = artifact.get("triples")
    if not isinstance(triples, list) or not triples:
        raise TrainingInputError("training artifact contains no triples")
    for row in triples:
        required = (
            row.get("query_id"), row.get("anchor_id"), row.get("split"),
            row.get("query"),
            (row.get("positive") or {}).get("chunk_id"),
            (row.get("positive") or {}).get("scoring_text"),
            (row.get("negative") or {}).get("chunk_id"),
            (row.get("negative") or {}).get("scoring_text"),
        )
        if not all(required):
            raise TrainingInputError("a triple is missing split/query/chunk scoring text")
        if row["split"] not in {"train", "dev"}:
            raise TrainingInputError(f"unsupported split: {row['split']!r}")
    train_anchors = {row["anchor_id"] for row in triples if row["split"] == "train"}
    dev_anchors = {row["anchor_id"] for row in triples if row["split"] == "dev"}
    if not train_anchors or not dev_anchors:
        raise TrainingInputError("both frozen train and dev anchor groups are required")
    if train_anchors & dev_anchors:
        raise TrainingInputError("anchor leakage between train and dev")
    return artifact


def _split_rows(artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in artifact["triples"] if row["split"] == "train"]
    dev = [row for row in artifact["triples"] if row["split"] == "dev"]
    return train, dev


def _preview(
    artifact: dict[str, Any],
    *,
    output: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    trainable_encoder_layers: int,
) -> dict[str, Any]:
    train, dev = _split_rows(artifact)
    return {
        "development_only": True,
        "private_blind_set_used": False,
        "base_model": base_model,
        "output": str(output),
        "train_triples": len(train),
        "dev_triples": len(dev),
        "train_anchors": len({row["anchor_id"] for row in train}),
        "dev_anchors": len({row["anchor_id"] for row in dev}),
        "train_pairs": len(labeled_pairs(train)),
        "dev_pairs": len(labeled_pairs(dev)),
        "train_triples_by_kind": triple_kind_counts(train),
        "dev_triples_by_kind": triple_kind_counts(dev),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pairwise_margin": margin,
        "trainable_encoder_layers": trainable_encoder_layers,
    }


def _model_card(
    *,
    artifact_path: Path,
    artifact: dict[str, Any],
    preview: dict[str, Any],
    training: dict[str, Any],
) -> str:
    return f"""# OfferClaw colloquial reranker candidate

> `development_only=true` — 仅供 Dev80 对照，不得直接切换生产。

## 数据边界

- 训练数据：`{artifact.get('base_set', '')}`
- 数据 SHA-256：`{_sha256_file(artifact_path)}`
- 原问题集 SHA-256：`{(artifact.get('provenance') or {}).get('cases_sha256', '')}`
- 索引指纹：`{json.dumps((artifact.get('provenance') or {}).get('index') or {}, ensure_ascii=False)}`
- Train / Dev 锚点：{preview['train_anchors']} / {preview['dev_anchors']}
- Blind80：**未读取、未复制、未参与训练**
- 文档输入格式：`compact32`（与当前 Dev80 最佳 arm 一致）

## 训练配置

- Base model: `{preview['base_model']}`
- Objective: `weighted_pairwise_logistic_margin`
- Epochs: {preview['epochs']}
- Batch size: {preview['batch_size']}
- Learning rate: {preview['learning_rate']}
- Pairwise margin: {preview['pairwise_margin']}
- Trainable top encoder layers: {preview['trainable_encoder_layers']}
- Device: `{training.get('device', '')}`
- Optimizer steps: {training.get('optimizer_steps', 0)}
- Train triples / Dev triples: {preview['train_triples']} / {preview['dev_triples']}
- Pre Dev ranking: `{json.dumps(training.get('pre_validation_metrics') or {}, sort_keys=True)}`
- Post Dev ranking: `{json.dumps(training.get('post_validation_metrics') or {}, sort_keys=True)}`

## 下一步

必须用 `eval_colloquial_rag.py --arm compact32 --reranker-model <本目录>`
完整运行 Dev80。只有端到端 R@1、R@3/R@5、Gate、负样本和延迟同时通过，才可把
候选模型拷回原电脑运行一次仓外 Blind80。
"""


def run_training(
    *,
    artifact_path: str | Path,
    output_path: str | Path,
    base_model: str = "BAAI/bge-reranker-base",
    epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    seed: int = 20260826,
    margin: float = 0.2,
    trainable_encoder_layers: int = 2,
    same_document_weight: float = 1.2,
    allow_formal_model_path: bool = False,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise TrainingInputError("epochs, batch_size, and learning_rate must be positive")
    if trainable_encoder_layers < 0:
        raise TrainingInputError("trainable_encoder_layers cannot be negative")
    if same_document_weight <= 0:
        raise TrainingInputError("same_document_weight must be positive")
    artifact_file = Path(artifact_path).expanduser().resolve()
    artifact = load_colloquial_training_artifact(artifact_file)
    output = assert_output_path_allowed(
        output_path, allow_formal_model_path=allow_formal_model_path,
    )
    train, dev = _split_rows(artifact)
    preview = _preview(
        artifact,
        output=output,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        margin=margin,
        trainable_encoder_layers=trainable_encoder_layers,
    )
    training = train_pairwise_cross_encoder(
        train_triples=train,
        validation_triples=dev,
        output=output,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        margin=margin,
        kind_weights={
            "adjudicated_hard_negative": 1.0,
            "same_document_wrong_section": same_document_weight,
        },
        trainable_encoder_layers=trainable_encoder_layers,
    )
    manifest = {
        "schema_version": "colloquial-reranker-development-model-v1",
        "development_only": True,
        "private_blind_set_used": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": artifact_file.name,
        "source_artifact_sha256": _sha256_file(artifact_file),
        "source_dataset_id": artifact.get("source_dataset_id"),
        "index": (artifact.get("provenance") or {}).get("index", {}),
        "configuration": preview,
        "training": training,
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "MODEL_CARD.md").write_text(
        _model_card(
            artifact_path=artifact_file,
            artifact=artifact,
            preview=preview,
            training=training,
        ),
        encoding="utf-8",
    )
    (output / "DEVELOPMENT_ONLY").write_text(
        "Not approved for production. Blind80 was not used.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--pairwise-margin", type=float, default=0.2)
    parser.add_argument("--trainable-encoder-layers", type=int, default=2)
    parser.add_argument("--same-document-weight", type=float, default=1.2)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--allow-formal-model-path", action="store_true")
    parser.add_argument(
        "--run-training", action="store_true",
        help="required opt-in; without it only validate and print the frozen split",
    )
    args = parser.parse_args()
    if args.device:
        os.environ["OFFERCLAW_TORCH_DEVICE"] = args.device

    artifact = load_colloquial_training_artifact(args.artifact)
    output = assert_output_path_allowed(
        args.output, allow_formal_model_path=args.allow_formal_model_path,
    )
    preview = _preview(
        artifact,
        output=output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.pairwise_margin,
        trainable_encoder_layers=args.trainable_encoder_layers,
    )
    if not args.run_training:
        print(json.dumps({**preview, "status": "validated_not_trained"}, ensure_ascii=False, indent=2))
        return 0
    manifest = run_training(
        artifact_path=args.artifact,
        output_path=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        margin=args.pairwise_margin,
        trainable_encoder_layers=args.trainable_encoder_layers,
        same_document_weight=args.same_document_weight,
        allow_formal_model_path=args.allow_formal_model_path,
    )
    print(json.dumps({**preview, "status": "trained", "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
