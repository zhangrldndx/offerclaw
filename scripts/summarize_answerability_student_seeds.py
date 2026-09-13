#!/usr/bin/env python3
"""Apply the pre-registered three-seed Pilot decision without blind labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SEEDS = (17, 29, 43)


def summarize(root: Path) -> dict:
    manifests = []
    for seed in EXPECTED_SEEDS:
        path = root / f"seed_{seed}" / "student_model_manifest.json"
        if not path.is_file():
            raise ValueError(f"missing seed manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("seed") != seed:
            raise ValueError(f"seed lineage mismatch in {path}")
        manifests.append((path, payload))
    dataset_hashes = {payload["dataset"]["sha256"] for _path, payload in manifests}
    hyperparameters = {
        json.dumps(payload["frozen_hyperparameters"], sort_keys=True)
        for _path, payload in manifests
    }
    devices = {payload.get("device") for _path, payload in manifests}
    if len(dataset_hashes) != 1 or len(hyperparameters) != 1 or len(devices) != 1:
        raise ValueError("three seeds did not use identical data/hyperparameters")
    ordered = sorted(
        manifests,
        key=lambda item: (
            item[1]["final_validation"]["post"]["r1"],
            item[1]["final_validation"]["post"]["ndcg5"],
            item[1]["final_validation"]["pair_flips"]["net_flips"],
        ),
    )
    median_path, median = ordered[1]
    official_cuda_run = devices == {"cuda"}
    all_nonnegative = all(
        payload["final_validation"]["post"]["r1"]
        >= payload["final_validation"]["base"]["r1"]
        for _path, payload in manifests
    )
    median_pass = all(median["pilot_model_checks"].values())
    return {
        "schema_version": "answerability-student-seed-summary-v1",
        "status": (
            "expand_labeling" if official_cuda_run and all_nonnegative and median_pass
            else "pilot_no_go_stop_spending" if official_cuda_run
            else "diagnostic_only_requires_cuda_rerun"
        ),
        "diagnostic_decision": (
            "would_expand_labeling" if all_nonnegative and median_pass
            else "would_stop_spending"
        ),
        "official_cuda_run": official_cuda_run,
        "device": next(iter(devices)),
        "dataset_sha256": next(iter(dataset_hashes)),
        "all_seeds_nonnegative_r1": all_nonnegative,
        "median_seed": median["seed"],
        "median_checkpoint": str(median_path.parent),
        "median_checkpoint_sha256": median["checkpoint_sha256"],
        "median_passes_all_model_checks": median_pass,
        "seeds": [
            {
                "seed": payload["seed"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "base_r1": payload["final_validation"]["base"]["r1"],
                "post_r1": payload["final_validation"]["post"]["r1"],
                "wins": payload["final_validation"]["wins"],
                "losses": payload["final_validation"]["losses"],
                "net_pair_flips": payload["final_validation"]["pair_flips"]["net_flips"],
                "status": payload["pilot_model_status"],
            }
            for _path, payload in manifests
        ],
        "blind_labels_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
