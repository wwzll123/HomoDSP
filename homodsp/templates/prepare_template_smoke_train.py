#!/usr/bin/env python
"""Prepare a tiny template-enabled DeepPBS training smoke test."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--num-train", type=int, default=4)
    parser.add_argument("--num-valid", type=int, default=1)
    parser.add_argument("--template-encoder-type", choices=["feature", "interface"], default="feature")
    parser.add_argument("--template-channels", type=int, default=14)
    parser.add_argument("--template-node-channels", type=int, default=92)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    aug_dir = work_dir / "aug_npz"
    files = sorted(path.name for path in aug_dir.glob("*.npz"))
    needed = args.num_train + args.num_valid
    if len(files) < needed:
        raise SystemExit(f"Need {needed} augmented files, found {len(files)}")

    train = files[: args.num_train]
    valid = files[args.num_train:needed]
    (work_dir / "train_smoke.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (work_dir / "valid_smoke.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")

    cfg = {
        "data_dir": str(aug_dir.resolve()),
        "output_path": str((work_dir / "train_out").resolve()),
        "nc": 4,
        "labels_key": "Y_pwm",
        "cache_dataset": False,
        "epochs": 1,
        "batch_size": 1,
        "condition": "prot_shape",
        "use_templates": True,
        "template_channels": args.template_channels,
        "template_encoder_type": args.template_encoder_type,
        "template_node_channels": args.template_node_channels,
        "template_hidden": 8,
        "ic_loss_weight": 0,
        "mse_loss_weight": 1,
        "remove_zero_class": False,
        "best_state_metric": "mae",
        "best_state_metric_goal": "min",
        "best_state_metric_threshold": 1.0,
        "best_state_metric_dataset": "validation",
        "model": {"transform_args": []},
        "optimizer": {
            "name": "adam",
            "kwargs": {"lr": 0.001, "weight_decay": 0.0001},
        },
        "scheduler": {"name": "", "kwargs": {}},
        "tensorboard": False,
        "write_test_predictions": False,
    }
    (work_dir / "config_template_smoke.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("train", train)
    print("valid", valid)
    print("config", work_dir / "config_template_smoke.json")


if __name__ == "__main__":
    main()
