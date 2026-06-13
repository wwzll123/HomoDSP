#!/usr/bin/env python
"""Run a single DeepPBS forward pass with lightweight template features."""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--template-hidden", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_root = repo_root / "run"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(run_root))

    from deeppbs.nn.utils import loadDataset
    from models.model_v2 import Model

    dataset, _, info, _ = loadDataset(
        [args.sample],
        4,
        "Y_pwm",
        args.data_dir,
        scale=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = dataset[0].to(device)

    model = Model(
        info["prot_features"],
        info["dna_features"],
        condition="prot_shape",
        use_templates=True,
        template_channels=int(data.template_x_dna.shape[-1]) if hasattr(data, "template_x_dna") else info["dna_features"],
        template_node_channels=int(data.template_node_x.shape[-1]) if hasattr(data, "template_node_x") else 92,
        template_encoder_type="interface" if hasattr(data, "template_node_x") else "feature",
        template_hidden=args.template_hidden,
    )
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data)

    print("input_len", data.x_dna.shape[0])
    print("output", tuple(out.shape), out.dtype)
    print("has_template", hasattr(data, "template_x_dna") or hasattr(data, "template_node_x"))
    if hasattr(data, "template_node_x"):
        print("template_node_x", tuple(data.template_node_x.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
