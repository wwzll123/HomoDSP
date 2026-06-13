#!/usr/bin/env python
"""Print a compact summary of template tensors stored in an NPZ."""

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    args = parser.parse_args()

    arrays = np.load(args.npz, allow_pickle=True)
    summary = {}
    for key in ("template_node_x", "template_node_mask", "template_scores", "template_dna_mappings"):
        if key in arrays.files:
            summary[key] = list(arrays[key].shape)
    if "template_node_feature_names" in arrays.files:
        names = arrays["template_node_feature_names"]
        summary["feature_count"] = int(len(names))
        summary["last_features"] = names[-8:].tolist()
    if "template_node_x" in arrays.files and "template_node_mask" in arrays.files:
        x = arrays["template_node_x"]
        mask = arrays["template_node_mask"]
        if x.shape[-1] >= 92 and mask.any():
            summary["aligned_node_count"] = float(x[..., -4][mask].sum())
            summary["aligned_same_aa_count"] = float(x[..., -1][mask].sum())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
