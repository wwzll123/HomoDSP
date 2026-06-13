#!/usr/bin/env python
"""Smoke-test loading an augmented DeepPBS NPZ file."""

import argparse
import sys
import types
from pathlib import Path


def install_import_stubs():
    """Install tiny stubs for preprocessing-only deps not needed by NPZ loading."""
    freesasa = types.ModuleType("freesasa")
    freesasa.Classifier = object
    freesasa.nowarnings = 0
    freesasa.setVerbosity = lambda _level: None
    sys.modules.setdefault("freesasa", freesasa)

    torch_cluster = types.ModuleType("torch_cluster")
    torch_cluster.radius = lambda *args, **kwargs: None
    torch_cluster.radius_graph = lambda *args, **kwargs: None
    sys.modules.setdefault("torch_cluster", torch_cluster)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--stub-preprocess-deps", action="store_true")
    args = parser.parse_args()

    if args.stub_preprocess_deps:
        install_import_stubs()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    try:
        from deeppbs.nn.utils import loadDataset
    except Exception as exc:
        print(f"IMPORT_ERROR {type(exc).__name__}: {exc}")
        return 2

    dataset, _, info, _ = loadDataset(
        [args.sample],
        4,
        "Y_pwm",
        args.data_dir,
        scale=False,
    )
    data = dataset[0]
    print("INFO", info)
    print("template_x_dna", tuple(data.template_x_dna.shape), data.template_x_dna.dtype)
    print("template_mask", tuple(data.template_mask.shape), data.template_mask.dtype)
    print("template_scores", tuple(data.template_scores.shape), data.template_scores.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
