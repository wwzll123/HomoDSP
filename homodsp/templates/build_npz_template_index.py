#!/usr/bin/env python
"""Build a lightweight JSONL index from processed DeepPBS NPZ files."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_npz(path):
    arrays = np.load(path, allow_pickle=True)
    x_dna = np.asarray(arrays["X_dna"], dtype=float)
    x_prot = np.asarray(arrays["X_prot"], dtype=float)
    contacts = np.asarray(arrays.get("contacts", [0]), dtype=float)
    y_pwm = np.asarray(arrays.get("Y_pwm", []), dtype=float)

    return {
        "id": path.name,
        "path": str(path),
        "dna_len": int(x_dna.shape[0]),
        "dna_features": int(x_dna.shape[1]) if x_dna.ndim == 2 else 0,
        "protein_atoms": int(x_prot.shape[0]),
        "contacts": float(contacts[0]) if contacts.size else 0.0,
        "x_dna_mean": np.nan_to_num(x_dna.mean(axis=0), nan=0.0).round(6).tolist(),
        "has_pwm": bool(y_pwm.size > 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing DeepPBS .npz files.")
    parser.add_argument("--out", required=True, help="Output JSONL index path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of files to index.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.npz"))
    if args.limit > 0:
        paths = paths[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    with out.open("w", encoding="utf-8") as handle:
        for path in paths:
            try:
                handle.write(json.dumps(summarize_npz(path), sort_keys=True) + "\n")
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                print(f"[WARN] skipped {path.name}: {exc}")

    print(f"indexed={n_ok} failed={n_fail} out={out}")


if __name__ == "__main__":
    main()
