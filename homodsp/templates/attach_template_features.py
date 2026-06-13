#!/usr/bin/env python
"""Attach lightweight template features to DeepPBS NPZ samples."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_hits(path):
    hits = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            hits[record["query"]] = record.get("templates", [])
    return hits


def resize_positions(array, length):
    """Linearly resample a [L, F] array to the requested length."""
    array = np.asarray(array, dtype=np.float32)
    if array.shape[0] == length:
        return array
    if array.shape[0] == 0:
        return np.zeros((length, array.shape[1]), dtype=np.float32)

    src = np.linspace(0.0, 1.0, array.shape[0])
    dst = np.linspace(0.0, 1.0, length)
    cols = [np.interp(dst, src, array[:, i]) for i in range(array.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def copy_with_templates(query_path, template_hits, data_dir, out_dir, top_k):
    query_arrays = np.load(query_path, allow_pickle=True)
    query_len = int(query_arrays["X_dna"].shape[0])
    x_dim = int(query_arrays["X_dna"].shape[1])

    template_x = np.zeros((top_k, query_len, x_dim), dtype=np.float32)
    template_mask = np.zeros((top_k, query_len), dtype=bool)
    template_scores = np.zeros((top_k,), dtype=np.float32)
    template_ids = np.array([""] * top_k, dtype="U256")

    for i, hit in enumerate(template_hits[:top_k]):
        template_path = data_dir / hit["id"]
        if not template_path.exists():
            print(f"[WARN] missing template file: {template_path}")
            continue
        template_arrays = np.load(template_path, allow_pickle=True)
        template_x[i] = resize_positions(template_arrays["X_dna"], query_len)
        template_mask[i, :] = True
        template_scores[i] = float(hit.get("score", 0.0))
        template_ids[i] = hit["id"]

    out_path = out_dir / query_path.name
    payload = {key: query_arrays[key] for key in query_arrays.files}
    payload.update({
        "template_x_dna": template_x,
        "template_mask": template_mask,
        "template_scores": template_scores,
        "template_ids": template_ids,
    })
    np.savez_compressed(out_path, **payload)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing original DeepPBS .npz files.")
    parser.add_argument("--hits", required=True, help="JSONL hits from retrieve_npz_templates.py.")
    parser.add_argument("--out-dir", required=True, help="Directory for augmented .npz files.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--copy-missing", action="store_true", help="Copy queries without hits unchanged.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = load_hits(args.hits)

    n_augmented = 0
    n_copied = 0
    for query_id, template_hits in hits.items():
        query_path = data_dir / query_id
        if not query_path.exists():
            print(f"[WARN] missing query file: {query_path}")
            continue
        if template_hits:
            copy_with_templates(query_path, template_hits, data_dir, out_dir, args.top_k)
            n_augmented += 1
        elif args.copy_missing:
            shutil.copy2(query_path, out_dir / query_path.name)
            n_copied += 1

    print(f"augmented={n_augmented} copied={n_copied} out_dir={out_dir}")


if __name__ == "__main__":
    main()
