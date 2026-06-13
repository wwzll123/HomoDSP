#!/usr/bin/env python
"""Attach protein-DNA complex template features from processed CIF templates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


GROUPS = {
    "phosphate": [0, 10],
    "sugar": [1, 9],
    "major": [2, 3, 4, 5],
    "minor": [6, 7, 8],
}


def target_to_cif_stem(target):
    stem = Path(target).name
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def load_hits(path, max_templates):
    hits = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            stem = target_to_cif_stem(row[1])
            score = float(row[2]) if len(row) > 2 else 0.0
            if stem not in [hit["stem"] for hit in hits]:
                hits.append({"stem": stem, "score": score})
            if max_templates and len(hits) >= max_templates:
                break
    return hits


def resize_positions(array, length):
    array = np.asarray(array, dtype=np.float32)
    if array.shape[0] == length:
        return array
    src = np.linspace(0.0, 1.0, array.shape[0])
    dst = np.linspace(0.0, 1.0, length)
    cols = [np.interp(dst, src, array[:, i]) for i in range(array.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def complex_features(npz_path, contact_cutoff=5.0):
    arrays = np.load(npz_path, allow_pickle=True)
    x_dna = np.asarray(arrays["X_dna"], dtype=np.float32)
    v_dna = np.asarray(arrays["V_dna"], dtype=np.float32)
    v_prot = np.asarray(arrays["V_prot"], dtype=np.float32)

    per_base = []
    for i in range(v_dna.shape[0]):
        beads = v_dna[i]
        dists = np.linalg.norm(v_prot[:, None, :] - beads[None, :, :], axis=-1)
        min_per_bead = dists.min(axis=0)
        min_dist = float(min_per_bead.min())
        all_contacts = float((dists < contact_cutoff).sum())
        group_contacts = []
        for indices in GROUPS.values():
            group_contacts.append(float((dists[:, indices] < contact_cutoff).sum()))
        geom = np.array([min_dist / 10.0, np.log1p(all_contacts)] + [np.log1p(v) for v in group_contacts], dtype=np.float32)
        per_base.append(np.concatenate([x_dna[i], geom], axis=0))
    return np.stack(per_base, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--hits", required=True)
    parser.add_argument("--template-npz-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    query_path = Path(args.query_npz)
    template_dir = Path(args.template_npz_dir)
    query_arrays = np.load(query_path, allow_pickle=True)
    query_len = int(query_arrays["X_dna"].shape[0])

    hits = load_hits(args.hits, args.top_k)
    feature_dim = 20
    template_x = np.zeros((args.top_k, query_len, feature_dim), dtype=np.float32)
    template_mask = np.zeros((args.top_k, query_len), dtype=bool)
    template_scores = np.zeros((args.top_k,), dtype=np.float32)
    template_ids = np.array([""] * args.top_k, dtype="U256")

    used = []
    for i, hit in enumerate(hits[: args.top_k]):
        npz_path = template_dir / f"{hit['stem']}.npz"
        if not npz_path.exists():
            print(f"[WARN] template NPZ missing: {npz_path}")
            continue
        feat = resize_positions(complex_features(npz_path), query_len)
        template_x[i] = feat
        template_mask[i, :] = True
        template_scores[i] = hit["score"]
        template_ids[i] = hit["stem"]
        used.append(hit["stem"])

    payload = {key: query_arrays[key] for key in query_arrays.files}
    payload.update({
        "template_x_dna": template_x,
        "template_mask": template_mask,
        "template_scores": template_scores,
        "template_ids": template_ids,
        "template_feature_names": np.array(
            list(query_arrays["dna_feature_names"]) + [
                "tmpl_min_prot_dist_over_10",
                "tmpl_log_all_contacts",
                "tmpl_log_phosphate_contacts",
                "tmpl_log_sugar_contacts",
                "tmpl_log_major_contacts",
                "tmpl_log_minor_contacts",
            ]
        ),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(json.dumps({"out": str(out), "used_templates": used, "template_shape": list(template_x.shape)}, indent=2))


if __name__ == "__main__":
    main()
