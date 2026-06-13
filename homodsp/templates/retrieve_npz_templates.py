#!/usr/bin/env python
"""Retrieve lightweight NPZ templates using index-level summary features."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_root(name):
    return Path(name).name.split("_")[0]


def score_candidate(query, candidate):
    q_shape = np.asarray(query["x_dna_mean"], dtype=float)
    c_shape = np.asarray(candidate["x_dna_mean"], dtype=float)
    n = min(q_shape.size, c_shape.size)
    shape_dist = float(np.linalg.norm(q_shape[:n] - c_shape[:n])) if n else 0.0
    len_penalty = abs(query["dna_len"] - candidate["dna_len"]) / max(query["dna_len"], candidate["dna_len"], 1)
    contact_bonus = np.log1p(candidate.get("contacts", 0.0)) / 10.0
    return contact_bonus - len_penalty - 0.05 * shape_dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-list", required=True, help="Text file with query NPZ file names.")
    parser.add_argument("--index", required=True, help="JSONL index from build_npz_template_index.py.")
    parser.add_argument("--out", required=True, help="Output JSONL hit file.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--allow-same-root", action="store_true", help="Allow candidates sharing the PDB root.")
    args = parser.parse_args()

    entries = load_jsonl(args.index)
    by_id = {entry["id"]: entry for entry in entries}
    query_ids = [line.strip() for line in Path(args.query_list).read_text(encoding="utf-8").splitlines() if line.strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("w", encoding="utf-8") as handle:
        for query_id in query_ids:
            query = by_id.get(Path(query_id).name)
            if query is None:
                print(f"[WARN] query not in index: {query_id}")
                continue

            hits = []
            for candidate in entries:
                if candidate["id"] == query["id"]:
                    continue
                if not args.allow_same_root and sample_root(candidate["id"]) == sample_root(query["id"]):
                    continue
                hits.append({
                    "id": candidate["id"],
                    "score": round(score_candidate(query, candidate), 6),
                    "dna_len": candidate["dna_len"],
                    "contacts": candidate["contacts"],
                })
            hits.sort(key=lambda item: item["score"], reverse=True)
            record = {"query": query["id"], "templates": hits[: args.top_k]}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            n_written += 1

    print(f"queries={n_written} out={out}")


if __name__ == "__main__":
    main()
