#!/usr/bin/env python
"""Batch interface-patch Foldseek retrieval and CIF template feature attachment."""

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from attach_cif_direct_complex_template_features import main as attach_contact_main
from attach_cif_interface_gnn_features import main as attach_interface_main
from extract_query_interface_patch import extract_patch


def read_queries(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_query_cif_map(path):
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            return json.load(handle)
        mapping = {}
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2:
                mapping[row[0]] = row[1]
        return mapping


def query_to_cif(query_id, query_cif_dir, query_cif_map):
    if query_id in query_cif_map:
        return Path(query_cif_map[query_id])
    stem = Path(query_id).stem
    for suffix in (".cif", ".pdb"):
        candidate = Path(query_cif_dir) / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    # Many DeepPBS names are pdb_chain_pwm.npz; try pdb root.
    root = stem.split("_")[0]
    for suffix in (".cif", ".pdb"):
        candidate = Path(query_cif_dir) / f"{root}{suffix}"
        if candidate.exists():
            return candidate
    return None


def run_foldseek(foldseek, query_patch, db_path, out_path, tmp_dir, threads):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        foldseek,
        "easy-search",
        str(query_patch),
        str(db_path),
        str(out_path),
        str(tmp_dir),
        "--format-output",
        "query,target,qtmscore,ttmscore,alnlen,evalue,bits,qstart,tstart,qaln,taln",
        "--threads",
        str(threads),
    ]
    return subprocess.run(cmd, text=True, capture_output=True)


def attach_features(query_npz, hits, cif_dir, out_npz, top_k, mode, max_nodes, cutoff, contact_cutoff, aligned_residue_mode):
    old_argv = sys.argv[:]
    script = "attach_cif_interface_gnn_features.py" if mode == "interface" else "attach_cif_direct_complex_template_features.py"
    try:
        sys.argv = [
            script,
            "--query-npz", str(query_npz),
            "--hits", str(hits),
            "--cif-dir", str(cif_dir),
            "--out", str(out_npz),
            "--top-k", str(top_k),
        ]
        if mode == "interface":
            sys.argv += [
                "--max-nodes", str(max_nodes),
                "--cutoff", str(cutoff),
                "--contact-cutoff", str(contact_cutoff),
                "--aligned-residue-mode", aligned_residue_mode,
            ]
            attach_interface_main()
        else:
            attach_contact_main()
    finally:
        sys.argv = old_argv


def process_query(query_id, args_dict, query_cif_map):
    query_npz = Path(args_dict["query_npz_dir"]) / query_id
    if not query_npz.exists():
        return {"query": query_id, "warning": f"missing query NPZ: {query_npz}"}
    query_cif = query_to_cif(query_id, args_dict.get("query_cif_dir") or ".", query_cif_map)
    if query_cif is None or not query_cif.exists():
        return {"query": query_id, "warning": f"missing query CIF/PDB for {query_id}"}

    out_dir = Path(args_dict["out_dir"])
    patch_dir = out_dir / "query_patches"
    hits_dir = out_dir / "hits"
    aug_dir = out_dir / "aug_npz"
    tmp_dir = out_dir / "tmp"

    stem = Path(query_id).stem
    patch_path = patch_dir / f"{stem}_interface.pdb"
    hits_path = hits_dir / f"{stem}.tsv"
    out_npz = aug_dir / query_id

    if args_dict.get("skip_existing") and out_npz.exists():
        return {"query": query_id, "patch": str(patch_path), "hits": str(hits_path), "out": str(out_npz), "skipped": True}

    try:
        extract_patch(query_cif, patch_path, args_dict["cutoff"], args_dict["flank"])
    except Exception as exc:
        return {"query": query_id, "warning": f"patch extraction failed for {query_id}: {exc}"}

    result = run_foldseek(
        args_dict["foldseek"],
        patch_path,
        args_dict["foldseek_db"],
        hits_path,
        tmp_dir / stem,
        args_dict["threads"],
    )
    if result.returncode != 0:
        return {"query": query_id, "warning": f"Foldseek failed for {query_id}: {result.stderr.strip()}"}

    attach_features(
        query_npz,
        hits_path,
        args_dict["template_cif_dir"],
        out_npz,
        args_dict["top_k"],
        args_dict["feature_mode"],
        args_dict["max_template_nodes"],
        args_dict["cutoff"],
        args_dict["contact_cutoff"],
        args_dict["aligned_residue_mode"],
    )
    return {"query": query_id, "patch": str(patch_path), "hits": str(hits_path), "out": str(out_npz), "skipped": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-list", required=True)
    parser.add_argument("--query-npz-dir", required=True)
    parser.add_argument("--query-cif-dir")
    parser.add_argument("--query-cif-map")
    parser.add_argument("--template-cif-dir", required=True)
    parser.add_argument("--foldseek-db", required=True)
    parser.add_argument("--foldseek", default="/root/foldseek/bin/foldseek")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--feature-mode", choices=["contact", "interface"], default="interface")
    parser.add_argument("--max-template-nodes", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument("--aligned-residue-mode", choices=["off", "soft", "hard"], default="soft")
    parser.add_argument("--flank", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1, help="Number of query-level workers.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip query outputs that already exist.")
    args = parser.parse_args()

    query_cif_map = load_query_cif_map(args.query_cif_map)
    out_dir = Path(args.out_dir)
    patch_dir = out_dir / "query_patches"
    hits_dir = out_dir / "hits"
    aug_dir = out_dir / "aug_npz"
    tmp_dir = out_dir / "tmp"
    for path in (patch_dir, hits_dir, aug_dir, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)

    queries = read_queries(args.query_list)
    args_dict = vars(args).copy()
    summary = []
    skipped = 0
    if args.jobs <= 1:
        iterator = (process_query(query_id, args_dict, query_cif_map) for query_id in queries)
    else:
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        futures = [executor.submit(process_query, query_id, args_dict, query_cif_map) for query_id in queries]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for item in iterator:
            if "warning" in item:
                print(f"[WARN] {item['warning']}", flush=True)
                continue
            if item.get("skipped"):
                skipped += 1
            summary.append({k: item[k] for k in ("query", "patch", "hits", "out")})
            if len(summary) % 25 == 0:
                print(f"processed={len(summary)} skipped={skipped}", flush=True)
    finally:
        if args.jobs > 1:
            executor.shutdown(wait=True, cancel_futures=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"processed={len(summary)} skipped={skipped} summary={summary_path}")


if __name__ == "__main__":
    main()
