#!/usr/bin/env python
"""Process Foldseek CIF hits into DeepPBS NPZ files."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def target_to_cif_stem(target):
    """Foldseek appends chain ids like _A to structure basenames."""
    stem = Path(target).name
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def read_hit_stems(hits_path, max_templates):
    stems = []
    with Path(hits_path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            stem = target_to_cif_stem(row[1])
            if stem not in stems:
                stems.append(stem)
            if max_templates and len(stems) >= max_templates:
                break
    return stems


def run_deeppbs_processing(repo_root, input_file, config_file):
    env = os.environ.copy()
    env["PATH"] = str(repo_root / "dependencies" / "bin") + os.pathsep + env.get("PATH", "")
    env["X3DNA"] = str(repo_root / "x3dna-v2.3-linux-64bit" / "x3dna-v2.3")
    cmd = [
        sys.executable,
        str(repo_root / "run" / "process_co_crystal.py"),
        str(input_file),
        str(config_file),
        "--no_pwm",
    ]
    return subprocess.run(cmd, cwd=repo_root, env=env, text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", required=True)
    parser.add_argument("--cif-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-templates", type=int, default=4)
    parser.add_argument("--include-query-cif", help="Optional query CIF to process alongside templates.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cif_dir = Path(args.cif_dir)
    out_dir = Path(args.out_dir)
    pdb_dir = out_dir / "pdb"
    npz_dir = out_dir / "npz"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    if args.include_query_cif:
        query_src = Path(args.include_query_cif)
        query_name = "query.cif"
        shutil.copy2(query_src, pdb_dir / query_name)
        copied.append(query_name)

    for stem in read_hit_stems(args.hits, args.max_templates):
        src = cif_dir / f"{stem}.cif"
        if not src.exists():
            print(f"[WARN] CIF not found for hit: {stem}")
            continue
        dst_name = f"{stem}.cif"
        shutil.copy2(src, pdb_dir / dst_name)
        copied.append(dst_name)

    input_file = out_dir / "input.txt"
    config_file = out_dir / "process_config.json"
    input_file.write_text("\n".join(copied) + "\n", encoding="utf-8")
    config_file.write_text(
        json.dumps({
            "PDB_FILES_PATH": str(pdb_dir.resolve()),
            "FEATURE_DATA_PATH": str(npz_dir.resolve()),
        }, indent=2),
        encoding="utf-8",
    )

    result = run_deeppbs_processing(repo_root, input_file, config_file)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    processed = sorted(path.name for path in npz_dir.glob("*.npz"))
    print("copied", len(copied))
    print("processed", len(processed))
    for name in processed:
        print(name)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
