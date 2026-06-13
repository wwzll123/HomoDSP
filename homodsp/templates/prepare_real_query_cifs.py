#!/usr/bin/env python
"""Download real query mmCIF files and write a DeepPBS sample-to-CIF map."""

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path


def read_queries(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def pdb_id_from_sample(sample):
    return Path(sample).stem.split("_")[0].lower()


def download_cif(pdb_id, out_path, retries=3):
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached"
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            if len(payload) < 100:
                raise RuntimeError("downloaded file is unexpectedly small")
            out_path.write_bytes(payload)
            return True, "downloaded"
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1 + attempt)
    return False, last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--map-out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = []
    summary = []
    for sample in read_queries(args.query_list):
        pdb_id = pdb_id_from_sample(sample)
        cif_path = out_dir / f"{pdb_id}.cif"
        ok, status = download_cif(pdb_id, cif_path)
        summary.append({"sample": sample, "pdb_id": pdb_id, "ok": ok, "status": status})
        if ok:
            rows.append([sample, str(cif_path.resolve())])

    map_out = Path(args.map_out)
    map_out.parent.mkdir(parents=True, exist_ok=True)
    with map_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    print(json.dumps({"mapped": len(rows), "summary": summary, "map": str(map_out)}, indent=2))


if __name__ == "__main__":
    main()
