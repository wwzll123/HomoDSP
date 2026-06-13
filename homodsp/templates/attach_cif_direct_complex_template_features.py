#!/usr/bin/env python
"""Attach direct protein-DNA contact features from Foldseek CIF hits."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser


DNA_RESNAMES = {
    "DA", "DC", "DG", "DT", "DI", "DU",
    "A", "C", "G", "T", "U",
}

AA_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
}

GROUP_ATOMS = {
    "phosphate": {"P", "OP1", "OP2", "O1P", "O2P"},
    "sugar": {"C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'"},
    "base": set(),
}


def target_to_cif_stem(target):
    stem = Path(target).name
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def load_hits(path, max_templates):
    hits = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            stem = target_to_cif_stem(row[1])
            if stem in seen:
                continue
            seen.add(stem)
            score = float(row[2]) if len(row) > 2 else 0.0
            hits.append({"stem": stem, "score": score})
            if max_templates and len(hits) >= max_templates:
                break
    return hits


def residue_atoms(residue):
    return [atom for atom in residue.get_atoms() if atom.element != "H"]


def parse_complex_cif(cif_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    dna_residues = []
    protein_coords = []

    for residue in structure.get_residues():
        resname = residue.get_resname().strip().upper()
        atoms = residue_atoms(residue)
        if not atoms:
            continue
        if resname in DNA_RESNAMES:
            dna_residues.append(residue)
        elif resname in AA_RESNAMES:
            protein_coords.extend(atom.coord for atom in atoms)

    if not dna_residues:
        raise ValueError(f"No DNA residues found in {cif_path}")
    if not protein_coords:
        raise ValueError(f"No protein atoms found in {cif_path}")
    return dna_residues, np.asarray(protein_coords, dtype=np.float32)


def group_coords(residue, group):
    atoms = residue_atoms(residue)
    if group == "base":
        exclude = GROUP_ATOMS["phosphate"] | GROUP_ATOMS["sugar"]
        coords = [atom.coord for atom in atoms if atom.get_name() not in exclude]
    else:
        names = GROUP_ATOMS[group]
        coords = [atom.coord for atom in atoms if atom.get_name() in names]
    if not coords:
        coords = [atom.coord for atom in atoms]
    return np.asarray(coords, dtype=np.float32)


def cif_contact_features(cif_path, contact_cutoff=5.0):
    dna_residues, protein_coords = parse_complex_cif(cif_path)
    rows = []
    for residue in dna_residues:
        residue_coords = np.asarray([atom.coord for atom in residue_atoms(residue)], dtype=np.float32)
        d_all = np.linalg.norm(protein_coords[:, None, :] - residue_coords[None, :, :], axis=-1)
        min_dist = float(d_all.min())
        all_contacts = float((d_all < contact_cutoff).sum())

        grouped = []
        for group in ("phosphate", "sugar", "base"):
            coords = group_coords(residue, group)
            d_group = np.linalg.norm(protein_coords[:, None, :] - coords[None, :, :], axis=-1)
            grouped.append(float((d_group < contact_cutoff).sum()))

        rows.append([
            min_dist / 10.0,
            np.log1p(all_contacts),
            np.log1p(grouped[0]),
            np.log1p(grouped[1]),
            np.log1p(grouped[2]),
            float(len(residue_coords)) / 30.0,
        ])
    return np.asarray(rows, dtype=np.float32)


def resize_positions(array, length):
    array = np.asarray(array, dtype=np.float32)
    if array.shape[0] == length:
        return array
    src = np.linspace(0.0, 1.0, array.shape[0])
    dst = np.linspace(0.0, 1.0, length)
    cols = [np.interp(dst, src, array[:, i]) for i in range(array.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--hits", required=True)
    parser.add_argument("--cif-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    query_path = Path(args.query_npz)
    cif_dir = Path(args.cif_dir)
    query_arrays = np.load(query_path, allow_pickle=True)
    query_len = int(query_arrays["X_dna"].shape[0])

    hits = load_hits(args.hits, args.top_k)
    feature_names = np.array([
        "tmpl_min_prot_dist_over_10",
        "tmpl_log_all_contacts",
        "tmpl_log_phosphate_contacts",
        "tmpl_log_sugar_contacts",
        "tmpl_log_base_contacts",
        "tmpl_dna_atom_count_over_30",
    ])
    template_x = np.zeros((args.top_k, query_len, len(feature_names)), dtype=np.float32)
    template_mask = np.zeros((args.top_k, query_len), dtype=bool)
    template_scores = np.zeros((args.top_k,), dtype=np.float32)
    template_ids = np.array([""] * args.top_k, dtype="U256")

    used = []
    for i, hit in enumerate(hits[: args.top_k]):
        cif_path = cif_dir / f"{hit['stem']}.cif"
        if not cif_path.exists():
            print(f"[WARN] template CIF missing: {cif_path}")
            continue
        try:
            features = resize_positions(cif_contact_features(cif_path), query_len)
        except Exception as exc:
            print(f"[WARN] failed to parse {cif_path.name}: {exc}")
            continue
        template_x[i] = features
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
        "template_feature_names": feature_names,
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(json.dumps({"out": str(out), "used_templates": used, "template_shape": list(template_x.shape)}, indent=2))


if __name__ == "__main__":
    main()
