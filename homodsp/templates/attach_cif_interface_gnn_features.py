#!/usr/bin/env python
"""Attach residue-level protein-DNA interface tensors from Foldseek CIF hits.

The output keeps the network input regular while preserving substantially more
template structure than simple contact counts:

    template_node_x:    [K, L_query, M, F]
    template_node_mask: [K, L_query, M]

For every template and DNA position, the M nodes are the closest protein
residues around the corresponding template DNA residue. Each residue node
contains amino-acid identity, physicochemical tags, closest atom type, DNA
region contacted, distance/RBF features, local-frame coordinates, and side-chain
orientation features.
"""

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
DNA_BASES = ["A", "C", "G", "T", "OTHER"]

AA_RESNAMES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {name: i for i, name in enumerate(AA_RESNAMES)}

AA_PROPERTY_NAMES = [
    "hydrophobic", "polar", "positive", "negative", "aromatic",
    "aliphatic", "small", "sulfur", "glycine", "proline",
]
AA_PROPERTIES = {
    "ALA": ["hydrophobic", "small", "aliphatic"],
    "ARG": ["polar", "positive"],
    "ASN": ["polar"],
    "ASP": ["polar", "negative"],
    "CYS": ["polar", "sulfur", "small"],
    "GLN": ["polar"],
    "GLU": ["polar", "negative"],
    "GLY": ["small", "glycine"],
    "HIS": ["polar", "positive", "aromatic"],
    "ILE": ["hydrophobic", "aliphatic"],
    "LEU": ["hydrophobic", "aliphatic"],
    "LYS": ["polar", "positive"],
    "MET": ["hydrophobic", "sulfur"],
    "PHE": ["hydrophobic", "aromatic"],
    "PRO": ["hydrophobic", "proline"],
    "SER": ["polar", "small"],
    "THR": ["polar", "small"],
    "TRP": ["hydrophobic", "aromatic"],
    "TYR": ["polar", "aromatic"],
    "VAL": ["hydrophobic", "aliphatic", "small"],
}

ELEMENTS = ["C", "N", "O", "S", "P", "OTHER"]
ELEMENT_TO_INDEX = {name: i for i, name in enumerate(ELEMENTS)}
ATOM_ROLES = ["backbone", "sidechain", "terminal", "other"]

GROUP_ATOMS = {
    "phosphate": {"P", "OP1", "OP2", "O1P", "O2P"},
    "sugar": {"C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'"},
}
GROUPS = ["phosphate", "sugar", "base"]
RBF_CENTERS = np.linspace(2.0, 12.0, 16).astype(np.float32)
RBF_WIDTH = 1.5


def build_feature_names():
    names = []
    names += [f"aa_{name}" for name in AA_RESNAMES]
    names += [f"aa_prop_{name}" for name in AA_PROPERTY_NAMES]
    names += [f"closest_atom_{name.lower()}" for name in ELEMENTS]
    names += [f"closest_atom_role_{name}" for name in ATOM_ROLES]
    names += [f"dna_group_{name}" for name in GROUPS]
    names += [f"dna_base_{name.lower()}" for name in DNA_BASES]
    names += [
        "min_dist_all_over_10",
        "min_dist_phosphate_over_10",
        "min_dist_sugar_over_10",
        "min_dist_base_over_10",
        "log_contacts_all",
        "log_contacts_phosphate",
        "log_contacts_sugar",
        "log_contacts_base",
        "residue_atom_count_over_30",
        "rank_norm",
    ]
    names += ["local_dx_over_10", "local_dy_over_10", "local_dz_over_10"]
    names += ["local_unit_dx", "local_unit_dy", "local_unit_dz"]
    names += [f"dist_rbf_{i:02d}" for i in range(len(RBF_CENTERS))]
    names += [
        "sidechain_lx", "sidechain_ly", "sidechain_lz",
        "ca_to_dna_lx", "ca_to_dna_ly", "ca_to_dna_lz",
        "sidechain_length_over_10", "has_sidechain_vector",
    ]
    names += [
        "is_foldseek_aligned",
        "aligned_query_pos_norm",
        "aligned_template_pos_norm",
        "aligned_same_aa",
    ]
    return np.asarray(names)


FEATURE_NAMES = build_feature_names()


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
            hit = {"stem": stem, "score": score, "target": row[1]}
            if len(row) >= 11:
                hit.update({
                    "qstart": row[7],
                    "tstart": row[8],
                    "qaln": row[9],
                    "taln": row[10],
                })
            hits.append(hit)
            if max_templates and len(hits) >= max_templates:
                break
    return hits


def one_hot(index, size):
    out = np.zeros((size,), dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def safe_normalize(vector, fallback):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def residue_atoms(residue):
    return [atom for atom in residue.get_atoms() if atom.element != "H"]


def dna_base_index(resname):
    base = resname.strip().upper().replace("D", "")
    if base == "U":
        base = "T"
    return DNA_BASES.index(base) if base in DNA_BASES else DNA_BASES.index("OTHER")


def dna_base_letter(resname):
    return DNA_BASES[dna_base_index(resname)]


def dna_atom_group(atom_name):
    if atom_name in GROUP_ATOMS["phosphate"]:
        return 0
    if atom_name in GROUP_ATOMS["sugar"]:
        return 1
    return 2


def atom_element_index(atom):
    elem = atom.element.strip().upper()
    return ELEMENT_TO_INDEX.get(elem, ELEMENT_TO_INDEX["OTHER"])


def protein_atom_role(atom_name):
    if atom_name in {"N", "CA", "C", "O"}:
        return 0
    if atom_name == "OXT":
        return 2
    if atom_name:
        return 1
    return 3


def residue_id(residue):
    chain = residue.get_parent().id
    hetflag, resseq, icode = residue.id
    return (chain, hetflag, int(resseq), str(icode).strip())


def residue_center(residue):
    coords = np.asarray([atom.coord for atom in residue_atoms(residue)], dtype=np.float32)
    return coords.mean(axis=0)


def get_atom_coord(residue, atom_name):
    return residue[atom_name].coord.astype(np.float32) if atom_name in residue else None


def dna_residue_frame(residue):
    atoms = residue_atoms(residue)
    coords = np.asarray([atom.coord for atom in atoms], dtype=np.float32)
    origin = coords.mean(axis=0)

    sugar = [atom.coord for atom in atoms if dna_atom_group(atom.get_name()) == 1]
    base = [atom.coord for atom in atoms if dna_atom_group(atom.get_name()) == 2]
    phosphate = [atom.coord for atom in atoms if dna_atom_group(atom.get_name()) == 0]

    sugar_center = np.asarray(sugar, dtype=np.float32).mean(axis=0) if sugar else origin
    base_center = np.asarray(base, dtype=np.float32).mean(axis=0) if base else origin + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    phosphate_center = np.asarray(phosphate, dtype=np.float32).mean(axis=0) if phosphate else sugar_center + np.array([0.0, 1.0, 0.0], dtype=np.float32)

    x_axis = safe_normalize(base_center - sugar_center, [1.0, 0.0, 0.0])
    p_vec = safe_normalize(phosphate_center - sugar_center, [0.0, 1.0, 0.0])
    z_axis = safe_normalize(np.cross(x_axis, p_vec), [0.0, 0.0, 1.0])
    y_axis = safe_normalize(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])
    frame = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    return origin.astype(np.float32), frame


def dna_residue_coords(residue):
    coords = []
    groups = []
    for atom in residue_atoms(residue):
        coords.append(atom.coord.astype(np.float32))
        groups.append(dna_atom_group(atom.get_name()))
    return np.asarray(coords, dtype=np.float32), np.asarray(groups, dtype=np.int64)


def aa_property_vector(resname):
    values = np.zeros((len(AA_PROPERTY_NAMES),), dtype=np.float32)
    for prop in AA_PROPERTIES.get(resname, []):
        values[AA_PROPERTY_NAMES.index(prop)] = 1.0
    return values


def residue_orientation_features(protein_residue, dna_origin, dna_frame):
    ca = get_atom_coord(protein_residue, "CA")
    cb = get_atom_coord(protein_residue, "CB")
    if ca is None:
        ca = residue_center(protein_residue)
    if cb is None:
        side_atoms = [
            atom.coord.astype(np.float32)
            for atom in residue_atoms(protein_residue)
            if protein_atom_role(atom.get_name()) == 1
        ]
        cb = np.asarray(side_atoms, dtype=np.float32).mean(axis=0) if side_atoms else None

    if cb is None:
        side_vec_global = np.zeros((3,), dtype=np.float32)
        side_len = 0.0
        has_sidechain = 0.0
    else:
        side_vec = cb - ca
        side_len = float(np.linalg.norm(side_vec))
        side_vec_global = safe_normalize(side_vec, [0.0, 0.0, 0.0])
        has_sidechain = 1.0

    ca_to_dna = safe_normalize(dna_origin - ca, [0.0, 0.0, 0.0])
    return np.concatenate([
        dna_frame.T @ side_vec_global,
        dna_frame.T @ ca_to_dna,
        np.asarray([side_len / 10.0, has_sidechain], dtype=np.float32),
    ]).astype(np.float32)


def parse_complex_cif(cif_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    dna_residues = []
    protein_residues = []

    for residue in structure.get_residues():
        resname = residue.get_resname().strip().upper()
        atoms = residue_atoms(residue)
        if not atoms:
            continue
        if resname in DNA_RESNAMES:
            dna_residues.append(residue)
        elif resname in AA_TO_INDEX:
            protein_residues.append(residue)

    if not dna_residues:
        raise ValueError(f"No DNA residues found in {cif_path}")
    if not protein_residues:
        raise ValueError(f"No protein residues found in {cif_path}")
    return dna_residues, protein_residues


def aa_three_to_one(resname):
    table = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return table.get(resname.strip().upper(), "X")


def alignment_to_template_map(hit, protein_residues):
    if not all(key in hit and hit[key] for key in ("qstart", "tstart", "qaln", "taln")):
        return {}
    try:
        q_pos = int(hit["qstart"]) - 1
        t_pos = int(hit["tstart"]) - 1
    except ValueError:
        return {}

    qaln = str(hit["qaln"])
    taln = str(hit["taln"])
    tlen = max(1, len(protein_residues) - 1)
    q_non_gap = max(1, sum(1 for char in qaln if char != "-") - 1)
    mapping = {}

    for q_char, t_char in zip(qaln, taln):
        current_q = q_pos if q_char != "-" else None
        current_t = t_pos if t_char != "-" else None
        if current_t is not None and 0 <= current_t < len(protein_residues):
            same_aa = 0.0
            if current_q is not None and q_char != "-" and t_char != "-":
                same_aa = 1.0 if q_char.upper() == aa_three_to_one(protein_residues[current_t].get_resname()) else 0.0
            mapping[residue_id(protein_residues[current_t])] = {
                "query_pos_norm": float(max(current_q or 0, 0)) / q_non_gap,
                "template_pos_norm": float(current_t) / tlen,
                "same_aa": same_aa,
            }
        if q_char != "-":
            q_pos += 1
        if t_char != "-":
            t_pos += 1
    return mapping


def residue_distance_summary(protein_residue, dna_coords, dna_groups, contact_cutoff):
    prot_atoms = residue_atoms(protein_residue)
    prot_coords = np.asarray([atom.coord for atom in prot_atoms], dtype=np.float32)
    dist = np.linalg.norm(prot_coords[:, None, :] - dna_coords[None, :, :], axis=-1)
    min_dist = dist.min(axis=1)
    closest_prot_atom = int(min_dist.argmin())
    closest_dna_atom = int(dist[closest_prot_atom].argmin())

    group_mins = []
    group_contacts = []
    for group_idx in range(len(GROUPS)):
        cols = np.where(dna_groups == group_idx)[0]
        if cols.size:
            group_dist = dist[:, cols]
            group_mins.append(float(group_dist.min()))
            group_contacts.append(float((group_dist < contact_cutoff).sum()))
        else:
            group_mins.append(20.0)
            group_contacts.append(0.0)

    return {
        "min_all": float(dist.min()),
        "group_mins": group_mins,
        "contacts_all": float((dist < contact_cutoff).sum()),
        "group_contacts": group_contacts,
        "closest_atom": prot_atoms[closest_prot_atom],
        "closest_group": int(dna_groups[closest_dna_atom]),
        "closest_coord": prot_coords[closest_prot_atom],
        "atom_count": len(prot_atoms),
    }


def rbf(distance):
    return np.exp(-((distance - RBF_CENTERS) / RBF_WIDTH) ** 2).astype(np.float32)


def make_residue_node_feature(protein_residue, dna_origin, dna_frame, dna_base, summary, rank, max_nodes, alignment_info=None):
    resname = protein_residue.get_resname().strip().upper()
    closest_atom = summary["closest_atom"]
    delta_global = summary["closest_coord"] - dna_origin
    delta_local = dna_frame.T @ delta_global
    unit_local = safe_normalize(delta_local, [0.0, 0.0, 0.0])

    feature = np.concatenate([
        one_hot(AA_TO_INDEX[resname], len(AA_RESNAMES)),
        aa_property_vector(resname),
        one_hot(atom_element_index(closest_atom), len(ELEMENTS)),
        one_hot(protein_atom_role(closest_atom.get_name()), len(ATOM_ROLES)),
        one_hot(summary["closest_group"], len(GROUPS)),
        one_hot(dna_base, len(DNA_BASES)),
        np.asarray([
            summary["min_all"] / 10.0,
            summary["group_mins"][0] / 10.0,
            summary["group_mins"][1] / 10.0,
            summary["group_mins"][2] / 10.0,
            np.log1p(summary["contacts_all"]),
            np.log1p(summary["group_contacts"][0]),
            np.log1p(summary["group_contacts"][1]),
            np.log1p(summary["group_contacts"][2]),
            float(summary["atom_count"]) / 30.0,
            float(rank) / max(1, max_nodes - 1),
        ], dtype=np.float32),
        delta_local.astype(np.float32) / 10.0,
        unit_local.astype(np.float32),
        rbf(summary["min_all"]),
        residue_orientation_features(protein_residue, dna_origin, dna_frame),
        np.asarray([
            1.0 if alignment_info else 0.0,
            alignment_info["query_pos_norm"] if alignment_info else 0.0,
            alignment_info["template_pos_norm"] if alignment_info else 0.0,
            alignment_info["same_aa"] if alignment_info else 0.0,
        ], dtype=np.float32),
    ]).astype(np.float32)
    if feature.shape[0] != FEATURE_NAMES.shape[0]:
        raise RuntimeError(f"Feature size mismatch: {feature.shape[0]} != {FEATURE_NAMES.shape[0]}")
    return feature


def interface_node_features(cif_path, max_nodes=32, cutoff=10.0, contact_cutoff=5.0, alignment_map=None, aligned_residue_mode="soft"):
    dna_residues, protein_residues = parse_complex_cif(cif_path)
    alignment_map = alignment_map or {}
    rows = np.zeros((len(dna_residues), max_nodes, FEATURE_NAMES.shape[0]), dtype=np.float32)
    mask = np.zeros((len(dna_residues), max_nodes), dtype=bool)

    for i, dna_residue in enumerate(dna_residues):
        dna_coords, dna_groups = dna_residue_coords(dna_residue)
        dna_origin, dna_frame = dna_residue_frame(dna_residue)
        dna_base = dna_base_index(dna_residue.get_resname())

        summaries = []
        for protein_residue in protein_residues:
            if aligned_residue_mode == "hard" and residue_id(protein_residue) not in alignment_map:
                continue
            summary = residue_distance_summary(protein_residue, dna_coords, dna_groups, contact_cutoff)
            if summary["min_all"] <= cutoff:
                summaries.append((protein_residue, summary))
        if not summaries:
            fallback_residues = protein_residues
            if aligned_residue_mode == "hard":
                fallback_residues = [
                    protein_residue
                    for protein_residue in protein_residues
                    if residue_id(protein_residue) in alignment_map
                ]
            all_summaries = [
                (protein_residue, residue_distance_summary(protein_residue, dna_coords, dna_groups, contact_cutoff))
                for protein_residue in fallback_residues
            ]
            summaries = sorted(all_summaries, key=lambda item: item[1]["min_all"])[:max_nodes]
        else:
            summaries = sorted(summaries, key=lambda item: item[1]["min_all"])[:max_nodes]

        for rank, (protein_residue, summary) in enumerate(summaries[:max_nodes]):
            rows[i, rank] = make_residue_node_feature(
                protein_residue,
                dna_origin,
                dna_frame,
                dna_base,
                summary,
                rank,
                max_nodes,
                alignment_map.get(residue_id(protein_residue)),
            )
            mask[i, rank] = True
    return rows, mask


def onehot_to_base(row):
    row = np.asarray(row)
    if row.size < 4:
        return "N"
    return "ACGT"[int(np.argmax(row[:4]))]


def query_sequence_from_arrays(query_arrays):
    if "Y_hard" not in query_arrays.files:
        return None
    y_hard = np.asarray(query_arrays["Y_hard"])
    if y_hard.ndim == 3:
        return "".join(onehot_to_base(row) for row in y_hard[0])
    if y_hard.ndim == 2:
        if y_hard.shape[0] == 2 and y_hard.shape[-1] != 4:
            return None
        rows = y_hard[0] if y_hard.shape[0] == 2 and y_hard.shape[-1] == 4 else y_hard
        return "".join(onehot_to_base(row) for row in rows)
    return None


def global_align_map(query_seq, template_seq, match=2.0, mismatch=-1.0, gap=-2.0):
    q = query_seq.upper().replace("U", "T")
    t = template_seq.upper().replace("U", "T")
    nq, nt = len(q), len(t)
    score = np.zeros((nq + 1, nt + 1), dtype=np.float32)
    trace = np.zeros((nq + 1, nt + 1), dtype=np.int8)
    for i in range(1, nq + 1):
        score[i, 0] = score[i - 1, 0] + gap
        trace[i, 0] = 1
    for j in range(1, nt + 1):
        score[0, j] = score[0, j - 1] + gap
        trace[0, j] = 2

    for i in range(1, nq + 1):
        for j in range(1, nt + 1):
            diag = score[i - 1, j - 1] + (match if q[i - 1] == t[j - 1] else mismatch)
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            best = max(diag, up, left)
            score[i, j] = best
            trace[i, j] = 0 if best == diag else (1 if best == up else 2)

    mapping = np.full((nq,), -1, dtype=np.int64)
    i, j = nq, nt
    while i > 0 or j > 0:
        step = trace[i, j]
        if i > 0 and j > 0 and step == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or step == 1):
            i -= 1
        else:
            j -= 1
    return mapping


def resize_template_nodes(node_x, node_mask, length):
    if node_x.shape[0] == length:
        return node_x, node_mask
    if length <= 1:
        indices = np.zeros((length,), dtype=np.int64)
    else:
        src_len = max(1, node_x.shape[0] - 1)
        indices = np.rint(np.linspace(0, src_len, length)).astype(np.int64)
    return node_x[indices], node_mask[indices]


def project_template_nodes(node_x, node_mask, template_seq, query_len, query_seq):
    if query_seq and template_seq and len(query_seq) == query_len:
        mapping = global_align_map(query_seq, template_seq)
        out_x = np.zeros((query_len, node_x.shape[1], node_x.shape[2]), dtype=node_x.dtype)
        out_mask = np.zeros((query_len, node_mask.shape[1]), dtype=node_mask.dtype)
        valid = 0
        for query_i, template_i in enumerate(mapping):
            if 0 <= template_i < node_x.shape[0]:
                out_x[query_i] = node_x[template_i]
                out_mask[query_i] = node_mask[template_i]
                valid += 1
        if valid:
            return out_x, out_mask, mapping

    node_x, node_mask = resize_template_nodes(node_x, node_mask, query_len)
    if query_len <= 1:
        mapping = np.zeros((query_len,), dtype=np.int64)
    else:
        src_len = max(1, len(template_seq) - 1 if template_seq else node_x.shape[0] - 1)
        mapping = np.rint(np.linspace(0, src_len, query_len)).astype(np.int64)
    return node_x, node_mask, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--hits", required=True)
    parser.add_argument("--cif-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument("--aligned-residue-mode", choices=["off", "soft", "hard"], default="soft")
    args = parser.parse_args()

    query_path = Path(args.query_npz)
    cif_dir = Path(args.cif_dir)
    query_arrays = np.load(query_path, allow_pickle=True)
    query_len = int(query_arrays["X_dna"].shape[0])
    query_seq = query_sequence_from_arrays(query_arrays)

    hits = load_hits(args.hits, args.top_k)
    template_node_x = np.zeros(
        (args.top_k, query_len, args.max_nodes, FEATURE_NAMES.shape[0]),
        dtype=np.float32,
    )
    template_node_mask = np.zeros((args.top_k, query_len, args.max_nodes), dtype=bool)
    template_scores = np.zeros((args.top_k,), dtype=np.float32)
    template_ids = np.array([""] * args.top_k, dtype="U256")
    template_dna_mappings = np.full((args.top_k, query_len), -1, dtype=np.int64)

    used = []
    for i, hit in enumerate(hits[: args.top_k]):
        cif_path = cif_dir / f"{hit['stem']}.cif"
        if not cif_path.exists():
            print(f"[WARN] template CIF missing: {cif_path}")
            continue
        try:
            template_dna_residues, protein_residues = parse_complex_cif(cif_path)
            alignment_map = {}
            if args.aligned_residue_mode != "off":
                alignment_map = alignment_to_template_map(hit, protein_residues)
            node_x, node_mask = interface_node_features(
                cif_path,
                max_nodes=args.max_nodes,
                cutoff=args.cutoff,
                contact_cutoff=args.contact_cutoff,
                alignment_map=alignment_map,
                aligned_residue_mode=args.aligned_residue_mode,
            )
            template_seq = "".join(dna_base_letter(res.get_resname()) for res in template_dna_residues)
            node_x, node_mask, mapping = project_template_nodes(
                node_x,
                node_mask,
                template_seq,
                query_len,
                query_seq,
            )
        except Exception as exc:
            print(f"[WARN] failed to parse {cif_path.name}: {exc}")
            continue
        template_node_x[i] = node_x
        template_node_mask[i] = node_mask
        template_scores[i] = hit["score"]
        template_ids[i] = hit["stem"]
        template_dna_mappings[i] = mapping
        used.append(hit["stem"])

    payload = {key: query_arrays[key] for key in query_arrays.files}
    payload.update({
        "template_node_x": template_node_x,
        "template_node_mask": template_node_mask,
        "template_scores": template_scores,
        "template_ids": template_ids,
        "template_node_feature_names": FEATURE_NAMES,
        "template_node_level": np.array(["protein_residue"]),
        "template_dna_mappings": template_dna_mappings,
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(json.dumps({
        "out": str(out),
        "used_templates": used,
        "template_node_shape": list(template_node_x.shape),
        "feature_count": int(FEATURE_NAMES.shape[0]),
    }, indent=2))


if __name__ == "__main__":
    main()
