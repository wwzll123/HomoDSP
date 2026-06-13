#!/usr/bin/env python
"""Extract a protein-only DNA-interface patch for Foldseek query."""

import argparse
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio


DNA_RESNAMES = {
    "DA", "DC", "DG", "DT", "DI", "DU",
    "A", "C", "G", "T", "U",
}

AA_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
}


def load_structure(path):
    atoms = strucio.load_structure(str(path))
    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]
    atoms = atoms[atoms.element != "H"]
    return atoms


def residue_keys(atoms):
    return list(zip(atoms.chain_id.tolist(), atoms.res_id.tolist(), atoms.ins_code.tolist()))


def expand_by_sequence(atoms, selected_keys, flank):
    if flank <= 0:
        return selected_keys
    expanded = set(selected_keys)
    for chain_id in sorted(set(atoms.chain_id.tolist())):
        chain_mask = atoms.chain_id == chain_id
        chain_atoms = atoms[chain_mask]
        ordered = []
        seen = set()
        for key in residue_keys(chain_atoms):
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        selected_positions = [i for i, key in enumerate(ordered) if key in selected_keys]
        for pos in selected_positions:
            lo = max(0, pos - flank)
            hi = min(len(ordered), pos + flank + 1)
            expanded.update(ordered[lo:hi])
    return expanded


def extract_patch(input_path, output_path, cutoff, flank):
    atoms = load_structure(input_path)
    dna = atoms[[res.upper() in DNA_RESNAMES for res in atoms.res_name]]
    protein = atoms[[res.upper() in AA_RESNAMES for res in atoms.res_name]]
    if dna.array_length() == 0:
        raise ValueError(f"No DNA atoms found in {input_path}")
    if protein.array_length() == 0:
        raise ValueError(f"No protein atoms found in {input_path}")

    cell_list = struc.CellList(dna, cell_size=cutoff)
    contacts = cell_list.get_atoms(protein.coord, radius=cutoff)
    contact_atom_mask = contacts != -1
    selected_atom_indices = contact_atom_mask.any(axis=1)
    selected_keys = set(key for key, keep in zip(residue_keys(protein), selected_atom_indices) if keep)
    selected_keys = expand_by_sequence(protein, selected_keys, flank)

    keep = [key in selected_keys for key in residue_keys(protein)]
    patch = protein[keep]
    if patch.array_length() == 0:
        raise ValueError(f"No protein interface atoms selected from {input_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strucio.save_structure(str(output_path), patch)
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"dna_atoms={dna.array_length()}")
    print(f"protein_atoms={protein.array_length()}")
    print(f"patch_atoms={patch.array_length()}")
    print(f"patch_residues={len(set(residue_keys(patch)))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--flank", type=int, default=5)
    args = parser.parse_args()
    extract_patch(args.input, args.output, args.cutoff, args.flank)


if __name__ == "__main__":
    main()
