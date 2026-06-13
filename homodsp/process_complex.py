#!/usr/bin/env python3
"""Construct DeepPBS graph features for protein-DNA complexes.

This is the inference-oriented version of the original HomoDSP co-crystal
preprocessor. With --no_pwm it stores the DNA sequence as the placeholder PWM
target so the existing DeepPBS data loader can build graph objects.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback

import numpy as np

from deeppbs import StructureData, splitEntities, cleanProtein, processDNA, cleanDNA
from deeppbs import makeDNACG, makeProteinGraph, loadPWM, computeYAndMask
from deeppbs import getAtomSASA, getAchtleyFactors, getCV, countContacts


INTRA_BP_PARAMETERS = ["buckle", "shear", "stretch", "stagger", "propeller", "opening"]
INTER_BP_PARAMETERS = ["shift", "slide", "rise", "tilt", "roll", "twist"]
BACKBONE_PARAMETERS = ["major_groove_3dna", "minor_groove_3dna"]


def process_one(line: str, config: dict, no_pwm: bool, no_cleanp: bool, debug_errors: bool) -> None:
    pdb_file = line.split(",")[0]
    pwm_id = None if no_pwm else line.split(",")[1]
    pwm = None if no_pwm else loadPWM(pwm_id)
    outdir = config.get("FEATURE_DATA_PATH", "./output")
    os.makedirs(outdir, exist_ok=True)

    try:
        structure = StructureData(os.path.join(config["PDB_FILES_PATH"], pdb_file), name="co_crystal")
    except Exception as exc:
        if debug_errors:
            print("ERROR: structure load failed", pdb_file, repr(exc), flush=True)
            traceback.print_exc()
        return

    protein, dna = splitEntities(structure)
    if not no_cleanp:
        try:
            protein, _ = cleanProtein(protein, add_charge_radius=True)
        except Exception as exc:
            print("ERROR: clean protein error", pdb_file, exc, flush=True)

    dna = cleanDNA(dna, fix_modified_nucleotide_hetflags=True)
    dna_data = processDNA(dna, quiet=False)
    dna_helices = []
    for entity in dna_data[0]["entities"]:
        dna_helices += entity["helical_segments"]
    if len(dna_helices) != 1:
        print("ERROR: helix count problem", len(dna_helices), pdb_file, flush=True)
        return
    dna_helix = dna_helices[0]

    try:
        V_dna, dna_seq, fn, dna_vectors = makeDNACG(dna, dna_helix)
    except Exception:
        print("ERROR: missing C5/OP1/OP1/OP2", pdb_file, flush=True)
        if debug_errors:
            traceback.print_exc()
        return

    if no_pwm:
        Y_pwm = dna_seq
        mask_shape = dna_seq.shape[:2]
        pwm_mask = np.ones(mask_shape, dtype=bool)
        dna_mask = np.ones(mask_shape, dtype=bool)
        aln_score = [None]
    else:
        Y_pwm, pwm_mask, dna_mask, aln_score = computeYAndMask(pwm, dna_seq)

    N = dna_helix["length"]
    X_dna = np.zeros((N, 14))
    dna_feature_names = []
    col = 0
    for param in INTRA_BP_PARAMETERS:
        X_dna[:, col] = np.array(dna_helix["shape_parameters"][param])
        dna_feature_names.append(param)
        col += 1
    for param in INTER_BP_PARAMETERS:
        values = list(dna_helix["shape_parameters"][param])
        values.append(np.mean(values))
        X_dna[:, col] = np.array(values)
        dna_feature_names.append(param)
        col += 1
    for param in BACKBONE_PARAMETERS:
        values = [0 if x == "NA" else x for x in dna_helix["shape_parameters"][param]]
        if "3dna" in param:
            old = list(values)
            values = [old[0]]
            for idx in range(len(old) - 1):
                values.append((old[idx] + old[idx + 1]) / 2)
            values.append(old[-1])
        X_dna[:, col] = np.array(values)
        dna_feature_names.append(param)
        col += 1

    X_dna_point = np.zeros((V_dna.shape[0] * V_dna.shape[1], V_dna.shape[1] + fn.shape[2] + X_dna.shape[1]))
    for i in range(V_dna.shape[0]):
        for j in range(V_dna.shape[1]):
            X_dna_point[i * V_dna.shape[1] + j, j] = 1
            X_dna_point[i * V_dna.shape[1] + j, V_dna.shape[1] : V_dna.shape[1] + fn.shape[2]] = fn[i, j, :]
            X_dna_point[i * V_dna.shape[1] + j, V_dna.shape[1] + fn.shape[2] :] = X_dna[i, :]

    pro_features = ["charge", "radius"]
    pro_features.append(getAtomSASA(protein, classifier=None))
    pro_features += getAchtleyFactors(protein)
    pro_features.append(getCV(protein, 7.5, feature_name="cv", impute_hydrogens=True))
    V_prot, X_prot, E_prot, prot_vectors = makeProteinGraph(protein, feature_names=pro_features)
    contacts = countContacts(protein, pdb_file, V_dna, dna_mask[0])
    print("CONTACT COUNT", contacts[0], pdb_file, flush=True)

    stem = os.path.basename(pdb_file).replace(".pdb", "").replace(".cif", "")
    suffix = "" if no_pwm else f"_{pwm_id}"
    np.savez_compressed(
        os.path.join(outdir, f"{stem}{suffix}.npz"),
        V_dna=V_dna,
        X_dna=X_dna,
        X_dna_point=X_dna_point,
        dna_feature_names=dna_feature_names,
        V_prot=V_prot,
        X_prot=X_prot,
        E_prot=E_prot,
        prot_feature_names=pro_features,
        Y_hard=dna_seq,
        Y_pwm=Y_pwm,
        pwm_mask=pwm_mask,
        dna_mask=dna_mask,
        aln_score=np.array([aln_score]),
        dna_vectors=dna_vectors,
        prot_vectors=prot_vectors,
        contacts=contacts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", help="CSV/list of structure files; add PWM id as second field for training mode.")
    parser.add_argument("config_file", help="JSON with PDB_FILES_PATH and FEATURE_DATA_PATH.")
    parser.add_argument("--no_pwm", action="store_true", default=False)
    parser.add_argument("--no_cleanp", action="store_true", default=False)
    parser.add_argument("--debug_errors", action="store_true", default=False)
    args = parser.parse_args()

    config = json.load(open(args.config_file, "r", encoding="utf-8"))
    for line in [x.strip() for x in open(args.data_file, "r", encoding="utf-8").readlines()]:
        if not line or line.startswith("#"):
            continue
        try:
            process_one(line, config, args.no_pwm, args.no_cleanp, args.debug_errors)
        except Exception as exc:
            if args.debug_errors:
                print("ERROR: unhandled processing error", line, repr(exc), flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()
