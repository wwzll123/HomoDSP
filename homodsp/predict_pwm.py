#!/usr/bin/env python3
"""Run HomoDSP PWM inference for one protein-DNA complex."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from torch_geometric.loader import DataLoader
except Exception:
    from torch_geometric.data import DataLoader

from deeppbs import oneHotToSeq
from deeppbs.nn import processBatch
from deeppbs.nn.utils import loadDataset
from homodsp.models.model_v2 import Model


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def resolve_path(value: str | None, base: Path = REPO_ROOT) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def run_command(cmd: list[str]) -> None:
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def write_pwm_csv(path: Path, pwm: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", "A", "C", "G", "T"])
        for idx, row in enumerate(pwm, start=1):
            writer.writerow([idx] + [float(x) for x in row])


def write_meme(path: Path, pwm: np.ndarray, motif_name: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("MEME version 4\n\n")
        handle.write("ALPHABET= ACGT\n\n")
        handle.write("strands: + -\n\n")
        handle.write("Background letter frequencies\n")
        handle.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        handle.write(f"MOTIF {motif_name}\n")
        handle.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} nsites= 20 E= 0\n")
        for row in pwm:
            handle.write(" ".join(f"{float(x):.8f}" for x in row) + "\n")


def process_complex(input_path: Path, sample_id: str, base_npz_dir: Path, no_cleanp: bool) -> Path:
    base_npz_dir.mkdir(parents=True, exist_ok=True)
    input_dir = base_npz_dir.parent / "input_structures"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied_input = input_dir / input_path.name
    if input_path.resolve() != copied_input.resolve():
        shutil.copy2(input_path, copied_input)

    list_path = base_npz_dir.parent / "input_list.csv"
    config_path = base_npz_dir.parent / "process_config.json"
    list_path.write_text(copied_input.name + "\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "PDB_FILES_PATH": str(input_dir),
                "FEATURE_DATA_PATH": str(base_npz_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "-m",
        "homodsp.process_complex",
        str(list_path),
        str(config_path),
        "--no_pwm",
        "--debug_errors",
    ]
    if no_cleanp:
        cmd.append("--no_cleanp")
    run_command(cmd)
    produced = base_npz_dir / f"{copied_input.stem}.npz"
    target = base_npz_dir / f"{sample_id}.npz"
    if produced.exists() and produced != target:
        produced.replace(target)
    if not target.exists():
        raise FileNotFoundError(f"DeepPBS feature generation did not produce {target}")
    return target


def attach_templates(base_npz: Path, input_path: Path, sample_id: str, out_dir: Path, config: dict) -> Path:
    template_cfg = config["templates"]
    template_cif_dir = resolve_path(template_cfg.get("template_cif_dir"))
    foldseek_db = resolve_path(template_cfg.get("foldseek_db"))
    foldseek = template_cfg.get("foldseek", "foldseek")
    if template_cif_dir is None or foldseek_db is None:
        raise ValueError("templates.template_cif_dir and templates.foldseek_db must be set")
    if not template_cif_dir.exists():
        raise FileNotFoundError(f"Template CIF directory does not exist: {template_cif_dir}")
    if not Path(str(foldseek_db) + ".dbtype").exists() and not foldseek_db.exists():
        raise FileNotFoundError(f"Foldseek DB prefix does not exist: {foldseek_db}")

    template_work = out_dir / "template_search"
    template_work.mkdir(parents=True, exist_ok=True)
    query_list = template_work / "queries.txt"
    query_map = template_work / "query_cif_map.csv"
    query_list.write_text(base_npz.name + "\n", encoding="utf-8")
    query_map.write_text(f"{base_npz.name},{input_path.resolve()}\n", encoding="utf-8")

    script = REPO_ROOT / "homodsp/templates/batch_cif_interface_templates.py"
    cmd = [
        sys.executable,
        str(script),
        "--query-list",
        str(query_list),
        "--query-npz-dir",
        str(base_npz.parent),
        "--query-cif-map",
        str(query_map),
        "--template-cif-dir",
        str(template_cif_dir),
        "--foldseek-db",
        str(foldseek_db),
        "--foldseek",
        str(foldseek),
        "--out-dir",
        str(template_work),
        "--top-k",
        str(template_cfg.get("top_k", 16)),
        "--feature-mode",
        "interface",
        "--max-template-nodes",
        str(template_cfg.get("max_template_nodes", 32)),
        "--cutoff",
        str(template_cfg.get("cutoff", 10.0)),
        "--contact-cutoff",
        str(template_cfg.get("contact_cutoff", 5.0)),
        "--aligned-residue-mode",
        str(template_cfg.get("aligned_residue_mode", "soft")),
        "--flank",
        str(template_cfg.get("flank", 5)),
        "--threads",
        str(template_cfg.get("threads", 4)),
        "--jobs",
        str(template_cfg.get("jobs", 1)),
    ]
    run_command(cmd)
    augmented = template_work / "aug_npz" / base_npz.name
    if not augmented.exists():
        raise FileNotFoundError(f"Template augmentation did not produce {augmented}")
    return augmented


def predict_one_member(npz_path: Path, model_cfg: dict, member_cfg: dict, device: torch.device) -> tuple[np.ndarray, str]:
    member_name = member_cfg.get("name", "model")
    checkpoint = resolve_path(member_cfg.get("checkpoint"))
    scaler_path = resolve_path(member_cfg.get("scaler"))
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist for {member_name}: {checkpoint}")
    if scaler_path is None or not scaler_path.exists():
        raise FileNotFoundError(f"Scaler does not exist for {member_name}: {scaler_path}")

    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    dataset, _, _, datafiles = loadDataset(
        [npz_path.name],
        int(model_cfg.get("nc", 4)),
        model_cfg.get("labels_key", "Y_pwm"),
        str(npz_path.parent),
        cache_dataset=False,
        balance=model_cfg.get("balance", "unmasked"),
        remove_mask=False,
        scale=True,
        scaler=scaler,
        pre_transform=None,
        feature_mask=None,
    )
    if len(dataset) != 1:
        raise RuntimeError(f"Expected one processed sample, got {len(dataset)} from {npz_path}")

    nF_prot = int(model_cfg.get("protein_channels", 13))
    nF_dna = int(model_cfg.get("dna_channels", 14))
    template_node_channels = int(model_cfg.get("template_node_channels", 92))
    template_encoder_type = model_cfg.get("template_encoder_type", "interface")
    if hasattr(dataset[0], "template_node_x"):
        template_node_channels = int(dataset[0].template_node_x.shape[-1])

    model = Model(
        nF_prot,
        nF_dna,
        condition=model_cfg.get("condition", "prot_shape"),
        readout=model_cfg.get("readout", "all"),
        use_templates=bool(model_cfg.get("use_templates", True)),
        template_channels=int(model_cfg.get("template_channels", nF_dna)),
        template_hidden=int(model_cfg.get("template_hidden", 8)),
        template_encoder_type=template_encoder_type,
        template_node_channels=template_node_channels,
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
    batch = next(iter(loader))
    seq = oneHotToSeq(batch.y_hard0.data.cpu().numpy())
    with torch.no_grad():
        batch_data = processBatch(device, batch)
        raw = model(batch_data["batch"])
        logits = raw["logits"] if isinstance(raw, dict) else raw
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    midpoint = probs.shape[0] // 2
    pwm = (probs[:midpoint, :] + np.flip(probs[midpoint:, :], axis=(0, 1))) / 2.0
    print(f"Loaded ensemble member {member_name}: {checkpoint}", flush=True)
    return pwm, seq


def predict_npz(npz_path: Path, sample_id: str, out_dir: Path, config: dict, device_name: str) -> tuple[np.ndarray, str]:
    model_cfg = config["model"]
    device = torch.device(device_name if device_name else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ensemble = model_cfg.get("ensemble")
    if not ensemble:
        ensemble = [
            {
                "name": "single",
                "checkpoint": model_cfg.get("checkpoint"),
                "scaler": model_cfg.get("scaler"),
            }
        ]

    pwms = []
    seq = None
    for member_cfg in ensemble:
        member_pwm, member_seq = predict_one_member(npz_path, model_cfg, member_cfg, device)
        if seq is None:
            seq = member_seq
        elif member_seq != seq:
            raise RuntimeError(f"Ensemble members produced inconsistent DNA sequences: {seq} vs {member_seq}")
        pwms.append(member_pwm)

    pwm = np.mean(np.stack(pwms, axis=0), axis=0)
    print(f"Averaged {len(pwms)} ensemble members", flush=True)
    return pwm, seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Protein-DNA complex in PDB or mmCIF format.")
    parser.add_argument("-o", "--out-dir", default="homodsp_prediction", help="Output directory.")
    parser.add_argument("-c", "--config", default=str(REPO_ROOT / "config/inference_config.json"))
    parser.add_argument("--sample-id", help="Optional sample id. Defaults to input stem.")
    parser.add_argument("--device", default="", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--skip-templates", action="store_true", help="Run the no-template model path if configured.")
    parser.add_argument("--no-clean-protein", action="store_true", help="Skip protein cleaning in DeepPBS preprocessing.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    config = load_config(Path(args.config).expanduser().resolve())
    sample_id = args.sample_id or input_path.stem
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = out_dir / "work"
    base_npz = process_complex(input_path, sample_id, work_dir / "base_npz", args.no_clean_protein)
    infer_npz = base_npz
    if config.get("model", {}).get("use_templates", True) and not args.skip_templates:
        infer_npz = attach_templates(base_npz, input_path, sample_id, work_dir, config)

    pwm, seq = predict_npz(infer_npz, sample_id, out_dir, config, args.device)
    np.save(out_dir / f"{sample_id}_pwm.npy", pwm)
    np.savez_compressed(out_dir / f"{sample_id}_prediction.npz", P=pwm, Seq=seq, source_npz=str(infer_npz))
    write_pwm_csv(out_dir / f"{sample_id}_pwm.csv", pwm)
    np.savetxt(out_dir / f"{sample_id}_pwm.tsv", pwm, delimiter="\t", header="A\tC\tG\tT", comments="")
    write_meme(out_dir / f"{sample_id}.meme", pwm, sample_id)
    summary = {
        "sample_id": sample_id,
        "input": str(input_path),
        "sequence": seq,
        "pwm_csv": str(out_dir / f"{sample_id}_pwm.csv"),
        "pwm_tsv": str(out_dir / f"{sample_id}_pwm.tsv"),
        "pwm_npy": str(out_dir / f"{sample_id}_pwm.npy"),
        "prediction_npz": str(out_dir / f"{sample_id}_prediction.npz"),
        "source_npz": str(infer_npz),
    }
    (out_dir / f"{sample_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
