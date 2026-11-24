"""Inference for the calibrated sample- and position-level PWM error heads."""

import copy
import csv
import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from homodsp.models.model_v2 import Model
from homodsp.templates.attach_cif_interface_gnn_features import target_to_cif_stem
from homodsp.uq_features import Capture, features, js, summarize


def _standardize(values, scaler):
    # Match the float32 input and in-place operations of sklearn StandardScaler.
    values = np.array(values, dtype=np.float32, copy=True)
    values -= scaler["mean"].numpy()
    values /= scaler["scale"].numpy()
    return values


def _auxiliary_probs(data, raw, mask, checkpoint, *, radial):
    settings = (
        dict(
            query_encoder_type="radial",
            query_prot_hidden=32,
            query_dna_hidden=32,
            fusion_hidden=32,
            fusion_depth=3,
        )
        if radial
        else {}
    )
    model = Model(13, 14, condition="prot_shape", use_templates=False, **settings)
    model.load_state_dict(checkpoint["weights"])
    device = data.x_prot.device
    model.to(device).eval()
    auxiliary = copy.copy(data)
    protein = raw["X_prot"].astype(np.float32, copy=True)
    protein[:, 4:] = _standardize(protein[:, 4:], checkpoint["protein"])
    auxiliary.x_prot = torch.as_tensor(protein, device=device)
    auxiliary.x_dna = torch.as_tensor(
        _standardize(raw["X_dna"], checkpoint["dna"]), device=device
    )
    return model(auxiliary).softmax(-1)[mask].cpu().numpy()


def _average_heads(heads, values):
    estimates = []
    for head in heads:
        scaler = head["scaler"]
        filled = np.where(np.isnan(values), scaler["median"].numpy(), values)
        x = torch.from_numpy(
            ((filled - scaler["mean"].numpy()) / scaler["scale"].numpy()).astype(
                np.float32
            )
        )
        state = head["state"]
        for layer in ("hidden.0", "hidden.3"):
            x = F.silu(F.linear(x, state[layer + ".weight"], state[layer + ".bias"]))
        logits = F.linear(x, state["output.weight"], state["output.bias"])
        estimates.append(float((2 * logits.sigmoid()).mean()))
    return float(np.mean(estimates))


class FoldErrorEstimator:
    def __init__(self, npz_path, hits_path, weights_dir):
        self.weights_dir = Path(weights_dir)
        with np.load(npz_path, allow_pickle=True) as arrays:
            self.raw = {key: arrays[key] for key in ("X_prot", "X_dna")}
            ids = arrays["template_ids"].astype(str)
            active = arrays["template_node_mask"].any(axis=(1, 2))
        used = set(ids[active])
        tm_scores = {}
        if used:
            with Path(hits_path).open(encoding="utf-8") as handle:
                for row in csv.reader(handle, delimiter="\t"):
                    name = target_to_cif_stem(row[1])
                    if name in used:
                        tm_scores.setdefault(name, min(float(row[2]), float(row[3])))
            if used != tm_scores.keys():
                raise ValueError(
                    "Foldseek hits do not match the templates supplied to UQ"
                )
        self.max_tm = max(tm_scores.values(), default=np.nan)
        iptm = []
        if used:
            with gzip.open(
                self.weights_dir / "template_iptm.tsv.gz", "rt", encoding="ascii"
            ) as handle:
                for name, value in csv.reader(handle, delimiter="\t"):
                    if name in used:
                        iptm.append(float(value))
        self.mean_iptm = np.mean(iptm) if len(iptm) == len(used) and used else np.nan
        self.fold_errors = []

    @torch.no_grad()
    def forward(self, model, batch, member_name):
        data = batch["batch"]
        mask = torch.cat([batch["dna_mask0"], batch["dna_mask1"]])
        bundle = torch.load(
            self.weights_dir / (member_name + ".pt"),
            map_location="cpu",
            weights_only=True,
        )
        capture = Capture(model)
        try:
            logits, readout = model(data, return_features=True)
        finally:
            capture.remove()
        prob = logits.softmax(-1)[mask]
        p = prob.cpu().numpy()
        deep = _auxiliary_probs(data, self.raw, mask, bundle["deep"], radial=False)
        query = _auxiliary_probs(data, self.raw, mask, bundle["query"], radial=True)

        # Compatibility uses the forward strand and templates with >=80% coverage.
        base = capture.rows[0]["base"][:, batch["dna_mask0"]].cpu().numpy()
        known = base.sum(-1) > 0.99
        likelihood = (base * query[None, : base.shape[1], :]).sum(-1)
        log_odds = np.log2((0.99 * likelihood + 0.01 / 4) / 0.25)
        eligible = known.mean(axis=1) >= 0.8
        compatibility = np.nan
        if eligible.any():
            compatibility = (
                (log_odds * known).sum(1)[eligible] / known.sum(1)[eligible]
            ).mean()
        meta = dict(
            JSD=float(js(p, deep).mean()),
            compatibility=compatibility,
            has_compatibility=bool(eligible.any()),
            mean_iptm=self.mean_iptm,
            max_min_tm=self.max_tm,
        )
        local, local_names = features(capture, readout, mask, p, deep, query, meta)
        streams = SimpleNamespace(
            **{
                key: [r[key] for r in capture.rows]
                for key in ("query", "template", "encoded", "score", "valid")
            }
        )
        pooled, names = summarize(streams, readout, mask, prob)
        pooled = np.r_[
            pooled, meta["JSD"], compatibility, float(eligible.any())
        ].astype(np.float32)
        names += ["JSD", "compatibility", "has_compatibility"]
        if (
            names != bundle["sample_features"]
            or local_names != bundle["position_features"]
        ):
            raise ValueError("UQ feature names do not match the checkpoint")
        error = (
            _average_heads(bundle["sample"], pooled[None])
            + _average_heads(bundle["position"], local)
        ) / 2
        if not np.isfinite(error):
            raise ValueError("Non-finite UQ estimate")
        self.fold_errors.append(error)
        return logits

    def mean_fold_mae(self):
        if len(self.fold_errors) != 5:
            raise ValueError("Expected five fold error estimates")
        return float(np.mean(self.fold_errors))
