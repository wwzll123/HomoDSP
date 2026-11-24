"""Frozen-representation features used by the standard HomoDSP error regressors."""

import numpy as np
import torch
from scipy.special import xlogy

from deeppbs.template_schema import DNA_INDICES
from homodsp.models.template_geometry import masked_softmax


class Capture:
    def __init__(self, model):
        self.rows = []

        def before(module, args, kwargs):
            x = args[0]
            mask = kwargs["template_node_mask"].bool()
            base = (x[..., DNA_INDICES[:4]] * mask[..., None]).sum(-2) / mask.sum(
                -1
            ).clamp_min(1)[..., None]
            self.rows.append(
                dict(
                    query=kwargs["query_features"].detach(),
                    valid=mask.any(-1),
                    base=base.detach(),
                )
            )

        def after(module, args, out):
            self.rows[-1]["template"] = out.detach()

        def encoded(module, args, out):
            self.rows[-1]["encoded"] = args[0][..., :32].detach()
            self.rows[-1]["score"] = out.squeeze(-1).detach()

        self.hooks = [
            model.template_encoder.register_forward_pre_hook(before, with_kwargs=True),
            model.template_encoder.register_forward_hook(after),
            model.template_encoder.template_compatibility.register_forward_hook(
                encoded
            ),
        ]

    def clear(self):
        self.rows.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


def js(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    m = (p + q) / 2
    return (0.5 * (xlogy(p, p / m) + xlogy(q, q / m))).sum(-1) / np.log(2)


def features(capture, readout, mask, p, d, q, meta):
    assert len(capture.rows) == 2
    n = len(p)
    tpl = []
    query = torch.cat([r["query"] for r in capture.rows])[mask].cpu().numpy()
    template = torch.cat([r["template"] for r in capture.rows])[mask].cpu().numpy()
    read = readout[mask].cpu().numpy()
    assert (query.shape[1], template.shape[1], read.shape[1]) == (46, 32, 32)
    arrays = [query, template, read, p, d, q]
    names = [
        f"{k}_{i:02d}"
        for k, size in [
            ("query", 46),
            ("template", 32),
            ("readout", 32),
            ("hprob", 4),
            ("dprob", 4),
            ("qprob", 4),
        ]
        for i in range(size)
    ]
    for key, prob in [("h", p), ("d", d), ("q", q)]:
        sortedp = np.sort(prob, axis=-1)
        arrays.extend(
            [
                -(prob * np.log(np.clip(prob, 1e-12, 1))).sum(-1)[:, None] / np.log(4),
                sortedp[:, -1, None],
                (sortedp[:, -1] - sortedp[:, -2])[:, None],
            ]
        )
        names.extend(
            [key + "_entropy", key + "_max_probability", key + "_probability_margin"]
        )
    for name, a, b in [("JS_h_d", p, d), ("JS_h_q", p, q), ("JS_d_q", d, q)]:
        arrays.append(js(a, b)[:, None])
        names.append(name)
    offset = 0
    length = capture.rows[0]["query"].shape[0]
    for strand, r in enumerate(capture.rows):
        use = mask[strand * length : (strand + 1) * length]
        v = r["valid"][:, use]
        e = r["encoded"][:, use]
        count = v.sum(0)
        mean = (e * v[..., None]).sum(0) / count.clamp_min(1)[:, None]
        var = ((e - mean[None]).square() * v[..., None]).sum(0) / count.clamp_min(1)[
            :, None
        ]
        weight = masked_softmax(r["score"][:, use], v, dim=0)
        ent = (
            -(weight * weight.clamp_min(1e-12).log()).sum(0)
            / count.clamp_min(2).float().log()
        )
        summary = (
            torch.stack(
                [var.mean(-1), ent, count.float() / 16, count.gt(0).float()], -1
            )
            .cpu()
            .numpy()
        )
        base = r["base"][:, use].cpu().numpy()
        assert np.all((base >= -1e-6) & (base <= 1 + 1e-6))
        known = (base.sum(-1) > 0.99) & v.cpu().numpy()
        nn = int(use.sum())
        qp = q[offset : offset + nn]
        offset += nn
        likelihood = (base * qp[None]).sum(-1)
        logodds = np.log2(((1 - 0.01) * likelihood + 0.01 / 4) / 0.25)
        cnt = known.sum(0)
        den = np.maximum(cnt, 1)
        comp = (logodds * known).sum(0) / den
        sd = np.sqrt((((logodds - comp[None]) ** 2) * known).sum(0) / den)
        comp[cnt == 0] = np.nan
        sd[cnt == 0] = np.nan
        tpl.append(np.c_[summary, comp, sd, cnt > 0])
    assert offset == n
    arrays.append(np.concatenate(tpl))
    names += [
        "template_latent_variance",
        "template_attention_entropy",
        "template_valid_fraction",
        "has_template",
        "local_template_query_compatibility",
        "local_template_query_compatibility_sd",
        "local_compatibility_available",
    ]
    global_values = [
        meta["JSD"],
        meta["compatibility"],
        float(meta["has_compatibility"]),
        meta["mean_iptm"],
        float(np.isfinite(meta["mean_iptm"])),
        meta["max_min_tm"],
        float(np.isfinite(meta["max_min_tm"])),
    ]
    arrays.append(np.tile(global_values, (n, 1)))
    names += [
        "global_JSD",
        "global_compatibility",
        "global_compatibility_available",
        "mean_iptm",
        "iptm_available",
        "max_min_tm",
        "tm_available",
    ]
    x = np.concatenate(arrays, -1).astype(np.float32)
    assert x.shape == (n, 148) and len(names) == 148 and not np.isinf(x).any()
    return x, names


def summarize(capture, readout, pm, prob):
    assert (
        len(capture.query)
        == len(capture.template)
        == len(capture.encoded)
        == len(capture.score)
        == 2
    )
    q = torch.cat(capture.query)[pm]
    t = torch.cat(capture.template)[pm]
    r = readout[pm]
    assert (q.shape[1], t.shape[1], r.shape[1]) == (46, 32, 32)
    pooled = torch.cat([q.mean(0), t.mean(0), r.mean(0)]).cpu().numpy()
    names = [
        f"{block}_mean_{i:02d}"
        for block, n in [("query", 46), ("template", 32), ("readout", 32)]
        for i in range(n)
    ]
    length = capture.query[0].shape[0]
    variance = []
    entropy = []
    coverage = []
    counts = []
    for s in range(2):
        mask = pm[s * length : (s + 1) * length]
        valid = capture.valid[s][:, mask]
        emb = capture.encoded[s][:, mask]
        score = capture.score[s][:, mask]
        count = valid.sum(0)
        mean = (emb * valid[..., None]).sum(0) / count.clamp_min(1)[:, None]
        var = ((emb - mean[None]).square() * valid[..., None]).sum(0) / count.clamp_min(
            1
        )[:, None]
        variance.append(var.mean(-1))
        w = masked_softmax(score, valid, dim=0)
        ent = -(w * w.clamp_min(1e-12).log()).sum(0) / count.clamp_min(2).float().log()
        entropy.append(ent)
        coverage.append(count.gt(0).float())
        counts.append(count.float() / 16)
    scalars = {
        "query_positional_std": float(q.std(0, unbiased=False).mean()),
        "template_positional_std": float(t.std(0, unbiased=False).mean()),
        "readout_positional_std": float(r.std(0, unbiased=False).mean()),
        "template_latent_variance": float(torch.cat(variance).mean()),
        "template_attention_entropy": float(torch.cat(entropy).mean()),
        "template_position_coverage": float(torch.cat(coverage).mean()),
        "mean_valid_template_fraction": float(torch.cat(counts).mean()),
        "PWM_entropy": float(
            -(prob * prob.clamp_min(1e-12).log()).sum(-1).mean() / np.log(4)
        ),
        "PWM_max_probability": float(prob.max(-1).values.mean()),
    }
    return np.r_[pooled, list(scalars.values())], names + list(scalars)
