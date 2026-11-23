# HomoDSP

HomoDSP predicts a DNA-binding PWM from a protein-DNA complex structure.

This repository is the inference-only package distilled from the training
workspace. It accepts a PDB or mmCIF protein-DNA complex, builds DeepPBS graph
features, retrieves structural templates with Foldseek, attaches template
features, and writes the predicted PWM.

## Inputs and Outputs

Input:

- One protein-DNA complex structure in `.pdb` or `.cif` format.
- The default assumption is one protein chain bound to one DNA double helix.

Outputs:

- `<sample>_pwm.csv`: PWM with columns `A,C,G,T`.
- `<sample>_pwm.tsv`: tab-separated PWM.
- `<sample>_pwm.npy`: NumPy PWM array.
- `<sample>_prediction.npz`: compressed NumPy output with `P` and `Seq`.
- `<sample>.meme`: MEME-format motif.
- `<sample>_summary.json`: paths and inferred DNA sequence.

## Models and checkpoints

Both variants use a DeepPBS-derived query encoder with radial distance features,
a query-conditioned template encoder with three geometric message-passing
blocks and latent coordinate and edge updates, and a residual convolutional PWM
prediction head. Both omit explicit DNA-base identity from the query inputs.

| Model | CLI variant | Explicit template DNA identity | Checkpoints |
| --- | --- | --- | --- |
| HomoDSP with template DNA identity (default) | `template-dna` | Included | [Five folds](checkpoints/template-dna) |
| HomoDSP without template DNA identity | `no-template-dna` | Masked | [Five folds](checkpoints/no-template-dna) |

The two models were trained independently with fold-specific filtered template
libraries. Each fold includes its best checkpoint and its DNA-shape and protein
feature scalers. Inference retrieves up to 16 templates from the FULL HomoDB
library for either variant. For each fold, the reverse-strand probabilities are
mapped back to the forward orientation and averaged with the forward-strand
probabilities. The reported PWM averages these probabilities across the five folds.

The original HQ model and instructions are available in the
[previous revision](https://github.com/wwzll123/HomoDSP/tree/5abdfb9a5c03e9b4af8f19f71b42b43f19334d57).

## External data

Download and extract [HomoDSP_DataSet](https://doi.org/10.5281/zenodo.20675636).
In `config/inference_config.json`, replace `/path/to/HomoDSP_DataSet` in
`templates.foldseek_db`, `templates.template_cif_dir` and
`templates.additional_cif_dirs`. The FULL Foldseek database prefix is
`foldseek_db/protenix_top100`. The distributed archive stores its CIFs in
both `protenix_top100_cifs` and `protenix_high_quality_cifs_iptm0.4_ptm0.5`.
If all FULL library CIFs are in a single directory, point `template_cif_dir`
to that directory and set `additional_cif_dirs` to `[]`.

## Environment

| package | version |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.7.1 |
| torch-geometric | 2.7.0 |
| torch-cluster | 1.6.3+pt27cu126 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| scikit-learn | 1.7.1 |
| biopython | 1.85 |
| biotite | 1.4.0 |
| matplotlib | 3.10.5 |
| seaborn | 0.13.2 |
| tqdm | 4.67.1 |

Runtime tools:

- Foldseek executable available as `foldseek` or configured by absolute path
- 3DNA-DSSR/3DNA tools available in `PATH`; `x3dna-dssr` and `analyze` are
  required by DeepPBS DNA shape preprocessing. Set `X3DNA` to the 3DNA
  installation directory containing `config/`.

DeepPBS structure preprocessing may also require the same structural biology
tools available in the original DeepPBS environment.

## Usage

Run from the repository root with your own query structure. Set
`templates.foldseek` to the executable path if Foldseek is outside `PATH`.

```bash
python -m homodsp.predict_pwm /path/to/query.cif --variant template-dna -o outputs/with_template_dna
python -m homodsp.predict_pwm /path/to/query.cif --variant no-template-dna -o outputs/without_template_dna
```

Use `--device cpu` to select the CPU or `--device cuda:0` for a GPU.
Use `--no-clean-protein` to skip protein cleaning on a prepared structure.

To debug the base DeepPBS feature generation alone:

```bash
python -m homodsp.process_complex input_list.csv process_config.json --no_pwm --debug_errors
```

## Cite

```
@article {Zeng2026.06.16.732781,
	author = {Zeng, Wenwu and Zou, Haitao and Li, Xiaoyu and Liu, Yaoyu and Xu, Liwen and Wang, Xiaoqi and Peng, Shaoliang},
	title = {DNA-binding specificity recognition from predicted homologous protein-DNA structures},
	elocation-id = {2026.06.16.732781},
	year = {2026},
	doi = {10.64898/2026.06.16.732781},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/06/17/2026.06.16.732781},
	eprint = {https://www.biorxiv.org/content/early/2026/06/17/2026.06.16.732781.full.pdf},
	journal = {bioRxiv}
}

```
