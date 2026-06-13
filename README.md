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

## External Data

Five HQ fold model checkpoints are included under `checkpoints/`. Inference
runs all five models and averages their PWM probabilities. The template library
is not included because it is large. Download [HomoDSP_DataSet](https://doi.org/10.5281/zenodo.20675636) separately and
set these paths in `config/inference_config.json`:

```json
{
  "templates": {
    "template_cif_dir": "/path/to/HomoDSP_DataSet/protenix_high_quality_cifs_iptm0.4_ptm0.5",
    "foldseek_db": "/path/to/HomoDSP_DataSet/foldseek_db/protenix_high_quality_iptm0.4_ptm0.5"
  }
}
```

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
  required by DeepPBS DNA shape preprocessing.

DeepPBS structure preprocessing may also require the same structural biology
tools available in the original DeepPBS environment.

## Usage

Edit `config/inference_config.json` first, especially the two template paths and
the Foldseek executable path if needed.

```bash
python -m homodsp.predict_pwm examples/example.cif -o outputs/example
```

To force CPU:

```bash
python -m homodsp.predict_pwm examples/example.cif -o outputs/example --device cpu
```

To skip protein cleaning if the cleaner fails on a structure:

```bash
python -m homodsp.predict_pwm examples/example.cif -o outputs/example --no-clean-protein
```

To debug the base DeepPBS feature generation alone:

```bash
python -m homodsp.process_complex input_list.csv process_config.json --no_pwm --debug_errors
```

## Configuration

`config/inference_config.json` records:

- five-fold model checkpoint and scaler paths
- template library path
- Foldseek DB prefix
- top-k templates
- interface patch cutoff and flanking residues
- per-DNA-position template node count

The included default corresponds to the five HQ fold models trained with top-16
interface templates.
