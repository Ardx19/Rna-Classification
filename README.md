# RNA Classification Submission Script

## Team Members

Group 45: Ayush Singh (2023163), Aryan Dutt (2023143), Aryan Dahiya (2023142)

This folder contains a standalone script pipeline for miRNA classification:

- `Group_45.py`

Competition Link : https://www.kaggle.com/competitions/bdmh-assignment-2-2026/

The script performs:

1. Feature engineering (taxonomy, structure, fuzzy matching, v9 extra features).
2. CPU model training and evaluation via GroupKFold.
3. Optional RNA-FM embedding branch.
4. Ensemble stacking and submission file generation.

## Model Architecture and How It Works

This is a multi-stage hybrid architecture that combines handcrafted biological features with pretrained sequence embeddings.

1. Input and preprocessing
  - Loads `train.csv`, `test.csv`, and `sample.csv`.
  - Normalizes RNA sequences to uppercase and converts T to U.
  - Uses GroupKFold splitting based on sequence similarity groups to reduce leakage from highly similar sequences.

2. Knowledge-driven matching and feature extraction
  - Exact + fuzzy matching against miRBase mature sequences (including near matches with small edit distance).
  - Taxonomy and seed-family features from matched entries.
  - Hairpin structural features using ViennaRNA (`RNA.fold`, `RNA.pf_fold`) such as MFE, loop/bulge patterns, overhang asymmetry, and MFE z-score.
  - Additional v9 features:
    - miRNA.dat-derived evidence features (references, experiments, reads, mature product counts).
    - Triplet structure-sequence features from precursor folds.

3. CPU base ensemble models (feature blocks)
  - Trains three model families on each feature block:
    - Logistic Regression (scaled/log features)
    - LightGBM (tree-based on raw features)
    - CatBoost (tree-based on raw features)
  - Averages these model probabilities to produce block-level predictions.
  - Compares feature blocks (`taxon_only`, `taxon_struct`, `taxon_struct_v9plus`) and keeps the best CPU branch.

4. Optional RNA-FM embedding branch
  - Loads pretrained RNA-FM and extracts frozen sequence embeddings (no fine-tuning).
  - Mean-pools token embeddings and applies PCA reduction.
  - Concatenates PCA embeddings with v9+ handcrafted features.
  - Trains the same LR + LightGBM + CatBoost blend under GroupKFold.

5. Final meta-ensemble and submission selection
  - Combines the best CPU branch and embedding branch using:
    - Logistic meta-stack
    - Rank average
    - Weighted blend
  - Selects the best validation AUC option automatically.
  - Writes all candidate outputs (`--all on`) or only best output (`--all off`).

## Environment

Recommended:

- Python `3.10+` (tested in notebook context with Python `3.12`)
- OS: Windows/Linux/macOS
- Optional CUDA-equipped GPU for faster RNA-FM embedding extraction

### GPU Notes

- GPU is **optional**.
- With `--gpu off` (default), the script runs on CPU.
- With `--gpu on`, CUDA is used if available; otherwise it falls back to CPU with a warning.
- With `--gpu auto`, CUDA is used when available.

## Dependencies

The script currently installs required packages at runtime (inside `Group_45.py`) using pip:

- `ViennaRNA`
- `multimolecule`
- `transformers`
- `torch`
- `catboost`

It also imports:

- `numpy`, `pandas`, `lightgbm`, `scikit-learn`, `scipy`

If these are not installed already, install them in your environment.

## requirements.txt

If you prefer environment setup via a requirements file, use the following:

```txt
numpy
pandas
scipy
scikit-learn
lightgbm
catboost
torch
transformers
multimolecule
ViennaRNA
```

Install with:

```bash
pip install -r requirements.txt
```

## Required Input Files

By default the script reads from `./input`.

Required:

- `train.csv`
- `test.csv`
- `sample.csv`
- `mature.fa`
- `hairpin.fa`

FASTA source:

- `mature.fa` and `hairpin.fa` were retrieved from: https://mirbase.org/download/

Optional:

- `miRNA.dat`
  - If missing, miRNA.dat-based features are zero-filled.

## Expected Input Directory Layout

```text
Group_45/
  Group_45.py
  input/
    train.csv
    test.csv
    sample.csv
    mature.fa
    hairpin.fa
    miRNA.dat          # optional
  output/
```

## CLI Flags

`Group_45.py` supports:

- `--input_path` (default: `./input`)
- `--output_path` (default: `./output`)
- `--gpu` (`off|on|auto`, default: `off`)
- `--all` (`on|off`, default: `on`)

## Run Commands

From this folder:

```bash
python Group_45.py
```

Custom paths:

```bash
python Group_45.py --input_path ./input --output_path ./output
```

GPU mode examples:

```bash
python Group_45.py --gpu off
python Group_45.py --gpu auto
python Group_45.py --gpu on
```

Control number of output files:

```bash
# Save all candidate submissions
python Group_45.py --all on

# Save only the best submission file
python Group_45.py --all off
```

## Output Files

All outputs are written to `--output_path`.

- With `--all on`:
  - CPU-only variants (from feature-block comparison)
  - Ensemble candidates
  - Best final file (`sub_v5_BEST.csv`)

- With `--all off`:
  - Only `sub_v5_BEST.csv`

## Notes

- The script currently sets `HF_TOKEN` inline. If needed, replace this with your own token management strategy (for example, environment variable injection outside the script).
- RNA-FM embedding extraction can be slow on CPU.
