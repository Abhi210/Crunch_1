# Broad Obesity-1 Perturbation Prediction

**CrunchDAO Hackathon Submission — Broad Obesity-1 Challenge**

Predicts single-cell gene expression responses to unseen genetic perturbations using a cross-attention transformer (DeltaInterpolationNetV9) with scGPT biological embeddings and a Retrieve-and-Shift cell generation procedure.

> **Phase 2:** See [Crunch_2](https://github.com/Abhi210/Crunch_2) for the extension to double-gene perturbations.

---

## Method Overview

The model takes a per-gene feature vector (cross-perturbation expression profile + control statistics) and a scGPT biological embedding, then attends over all 122 training perturbation deltas to predict the mean expression shift for the query gene.

Key components:
- **Cross-attention over training deltas** — interpolates training perturbation responses rather than predicting from scratch, suited for the low-sample regime (122 perturbations)
- **Enriched query and key paths** — both fuse expression features with scGPT embeddings via learned gating, enabling the model to find biologically similar training perturbations even when expression profiles differ
- **Auxiliary std head** — predicts per-HVG gene standard deviations for distribution-accurate cell generation
- **Confidence calibration** — conditioned on attention entropy and embedding outlier distance, shrinks predictions toward the population mean for out-of-distribution genes
- **Retrieve-and-Shift** — samples cells from KNN-selected training perturbations and affine-transforms them to match predicted mean and HVG standard deviations
- **Ensemble of 16 models** — 8 trained on full data, 8 with leave-one-out validation, weighted by held-out Pearson correlation
- **Blended proportion prediction** — neural proportion head + KNN lookup on predicted mean deltas

See [Method description.md](Method%20description.md) for the full rationale.

---

## Repository Structure

```
.
├── main.py                        # All model code: train(), infer(), evaluate()
├── Method description.md          # Full method and rationale write-up
├── notebook.ipynb                 # Development notebook
├── resources/
│   ├── embeddings/
│   │   ├── embedding_info.json    # Metadata for the combined embedding file
│   │   ├── gene_embedding_order.txt  # Gene names aligned to embedding matrix
│   │   └── gene_embeddings_combined.npy  # [NOT TRACKED] 512-dim scGPT gene embeddings
│   └── scGPT_human/
│       ├── args.json              # scGPT whole-human checkpoint training config
│       └── vocab.json             # scGPT gene vocabulary
└── data/                          # [NOT TRACKED] Competition data (see below)
```

---

## Data

The model trains on the **Broad Obesity-1 Challenge dataset** (not included — available from CrunchDAO):

| File | Description |
|------|-------------|
| `data/obesity_challenge_1.h5ad` | scRNA-seq matrix: 88,202 cells × 21,592 genes, 122 gene knockouts + NC controls |
| `data/program_proportion.csv` | Cell program annotations (pre_adipo, adipo, lipo, other) per perturbation |
| `data/predict_perturbations.txt` | List of genes to predict at inference time |
| `data/genes_to_predict.txt` | List of output genes (defaults to all if absent) |
| `data/obesity_challenge_1_local_gtruth.h5ad` | Local ground truth for `evaluate()` (optional) |
| `data/program_proportion_local_gtruth.csv` | Ground truth proportions for `evaluate()` (optional) |

---

## Resources

Large binary files are not tracked in git. You need:

**Gene embeddings** — extracted from the scGPT whole-human checkpoint:
```bash
python extract_foundation_embeddings.py \
    --gene_list data/obesity_challenge_1.h5ad \
    --out_dir resources/embeddings/
```
This produces `resources/embeddings/gene_embeddings_combined.npy` (512-dim, ~42 MB) and `gene_embedding_order.txt`.

**Model checkpoint** — produced by `train()`:
```
resources/ensemble_v9.pt
```

---

## Setup

```bash
pip install torch numpy pandas anndata scanpy scikit-learn scipy tqdm matplotlib
```

The scGPT vocabulary (`resources/scGPT_human/vocab.json`) is included. The full scGPT model weights are only needed to re-extract embeddings.

---

## Usage

All entry points are in `main.py`:

### Train
```python
from main import train
train(
    data_directory_path  = "data/",
    model_directory_path = "resources/",
)
# Saves: resources/ensemble_v9.pt
```

### Infer
```python
from main import infer
import pandas as pd, anndata as ad

perts = pd.read_csv("data/predict_perturbations.txt", header=None)[0].tolist()
genes = ad.read_h5ad("data/obesity_challenge_1.h5ad").var_names.tolist()

infer(
    data_directory_path              = "data/",
    prediction_directory_path        = "outputs/",
    prediction_h5ad_file_path        = "outputs/prediction.h5ad",
    program_proportion_csv_file_path = "outputs/predict_program_proportion.csv",
    model_directory_path             = "resources/",
    predict_perturbations            = perts,
    genes_to_predict                 = genes,
)
```

### Evaluate (requires local ground truth)
```python
from main import evaluate
results = evaluate(
    data_directory_path   = "data/",
    model_directory_path  = "resources/",
    output_directory_path = "outputs/",
)
```

---

## Key Hyperparameters

All hyperparameters are defined in the `Config` dataclass at the top of `main.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 256 | Transformer hidden dim |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 4 | Transformer encoder layers |
| `epochs` | 1200 | Max training epochs |
| `n_ensemble` | 16 | Ensemble size (8 full + 8 LOO) |
| `k_retrieve` | 10 | KNN neighbours for Retrieve-and-Shift |
| `cells_per_pert` | 100 | Synthetic cells generated per perturbation |
| `hvg_extra_w` | 4.0 | Extra loss weight on highly variable genes |

---

## Official Leaderboard Scores

Scores from the CrunchDAO Broad Obesity-1 leaderboard (held-out test set — not reproducible locally):

| Metric | Leaderboard Score |
|--------|-------------------|
| Pearson Delta | 0.081 |
| MMD | 0.083 |
| L1 Distance | 0.150 |
