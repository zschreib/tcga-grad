# tcga-grad

Modular deep learning pipeline for cancer subtype classification and gradient attribution 
from bulk RNA-seq expression data. Demonstrated on GSE96058 (SCAN-B breast cancer cohort).


## Background

PAM50 molecular subtyping is a clinically validated method for classifying breast tumors into five intrinsic subtypes (Luminal A, Luminal B, Basal-like, HER2-enriched, and Normal-like), each associated with distinct prognoses and treatment pathways. Established tools like the Prosigna assay and the genefu R package implement centroid-based classification using the 50 PAM50 marker genes, and machine learning approaches to PAM50 classification are well documented in the literature. The novel focus of this project is interpretability: after training a neural network on full transcriptome RNA-seq data (30,865 measured features in GSE96058), gradient attribution is applied to examine which PAM50 marker genes the model relies on, and whether the signal is concentrated in known subtype-specific markers or distributed across the broader expression program.

## Approach

A multilayer perceptron (MLP) was trained on RNA-seq expression data from GSE96058 (SCAN-B cohort, 3,273 breast cancer patients plus 136 replicates, 3,409 total samples) to classify PAM50 subtypes directly from normalized gene expression profiles. The pipeline includes stratified train/validation/holdout splits, class-weighted loss to handle subtype imbalance, hyperparameter grid search, and gradient-based attribution analysis filtered to known PAM50 marker genes.

## Results

The model achieves **88% accuracy** and a **macro F1 of 0.85** on a held-out patient set never seen during training or hyperparameter tuning.

| Subtype | Precision | Recall | F1 |
|---------|-----------|--------|----|
| LumA    | 0.93      | 0.90   | 0.91 |
| LumB    | 0.83      | 0.85   | 0.84 |
| Basal   | 0.91      | 0.96   | 0.94 |
| Her2    | 0.79      | 0.87   | 0.83 |
| Normal  | 0.74      | 0.68   | 0.71 |

AUROC: **0.97** across all five subtypes.

## Gradient Attribution

Where this project differs from centroid-based tools is interpretability. After training, gradient attribution was applied to identify which PAM50 genes most influenced each subtype prediction. Gradients were computed with respect to the input expression values and averaged across correctly predicted holdout samples per class.

Attribution reveals diffuse signal across PAM50 genes rather than concentration on a small number of subtype-specific markers, consistent with the known biology of PAM50 subtypes, which are defined by coordinated expression programs rather than single-gene thresholds. This aligns with the validation loss plateau observed during training, suggesting the model learned broad transcriptomic patterns rather than marker-specific rules.

This finding motivates future work using more precise attribution methods such as integrated gradients or SHAP, which may better resolve gene-level contributions in high-dimensional expression data.

## Project Structure
```
tcga-grad/
├── data/               # downloaded automatically, gitignored
├── results/            # plots, grid search log, attribution CSVs
├── src/
│   ├── dataset.py      # GEO data download, alignment, label encoding
│   ├── model.py        # TcgaNet MLP architecture
│   ├── trainer.py      # training loop, evaluation, metrics, plotting
│   ├── train.py        # single run entry point
│   ├── search.py       # hyperparameter grid search
│   ├── attribution.py  # gradient attribution analysis
│   └── tests/
│       └── test_model.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/YOURUSERNAME/tcga-grad.git
cd tcga-grad
pip install -r requirements.txt
```

## Usage

```bash
# single training run with best hyperparameters
python src/train.py

# hyperparameter grid search
python src/search.py

# gradient attribution analysis
python src/attribution.py
```

Data downloads automatically from GEO on first run.

## Model

**Architecture:** 2-layer MLP (30,865 input features -> 64 hidden nodes -> 5 subtype classes)

**Optimizer:** Adam with learning rate 0.001 and weight decay 1e-4

**Regularization:** Dropout (p=0.3) and class-weighted CrossEntropyLoss to handle subtype imbalance

**Hyperparameters:** Selected via 16-run grid search across hidden dim, dropout, learning rate, and epoch count. Full results in `results/grid_search_log.txt`

## Limitations and Future Work

- Vanilla gradient attribution produces diffuse signal on high-dimensional expression data. Integrated gradients or SHAP would improve gene-level resolution and better resolve subtype-specific marker contributions.

- The model overfits to training data (train loss approaches 0, val loss plateaus around 0.65), suggesting that feature selection or dimensionality reduction prior to training is a natural next step. Filtering to biologically relevant gene sets before training may produce both better generalization and cleaner attribution.

- Normal subtype performance (F1 0.71) reflects the smallest patient cohort in the dataset (225 samples). Data augmentation or oversampling strategies may improve minority class prediction.

## Data

GSE96058: SCAN-B breast cancer RNA-seq cohort. Downloaded automatically via GEOparse. Raw data not included in this repository per GEO data use policy.