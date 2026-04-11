import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import TcgaNet
from trainer import RESULTS_DIR, set_seed
from dataset import load_dataset

PAM50_CLASSES = ['LumA', 'LumB', 'Basal', 'Her2', 'Normal']

PAM50_GENES = [
    'UBE2T', 'BIRC5', 'NUF2', 'CDC6', 'CCNB1', 'TYMS', 'MYBL2',
    'CEP55', 'MELK', 'NDC80', 'RRM2', 'UBE2C', 'CENPF', 'PTTG1',
    'EXO1', 'ORC6L', 'ANLN', 'CCNE1', 'CDC20', 'MKI67', 'KIF2C',
    'ACTR3B', 'MYC', 'EGFR', 'KRT5', 'PHGDH', 'CDH3', 'MIA',
    'KRT17', 'FOXC1', 'SFRP1', 'KRT14', 'ESR1', 'SLC39A6', 'BAG1',
    'MAPT', 'PGR', 'CXXC5', 'MLPH', 'BCL2', 'MDM2', 'NAT1',
    'CXCL10', 'BLVRA', 'MMP11', 'GPR160', 'FGFR4', 'GRB7',
    'TMEM45B', 'ERBB2'
]


def compute_gradients(model, sample, class_idx, gene_names):
    """
    Compute gradient of target class score with respect to input genes.
    High gradient = gene strongly influenced the prediction.
    """
    model.eval()
    sample = sample.unsqueeze(0).requires_grad_(True)

    # forward pass
    output = model(sample)

    # backward on target class score only
    output[0, class_idx].backward()

    # collect gradients magnitude only, direction doesn't matter
    gradients = sample.grad.squeeze().detach().numpy()
    gradients = np.abs(gradients)

    # pair gene names with gradient scores and rank
    gene_scores = pd.Series(gradients, index=gene_names)
    gene_scores = gene_scores.sort_values(ascending=False)

    return gene_scores


def plot_top_genes(gene_scores, class_name, n_genes=20):
    """
    Bar chart of top N genes by gradient score for a given PAM50 class.
    """
    top_genes = gene_scores.head(n_genes)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(top_genes.index[::-1], top_genes.values[::-1], color='steelblue')
    ax.set_xlabel('Gradient Magnitude')
    ax.set_ylabel('Gene')
    ax.set_title(f'Top {n_genes} Genes by Gradient Attribution — {class_name}')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f'attribution_{class_name}.png'
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {RESULTS_DIR}/{filename}")


def run_attribution(model, X_test, y_test, gene_names, n_genes=20):
    """
    Run gradient attribution for each PAM50 class.
    Computes gradients on full input then filters to PAM50 genes only.
    Averages gradients across all correctly predicted samples of that class.
    """
    results = {}

    for class_idx, class_name in enumerate(PAM50_CLASSES):

        # find correctly predicted samples of this class
        model.eval()
        with torch.no_grad():
            output = model(X_test)
            preds = torch.argmax(output, dim=1)

        correct_mask = (y_test == class_idx) & (preds == class_idx)
        correct_samples = X_test[correct_mask]

        if len(correct_samples) == 0:
            print(f"No correct predictions for {class_name} — skipping")
            continue

        print(f"\nComputing attribution for {class_name} "
              f"({len(correct_samples)} correct samples)...")

        # average gradients across all correct samples
        all_gradients = []
        for i in range(len(correct_samples)):
            sample = correct_samples[i].clone()
            gene_scores = compute_gradients(model, sample, class_idx, gene_names)
            all_gradients.append(gene_scores.values)

        mean_gradients = np.mean(all_gradients, axis=0)

        # build full gene score series
        gene_scores = pd.Series(mean_gradients, index=gene_names)

        # filter to PAM50 genes only — report only known markers
        gene_scores = gene_scores[gene_scores.index.isin(PAM50_GENES)]
        gene_scores = gene_scores.sort_values(ascending=False)

        print(f"PAM50 genes found: {len(gene_scores)}")

        results[class_name] = gene_scores
        plot_top_genes(gene_scores, class_name, n_genes=min(n_genes, len(gene_scores)))

        print(f"Top 10 PAM50 genes for {class_name}:")
        print(gene_scores.head(10).to_string())

    return results


if __name__ == "__main__":
    set_seed(42)
    expression, labels, label_map = load_dataset()

    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        expression, labels, test_size=0.15, stratify=labels, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_temp)
    X_holdout_scaled = scaler.transform(X_holdout)

    X_test = torch.tensor(X_holdout_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_holdout.values, dtype=torch.long)

    # gene names from expression matrix columns
    gene_names = expression.columns.tolist()

    # load best model architecture must match train.py
    model = TcgaNet(input_dim=30865, hidden_dim=64, output_dim=5, dropout=0.3)
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, 'best_model.pth'),
                   weights_only=True)
    )
    print("Model loaded from best_model.pth")

    # run attribution gradients on full input, report PAM50 genes only
    results = run_attribution(model, X_test, y_test, gene_names, n_genes=20)

    # save ranked PAM50 gene lists to csv
    for class_name, gene_scores in results.items():
        csv_path = os.path.join(RESULTS_DIR, f'pam50_attribution_{class_name}.csv')
        gene_scores.to_csv(csv_path, header=['gradient_score'])
        print(f"Attribution saved to {csv_path}")
