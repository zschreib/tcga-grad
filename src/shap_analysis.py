import torch
import shap
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


def load_model_and_data():
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

    # background dataset for SHAP
    X_temp_scaled = scaler.transform(X_temp)
    X_train_tensor = torch.tensor(X_temp_scaled, dtype=torch.float32)
    X_background = X_train_tensor[:100]  # 100 samples

    return model, X_test, y_test, gene_names, X_background


def run_shap(model, X_background, X_test, gene_names):
    """
    Compute SHAP values for each PAM50 class using DeepExplainer.
    Returns mean absolute SHAP values filtered to PAM50 genes.
    """
    model.eval()

    # init explainer with background dataset
    explainer = shap.DeepExplainer(model, X_background)

    # shape [n_samples, n_features]
    print("Computing SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    results = {}

    for class_idx, class_name in enumerate(PAM50_CLASSES):
        print(f"\nProcessing SHAP values for {class_name}...")

        # mean abs shap value per gene across all samples
        mean_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)

        # pair gene names with shap scores
        gene_scores = pd.Series(mean_shap, index=gene_names)

        # PAM50 filter
        gene_scores = gene_scores[gene_scores.index.isin(PAM50_GENES)]
        gene_scores = gene_scores.sort_values(ascending=False)

        print(f"Top 10 PAM50 genes for {class_name}:")
        print(gene_scores.head(10).to_string())

        results[class_name] = gene_scores

    return results, shap_values


def plot_shap(shap_values, X_test, gene_names, class_idx, class_name,
              n_genes=20, save_dir=None):
    save_dir = save_dir or RESULTS_DIR

    # get PAM50 gene indices
    pam50_idx = [i for i, g in enumerate(gene_names) if g in PAM50_GENES]
    pam50_names = [gene_names[i] for i in pam50_idx]
    X_pam50 = X_test[:, pam50_idx].numpy()
    shap_pam50 = shap_values[:, pam50_idx, class_idx]

    shap.summary_plot(
        shap_pam50,
        X_pam50,
        feature_names=pam50_names,
        max_display=n_genes,
        show=False,
        plot_type='dot'
    )

    plt.title(f'SHAP Summary — {class_name}')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = f'shap_summary_{class_name}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_dir}/{filename}")


if __name__ == "__main__":

    # load model and data
    model, X_test, y_test, gene_names, X_background = load_model_and_data()

    # run SHAP
    results, raw_shap_values = run_shap(model, X_background, X_test, gene_names)

    # shap results
    shap_dir = os.path.join(RESULTS_DIR, 'shap')
    os.makedirs(shap_dir, exist_ok=True)

    # plots
    for class_idx, class_name in enumerate(PAM50_CLASSES):
        plot_shap(raw_shap_values, X_test, gene_names, class_idx, class_name,
                  save_dir=shap_dir)

        # save ranked gene csv to shap subfolder
        csv_path = os.path.join(shap_dir, f'shap_{class_name}.csv')
        results[class_name].to_csv(csv_path, header=['shap_score'])
        print(f"SHAP scores saved to {csv_path}")
