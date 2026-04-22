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
    X_background = X_train_tensor[:100] #100 samples

    return model, X_test, y_test, gene_names, X_background

