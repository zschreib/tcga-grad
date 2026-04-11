import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import load_dataset
from model import TcgaNet
from trainer import train, evaluate, plot_metrics, set_seed, RESULTS_DIR

if __name__ == "__main__":
    set_seed(42)
    expression, labels, label_map = load_dataset()

    # split off holdout set (15%) not seen by model
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        expression, labels, test_size=0.15, stratify=labels, random_state=42
    )

    # normalize AFTER splitting — fit on train only, transform both
    scaler = StandardScaler()
    X_temp_scaled = scaler.fit_transform(X_temp)
    X_holdout_scaled = scaler.transform(X_holdout)

    # split remainder into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp_scaled, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )

    # convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_holdout_scaled, dtype=torch.float32)

    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_val = torch.tensor(y_val.values, dtype=torch.long)
    y_test = torch.tensor(y_holdout.values, dtype=torch.long)

    # best params from grid search: see results/grid_search_log.txt
    model = TcgaNet(input_dim=30865, hidden_dim=64, output_dim=5, dropout=0.3)

    train_losses, val_losses, accuracies = train(
        model, X_train, y_train, X_val, y_val,
        epochs=100, lr=0.001
    )

    precision, recall, f1, auroc = evaluate(model, X_test, y_test)
    plot_metrics(train_losses, val_losses, accuracies)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
    print(f"Model saved to {RESULTS_DIR}/best_model.pth")