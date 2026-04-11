import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product
from dataset import load_dataset
from model import TcgaNet
from trainer import train, evaluate, plot_metrics, set_seed, RESULTS_DIR

# grid search over hyperparameters: runs all combinations and saves best model
# see results/grid_search_log.txt for full results
if __name__ == "__main__":
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # data setup
    expression, labels, label_map = load_dataset()

    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        expression, labels, test_size=0.15, stratify=labels, random_state=42
    )

    scaler = StandardScaler()
    X_temp_scaled = scaler.fit_transform(X_temp)
    X_holdout_scaled = scaler.transform(X_holdout)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp_scaled, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_holdout_scaled, dtype=torch.float32)

    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_val = torch.tensor(y_val.values, dtype=torch.long)
    y_test = torch.tensor(y_holdout.values, dtype=torch.long)

    # grid params
    param_grid = {
        'hidden_dim': [64, 128],
        'dropout': [0.3, 0.5],
        'lr': [0.001, 0.0001],
        'epochs': [50, 100],
    }

    combinations = list(product(
        param_grid['hidden_dim'],
        param_grid['dropout'],
        param_grid['lr'],
        param_grid['epochs']
    ))

    best_f1 = 0.0
    best_params = None
    best_metrics = None
    results_log = []

    # loop run
    for hidden_dim, dropout, lr, epochs in combinations:
        print(f"\nRunning: hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}, epochs={epochs}")

        model = TcgaNet(input_dim=30865, hidden_dim=hidden_dim, output_dim=5, dropout=dropout)

        train_losses, val_losses, accuracies = train(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, lr=lr
        )

        precision, recall, f1, auroc = evaluate(model, X_test, y_test)

        results_log.append({
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'lr': lr,
            'epochs': epochs,
            'f1': f1,
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_params = (hidden_dim, dropout, lr, epochs)
            best_metrics = (train_losses, val_losses, accuracies)
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, 'best_model.pth'))

    # post run logging
    title = (f"Best Run — hidden_dim={best_params[0]}, dropout={best_params[1]}, "
             f"lr={best_params[2]}, epochs={best_params[3]}")
    plot_metrics(best_metrics[0], best_metrics[1], best_metrics[2], title=title, filename='GSE96058_training_best_grid_result.png')

    with open(os.path.join(RESULTS_DIR, 'grid_search_log.txt'), 'w') as f:

        for r in results_log:
            f.write(
                f"hidden_dim={r['hidden_dim']} | dropout={r['dropout']} | "
                f"lr={r['lr']} | epochs={r['epochs']} | "
                f"f1={r['f1']:.4f} | auroc={r['auroc']:.4f} | "
                f"precision={r['precision']:.4f} | recall={r['recall']:.4f}\n"
            )

        f.write("\n======== BEST RUN ==========\n")
        f.write(
            f"hidden_dim={best_params[0]} | dropout={best_params[1]} | "
            f"lr={best_params[2]} | epochs={best_params[3]} | "
            f"f1={best_f1:.4f}\n"
        )

    print("\n========= GRID SEARCH COMPLETE =========")
    print(f"Best F1:         {best_f1:.4f}")
    print(f"Best hidden_dim: {best_params[0]}")
    print(f"Best dropout:    {best_params[1]}")
    print(f"Best lr:         {best_params[2]}")
    print(f"Best epochs:     {best_params[3]}")
    print(f"Log saved to:    {RESULTS_DIR}/grid_search_log.txt")
    print("=========================================")
