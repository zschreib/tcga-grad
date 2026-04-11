import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):

    # class counts from GSE96058 full dataset. todo: update dynamic if dataset changes
    class_counts = torch.tensor([1709, 767, 360, 348, 225], dtype=torch.float32)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # track losses for plotting
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(epochs):

        # forward pass
        model.train()
        optimizer.zero_grad()
        output = model(X_train)

        # loss
        loss = criterion(output, y_train)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # validation every epoch
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            predictions = torch.argmax(val_output, dim=1)
            accuracy = (predictions == y_val).float().mean()

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        accuracies.append(accuracy.item())

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss:   {val_loss.item():.4f} | "
                f"Accuracy:   {accuracy.item():.4f}"
            )

    return train_losses, val_losses, accuracies


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        probs = F.softmax(output, dim=1)
        preds = torch.argmax(output, dim=1)

    y_true = y_test.numpy()
    y_pred = preds.numpy()
    y_prob = probs.numpy()

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    print(classification_report(y_true, y_pred,
                                target_names=['LumA', 'LumB', 'Basal', 'Her2', 'Normal']))

    return precision, recall, f1, auroc

def plot_metrics(train_losses, val_losses, accuracies,
                 title='TCGA MLP - Training Results',
                 filename='GSE96058_training_results.png'):

    epochs = range(len(train_losses))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # loss curve
    ax1.plot(epochs, train_losses, label='Train Loss', color='steelblue')
    ax1.plot(epochs, val_losses,   label='Val Loss',   color='coral')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # accuracy curve
    ax2.plot(epochs, accuracies, label='Val Accuracy', color='seagreen')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {RESULTS_DIR}/{filename}")