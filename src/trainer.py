import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def train(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
    class_counts = torch.tensor([1709, 767, 360, 348, 225], dtype=torch.float32)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # track losses for plotting
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(epochs):

        # ── forward pass ──
        model.train()
        optimizer.zero_grad()
        output = model(X_train)

        # ── loss ──
        loss = criterion(output, y_train)

        # ── backward pass ──
        loss.backward()

        # ── update weights ──
        optimizer.step()

        # ── validation every epoch ──
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
