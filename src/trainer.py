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


def train(model, X_train, y_train, X_test, y_test, X_validate, y_validate, epochs=100, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # track losses for plotting
    train_losses = []
    test_losses = []
    validate_losses = []
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

        # ── evaluation every epoch for smooth curve ──
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            val_output = model(X_validate)

            test_loss = criterion(test_output, y_test)
            val_loss = criterion(val_output, y_validate)
            predictions = torch.argmax(test_output, dim=1)
            accuracy = (predictions == y_test).float().mean()

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        validate_losses.append(val_loss.item())
        accuracies.append(accuracy.item())

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"Accuracy: {accuracy.item():.4f}"
            )

    return train_losses, test_losses, validate_losses, accuracies

def evaluate(model, X_test , y_test):
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

    print(classification_report(y_true, y_pred, target_names=['LumA', 'LumB', 'Basal', 'Her2', 'Normal']))

    return precision, recall, f1, auroc