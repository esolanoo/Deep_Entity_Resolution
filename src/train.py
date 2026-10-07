import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from funcs import get_device
from feature_extractor import data, EntityMatcher

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


device = get_device()
train_data, test_data, val_data, train_loader, test_loader, val_loader= data()

# Get fields from your dataset
features_A, features_B, label = train_data[0]
fields = list(features_A.keys())
"""
print("Fields:")
for field in fields:print(f"  - {field}")

print(f"\nNumber of fields: {len(fields)}")
print(f"Train samples: {len(train_data)}")
print(f"Test samples:  {len(test_data)}")
print(f"Train batches: {len(train_loader)}")
print(f"Test batches:  {len(test_loader)}")
"""

model = EntityMatcher(fields=fields, embedding_dim=128).to(device)
"""
print("\nModel:")
print(model)

total_params = sum( p.numel() for p in model.parameters())
trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
"""

pos_weight = torch.tensor([36.87], dtype=torch.float32, device=device) # pos/neg ratio I found in the notebooks
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

### Single batch run
"""
model.train()

batch_A, batch_B, labels = next(iter(train_loader))
batch_A = {field: batch_A[field].to(device)for field in fields}
batch_B = {field: batch_B[field].to(device)for field in fields}
labels = labels.float().to(device)

print("Batch check:")
for field in fields:print(f"{field:15s}", batch_A[field].shape, batch_B[field].shape)

print("Labels:", labels.shape)
print("Positive labels:", labels.sum().item())
print("Negative labels:", (labels == 0).sum().item())

optimizer.zero_grad()

logits = model(batch_A,batch_B)

loss = criterion(logits,labels)

print("\nForward pass:")
print("Logits shape:", logits.shape)
print("Loss:", loss.item())

loss.backward()

print("\nGradient check:")

for name, parameter in model.named_parameters():
    if parameter.grad is None:    
        print(f"WARNING: No gradient -> {name}")
    else:
        grad_mean = parameter.grad.abs().mean().item()    
        grad_max = parameter.grad.abs().max().item()
        print(f"{name:50s} "
              f"mean={grad_mean:.6e} "
              f"max={grad_max:.6e}")
"""

def train_one_epoch(model,dataloader,optimizer,criterion,device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probs = []
    start_time = time.perf_counter()

    for batch_idx, (batch_A, batch_B, labels) in enumerate(dataloader):

        # Move fields to GPU/CPU
        batch_A = {field: batch_A[field].to(device)for field in model.fields}
        batch_B = {field: batch_B[field].to(device)for field in model.fields}
        labels = labels.float().to(device)

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------
        optimizer.zero_grad()
        logits = model(batch_A,batch_B)

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------
        loss = criterion(logits,labels)

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------
        loss.backward()
        if batch_idx == 0:
            grad_info = gradient_diagnostics(model)
            print("\nGradient diagnostics:")
            print(
                f"Gradient norm : "
                f"{grad_info['total_grad_norm']:.6e}"
            )
            print(
                f"Non-zero grads: "
                f"{grad_info['nonzero_grads']}"
            )
            print(
                f"Zero grads    : "
                f"{grad_info['zero_grads']}"
            )
            print(
                f"NaN gradients : "
                f"{grad_info['nan_grads']}"
            )

        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
        optimizer.step()

        # --------------------------------------------------
        # Monitoring
        # --------------------------------------------------
        probs = torch.sigmoid(logits)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        # --------------------------------------------------
        # Live batch monitoring
        # --------------------------------------------------

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0:

            batch_predictions = (probs >= 0.5).long()
            batch_accuracy = (batch_predictions == labels.long()).float().mean().item()
            elapsed = time.perf_counter() - start_time

            print(
                f"  Batch {batch_idx + 1:4d}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {batch_accuracy:.4f} | "
                f"Mean P: {probs.mean().item():.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    epoch_loss = running_loss / total_samples
    return (epoch_loss,  np.array(all_labels), np.array(all_probs))


def evaluate(model,dataloader,criterion,device,threshold=0.5):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_A, batch_B, labels in dataloader:
            batch_A = { field: batch_A[field].to(device) for field in model.fields}
            batch_B = { field: batch_B[field].to(device) for field in model.fields}
            labels = labels.float().to(device)
            logits = model( batch_A, batch_B)
            loss = criterion( logits, labels)
            probs = torch.sigmoid(logits)
            batch_size = labels.size(0)
            running_loss += ( loss.item() * batch_size)
            total_samples += batch_size
            all_labels.extend( labels.cpu().numpy())
            all_probs.extend( probs.cpu().numpy())

    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)

    predictions = (probs_np >= threshold).astype(int)

    metrics = {
        "loss": running_loss / total_samples,
        "accuracy": accuracy_score(labels_np,predictions),
        "precision": precision_score(labels_np,predictions,zero_division=0),
        "recall": recall_score(labels_np,predictions,zero_division=0),
        "f1": f1_score(labels_np,predictions,zero_division=0),
        "roc_auc": roc_auc_score(labels_np,probs_np),
        "pr_auc": average_precision_score(labels_np,probs_np)
    }

    return metrics, labels_np, probs_np


def gradient_diagnostics(model):

    total_norm = 0.0
    nonzero_grads = 0
    zero_grads = 0
    nan_grads = 0

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        if torch.isnan(grad).any():
            nan_grads += 1

        grad_norm = grad.norm(2).item()

        total_norm += grad_norm ** 2

        if grad_norm > 0:
            nonzero_grads += 1
        else:
            zero_grads += 1

    total_norm = total_norm ** 0.5

    return {
        "total_grad_norm": total_norm,
        "nonzero_grads": nonzero_grads,
        "zero_grads": zero_grads,
        "nan_grads": nan_grads
    }


num_epochs = 3

history = {
    "train_loss": [],
    "test_loss": [],
    "test_accuracy": [],
    "test_precision": [],
    "test_recall": [],
    "test_f1": [],
    "test_roc_auc": [],
    "test_pr_auc": []
}

best_f1 = -1.0

for epoch in range(num_epochs):
    print("\n")
    print("=" * 80)
    print(f"EPOCH {epoch + 1}/{num_epochs}")
    print("=" * 80)

    # ======================================================
    # TRAIN
    # ======================================================
    train_start = time.perf_counter()

    train_loss, train_labels, train_probs = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    train_time = time.perf_counter() - train_start

    # ======================================================
    # TEST
    # ======================================================
    test_metrics, test_labels, test_probs = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        threshold=0.5
    )

    # ======================================================
    # SAVE HISTORY
    # ======================================================
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_metrics["loss"])
    history["test_accuracy"].append(test_metrics["accuracy"])
    history["test_precision"].append(test_metrics["precision"])
    history["test_recall"].append(test_metrics["recall"])
    history["test_f1"].append(test_metrics["f1"])
    history["test_roc_auc"].append(test_metrics["roc_auc"])
    history["test_pr_auc"].append(test_metrics["pr_auc"])

    # ======================================================
    # PRINT SUMMARY
    # ======================================================
    print("\n")
    print("-" * 80)
    print(f"Epoch {epoch + 1} Summary")
    print("-" * 80)

    print(f"Train Loss : {train_loss:.6f}")
    print(f"Test Loss  : {test_metrics['loss']:.6f}")

    print(f"Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"Precision  : {test_metrics['precision']:.4f}")
    print(f"Recall     : {test_metrics['recall']:.4f}")
    print(f"F1         : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC    : {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC     : {test_metrics['pr_auc']:.4f}")

    print(f"Epoch time : {train_time:.2f} sec")

    # ======================================================
    # SAVE BEST MODEL
    # ======================================================
    if test_metrics["f1"] > best_f1:
        best_f1 = test_metrics["f1"]
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "history": history
            },
            "cnn_mlp_baseline_best.pt"
        )
        print(f"\n✓ New best model saved! "
              f"F1 = {best_f1:.4f}"
        )
        
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["test_loss"],
        label="Test Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN + MLP Training Loss")

    plt.legend()
    plt.grid(True)

    plt.show()