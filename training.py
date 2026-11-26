"""Training machinery: loss, optimiser, training loop, evaluation and metrics.

Nothing here knows about file paths or config values: main.py passes everything in.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


# -------------- LOSS --------------
def make_loss_fn(train_labels, pos_weight_cap, device):
    #Sigmoid + binary cross-entropy for each category (a paper can have several).

    #pos_weight = negatives / positives makes rare categories count more. 
    #Computed on the training split only, and capped so a category seen once doesn't get a weight of ~8000.

    train_labels = torch.as_tensor(train_labels, dtype=torch.float32)
    pos = train_labels.sum(dim=0).clamp(min=1)
    pos_weight = ((len(train_labels) - pos) / pos).clamp(max=pos_weight_cap)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))


# -------------- OPTIMISER --------------
def make_optimizer(model, lr_head, lr_encoder, epochs, steps_per_epoch, weight_decay=1e-2):
    """AdamW updates the weights after each batch, with two learning rates: a larger one for
    the head (random at the start) and a smaller one for the unfrozen encoder layers.
    OneCycleLR warms the learning rate up, then decays it to ~0 by the end of training."""
    groups = [{"params": list(model.head.parameters()), "lr": lr_head}]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": lr_encoder})
    optimizer = AdamW(groups, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )
    return optimizer, scheduler


# -------------- METRICS --------------
def compute_metrics(logits, targets, threshold=0.5):
    """logits, targets: (n_papers, n_cats) for a whole dataset, not a single batch."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    exact_match = (preds == targets).all(dim=1).float().mean().item()

    tp = (preds * targets).sum(dim=0)        # true positives per category
    fp = (preds * (1 - targets)).sum(dim=0)  # false positives
    fn = ((1 - preds) * targets).sum(dim=0)  # false negatives

    # micro-F1: pool all categories together (dominated by the common ones)
    micro_f1 = (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp(min=1e-9)).item()
    # macro-F1: F1 per category, averaged over the categories that occur in this set
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-9)
    present = targets.sum(dim=0) > 0
    macro_f1 = f1[present].mean().item()
    return {"exact_match": exact_match, "micro_f1": micro_f1, "macro_f1": macro_f1}


# -------------- ONE EPOCH / EVALUATION --------------
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device):
    """One pass over the training data. Returns the average training loss."""
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_loss = 0.0
    for input_ids, attention_mask, targets in loader:
        input_ids, attention_mask, targets = (
            input_ids.to(device), attention_mask.to(device), targets.to(device)
        )
        optimizer.zero_grad()
        loss = loss_fn(model(input_ids, attention_mask), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, threshold=0.5):
    """Loss and metrics on a dataset the model doesn't train on (dropout off)."""
    model.eval()
    all_logits, all_targets, total_loss = [], [], 0.0
    for input_ids, attention_mask, targets in loader:
        input_ids, attention_mask, targets = (
            input_ids.to(device), attention_mask.to(device), targets.to(device)
        )
        logits = model(input_ids, attention_mask)
        total_loss += loss_fn(logits, targets).item() * len(targets)
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_targets), threshold)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# -------------- FULL TRAINING --------------
def fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, device, threshold=0.5):
    """Train for `epochs` epochs, evaluating on the validation set after each one.

    Keeps a copy of the weights from the best epoch (highest validation micro-F1) and puts
    them back into the model at the end. Returns that epoch's validation metrics.
    """
    best, best_weights = None, None
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        m = evaluate(model, val_loader, loss_fn, device, threshold)
        print(
            f"Epoch {epoch:>3}/{epochs} | train loss {train_loss:.4f} | val loss {m['loss']:.4f} | "
            f"exact {m['exact_match']:.3f} | micro-F1 {m['micro_f1']:.3f} | macro-F1 {m['macro_f1']:.3f}"
        )
        if best is None or m["micro_f1"] > best["micro_f1"]:
            best = {**m, "epoch": epoch}
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    return best