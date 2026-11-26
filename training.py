"""Training machinery: loss, optimiser, training loop, evaluation and metrics.

Nothing here knows about file paths or config values: main.py passes everything in.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


def format_time(seconds):
    """3725 -> '1h02m', 95 -> '1m35s'."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


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
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, log_every=100):
    """One pass over the training data. Returns the average training loss.
    Every `log_every` batches, prints progress and the time left in this epoch."""
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_loss, n_batches, start = 0.0, len(loader), time.time()
    for step, (input_ids, attention_mask, targets) in enumerate(loader, start=1):
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

        if step % log_every == 0 or step == n_batches:
            elapsed = time.time() - start
            left = elapsed / step * (n_batches - step)
            print(f"\r  batch {step}/{n_batches} | {format_time(elapsed)} elapsed | "
                  f"~{format_time(left)} left in this epoch ", end="", flush=True)
    print()
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
def fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, device,
        threshold=0.5, start_epoch=1, best=None, seed=0, on_epoch_end=None):
    """Train from `start_epoch` to `epochs`, evaluating on the validation set after each epoch.

    best:          validation metrics of the best epoch so far (None for a new run).
    on_epoch_end:  called as on_epoch_end(epoch, best, is_best) after every epoch;
                   main.py uses it to save the model and the training state.
    Returns the metrics of the best epoch (highest validation micro-F1).
    """
    run_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        # Seeding per epoch gives the same shuffling and dropout whether or not the run was resumed
        torch.manual_seed(seed + epoch)
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        m = evaluate(model, val_loader, loss_fn, device, threshold)

        is_best = best is None or m["micro_f1"] > best["micro_f1"]
        if is_best:
            best = {**m, "epoch": epoch}

        epochs_done = epoch - start_epoch + 1
        time_left = (time.time() - run_start) / epochs_done * (epochs - epoch)
        print(
            f"Epoch {epoch:>3}/{epochs} | train loss {train_loss:.4f} | val loss {m['loss']:.4f} | "
            f"exact {m['exact_match']:.3f} | micro-F1 {m['micro_f1']:.3f} | macro-F1 {m['macro_f1']:.3f} | "
            f"{format_time(time.time() - epoch_start)}"
            + (f" (about {format_time(time_left)} left)" if epoch < epochs else "")
            + (" *best*" if is_best else "")
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, best, is_best)
    return best


def save_training_state(path, model, optimizer, scheduler, epoch, best, config):
    """Everything needed to continue training after `epoch`: weights, optimiser and
    learning-rate-schedule state, best score so far, and the settings of the run."""
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "best": best,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)  # write then rename: a crash mid-save never leaves a broken file


def load_training_state(path, model, optimizer, scheduler, config):
    """Load a saved state into model/optimizer/scheduler. Returns (next epoch, best metrics).
    Refuses if the run's settings changed, because the data split and the learning-rate
    schedule would no longer match."""
    # weights_only=False because the file also holds optimiser/scheduler settings, not just
    # tensors. That is fine for files you saved yourself; never load one from someone else.
    state = torch.load(path, map_location="cpu", weights_only=False)
    changed = sorted(k for k in set(config) | set(state["config"])
                     if config.get(k) != state["config"].get(k))
    if changed:
        raise ValueError(
            f"{path} was saved by a run with different settings: {', '.join(changed)}.\n"
            f"Put those settings back to resume, or delete {path} (or set RESUME = False) "
            f"to start a new run."
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"] + 1, state["best"]