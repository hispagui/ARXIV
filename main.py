#Fine-tune MiniLM to predict arXiv categories from a paper's title + abstract

import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer


from dataprocess import Data
from modelclassifier import ArxivClassifier, get_device, make_loaders, save_checkpoint
from training import fit, load_training_state, make_loss_fn, make_optimizer, save_training_state

#--------------CONFIG----------------
DATA_PATH = "arxiv_metadata.json"      # The Kaggle download
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NB_PAPERS = 300000         # None = the whole file
RANDOM_SAMPLE = True
MAX_LEN = 128              # tokens kept per paper (title + start of the abstract)
BATCH_SIZE = 32            
EPOCHS = 3
LR_ENCODER = 5e-5          # small: the unfrozen transformer layers are already trained
LR_HEAD = 1e-3             # larger: the head starts from random weights

N_TRAINABLE_LAYERS = 2     # last 2 of MiniLM's 6 layers are fine-tuned  
VAL_FRAC = 0.1             # validation percentage
POS_WEIGHT_CAP = 50.0      # limit on how much rare categories are up-weighted in the loss
THRESHOLD = 0.5            # probability above which a category is predicted
SEED = 0
CHECKPOINT = "arxiv_classifier.pt"
STATE_PATH = "arxiv_classifier_last.pt"
RESUME = True
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():   # Apple Silicon GPU (M1, M2, …)
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    set_seed(SEED)

    # 1. Data
    data = Data(DATA_PATH, NB_PAPERS, RANDOM_SAMPLE, SEED)
    print(f"{len(data.texts)} papers | {data.n_cats} categories | device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader = make_loaders(
        tokenizer, data.texts, data.labels, MAX_LEN, BATCH_SIZE, VAL_FRAC, SEED
    )

    # 2. Model
    model = ArxivClassifier(MODEL_NAME, data.n_cats, N_TRAINABLE_LAYERS).to(DEVICE)

    # 3. Loss and optimiser
    train_labels = data.labels[train_loader.dataset.indices]  # training split only
    loss_fn = make_loss_fn(train_labels, POS_WEIGHT_CAP, DEVICE)
    optimizer, scheduler = make_optimizer(model, LR_HEAD, LR_ENCODER, EPOCHS, len(train_loader))

    # 4. Resume from the last finished epoch, if there is one.
    # These settings must match for a resume to make sense (same papers, same split, same schedule).
    run_config = {
        "data_path": DATA_PATH, "model_name": MODEL_NAME, "nb_papers": NB_PAPERS,
        "random_sample": RANDOM_SAMPLE, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr_encoder": LR_ENCODER, "lr_head": LR_HEAD,
        "n_trainable_layers": N_TRAINABLE_LAYERS, "val_frac": VAL_FRAC,
        "pos_weight_cap": POS_WEIGHT_CAP, "seed": SEED,
    }
    start_epoch, best = 1, None
    if RESUME and os.path.exists(STATE_PATH):
        start_epoch, best = load_training_state(STATE_PATH, model, optimizer, scheduler, run_config)
        if start_epoch > EPOCHS:
            print(f"This run already finished all {EPOCHS} epochs (best: epoch {best['epoch']}, "
                  f"micro-F1 {best['micro_f1']:.3f}). Delete {STATE_PATH} to train again.")
            return
        print(f"Resuming from {STATE_PATH}: starting at epoch {start_epoch}/{EPOCHS}")

    # 5. Train, saving after every epoch
    def on_epoch_end(epoch, best, is_best):
        if is_best:
            save_checkpoint(CHECKPOINT, model, MODEL_NAME, data.categories, MAX_LEN)
        save_training_state(STATE_PATH, model, optimizer, scheduler, epoch, best, run_config)

    best = fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, EPOCHS, DEVICE,
               THRESHOLD, start_epoch=start_epoch, best=best, seed=SEED, on_epoch_end=on_epoch_end)

    print(f"Done. Best epoch: {best['epoch']} (validation micro-F1 {best['micro_f1']:.3f}), "
          f"saved in {CHECKPOINT}")







