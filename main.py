#Fine-tune MiniLM to predict arXiv categories from a paper's title + abstract

import random
import numpy as np
import torch
from transformers import AutoTokenizer


from dataprocess import Data
from modelclassifier import ArxivClassifier, make_loaders, save_checkpoint
from training import fit, make_loss_fn, make_optimizer

#--------------CONFIG----------------
DATA_PATH = "arxiv_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NB_PAPERS = 8191           # None = the whole file
MAX_LEN = 128              # tokens kept per paper (title + start of the abstract)
BATCH_SIZE = 32            
EPOCHS = 10
LR_ENCODER = 5e-5          # small: the unfrozen transformer layers are already trained
LR_HEAD = 1e-3             # larger: the head starts from random weights

N_TRAINABLE_LAYERS = 2     # last 2 of MiniLM's 6 layers are fine-tuned  
VAL_FRAC = 0.1             # validation percentage
POS_WEIGHT_CAP = 50.0      # limit on how much rare categories are up-weighted in the loss
THRESHOLD = 0.5            # probability above which a category is predicted
SEED = 0
CHECKPOINT = "arxiv_classifier.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    set_seed(SEED)

    # 1. Data
    data = Data(DATA_PATH, NB_PAPERS)
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

    # 4. Train (the model comes back with the weights of its best epoch)
    best = fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, EPOCHS, DEVICE, THRESHOLD)

    # 5. Save
    save_checkpoint(CHECKPOINT, model, MODEL_NAME, data.categories, MAX_LEN)
    print(f"Best epoch: {best['epoch']} (validation micro-F1 {best['micro_f1']:.3f}). Saved to {CHECKPOINT}")


if __name__ == "__main__":
    main()













#--------------DATA----------------
metadata = Data(data_path, nb_papers)
texts, labels = metadata.load_arxiv(nb_papers)
n_cats = metadata.n_cats

tokenizer     = Token(model_name, texts, max_len=max_len)
labels_tensor = torch.tensor(labels, dtype=torch.float32)
loader        = tokenizer.make_dataset(labels_tensor)

#--------------MODEL----------------
model = ArxivClassifier(model_name, n_cats)
model.to(device) # where to allocate computation, default is CPU

#--------------LOSS----------------
'''sigmoid activation layer + BinaryCrossEntroy'''

pos_counts = labels_tensor.sum(dim=0).clamp(min=1)
neg_counts  = len(labels_tensor) - pos_counts
pos_weight  = (neg_counts / pos_counts).to(device)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#--------------OPTIMISER----------------
''' Optimiser, each batch : 1. compute gradients, 2. update weights
AdamW is an improved gradient descent method
OneCycleLR is scheduler, controls how learning rate changes
'''
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=lr, weight_decay=1e-2)

scheduler = OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=len(loader),
    epochs=epochs,
)

#--------------METRICS----------------
def compute_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()  #transforms raw scores [-1.2, 3.5, ...] into proba between [0,1]

    exact_match = (preds == targets).all(dim=1).float().mean().item()

    tp = (preds * targets).sum(dim=0)          #true positive
    fp = (preds * (1 - targets)).sum(dim=0)    #false positive
    fn = ((1 - preds) * targets).sum(dim=0)    #false negative
    precision = tp / (tp + fp).clamp(min=1e-9) #how many of the labels predicted are correct
    recall    = tp / (tp + fn).clamp(min=1e-9) #of true labels how many successfully found 
    
    f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-9)  
    macro_f1  = f1.mean().item()

    return exact_match, macro_f1

#--------------TRAINING----------------
def train(model, loader, loss_fn, optimizer, scheduler, epochs, device):
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss  = 0.0
        total_exact = 0.0
        total_f1    = 0.0

        for input_ids, attention_mask, targets in loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets        = targets.to(device)

            optimizer.zero_grad()
            logits = model.forward(input_ids, attention_mask)
            loss   = loss_fn(logits, targets)
            loss.backward()
            
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            exact, f1 = compute_metrics(logits.detach(), targets.detach())
            total_loss  += loss.item()
            total_exact += exact
            total_f1    += f1

        n = len(loader)
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Loss: {total_loss/n:.4f} | "
            f"Exact-match: {total_exact/n:.4f} | "
            f"Macro-F1: {total_f1/n:.4f}"
        )

    return model


if __name__ == "__main__":
    print(f"Training on {device}  |  {n_cats} categories  |  {len(texts)} papers")
    model = train(model, loader, loss_fn, optimizer, scheduler, epochs, device)
    torch.save(model.state_dict(), "arxiv_classifier.pt")
    print("Model saved to arxiv_classifier.pt")
