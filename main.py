from dataprocess import Data
from modelclassifier import ArxivClassifier, Token

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


#--------------CONFIG----------------
data_path = "arxiv_metadata.json"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
nb_papers = 8191   # max is 1700000
max_len = 128 
lr = 2e-4    #learning rate
epochs = 10 
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
