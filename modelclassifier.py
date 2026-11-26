# MiniLM encoder + classification head, data loaders, and checkpoint save/load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import AutoModel



# -------------- EMBEDDINGS --------------
def mean_pooling(last_hidden_state, attention_mask):
    """Average the token vectors of a text, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(encoder, input_ids, attention_mask):
    """One L2-normalised vector per text: the same embedding sentence-transformers produces
    for all-MiniLM-L6-v2. The dot product of two of them is their cosine similarity."""
    out = encoder(input_ids=input_ids, attention_mask=attention_mask)
    return F.normalize(mean_pooling(out.last_hidden_state, attention_mask), p=2, dim=1)


# -------------- DATA LOADERS --------------
def make_loaders(tokenizer, texts, labels, max_len=128, batch_size=32, val_frac=0.1, seed=0):
    """Tokenise all texts, then split into a training and a validation loader."""
    enc = tokenizer(texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.as_tensor(labels, dtype=torch.float32))
    n_val = max(1, int(len(dataset) * val_frac))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


# -------------- MODEL --------------
class ArxivClassifier(nn.Module):
    def __init__(self, model_name, n_cats, n_trainable_layers=2, head_hidden=256, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        dim = self.encoder.config.hidden_size  # 384 for MiniLM-L6

        self.head = nn.Sequential(
            nn.Linear(dim, head_hidden),     # 384 -> 256: small hidden layer
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_cats),  # one score (logit) per category
            )
        self.freeze_encoder(n_trainable_layers)

    def freeze_encoder(self, n_trainable_layers):
        """Freeze the encoder, then unfreeze its last n layers (MiniLM-L6 has 6: 0..5).
        The head is never frozen."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        layers = self.encoder.encoder.layer
        for layer in layers[len(layers) - n_trainable_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def encode(self, input_ids, attention_mask):
        return embed(self.encoder, input_ids, attention_mask)

    def forward(self, input_ids, attention_mask):
        return self.head(self.encode(input_ids, attention_mask))  # logits, (batch, n_cats)



# -------------- CHECKPOINTS --------------
def save_checkpoint(path, model, model_name, categories, max_len):
    """Save the weights (encoder + head) together with what is needed to rebuild the model."""
    torch.save(
        {
            "model_name": model_name,
            "categories": list(categories),  # index i of the output = categories[i]
            "max_len": max_len,
            "head_hidden": model.head[0].out_features,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Rebuild the model from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" not in ckpt:
        raise ValueError(
            f"{path} is in the old format (encoder weights only, no head, no category list). "
            "Retrain with main.py to get a usable checkpoint."
        )
    model = ArxivClassifier(ckpt["model_name"], len(ckpt["categories"]), head_hidden=ckpt["head_hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


