import torch
from dataprocess import Data
from modelclassifier import ArxivClassifier, Token


#--------------CONFIG (mirror training)----------------
data_path = "arxiv_metadata.json"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
nb_papers = 50
max_len = 128 
threshold = 0.5
weights    = "arxiv_classifier.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------LOAD VOCAB & MODEL----------------
# We re-use Data just to rebuild the same cat2idx mapping used during training.
metadata  = Data(data_path, nb_papers)
n_cats    = metadata.n_cats
idx2cat   = {i: cat for cat, i in metadata.cat2idx.items()}

model = ArxivClassifier(model_name, n_cats)
model.load_state_dict(torch.load(weights, map_location=device))
model.to(device)
model.eval()


# --------------PREDICT----------------
def predict(title: str, threshold: float = threshold) -> list[str]:
    """
    Given a paper title string, return the predicted arXiv category labels.
    """
    tokenizer = Token(model_name, [title], max_len=max_len)
    enc = tokenizer.get()

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model.forward(input_ids, attention_mask)       # (1, n_cats)
        probs  = torch.sigmoid(logits).squeeze(0)               # (n_cats,)

    predicted = [
        (idx2cat[i], round(probs[i].item(), 3))
        for i in range(n_cats)
        if probs[i] >= threshold
    ]

    # Sort by confidence descending
    predicted.sort(key=lambda x: x[1], reverse=True)
    return predicted


# --------------ENTRY POINT----------------
if __name__ == "__main__":
    title = "Étale Fundamental Groups -- a geometric and topological approach to fundamental groups in algebraic geometry"

    results = predict(title)

    print(f"\nTitle: '{title}'")
    print(f"Predicted categories (threshold={threshold}):")
    if results:
        for cat, conf in results:
            print(f"  {cat:<20} {conf:.3f}")
    else:
        print("  No category exceeded the threshold.")
        print("\n  Top 3 regardless of threshold:")
        tokenizer = Token(model_name, [title], max_len=max_len)
        enc       = tokenizer.get()
        with torch.no_grad():
            logits = model.forward(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs  = torch.sigmoid(logits).squeeze(0)
        top3 = probs.topk(3)
        for score, idx in zip(top3.values, top3.indices):
            print(f"  {idx2cat[idx.item()]:<20} {score.item():.3f}")