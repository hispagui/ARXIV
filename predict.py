"""Predict arXiv categories for a paper title (and optionally its abstract).

    python predict.py "Étale fundamental groups in algebraic geometry"
    python predict.py "Some title" --abstract "The abstract..." --threshold 0.3
"""

import argparse
import torch
from transformers import AutoTokenizer
from dataprocess import paper_text
from modelclassifier import load_checkpoint


CHECKPOINT = "arxiv_classifier.pt"
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Predictor:
    """Loads the model, tokenizer and category list once; then predict() can be called many times."""

    def __init__(self, checkpoint=CHECKPOINT, device=DEVICE):
        self.device = device
        self.model, ckpt = load_checkpoint(checkpoint, device)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])
        self.categories = ckpt["categories"]  # the exact list used during training
        self.max_len = ckpt["max_len"]

    @torch.no_grad()
    def probabilities(self, title, abstract=""):
        """All categories with their probability, most likely first."""
        enc = self.tokenizer(
            [paper_text(title, abstract)],
            max_length=self.max_len, truncation=True, padding=True, return_tensors="pt",
        ).to(self.device)
        probs = torch.sigmoid(self.model(enc["input_ids"], enc["attention_mask"]))[0].cpu()
        return sorted(zip(self.categories, probs.tolist()), key=lambda x: x[1], reverse=True)

    def predict(self, title, abstract="", threshold=THRESHOLD):
        """Categories whose probability is at least `threshold`."""
        return [(cat, p) for cat, p in self.probabilities(title, abstract) if p >= threshold]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("title", nargs="?", default=(
        "Étale Fundamental Groups -- a geometric and topological approach "
        "to fundamental groups in algebraic geometry"))
    parser.add_argument("--abstract", default="")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint)
    ranked = predictor.probabilities(args.title, args.abstract)
    above = [(cat, p) for cat, p in ranked if p >= args.threshold]

    print(f"\nTitle: '{args.title}'")
    print(f"Predicted categories (threshold={args.threshold}):")
    if above:
        for cat, p in above:
            print(f"  {cat:<20} {p:.3f}")
    else:
        print("  No category exceeded the threshold. Top 3:")
        for cat, p in ranked[:3]:
            print(f"  {cat:<20} {p:.3f}")


if __name__ == "__main__":
    main()
