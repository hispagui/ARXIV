"""Semantic search over arXiv papers: find the papers closest in meaning to a query.

Step 1, build the index once (embeds every paper and saves the vectors):
    python search.py build                                   # base MiniLM model
    python search.py build --checkpoint arxiv_classifier.pt  # your fine-tuned encoder

Step 2, search (as often as you like):
    python search.py query "étale fundamental group"
    python search.py query            # interactive: type queries, empty line to quit
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from dataprocess import clean, iter_papers, paper_text
from modelclassifier import embed, load_checkpoint

DATA_PATH = "arxiv_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NB_PAPERS = 8192          # None = the whole file
MAX_LEN = 256             # MiniLM-L6 was trained on up to 256 tokens
BATCH_SIZE = 128
INDEX = "arxiv_index"     # writes arxiv_index.npy (vectors) + arxiv_index.json (ids, titles)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(checkpoint=None):
    """The base MiniLM model, or the encoder part of a fine-tuned checkpoint."""
    if checkpoint:
        model, ckpt = load_checkpoint(checkpoint, DEVICE)
        encoder, name = model.encoder, ckpt["model_name"]
    else:
        encoder, name = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval(), MODEL_NAME
    return encoder, AutoTokenizer.from_pretrained(name), name


@torch.no_grad()
def encode(encoder, tokenizer, texts, max_len=MAX_LEN):
    enc = tokenizer(texts, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
    return embed(encoder, enc["input_ids"], enc["attention_mask"]).cpu().numpy()


def build(data_path, nb_papers, checkpoint, index):
    encoder, tokenizer, name = load_encoder(checkpoint)
    vectors, papers, batch = [], [], []

    def flush():
        vectors.append(encode(encoder, tokenizer, batch).astype(np.float16))  # float16 halves the size
        batch.clear()
        print(f"\r  embedded {len(papers)} papers", end="", flush=True)

    for p in iter_papers(data_path, nb_papers):  # streamed: the whole file never sits in memory
        batch.append(paper_text(p["title"], p["abstract"]))
        papers.append({"id": p["id"], "title": clean(p["title"])})
        if len(batch) == BATCH_SIZE:
            flush()
    if batch:
        flush()
    print()

    np.save(f"{index}.npy", np.concatenate(vectors))
    meta = {"model_name": name, "checkpoint": checkpoint, "max_len": MAX_LEN, "papers": papers}
    with open(f"{index}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"Index of {len(papers)} papers saved to {index}.npy / {index}.json")


class Searcher:
    def __init__(self, index=INDEX):
        self.vectors = np.load(f"{index}.npy", mmap_mode="r")  # read from disk as needed
        with open(f"{index}.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.papers, self.max_len = meta["papers"], meta["max_len"]
        # Queries must be embedded with the same model that built the index
        self.encoder, self.tokenizer, _ = load_encoder(meta["checkpoint"])

    def search(self, query, k=10):
        q = encode(self.encoder, self.tokenizer, [query], self.max_len)[0]
        # Vectors are normalised, so a dot product = cosine similarity. Done in chunks
        # so a large index never has to be converted to float32 all at once.
        scores = np.empty(len(self.vectors), dtype=np.float32)
        for s in range(0, len(scores), 100_000):
            scores[s:s + 100_000] = self.vectors[s:s + 100_000].astype(np.float32) @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.papers[i]) for i in top]


def print_results(results):
    for score, p in results:
        print(f"  {score:.3f}  {p['id']:<12} {p['title']}")
        print(f"         https://arxiv.org/abs/{p['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="embed papers and save the index")
    b.add_argument("--data", default=DATA_PATH)
    b.add_argument("--n", type=int, default=NB_PAPERS, help="number of papers (default %(default)s)")
    b.add_argument("--all", action="store_true", help="index the whole file")
    b.add_argument("--checkpoint", default=None, help="use the encoder of a fine-tuned checkpoint")
    b.add_argument("--index", default=INDEX)
    q = sub.add_parser("query", help="search the index")
    q.add_argument("text", nargs="?", help="leave out for interactive mode")
    q.add_argument("-k", type=int, default=10, help="number of results")
    q.add_argument("--index", default=INDEX)
    args = parser.parse_args()

    if args.command == "build":
        build(args.data, None if args.all else args.n, args.checkpoint, args.index)
        return

    searcher = Searcher(args.index)
    if args.text:
        print_results(searcher.search(args.text, args.k))
        return
    while True:
        text = input("\nQuery (empty to quit): ").strip()
        if not text:
            break
        print_results(searcher.search(text, args.k))


if __name__ == "__main__":
    main()
