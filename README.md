# arXiv semantic search and category classifier

Uses the arXiv metadata dataset from Kaggle and the `sentence-transformers/all-MiniLM-L6-v2` model to:

Search: find the papers closest in meaning to a query (`search.py`).

Classify: predict a paper's arXiv categories from its title/abstract (python `main.py` trains, `predict.py` predicts).


`all-MiniLM-L6-v2` is already trained for semantic similarity, so search works without any training. Fine-tuning on categories (`main.py`) is optional for search: you can build a second index with the fine-tuned encoder and compare the results.

## Setup
python -m venv .venv && source .venv/bin/activate

And have torch>=2.1 transformers>=4.36 numpy>=1.24



## Data

Download from Kaggle (5.3 GB): https://www.kaggle.com/datasets/Cornell-University/arxiv

The download is named `arxiv-metadata-oai-snapshot.json`. 
Either rename it to `arxiv_metadata.json` or change `DATA_PATH` at the top of `main.py` and `search.py`.

The file is JSON Lines: one paper per line. 
A smaller test file must keep that format:

```bash
head -n 2000 arxiv-metadata-oai-snapshot.json > small_data_test.jsonl
```

`small_data_test.jsonl` (2,240 papers) is included for quick tests.

## Usage

```bash

# Semantic search
python search.py build                         # embeds NB_PAPERS papers -> arxiv_index.npy/.json
python search.py build --all                   # the whole file (slow on CPU; use a GPU)
python search.py query "étale fundamental group"
python search.py query                         # interactive mode

# Category classifier
python main.py                                 # trains, keeps the best epoch in arxiv_classifier.pt
python predict.py "Étale fundamental groups in algebraic geometry"
python search.py build --checkpoint arxiv_classifier.pt   # search with the fine-tuned encoder
```

To try everything on the small file, use --data small_data_test.jsonl with search.py build, and set DATA_PATH = "small_data_test.jsonl" in main.py.



arxiv_metadata.json available (5.33 GB) avaialble at kaggle : 
https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download
