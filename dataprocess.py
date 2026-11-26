import json
import numpy as np



def clean(text):
    #Collapse newlines and repeated spaces 'A long\\n  title' -> 'A long title'
    return " ".join(text.split())


def paper_text(title, abstract=""):
    #One text format used everywhere (training, prediction and search)
    title, abstract = clean(title), clean(abstract)
    return f"{title}. {abstract}" if abstract else title



def iter_papers(path, max_papers=None):
    #Yield one paper (a dict) per line of the file
    #Kaggle file is JSON Lines, exactly one JSON object per line 
    #To make a small test file, keep it in that format, e.g.  head -n 2000 arxiv-metadata-oai-snapshot.json > sample.jsonl

    n = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_papers is not None and n >= max_papers:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}, line {line_no} is not a JSON object. The file must have one paper "
                    f"per line (JSON Lines). RTF files or pretty-printed JSON won't work."
                ) from e
            n += 1





class Data:
    #Reads file once and keeps paper ids, texts, the category vocabulary and the labels
    def __init__(self, path, max_papers=None):
        self.ids, self.texts, cat_lists = [], [], []
        for paper in iter_papers(path, max_papers):
            self.ids.append(paper["id"])
            self.texts.append(paper_text(paper["title"], paper["abstract"]))
            cat_lists.append(paper["categories"].split())

        # Sorted so each category gets the same index on every run
        self.categories = sorted({cat for cats in cat_lists for cat in cats})
        self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}
        self.labels = self.encode_labels(cat_lists)

    @property
    def n_cats(self):
        return len(self.categories)

    def encode_labels(self, cat_lists):
        #List of category lists -> (n_papers, n_cats) array of 0/1 (multi-hot)
        labels = np.zeros((len(cat_lists), self.n_cats), dtype=np.float32)
        for row, cats in enumerate(cat_lists):
            for cat in cats:
                if cat in self.cat2idx:
                    labels[row, self.cat2idx[cat]] = 1.0
        return labels













    def build_cat_vocab(self, max_papers=5):
        """ 
        Scan the dataset once to collect all categories.
        Returns dictionnary with each {category : index}
        """
        cats = set()
        with open(self.path) as f:
            for i, line in enumerate(f):
                if i >= max_papers:
                    break
                metadata = json.loads(line)
                for cat in metadata["categories"].strip().split():
                    cats.add(cat)
        # Sort for determinism — same order every run
        self.cat2idx = {cat: i for i, cat in enumerate(sorted(cats))}
        return self.cat2idx
    






    

    def load_arxiv(self, max_papers = 5):
        texts = []
        labels = []
        with open(self.path) as f:
            for i, line in enumerate(f):
                if i >= max_papers:
                    break
                metadata = json.loads(line)
                metadata["title"] = metadata["title"].replace('\n', ' ')
                metadata["abstract"] = metadata["abstract"].replace('\n', ' ')
                text = (
                    f"Title: {metadata['title'].strip()}. "
                    f"Abstract: {metadata['abstract'].strip()}."
                )
                label = np.zeros(self.n_cats, dtype = np.float32)
                for cat in metadata["categories"].strip().split():
                    if cat in self.cat2idx:
                        label[self.cat2idx[cat]] = 1.0

                texts.append(text)
                labels.append(label)

        return texts, np.array(labels)
    