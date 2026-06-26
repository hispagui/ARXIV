import json
import numpy as np

class Data:
    def __init__(self, path, max_papers=5):
        self.path = path
        self.max_papers = max_papers 
        self.cat2idx = self.build_cat_vocab(self.max_papers)
        self.n_cats = len(self.cat2idx)

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
    