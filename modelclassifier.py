from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


#--------------TOKENIZATION----------------
class Token:

    def __init__(self, model, texts, max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.texts = texts
        self.max_len = max_len
        self.enc = self.tokenizer(         #The Tokens
            self.texts,
            max_length = self.max_len, 
            padding = "max_length", 
            truncation = True, 
            return_tensors = "pt")

    def get(self):
        return self.enc

    def make_dataset(self, labels):
        dataset = TensorDataset(
        self.enc["input_ids"],
        self.enc["attention_mask"],
        labels)

        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return loader


    '''
    # 80/10/10 split — train, validation, test
    total    = len(texts) // BATCH_SIZE
    n_test   = total // 10
    n_val    = total // 10
    test_ds  = dataset.take(n_test)
    val_ds   = dataset.skip(n_test).take(n_val)
    train_ds = dataset.skip(n_test + n_val)
    '''


#--------------MODEL----------------
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)



class ArxivClassifier(nn.Module):          #PyTorch Model format
    def __init__(self, model_name, n_cats):
        super().__init__() 

        self.st_model = SentenceTransformer(model_name)
        self.transformer = self.st_model[0]
        hidden = self.transformer.config.hidden_size

        #different layers
        self.fc1 = nn.Linear(hidden, 256)     #explain numbers
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, n_cats)
        
        self.freeze_transformer()

    def freeze_transformer(self):
        for param in self.transformer.parameters(): #freezes all layers
            param.requires_grad = False
        for i in [4,5]:
            for param in self.transformer.auto_model.encoder.layer[i].parameters(): #unfreezes layers 4 and 5 (which are the relevant ones)
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        #kind of network architecture
        outputs = self.transformer(features)

        x = mean_pooling(outputs["token_embeddings"], attention_mask)
        x = F.normalize(x, p=2, dim=1)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm(x)
        logits = self.fc2(x)
        return logits #predictions 

    def parameters(self):
        """Yield all trainable parameters for the optimizer."""
        yield from self.transformer.parameters()

    def train(self):
        self.transformer.train()


    def eval(self):
        self.transformer.eval()

    def to(self, device):
        self.transformer.to(device)
        return self

    def state_dict(self):
        return {
            "transformer": self.transformer.state_dict(),
        }

    def load_state_dict(self, sd):
        self.transformer.load_state_dict(sd["transformer"])
