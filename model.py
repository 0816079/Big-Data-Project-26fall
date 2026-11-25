import numpy as np
import copy


class BERT:
    def __init__(self, corpus =None, name="bert-base-uncased", epochs=3, max_len=256, lr=2e-5, batch=16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.epochs, self.max_len, self.lr, self.batch = torch, epochs, max_len, lr, batch
        self.tokenize = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)  
        self.dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def _inputs(self , texts):
        return   self.tokenize(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.dev)

    def fit(self, texts, y):
        torch = self.torch
        new = copy.copy(self)
        new.model = copy.deepcopy(self.model).to(self.dev).train()
        y= torch.tensor(np.asarray(y))
        n_pos = int(y.sum())
        unbalanced = n_pos ==0 or (len(y) - n_pos)/ n_pos >=2
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device=self.dev) if unbalanced else None)
        optimizer = torch.optim.AdamW(new.model.parameters(), lr=self.lr)
        texts = list(texts)
        for _ in range(self.epochs):
            order = np.random.permutation(len(texts))
            for start in range(0, len(texts), self.batch):
                idx = order[start:start+self.batch]
                logits = new.model(**self._inputs([texts[i] for i in idx])).logits
                loss = loss_fn(logits, y[idx].to(self.dev))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        new.model.eval()
        return new        