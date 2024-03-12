import torch
from torch import nn
from torch.nn import functional as F
class NGramModel(nn.Module):
  def __init__(self, n_gram: int, vocab_size:int, embedding_dim:int):
    super().__init__()
    self.n_gram = n_gram
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim

    self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

    self.linear = nn.Linear(self.n_gram * self.embedding_dim, self.vocab_size)

  def forward(self, x):
    x = self.embed(x).view(x.size(0), -1) # (B, T * C)
    x = self.linear(x) # (B, vocab)

    return x

  def generate(self, x, max_new_tokens:int, temperature:float = 1.0, top_k:int = None, top_p:float = None):
      for _ in range(max_new_tokens):
          logits = self(x[:, -self.n_gram:])
          logits = logits / temperature

          if top_k is not None:
              vals, _ = torch.topk(logits, 5)
              logits[logits < torch.min(vals)] = float('-inf')

          if top_p is not None:
              val, idx = torch.sort(logits,descending=True)
              logits[:, idx[torch.cumsum(F.softmax(val, dim=-1), dim=-1) >= top_p]] = float('-inf')
          
          probs = F.softmax(logits, dim=-1)

          idx_next = torch.multinomial(probs, num_samples=1)
          x = torch.cat((x, idx_next),dim=1)
      return x