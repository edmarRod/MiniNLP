import torch
from torch import nn


class CBOW(nn.Module):
    def __init__(self, vocab_size: int = 20000, embedding_dim: int = 256, context_size: int = 4, dim_ff:int = 1024) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(context_size * embedding_dim, dim_ff, bias=False)
        self.out = nn.Linear(dim_ff, vocab_size, bias=False)

    def forward(self, x):
        x = self.emb(x) # (B, T, C)
        x = self.proj(x.view(x.size(0), -1)) # (B, T * C)
        x = self.out(x) # (B, vocab_size)
        return x
    
if __name__ == '__main__':
    model = CBOW(embedding_dim=256, context_size=8)
    out = model(torch.ones(4, 8, dtype=torch.long))
    #x = model.generate(torch.ones(1, 8, dtype=torch.long), max_new_tokens=10)
    print(out.shape)