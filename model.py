import torch
from torch import nn


class MiniLLM(nn.Module):
    def __init__(self, vocab_size: int = 1024, embedding_dim: int = 256, context_size: int = 256, num_heads: int = 8, dim_feedforward:int = 1024) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

        self.positional_embedding = torch.tensor(self.context_size)

        self.attn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads)
        self.layer_norm1 = nn.LayerNorm()
        self.ff = nn.Linear(in_features=self.dim_feedforward, out_features=self.vocab_size)
        self.layer_norm2 = nn.LayerNorm()



    def forward(self, x):

        x = self.embedding(x)
        x = x + self.positional_embedding

        attn = self.attn(x)
        x = self.layer_norm1(x + attn)
        ff = self.ff(x)
        x = self.layer_norm2(x+ff)
        return x