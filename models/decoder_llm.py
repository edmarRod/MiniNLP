import math
import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadSelfAttentionLayer(nn.Module):
  def __init__(self, n_heads: int = 8, embedding_dim: int = 256, context_size: int = 256) -> None:
      super().__init__()

      self.n_heads = n_heads
      self.embedding_dim = embedding_dim
      self.head_size = embedding_dim//n_heads
      self.context_size = context_size

      self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
      self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
      self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
      self.w_o = nn.Linear(embedding_dim, embedding_dim, bias=False)

      self.register_buffer("mask", torch.tril(torch.ones(self.context_size,self.context_size)).view(1, 1, self.context_size, self.context_size))

  def forward(self, x):
    B, T, C = x.size()

    q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, T, head_size)
    k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, T, head_size)
    v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, n_heads, T, head_size)

    q_k = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_size) # (B, n_heads, T, head_size)

    causal_q_k = q_k.masked_fill(self.mask[:,:,:T,:T]== 0, float('-inf')) # (B, n_heads, T, head_size)

    causal_q_k = F.softmax(causal_q_k, dim=-1) # (B, n_heads, T, T)

    att = causal_q_k @ v # (B, n_heads, T, head_size)
    att = self.w_o(att.transpose(1, 2).contiguous().view(B, T, C))# (B, T, C)
    return att 

class FeedForward(nn.Module):
    def __init__(self, embedding_dim:int = 256, dim_ff:int = 1024) -> None:
        super().__init__()
        self.dim_ff = dim_ff
        self.embedding_dim = embedding_dim
        self.ff1 = nn.Linear(in_features=self.embedding_dim, out_features=self.dim_ff)
        self.ff2 = nn.Linear(in_features=self.dim_ff, out_features=self.embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        return x


    
class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim:int = 256, n_heads:int = 8,dim_ff:int = 1024) -> None:
        super().__init__()

        self.attn = MultiHeadSelfAttentionLayer(n_heads=n_heads, embedding_dim=embedding_dim, context_size=embedding_dim)
        self.ff = FeedForward(embedding_dim=embedding_dim, dim_ff=dim_ff)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.ff(self.layer_norm2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size: int = 20000, embedding_dim: int = 256, context_size: int = 256, num_heads: int = 8, dim_feedforward:int = 1024, num_blocks: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.positional_embedding = nn.Linear(in_features=self.embedding_dim, out_features=self.context_size)

        self.blocks = nn.Sequential(*[DecoderBlock(embedding_dim=self.embedding_dim, n_heads=self.num_heads, dim_ff=self.dim_feedforward) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.lm_head = nn.Linear(in_features=self.embedding_dim, out_features=self.vocab_size)

    def forward(self, x):
        x = self.embedding(x) # (B, T, C)
        x = x + self.positional_embedding.weight # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.lm_head(self.layer_norm(x)) # (B, T, vocab_size)
        return x


    def generate(self, x, max_new_tokens:int, temperature:float = 1.0, top_k:int = None, top_p:float = None):
        for _ in range(max_new_tokens):
            logits = self(x[:, -self.context_size:])
            logits = logits[:, -1, :] / temperature

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
if __name__ == '__main__':
    model = MiniLLM(context_size=8)
    out = model(torch.ones(4, 8, dtype=torch.long))
    x = model.generate(torch.ones(1, 8, dtype=torch.long), max_new_tokens=10)
    print(x.shape)