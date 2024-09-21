import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_dim = 128
num_heads = 8
num_layers = 8
num_tokens = 200
dropout = 0.2

class embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Embedding(num_tokens, embedding_dim)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        
