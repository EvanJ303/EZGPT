import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_dim = 128
num_heads = 12
num_tokens = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Embedding(num_tokens, embedding_dim)

    def forward(self, tokens):
        embeddings = self.embedding(tokens, dtype=torch.float32)
        positions = torch.arange(0, num_tokens, dtype=torch.long)
        positions = self.positional_encoding(positions)
        positional_embeddings = torch.add(embeddings, positions)
        return positional_embeddings
    
class FFN

