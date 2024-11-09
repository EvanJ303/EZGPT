import torch
import torch.nn as nn
import torch.nn.functional as F

#Params

embedding_dim = 768
num_heads = 12
head_dim = 64
num_tokens = 200

#Detect device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Token embedder NN

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
    
#Attention block NN

class attention_block(nn.Module):
    def __init__(self, num_heads, embedding_dim, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim

        self.q_mats = nn.ModuleList([nn.Linear(self.embedding_dim, self.head_dim, bias=False) for i in range(self.num_heads)])
        self.k_mats = nn.ModuleList([nn.Linear(self.embedding_dim, self.head_dim, bias=False) for i in range(self.num_heads)])
        self.v_mats = nn.ModuleList([nn.Linear(self.embedding_dim, self.head_dim, bias=False) for i in range(self.num_heads)])
        self.merge_heads = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, embedding):

        head_outs = []
    
        for i in range(self.num_heads):
            q_vec = self.q_mats[i](embedding)
            k_vec = self.k_mats[i](embedding)
            v_vec = self.v_mats[i](embedding)

            k_transposed = torch.transpose(k_vec, 1, 2)

            att_scores = torch.matmul(q_vec, k_transposed)
            att_scores = torch.div(att_scores, self.head_dim ** 0.5)
            att_scores = F.softmax(att_scores, 1)

            v_transposed = torch.transpose(v_vec, 1, 2)

            output = torch.matmul(att_scores, v_transposed)
            head_outs.append(output)

        att_out = torch.cat(head_outs)
        att_out = self.merge_heads(att_out)

        return att_out
            

class ffn(nn.Module):
    def __init__(self)









