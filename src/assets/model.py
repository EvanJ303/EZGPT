import torch
import torch.nn as nn
import torch.nn.functional as F

#Constants
embedding_dim = 768
num_heads = 12
num_blocks = 12
num_tokens = 250
head_dim = embedding_dim / num_heads
hs_dim = embedding_dim * 4

#Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("(SYSTEM) USING DEVICE: CUDA")
else:
    device = torch.device("cpu")
    print("(SYSTEM) USING DEVICE: CPU")

#Token embedder NN:
#250 token maximum
class embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Embedding(num_tokens, embedding_dim)

    def forward(self, tokens):
        embeddings = self.embedding(tokens, dtype = torch.float32)
        positions = torch.arange(0, num_tokens, dtype = torch.long)
        positions = self.positional_encoding(positions)
        positional_embeddings = torch.add(embeddings, positions)
        return positional_embeddings
    
#Attention block NN:
#64 heads
#Embedding dimension of 768
class attention_block(nn.Module):
    def __init__(self, num_heads, embedding_dim, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim

        self.q_mats = nn.ModuleList([nn.Linear(self.embedding_dim, self.head_dim, bias = False) for i in range(self.num_heads)])
        self.k_mats = nn.ModuleList([nn.Linear(self.embedding_dim, self.head_dim, bias = False) for i in range(self.num_heads)])
        self.v_mats = nn.ModuleList([nn.Linear(self.embedding_dim, self.head_dim, bias = False) for i in range(self.num_heads)])
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
    def __init__(self, embedding_dim, hs_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hs_dim = hs_dim

        self.l1 = nn.Linear(self.embedding_dim, self.hs_dim)
        self.l2 = nn.Linear(self.hs_dim, embedding_dim)

    def forward(self, embedding):
        hidden_state = self.l1(embedding)
        hidden_state = F.gelu(hidden_state)
        output = self.l2(hidden_state)

        return output

class gpt(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks









