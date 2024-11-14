import torch
import torch.nn as nn
import torch.nn.functional as F

#Constants/Hyperparameters
#1600 dimension embedding space
#25 heads per self-attention block
#48 blocks
#1024 token maximum
d_model = 1600
num_heads = 25
num_blocks = 48
num_tokens = 1024
d_head = 64
hs_dim = 6400
vocab_size = 69420 #placeholder

#Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("(SYSTEM) USING DEVICE: CUDA")
else:
    device = torch.device("cpu")
    print("(SYSTEM) USING DEVICE: CPU")

#Token embedder NN:
class embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(num_tokens, d_model)

    def forward(self, tokens):
        embeddings = self.embedding(tokens, dtype = torch.float32)
        positions = torch.arange(0, num_tokens, dtype = torch.long)
        positions = self.positional_encoding(positions)
        positional_embeddings = torch.add(embeddings, positions)
        return positional_embeddings
    
#Attention block
class attention_block(nn.Module):
    def __init__(self, num_heads, d_model, d_head):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_head

        self.q_mat = nn.Linear(self.d_model, self.d_head)
        self.k_mat = nn.Linear(self.d_model, self.d_head)
        self.v_mat = nn.Linear(self.d_model, self.d_head)
        self.merge_heads = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):

        head_outs = []
    
        for _ in range(self.num_heads):
            q_vec = self.q_mats(x)
            k_vec = self.k_mats(x)
            v_vec = self.v_mats(x)

            k_transposed = torch.transpose(k_vec, 1, 2)

            att_scores = torch.matmul(q_vec, k_transposed)
            att_scores = torch.div(att_scores, self.d_head ** 0.5)
            att_scores = F.softmax(att_scores, 1)

            v_transposed = torch.transpose(v_vec, 1, 2)

            output = torch.matmul(att_scores, v_transposed)
            head_outs.append(output)

        att_out = torch.cat(head_outs, 2)
        att_out = self.merge_heads(att_out)

        return att_out
            

class ffn(nn.Module):
    def __init__(self, d_model, hs_dim):
        super().__init__()
        self.d_model = d_model
        self.hs_dim = hs_dim

        self.l1 = nn.Linear(self.d_model, self.hs_dim)
        self.l2 = nn.Linear(self.hs_dim, d_model)

    def forward(self, embedding):
        hidden_state = self.l1(embedding)
        hidden_state = F.gelu(hidden_state)
        output = self.l2(hidden_state)

        return output
    
class output_layer(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.d_model = d_model

class gpt(nn.Module):
    def __init__(self, num_blocks, num_heads, d_model, d_head, vocab_size, hs_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_head
        self.vocab_size = vocab_size
        self.hs_dim = hs_dim

        self.embedder = embedder(self.vocab_size, self.d_model)
        self.attention_block = attention_block(self.num_heads, self.d_model, self.d_head)
        self.ffn = ffn(self.d_model, self.hs_dim)

    def forward(self, x):
        for _ in range(num_blocks):
            x
        









