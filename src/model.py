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
d_hs = 6400
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

    def forward(self, x):
        x = self.embedding(x, dtype = torch.float32)
        positions = torch.arange(0, num_tokens, dtype = torch.long)
        positions = self.positional_encoding(positions)
        x = torch.add(x, positions)
        return x
    
#Attention block
class attention_block(nn.Module):
    def __init__(self, num_heads, d_model, d_head):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_head

        self.query = nn.ModuleList([nn.Linear(self.d_model, self.d_head) for _ in range(self.num_heads)])
        self.key = nn.ModuleList([nn.Linear(self.d_model, self.d_head) for _ in range(self.num_heads)])
        self.value = nn.ModuleList([nn.Linear(self.d_model, self.d_head) for _ in range(self.num_heads)])
        self.merge_heads = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):

        head_outs = []
    
        for i in range(self.num_heads):
            q = self.query[i](x)
            k = self.key[i](x)
            v = self.value[i](x)

            k = torch.transpose(k, 1, 2)

            x = torch.matmul(q, k)
            x = torch.div(x, self.d_head ** 0.5)
            x = F.softmax(x, 1)

            v = torch.transpose(v, 1, 2)

            x = torch.matmul(x, v)
            head_outs.append(x)

        x = torch.cat(head_outs, 2)
        x = self.merge_heads(x)

        return x
            
class ffn(nn.Module):
    def __init__(self, d_model, d_hs):
        super().__init__()
        self.d_model = d_model
        self.d_hs = d_hs

        self.l1 = nn.Linear(self.d_model, self.d_hs)
        self.l2 = nn.Linear(self.d_hs, d_model)

    def forward(self, x):
        x = self.l1(x)
        x = F.gelu(x)
        x = self.l2(x)

        return x
    
class output_layer(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.d_model = d_model

class gpt(nn.Module):
    def __init__(self, num_blocks, num_heads, d_model, d_head, vocab_size, d_hs):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_head
        self.vocab_size = vocab_size
        self.d_hs = d_hs

        self.embedder = embedder(self.vocab_size, self.d_model)
        self.attention_block = nn.ModuleList([attention_block(self.num_heads, self.d_model, self.d_head) for _ in range(self.num_blocks)])
        self.ffn = nn.ModuleList([ffn(self.d_model, self.d_hs) for _ in range(self.num_blocks)])

    def forward(self, x):
        x = self.embedder(x)

        for i in range(num_blocks):
            output = self.attention_block[i](x)
            x = torch.add(x, output)

            output = self.ffn[i](x)
            x = torch.add(x, output)

        
            
        









