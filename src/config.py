"""
Constants/Hyperparameters:

GPT2 Small:
768 dimension embedding space
12 heads per self-attention block
12 blocks
1024 tokens maximum

GPT2 Medium:
1024 dimension embedding space
16 heads per self-attention block
24 blocks
1024 token maximum

GPT2 Large:
1280 dimension embedding space
20 heads per self-attention block
24 blocks
1024 token maximum

Custom model configurations should be written below this comment

Required parameters:
d_model
num_heads
num_blocks
num_tokens
d_head
d_hs
vocab_size
dropout
"""

class GPT2_small_cfg:
    d_model = 768
    num_heads = 12
    num_blocks = 12
    num_tokens = 1024
    d_head = 64
    d_hs = 3072
    vocab_size = 50257
    dropout = 0.1

class GPT2_med_CFG:
    d_model = 1024
    num_heads = 16
    num_blocks = 24
    num_tokens = 1024
    d_head = 64
    d_hs = 4096
    vocab_size = 50257
    dropout = 0.1

class GPT2_large_cfg:
    d_model = 1280
    num_heads = 20
    num_blocks = 24
    num_tokens = 1024
    d_head = 64
    d_hs = 5120
    vocab_size = 50257
    dropout = 0.1

"""
Custom training configurations should be written below this comment

Required parameters:
learn_rate
batch_size
device
num_workers
num_epochs
max_norm
"""