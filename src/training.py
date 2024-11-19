import torch
size= 10
mask = torch.triu(torch.ones(size, size), diagonal=0)
print(mask)