import torch

a = torch.randn(5, 3)
print(a)

b = a.unsqueeze(0)
print(b)
print(a)