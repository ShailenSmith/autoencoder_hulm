import torch

t = torch.zeros(10, 10)
t = t.normal_(mean=0, std=0.02)
for i in range(t.shape[0]):
    print(torch.std(t[i]))