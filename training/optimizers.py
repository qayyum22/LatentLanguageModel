# Optimizer configurations


import torch

def get_optimizer(model, lr=0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)
