# Helper functions (e.g., value function, embeddings visualization)

import torch

def calculate_value_function(hidden_state):
    return torch.norm(hidden_state, dim=-1).item()
