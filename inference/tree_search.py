# Tree search and BFS logic for latent reasoning


import torch

def tree_search(model, hidden_state, num_paths=3):
    paths = [hidden_state.clone() for _ in range(num_paths)]
    value_scores = [model.calculate_value_function(path) for path in paths]
    return sorted(zip(paths, value_scores), key=lambda x: x[1], reverse=True)
