# Model export and deployment


import torch

def export_model(model, export_path):
    torch.save(model.state_dict(), export_path)
    print(f"Model saved to {export_path}")
