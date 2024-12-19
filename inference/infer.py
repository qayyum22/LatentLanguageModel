# Inference and evaluation logic



import torch

def infer(model, input_tokens, latent_steps):
    model.eval()
    with torch.no_grad():
        mode = "latent" if "<bot>" in input_tokens else "language"
        outputs = model(input_tokens, mode=mode, latent_steps=latent_steps)
    return outputs