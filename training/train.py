# Training pipeline entry point


import torch
from data.loaders import get_dataloader
from models.latent_language_model import LatentLanguageModel
from training.curriculum import train_with_curriculum

def main():
    vocab_size, hidden_size, num_layers, max_len = 100, 16, 2, 50
    model = LatentLanguageModel(vocab_size, hidden_size, num_layers, max_len)
    dataloader = get_dataloader("data/train", batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_with_curriculum(model, dataloader, optimizer, stages=3)

if __name__ == "__main__":
    main()
