# Implementation of LatentLanguageModel

import torch
import torch.nn as nn

class LatentLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, max_len):
        super(LatentLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_tokens, mode="language", latent_steps=0):
        token_embeddings = self.token_embedding(input_tokens)
        position_ids = torch.arange(input_tokens.size(1), device=input_tokens.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        if mode == "language":
            hidden_states = self.transformer_encoder(embeddings)
            return self.output_layer(hidden_states)
        elif mode == "latent":
            hidden_state = embeddings[:, -1, :]
            for _ in range(latent_steps):
                hidden_state = self.transformer_encoder(hidden_state.unsqueeze(1)).squeeze(1)
            return hidden_state
        else:
            raise ValueError("Invalid mode.")