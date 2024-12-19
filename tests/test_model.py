# Unit tests for LatentLanguageModel


import unittest
import torch
from models.latent_language_model import LatentLanguageModel

class TestLatentLanguageModel(unittest.TestCase):
    def test_forward_language_mode(self):
        model = LatentLanguageModel(vocab_size=100, hidden_size=16, num_layers=2, max_len=50)
        input_tokens = torch.randint(0, 100, (2, 50))
        outputs = model(input_tokens, mode="language")
        self.assertEqual(outputs.shape, (2, 50, 100))

if __name__ == "__main__":
    unittest.main()
