 # Scripts for loading and preprocessing data
 
import torch
from torch.utils.data import DataLoader, Dataset

class ReasoningDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def load_data(self, path):
        # Placeholder: Load and preprocess the dataset
        return [
            {"input": "<bot> question <eot>", "target": "answer"}
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_tensor = torch.tensor([ord(char) for char in example["input"]], dtype=torch.long)
        target_tensor = torch.tensor([ord(char) for char in example["target"]], dtype=torch.long)
        return input_tensor, target_tensor


def get_dataloader(data_path, batch_size):
    dataset = ReasoningDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)