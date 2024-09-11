import torch
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        ecg_data = torch.tensor(item['ecg'].transpose(), dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        return ecg_data, labels