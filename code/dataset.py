import torch
from torch.utils.data import Dataset

class SentencePackDataset(Dataset):
    def __init__(self, data_path='data/sentence_pack.pt', label_path='data/label_pack.pt'):
        self.data = torch.load(data_path)
        self.labels = torch.load(label_path)
        
        assert len(self.data) == len(self.labels), "Data and labels must have same batch size"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]