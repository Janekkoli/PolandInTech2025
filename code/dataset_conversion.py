import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def _load_file(path : str) -> torch.Tensor:
    with open(path, 'r') as f:
        data = f.read().splitlines()
    data = [data_line.split() for data_line in data]
    max_len = max(int(line[0]) for line in data)
    count = len(data)
    token_dim = int(data[0][1])
    print(f"Count: {count}, Max Length: {max_len}, Token Dimension: {token_dim}")
    data = [[float(x) for x in line[2:]] for line in data]
    arr = torch.zeros((count, max_len*token_dim), dtype=torch.float32)
    for i, line in enumerate(data):
        arr[i, :len(line)] = torch.tensor(line, dtype=torch.float32)
    arr = arr.view(count, max_len, token_dim)
    return arr


def save_file_as_torch_tensor(input_path: str, output_path: str):
    data = _load_file(input_path)
    torch.save(data, output_path)


save_file_as_torch_tensor('data/sentence_pack_1.dat', 'data/sentence_pack_1.pt')
save_file_as_torch_tensor('data/sentence_pack_2.dat', 'data/sentence_pack_2.pt')


# data = _load_file('data/sentence_pack_1.dat')
# print("First item:", data[0][0][0:20])
# print("data dimensions:", data.shape)

class SentenceDataset(Dataset):
    def __init__(self, path='data/sentence_pack_1.dat', transform=None):
        self.data = _load_file(path)
        self.transform = transform

    def __len__(self):
        if isinstance(self.data, list):
            return len(self.data)
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item


def make_loader(path='data/sentence_pack_1.dat', batch_size=32, shuffle=True, num_workers=0):
    return DataLoader(SentenceDataset(path), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
