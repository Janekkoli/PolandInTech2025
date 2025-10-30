import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def _load_file(path : str, padding_to = 68):
    with open(path, 'r') as f:
        data = f.read().splitlines()
    data = [data_line.split() for data_line in data]
    max_len = max(max(int(line[0]) for line in data),padding_to)
    count = len(data)
    token_dim = int(data[0][1])
    print(f"Count: {count}, Max Length: {max_len}, Token Dimension: {token_dim}")
    data = [[float(x) for x in line[2:]] for line in data]
    arr = torch.zeros((count, max_len*token_dim), dtype=torch.float32)
    for i, line in enumerate(data):
        arr[i, :len(line)] = torch.tensor(line, dtype=torch.float32)
    arr = arr.view(count, max_len, token_dim)
    return arr

def save_files_as_torch_tensor(input_path1: str,input_path2: str,  output_path: str):
    data1 = _load_file(input_path1)
    data2 = _load_file(input_path2)

    data = torch.cat((data1, data2), dim=1)
    data = data.view(data.shape[0], -1, 2, data.shape[2])  # (count, seq_len, 2, token_dim)
    print("data dimensions: ", data.shape, " saving to:", output_path)
    torch.save(data, output_path)

def load_y(path: str) -> torch.Tensor:
    with open(path, 'r') as f:
        data = f.read().splitlines()
    data = [float(line) for line in data]
    return torch.tensor(data, dtype=torch.float32)

def save_y_as_torch_tensor(input_path: str, output_path: str):
    data = load_y(input_path)
    print("label data dimensions: ", data.shape, " saving to:", output_path)
    torch.save(data, output_path)


save_files_as_torch_tensor('data/sentence_pack_1.dat', 'data/sentence_pack_2.dat','data/sentence_pack.pt')


save_y_as_torch_tensor('data/label_pack.dat', 'data/label_pack.pt')