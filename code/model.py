import torch
from torch import nn

class MySentenceEmbedding(nn.Module):
    def __init__(self, input_size=768, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 768) -> out: (batch, output_size)
        # Mean pooling over sequence dimension
        x = torch.mean(x, dim=1)  # (batch, seq_len, 768) -> (batch, 768)
        return self.fc2(torch.relu(self.fc1(x)))

def get_model():
    model = MySentenceEmbedding()
    # Do not load weights here, they will be loaded by the platform automatically
    return model