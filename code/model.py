import torch
from torch import nn
class MySentenceEmbedding(nn.Module):
    def __init__(self, input_size=768, output_size=30):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 768) -> out: (batch, output_size)
        # Mean pooling over sequence dimension
        x = torch.mean(x, dim=1)  # (batch, seq_len, 768) -> (batch, 768)
        return self.fc2(torch.relu(self.fc1(x)))


class SentenceEmbedingClosnesnessModel(nn.Module):
    def __init__(self, input_size=768, SE_output_size=30):
        super().__init__()
        self.sentence_embedding = MySentenceEmbedding(input_size, SE_output_size)
        self.linear = nn.Linear(SE_output_size*2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, seq_len, 768) -> out: (batch, 1)
        sent1 = x[:,0,:,:]  # (batch, seq_len, 768)
        sent2 = x[:,1,:,:]  # (batch, seq_len, 768
        emb1 = self.sentence_embedding(sent1)  # (batch, SE_output_size)
        emb2 = self.sentence_embedding(sent2)  # (batch, SE_output_size
        combined = torch.cat((emb1, emb2), dim=1)  # (batch, SE_output_size*2)
        out = self.linear(combined)  # (batch, 1)
        out = out.squeeze(1)  # (batch,)
        return out

def get_model():
    model = MySentenceEmbedding()
    # Do not load weights here, they will be loaded by the platform automatically
    return model