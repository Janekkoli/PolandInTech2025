import torch
from torch import nn


class MySentenceEmbedding(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=40):
        super().__init__()
        hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [batch, seq, embed_dim]
        scores = self.attention(x).squeeze(-1)          # [batch, seq]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [batch, seq, 1]
        return (x * weights).sum(dim=1)  # [batch, embed_dim]
    

def get_model():
    model = MySentenceEmbedding()
    # Do not load weights here, they will be loaded by the platform automatically
    return model