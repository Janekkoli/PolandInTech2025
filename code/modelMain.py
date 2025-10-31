import torch
from torch import nn

class MySentenceEmbedding(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim = 30, reduce_size = 64):
        super().__init__()
        self.reduceLayer = nn.Linear(embed_dim, reduce_size)
        self.gru = nn.GRU(reduce_size, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x: [batch, seq, embed_dim]
        x = self.reduceLayer(x)  # [batch, seq, reduce_size]
        _, h = self.gru(x)           # h: [2, batch, hidden_dim]
        h = torch.cat(list(h), dim=-1)  # [batch, hidden_dim*2]
        return self.projection(h)

def get_model():
    model = MySentenceEmbedding()
    # Do not load weights here, they will be loaded by the platform automatically
    return model