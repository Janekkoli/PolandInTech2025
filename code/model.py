import torch
from torch import nn
from modelMain import MySentenceEmbedding


class SentenceEmbedingClosnesnessModel(nn.Module):
    def __init__(self, input_size=768, SE_output_size=768):
        super().__init__()
        self.sentence_embedding = MySentenceEmbedding(input_size, 40)
        #self.linear = nn.Linear(SE_output_size*2, 1)
        self.weight = nn.Linear(1,1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, seq_len, 768) -> out: (batch, 1)
        sent1 = x[:,0,:,:]  # (batch, seq_len, 768)
        sent2 = x[:,1,:,:]  # (batch, seq_len, 768
        emb1 = self.sentence_embedding(sent1)  # (batch, SE_output_size)
        emb2 = self.sentence_embedding(sent2)  # (batch, SE_output_size
        #combined = torch.cat((emb1, emb2), dim=1)  # (batch, SE_output_size*2)
        combined = torch.norm(emb1 - emb2, dim=1, keepdim=True)  # (batch, 1)
        #print(emb1.shape, emb2.shape)
        out = self.weight(combined)  # (batch, 1)
        out = out.squeeze(1)  # (batch,)
        return out

def get_model():
    model = MySentenceEmbedding()
    # Do not load weights here, they will be loaded by the platform automatically
    return model