

import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Inspired by https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-With-PyTorch-Lightning--VmlldzoyMzg4Njk1#implementing-the-model
    """
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()


        self.projection = nn.Linear(in_features=embedding_dim, out_features=projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(in_features=projection_dim, out_features=projection_dim)


        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)


    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)


        x += projected

        return self.layer_norm(x)
