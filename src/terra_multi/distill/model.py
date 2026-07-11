"""UNI2 -> TERRA cell-embedding projector and cosine distillation loss."""

import torch.nn as nn
import torch.nn.functional as F


class TerraProjector(nn.Module):
    def __init__(self, in_dim=1536, out_dim=384, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def cosine_distill_loss(student, teacher):
    teacher = F.normalize(teacher, dim=-1)
    return 1.0 - (student * teacher).sum(dim=-1).mean()
