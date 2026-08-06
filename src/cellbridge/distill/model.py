"""UNI2 -> TERRA cell-embedding projector and distillation losses."""

import torch
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

    def forward(self, x, return_raw=False):
        raw = self.net(x)
        normed = F.normalize(raw, dim=-1)
        if return_raw:
            return normed, raw
        return normed


def cosine_distill_loss(student_normed, teacher):
    teacher = F.normalize(teacher, dim=-1)
    return 1.0 - (student_normed * teacher).sum(dim=-1).mean()


def variance_loss(z, gamma=1.0, eps=1e-4):
    z = z - z.mean(dim=0, keepdim=True)
    std = (z.var(dim=0, unbiased=False) + eps).sqrt()
    return F.relu(gamma - std).mean()


def covariance_loss(z):
    batch_size, dim = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    covariance = (z.T @ z) / max(batch_size - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return off_diagonal.square().sum() / dim
