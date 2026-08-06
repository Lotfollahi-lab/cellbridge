"""Molecular prototypes and the UNI2 prototype student."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Dataset


class PrototypeStudent(nn.Module):
    def __init__(
        self,
        in_dim=1536,
        hidden_dim=1024,
        feat_dim=512,
        num_prototypes=256,
        dropout=0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_prototypes = num_prototypes
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(feat_dim, num_prototypes)

    def forward(self, x, return_features=False):
        features = self.trunk(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


class PrototypeDataset(Dataset):
    def __init__(self, uni, prototype_ids):
        if len(uni) != len(prototype_ids):
            raise ValueError(
                f"UNI has {len(uni)} rows; assignments have {len(prototype_ids)}"
            )
        self.uni = uni
        self.prototype_ids = prototype_ids

    def __len__(self):
        return len(self.uni)

    def __getitem__(self, index):
        uni = torch.from_numpy(np.asarray(self.uni[index], dtype=np.float32))
        return uni, int(self.prototype_ids[index])


def l2_normalize(values):
    values = np.array(values, dtype=np.float32, copy=True, order="C")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values /= np.maximum(norms, 1e-12)
    return values


def sample_kmeans_rows(n_cells, max_cells, seed):
    if max_cells is None or n_cells <= max_cells:
        return np.arange(n_cells)
    rows = np.random.default_rng(seed).choice(n_cells, max_cells, replace=False)
    return np.sort(rows)


def fit_prototypes(
    terra,
    num_prototypes,
    max_cells,
    seed,
    batch_size=8192,
    max_iter=100,
    n_init=3,
):
    rows = sample_kmeans_rows(len(terra), max_cells, seed)
    fit_data = l2_normalize(terra[rows])
    model = MiniBatchKMeans(
        n_clusters=num_prototypes,
        init="k-means++",
        n_init=n_init,
        random_state=seed,
        batch_size=batch_size,
        max_iter=max_iter,
        max_no_improvement=10,
        reassignment_ratio=0.01,
        compute_labels=False,
    )
    model.fit(fit_data)
    return model, rows


def assign_prototypes(model, terra, batch_size=65_536):
    assignments = np.empty(len(terra), dtype=np.int32)
    for start in range(0, len(terra), batch_size):
        batch = l2_normalize(terra[start : start + batch_size])
        assignments[start : start + len(batch)] = model.predict(batch)
    return assignments


def inverse_sqrt_class_weights(prototype_ids, num_prototypes, clip=(0.25, 4.0)):
    counts = np.bincount(prototype_ids, minlength=num_prototypes)
    present = counts > 0
    weights = np.zeros(num_prototypes, dtype=np.float32)
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    weights[present] = np.clip(weights[present], clip[0], clip[1])
    return torch.from_numpy(weights)


@torch.inference_mode()
def predict_student_features(model, uni, device, batch_size=8192):
    model.eval()
    features, probabilities, logits_out = [], [], []
    for start in range(0, len(uni), batch_size):
        batch = np.asarray(uni[start : start + batch_size], dtype=np.float32)
        logits, batch_features = model(
            torch.from_numpy(batch).to(device), return_features=True
        )
        features.append(batch_features.cpu().numpy())
        probabilities.append(logits.softmax(dim=-1).cpu().numpy())
        logits_out.append(logits.cpu().numpy())
    return {
        "features_penultimate": np.concatenate(features).astype(np.float32),
        "prototype_probs": np.concatenate(probabilities).astype(np.float32),
        "prototype_logits": np.concatenate(logits_out).astype(np.float32),
    }
