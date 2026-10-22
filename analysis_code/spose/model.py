"""
SPoSE model definition.

Adapted from the official implementation:
  https://github.com/ViCCo-Group/SPoSE
  ViCCo-Group, Max Planck Institute for Human Cognitive and Brain Sciences
  Muttenthaler et al. (2022) "Human alignment of neural network representations"

Key design:
  - A single linear layer (no bias) of shape (embed_dim, n_items).
  - Non-negativity is NOT enforced architecturally; instead a soft positivity
    penalty (0.01 * relu(-W).sum()) is added to the loss during training.
  - The final non-negative embedding is recovered by clamping: relu(W.T).
"""

import torch
import torch.nn as nn


class SPoSE(nn.Module):
    """Sparse Positive Object Similarity Embedding model.

    Parameters
    ----------
    in_size : int
        Number of items (vocabulary size).  The model uses a linear projection
        from one-hot item vectors to the embedding space.
    out_size : int
        Embedding dimensionality (number of latent dimensions).
    init_weights : bool
        If True, initialise weights with N(0, 0.1).
    """

    def __init__(self, in_size: int, out_size: int, init_weights: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size, bias=False)
        if init_weights:
            nn.init.normal_(self.fc.weight, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_size * 3, n_items)
            One-hot encoded triplet items (3 items per trial, flattened).

        Returns
        -------
        Tensor of shape (batch_size * 3, embed_dim)
            Embedding vectors; caller reshapes to (-1, 3, embed_dim).
        """
        return self.fc(x)


def l1_regularization(model: SPoSE) -> torch.Tensor:
    """Sum of absolute values of the embedding weight matrix (L1 norm).

    In the official SPoSE training loop this is scaled by (lmbda / n_items)
    before being added to the loss.
    """
    return model.fc.weight.abs().sum()


def get_embedding(model: SPoSE) -> torch.Tensor:
    """Extract the non-negative embedding matrix.

    Returns
    -------
    Tensor of shape (n_items, embed_dim), values clamped to [0, ∞).
    """
    return torch.relu(model.fc.weight).T.detach().cpu()


def get_nneg_dims(model: SPoSE, eps: float = 0.1) -> torch.Tensor:
    """Return indices of dimensions whose maximum weight across items > eps.

    Mirrors official SPoSE pruning criterion (utils.get_nneg_dims / remove_zeros).
    """
    W = model.fc.weight  # (embed_dim, n_items)
    w_max = W.max(dim=1).values  # max activation per dimension
    return torch.where(w_max > eps)[0]
