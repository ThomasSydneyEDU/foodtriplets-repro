"""
SPoSE triplet loss and one-hot encoding utilities.

Adapted from the official implementation:
  https://github.com/ViCCo-Group/SPoSE  (utils.py)
  ViCCo-Group, Max Planck Institute for Human Cognitive and Brain Sciences

The trinomial loss maximises the probability that the anchor–positive pair is
identified as most similar in an odd-one-out (3-AFC) trial.
"""

import torch


def encode_as_onehot(I: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
    """Encode integer triplets as stacked one-hot vectors.

    Parameters
    ----------
    I : Tensor of shape (n_items, n_items)
        Identity matrix used for one-hot lookup.
    triplets : LongTensor of shape (batch, 3)
        Each row is [pair_item1, pair_item2, odd_item].

    Returns
    -------
    Tensor of shape (batch * 3, n_items)
        Three one-hot vectors per trial, in order [pair1, pair2, odd].
    """
    return I[triplets.flatten(), :]


def _compute_similarities(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    distance_metric: str = "dot",
):
    """Compute the three pairwise similarities for an odd-one-out triplet.

    Returns (sim_anchor_positive, sim_anchor_negative, sim_positive_negative).
    """
    if distance_metric == "dot":
        return (
            torch.sum(anchor * positive, dim=1),
            torch.sum(anchor * negative, dim=1),
            torch.sum(positive * negative, dim=1),
        )
    if distance_metric == "euclidean":
        return (
            -torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1)),
            -torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1)),
            -torch.sqrt(torch.sum((positive - negative) ** 2, dim=1)),
        )
    raise ValueError(f"Unknown distance_metric: '{distance_metric}'. Use 'dot' or 'euclidean'.")


def trinomial_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    t: torch.Tensor,
    distance_metric: str = "dot",
) -> torch.Tensor:
    """SPoSE cross-entropy loss for the odd-one-out (3-AFC) task.

    Mirrors `utils.trinomial_loss` from the official SPoSE repo.  The loss is
    the mean negative log-probability that the anchor–positive pair is the most
    similar, under a softmax with temperature *t*.

    Parameters
    ----------
    anchor, positive : Tensor of shape (batch, embed_dim)
        Embeddings of the two similar items.
    negative : Tensor of shape (batch, embed_dim)
        Embedding of the odd item.
    t : scalar Tensor
        Softmax temperature (1.0 = no scaling, matching SPoSE default).
    distance_metric : {'dot', 'euclidean'}
    """
    sim_ap, sim_an, sim_pn = _compute_similarities(anchor, positive, negative, distance_metric)

    # Manual softmax + cross-entropy — faithful to official utils.cross_entropy_loss
    exp_ap = torch.exp(sim_ap / t)
    exp_an = torch.exp(sim_an / t)
    exp_pn = torch.exp(sim_pn / t)
    p_correct = exp_ap / (exp_ap + exp_an + exp_pn)
    return torch.mean(-torch.log(p_correct.clamp(min=1e-12)))
