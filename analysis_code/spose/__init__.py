# SPoSE — Sparse Positive Object Similarity Embeddings
# Vendored from https://github.com/ViCCo-Group/SPoSE (ViCCo-Group, Max Planck Institute)
# Only the model definition and core loss utilities are included here.
# The full repo should be cited as: Muttenthaler et al. (2022).
from .model import SPoSE, l1_regularization
from .triplet_loss import trinomial_loss, encode_as_onehot
