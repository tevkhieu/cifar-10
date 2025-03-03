"""Loss functions for training the model."""

import torch


class Loss:
    def __init__(self):
        pass

    def cross_entropy_loss(self):
        return torch.nn.CrossEntropyLoss()
