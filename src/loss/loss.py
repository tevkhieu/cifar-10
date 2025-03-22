"""Loss functions for training the model."""

import torch


class Loss:
    def __init__(self, teacher_model, use_distilled):
        if use_distilled:
            self.teacher_model = teacher_model
        else:
            self.teacher_model = None

    @staticmethod
    def cross_entropy_loss(model, input_batch, target_batch):
        return torch.nn.functional.cross_entropy(model(input_batch), target_batch)

    def distilled_cross_entropy(self, model, input_batch, target_batch):
        return torch.nn.funtional.cross_entropy(
            model(input_batch), target_batch
        ) + 0.5 * torch.linalg.vector_norm(
            self.teacher_model(input_batch - model(input_batch), ord=2)
        )
