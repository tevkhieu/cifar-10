"""Loss functions for training the model."""

import torch
import torch.nn as nn

T = 20
ce_loss = nn.CrossEntropyLoss()
soft_target_loss_weight = 0.5
ce_loss_weight = 0.5

class Loss:
    def __init__(self, teacher_model, use_distilled, device):
        if use_distilled:
            self.teacher_model = teacher_model
            self.teacher_model.eval()
            self.teacher_model.to(device)
            self.teacher_model.half()            
        else:
            self.teacher_model = None

    @staticmethod
    def cross_entropy_loss(model, input_batch, target_batch):
        return torch.nn.functional.cross_entropy(model(input_batch), target_batch)

    def distilled_cross_entropy(self, model, input_batch, target_batch):
        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        with torch.no_grad():
            teacher_logits = self.teacher_model(input_batch)

        # Forward pass with the student model
        student_logits = model(input_batch)

        #Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

        # Calculate the true label loss
        label_loss = ce_loss(student_logits, target_batch)

        # Weighted sum of the two losses
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
        
        return loss