import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from tqdm import tqdm

import wandb


class Trainer:
    def __init__(
        self,
        model,
        device,
        optimizer,
        loss_class,
        max_epochs,
        train_loader,
        test_loader,
        root_dir,
        experiment_name,
        scheduler,
        iterative_global_prune,
        global_prune_iteration,
        iterative_structured_prune,
        structured_prune_iteration,
        use_distillation,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_class = loss_class
        self.max_epochs = max_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.iterative_global_prune = iterative_global_prune
        self.global_prune_iteration = global_prune_iteration
        self.iterative_structured_prune = iterative_structured_prune
        self.structured_prune_iteration = structured_prune_iteration
        self.use_distillation = use_distillation
        model.half()
        model.to(device)

        wandb.init(
            project="effdl",
            config={"model": model, "optimizer": optimizer, "max_epochs": max_epochs},
        )
        wandb.watch(self.model)

    def global_prune(self):
        parameters_to_prune = [
            (module, "weight")
            for module in self.model.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]

        prune.global_unstructured(
            parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.3
        )

    def remove_pruning(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.remove(module, "weight")

    def structured_prune(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=0.3, n=1, dim=1)

    def train_loop_one_step(self):
        self.train_one_step()
        self.scheduler.step()
        new_acc = self.test_one_step()
        return new_acc

    def train(self):
        """
        Train and test loop for a model
        """
        if self.iterative_global_prune:
            acc = 0
            for _ in tqdm(range(self.global_prune_iteration)):
                self.global_prune()
                for epoch in range(self.max_epochs):
                    acc = self.train_loop_one_step()
            self.remove_pruning()
            self.save()
        elif self.iterative_structured_prune:
            acc = 0
            for _ in tqdm(range(self.structured_prune_iteration)):
                self.structured_prune()
                for epoch in range(self.max_epochs):
                    acc = self.train_loop_one_step()
            self.remove_pruning()
            self.save()

        else:
            best_acc = 0
            for epoch in tqdm(range(self.max_epochs)):
                acc = self.train_loop_one_step()
                if acc > best_acc:
                    self.save()

        wandb.finish()

    @torch.no_grad()
    def test_one_step(self):
        acc = 0
        total = 0
        correct = 0
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.half().to(self.device), targets.half().to(
                self.device
            )
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total
        wandb.log({"Test Accuracy": acc})
        return acc

    def train_one_step(self):
        self.model.train()
        if self.use_distillation:
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.half().to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_class.cross_entropy_loss(self.model, inputs, targets)
                loss.backward()
                self.optimizer.step()
        else:
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.half().to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_class.distilled_cross_entropy(self.model, inputs, targets)
                loss.backward()
                self.optimizer.step()

    def save(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.root_dir, "checkpoints", self.experiment_name + ".pt"),
        )
