import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

import wandb


class Trainer:
    def __init__(
        self,
        model,
        device,
        optimizer,
        criterion,
        max_epochs,
        train_loader,
        test_loader,
        root_dir,
        experiment_name,
        scheduler,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        wandb.init(
            project="effdl",
            config={"model": model, "optimizer": optimizer, "max_epochs": max_epochs},
        )
        wandb.watch(self.model)

    def train(self):
        """
        Train and test loop for a model
        """
        best_acc = 0
        for epoch in tqdm(range(self.max_epochs)):
            self.train_one_step()
            self.scheduler.step()
            new_acc = self.test_one_step()
            if new_acc > best_acc:
                best_acc = new_acc
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.root_dir, "checkpoints", self.experiment_name + ".pt"
                    )
                )
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
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.half().to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
