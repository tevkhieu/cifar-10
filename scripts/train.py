import os

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune

import argparse

import effdl


def create_arg_parser():
    """
    Create an argparser to parse command line arguments
    """

    parser = argparse.ArgumentParser(description="Quantization Lab")
    parser.add_argument(
        "--state_dict_path",
        type=str,
        default="checkpoints/ckpt_sgd_0.1_0.9_0.9_0.999_0.0001",
        help="Path to the state dict",
    )
    parser.add_argument("--model", type=str, default="densenet", help="Model to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer to use")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--beta_1", type=float, default=0.9, help="Beta 1")
    parser.add_argument("--beta_2", type=float, default=0.999, help="Beta 2")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="Path to the base of directory"
    )
    parser.add_argument("--global_prune", action="store_true", help="Use Global Pruning")

    return parser


def main():
    args = create_arg_parser().parse_args()
    match args.model:
        case "densenet":
            model = effdl.densenet_cifar()
        case _:
            raise ValueError("Model not supported")

    state_dict = torch.load(args.state_dict_path)
    new_state_dict = {}
    for key, value in state_dict["net"].items():
        new_state_dict[key[7:]] = value
    model.load_state_dict(new_state_dict)

    parameters_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]

    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.3
    )

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, "weight")

    model.half()
    model.cuda()

    normalize_scratch = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_scratch,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize_scratch,
        ]
    )

    dataset_dir = os.path.join(args.root_dir, "datasets")
    c10train = CIFAR10(
        dataset_dir, train=True, download=True, transform=transform_train
    )
    c10test = CIFAR10(dataset_dir, train=False, download=True, transform=transform_test)

    trainloader = DataLoader(c10train, batch_size=32)
    testloader = DataLoader(c10test, batch_size=32)

    match args.optimizer:
        case "sgd":
            optimizer = torch.optim.SGD(
                [param for param in model.parameters() if param.requires_grad],
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        case "adam":
            optimizer = torch.optim.Adam(
                [param for param in model.parameters() if param.requires_grad],
                lr=args.lr,
                betas=(args.beta_1, args.beta_2),
                weight_decay=args.weight_decay,
            )
        case _:
            raise ValueError("Optimizer not supported")

    loss_class = effdl.Loss()
    criterion = loss_class.cross_entropy_loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)
    trainer = effdl.Trainer(
        model,
        args.device,
        optimizer,
        criterion,
        args.max_epochs,
        trainloader,
        testloader,
        args.root_dir,
        args.experiment_name,
        scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
