import argparse
import torch
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    QConfigMapping,
)
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch.ao.quantization.quantize_fx as quantize_fx
import copy
import os
import src


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Compute accuracy")
    parser.add_argument("--model", type=str, default="densenet", help="Model to use")
    parser.add_argument("--state_dict_path", type=str, help="Path to the state dict")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    return parser


@torch.no_grad()
def compute_accuracy(model, state_dict_path, dataset_path, device):
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.to(device)
    transform_class = src.Transforms()

    transform_test = transform_class.transform_test()

    if dataset_path is None:
        dataset_dir = os.path.join(os.getcwd(), "datasets")
    else:
        dataset_dir = dataset_path

    c10test = CIFAR10(dataset_dir, train=False, download=True, transform=transform_test)

    dataloader = DataLoader(c10test, batch_size=64)

    example_inputs = next(iter(dataloader))[0].to(device)
    quantized_model = copy.deepcopy(model)
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    quantized_model.eval()
    # Prepare model for quantization
    model_prepared = quantize_fx.prepare_fx(
        quantized_model, qconfig_mapping, example_inputs
    )

    # **Calibration Step: Run a few batches to collect statistics**
    for data in tqdm(dataloader):
        images, _ = data  # We only need inputs, not labels
        images = images.to(device)
        model_prepared(images)  # This step collects statistics for quantization

    # Convert to quantized model
    model_quantized = quantize_fx.convert_fx(model_prepared)

    model_quantized.eval()
    correct = 0
    total = 0
    for data in tqdm(dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model_quantized(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


def main():
    args = create_arg_parser().parse_args()
    match args.model:
        case "densenet":
            model = src.densenet_cifar()
        case "custom_densenet":
            model = src.densenet_custom_cifar()
        case "depth_densenet":
            model = src.depth_densenet_cifar()
        case _:
            raise ValueError("Model not supported")

    accuracy = compute_accuracy(
        model, args.state_dict_path, args.dataset_path, args.device
    )
    print(accuracy)


if __name__ == "__main__":
    main()
