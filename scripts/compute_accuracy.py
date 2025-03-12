import argparse
import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy 

import src

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Compute accuracy")
    parser.add_argument("--model", type=str, default="densenet", help="Model to use")
    parser.add_argument("--state_dict_path", type=str, help="Path to the state dict")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--quantization", type=int, default=32, help="Quantization rate")
    return parser

@torch.no_grad()
def compute_accuracy(model, state_dict_path, dataset_path, device, quantization):
    state_dict = torch.load(state_dict_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace("module.", "")] = value
    model.load_state_dict(new_state_dict)
    model.to(device)

    dataset = src.datasets.CIFAR10(dataset_path, quantization)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    example_inputs = next(iter(dataloader))[0].to(device)
    quantized_model = copy.deepcopy(model)
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    quantized_model.eval()
    # Prepare model for quantization
    model_prepared = quantize_fx.prepare_fx(quantized_model, qconfig_mapping, example_inputs)

    # **Calibration Step: Run a few batches to collect statistics**
    for data in dataloader:
        images, _ = data  # We only need inputs, not labels
        images = images.to(device)
        model_prepared(images)  # This step collects statistics for quantization

    # Convert to quantized model
    model_quantized = quantize_fx.convert_fx(model_prepared)

    model_quantized.eval()
    correct = 0
    total = 0
    for data in dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model_quantized(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total