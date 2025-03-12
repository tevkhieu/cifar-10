import argparse

from torchinfo import summary;
import torch

import src

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="densenet", help="Model to use")
    parser.add_argument("--pruning", type=float, default=0.0, help="Pruning rate")
    parser.add_argument("--quantization", type=float, nargs='+', default=[16, 16], help="Quantization rate")
    return parser

def compute_score(model, quantization, pruning):
    paramRef = 5.6*(10**6)
    opsRef = 2.8*(10**8)
    nonzero = torch.count_nonzero(torch.cat([param.view(-1) for param in model.parameters()]))
    modelSummary = summary(model, (1,3,32,32))
    score_param = quantization[0]*nonzero/32/paramRef
    score_ops = (1-pruning)*max(quantization)*modelSummary.total_mult_adds/32/opsRef
    print(score_param, score_ops)
    return score_param + score_ops

def main():
    args = create_arg_parser().parse_args()
    match args.model:
        case "densenet":
            model = src.densenet_cifar()
        case "custom_densenet":
            model = src.densenet_custom_cifar()
        case _:
            raise ValueError("Model not supported")
    score = compute_score(model, args.quantization, args.pruning)
    print(score)

if __name__ == "__main__":
    main()