from torchinfo import summary;
import torch
import models
import models.resnet
paramRef = 5.6*(10**6)
opsRef = 2.8*(10**8)

#Compute the score of the model used according to its quantization (weights, activations) and pruning (structured, unstructured)
def computeScore(model, batch_size , quantization, pruning):
    nonzero = torch.count_nonzero(torch.cat([param.view(-1) for param in model.parameters()]))
    modelSummary = summary(model, (batch_size,3,32,32))
    score_param = quantization[0]*nonzero/32/paramRef
    score_ops = (1-pruning)*max(quantization)*modelSummary.total_mult_adds/32/opsRef
    print(score_param, score_ops)
    return score_param + score_ops

#base score with binary conect is 1.

state_dict = torch.load(r"EFFDL\lab\checkpoints\iterative_structured_pruning.pt")

# new_state_dict = {}
# for key, value in state_dict["net"].items():
#     new_state_dict[key[7:]] = value

model = models.densenet_cifar()

model.load_state_dict(state_dict)

print(computeScore(model, 1, (16,16),((1-0.9**7),0)))