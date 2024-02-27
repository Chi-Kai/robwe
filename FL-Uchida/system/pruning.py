# parameter pruning: 25% to 85% of model weights with lowest absolute values are set to zero
# see table 1 in adversarial frontier stitching
# pruning rate vs extraction rate vs accuracy after(gotta be plausible)

# Methodology:
# # take pretrained model
# # watermark the model with certain parameters (epsilon, size of the key set)
# # prune by 0.25 rate (additionally 0.50, 0.75, 0.85)
# # extraction rate (check how many watermarks are verified as such)
# # accuracy after pruning (fidelity)

import numpy as np
import torch
import torch.nn.utils.prune
import logging

def get_params_to_prune(arch, net):
    """
    get parameters which are going to be pruned. Maybe there would have been a better way to do this, but I could not find one
    """
    if arch == "cnncifar10":
        return [
            (net.head[0], 'weight'),
            (net.head[2], 'weight'),
            (net.head[4], 'weight')
        ]

def get_modules(arch, net):
    if arch == "cnncifar10":
        return [net.head[0],
                net.head[2],
                net.head[4]
                ]

def prune_attack(net, arch, pruning_rate):
    """
    Run Pruning Attack on model.
    """
    logging.info('Set parameters to prune')
    parameters_to_prune = get_params_to_prune(arch, net)

    logging.info('Prune...')
    torch.nn.utils.prune.global_unstructured(parameters_to_prune, pruning_method=torch.nn.utils.prune.L1Unstructured,
                                             amount=pruning_rate)

    for module in get_modules(arch, net):
        torch.nn.utils.prune.remove(module, "weight")

class GlobalRandomPruningMethod(torch.nn.utils.prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        self.amount = amount

    def compute_mask(self, tensor, default_mask):
        mask = torch.ones_like(tensor)

        num_prune = int(self.amount * tensor.numel())

        mask.view(-1)[torch.randperm(tensor.numel())[:num_prune]] = 0

        return mask

def prune_model(net, arch, pruning_rate):
    """
    Run Pruning Attack on model.
    """
    logging.info('Set parameters to prune')
    parameters_to_prune = get_params_to_prune(arch, net)

    logging.info('Prune...')
    torch.nn.utils.prune.global_unstructured(parameters_to_prune, pruning_method=GlobalRandomPruningMethod,
                                             amount=pruning_rate)

    # Apply the temporary masks on the model weights
    for module in get_modules(arch, net):
        torch.nn.utils.prune.remove(module, "weight")

def pruning_resnet(model, pruning_perc):
    if pruning_perc == 0:
        return

    allweights = []
    for p in model.parameters():
        allweights += p.data.cpu().abs().numpy().flatten().tolist()

    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)
    for p in model.parameters():
        mask = p.abs() > threshold
        p.data.mul_(mask.float())
# 对于某个层进行剪枝
def pruning_resnet_layer(model, pruning_perc,layers):

    if pruning_perc == 0:
        return
    allweights = []
    # 如果在指定的层中，则进行剪枝
    for name, p in model.named_parameters():
        if name in layers:
            allweights += p.data.cpu().abs().numpy().flatten().tolist()
    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)
    for name, p in model.named_parameters():
        if name in layers:
            mask = p.abs() > threshold
            p.data.mul_(mask.float())

def pruning_linear_layers(model, pruning_perc):
    if pruning_perc == 0:
        return
    all_weights = []
    layers_to_prune = [0,2,4]  # 线性层的索引为0-4
    for name, p in model.named_parameters():
        if name.split('.')[1] in layers_to_prune:
            all_weights += p.data.cpu().abs().numpy().flatten().tolist()
    all_weights = np.array(all_weights)
    threshold = np.percentile(all_weights, pruning_perc)
    for name, p in model.named_parameters():
        if name.split('.')[1] in layers_to_prune:
            mask = p.abs() > threshold
            p.data.mul_(mask.float())