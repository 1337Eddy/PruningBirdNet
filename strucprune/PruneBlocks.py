from collections import OrderedDict
import numpy as np
import torch
import model 
from torch import nn

def prune_blocks(model_state_dict, filters, ratio):
    remove_index = []
    remove_list = []
    softmax = nn.Softmax(dim=0)
    #Create list of scaling factors to remove the least important
    scaling_factors = []
    for i in range(1, len(filters) - 1):
        for j in range (1, len(filters[i])):
            name = f"module.classifier.{i}.classifier.{j}."
            W = model_state_dict[name + "W"]
            W = softmax(W)
            scaling_factors.append(W)

    scaling_factors = sorted(scaling_factors, key=lambda x: x[1], reverse=True)

    if ratio > len(scaling_factors) or ratio < 0:
        raise RuntimeError(f'{ratio} is no valid argument. It has to be an Integer between 0 and {len(scaling_factors)-1}')

    
    if ratio >= len(scaling_factors) - 1:
        threshold = 1.0
    elif ratio <= 0:
        threshold = 0.0
    else: 
        threshold = scaling_factors[ratio][0]

    #Iterate over filters to build names of saved layers and find layers to drop
    for i in range(1, len(filters) - 1):
        for j in range (1, len(filters[i])):
            name = f"module.classifier.{i}.classifier.{j}."
            W = softmax(model_state_dict[name + "W"])
            #If the condition to the custom weights is True drop the whole layer
            if (W[0] < threshold):
                layers = list(model_state_dict.keys())
                remove_index.insert(0, (i,j)) 
                for layer in layers:
                    if layer.startswith(name):
                        remove_list.append(layer)

    #Remove elements from channel structure 
    for i, j in remove_index:
        del filters[i][j]

    #Remove weights from model
    for elem in remove_list:
        model_state_dict.pop(elem)
    
    return model_state_dict, filters

def fix_dim_problems_block_pruning(new_state_dict, state_dict):
    for key, value in new_state_dict.items():
        shape = np.shape(state_dict[key])
        if value.dim() > 1:         
            new_state_dict[key] = value[:shape[0],:shape[1],:shape[2],:shape[3]]
        elif shape != torch.Size([]): 
            new_state_dict[key] = value[:shape[0]]
    return new_state_dict

def rename_parameter_keys(new_state_dict, state_dict):
    model_state_dict = OrderedDict()
    for (_, value), (key, _) in zip(new_state_dict.items(), state_dict.items()):
        model_state_dict[key] = value

    return model_state_dict

def prune(model_state_dict, filters, ratio):
    #print("prune blocks")
    model_state_dict, filters = prune_blocks(model_state_dict, filters, ratio)
    #Build new pruned model
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()

    #Prepare weights after pruning for new model
    model_state_dict = rename_parameter_keys(model_state_dict, birdnet.state_dict())
    model_state_dict = fix_dim_problems_block_pruning(model_state_dict, birdnet.state_dict())

    return model_state_dict, filters