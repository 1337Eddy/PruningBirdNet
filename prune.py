import collections
from hmac import new
from re import L
from typing import OrderedDict
import torch
import torch.optim as optim
from torch import device, nn, softmax, threshold
from analyze_birdnet import AnalyzeBirdnet
import model 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import CallsDataset
from data import labels_to_one_hot_encondings as map_labels
from data import id_to_label
import numpy as np
from torchsummary import summary
from pathlib import Path
from metrics import AverageMeter
from metrics import accuracy
import monitor
import argparse
from utils import audio
import re


filters = [[[32]], 
[[16, 16, 32], [64, 64], [64, 64], [64, 64]], 
[[32, 32, 64], [128, 128], [128, 128], [128, 128]], 
[[64, 64, 128], [256, 256], [256, 256], [256, 256]], 
[[128, 128, 256], [512, 512], [512, 512], [512, 512]],
[512, 512, 83]]
    
# def has_successor(model_state_dict, elem):
#     prefix = elem[:35]
#     number = int(elem[35])
#     suffix = elem [36:]

#     for i in range(1, 10):
#         name = prefix + str(number + i) + suffix 
#         if name in model_state_dict:
#             return True, prefix + str(number + 1) + suffix
#     return False, ""


def rename_parameter_keys(model_state_dict):
    """
    Pytorch stores the parameters in a OrderdDict with specific key names
    After removing parameters from the model, the names have to be rearranged because the continous
    numbering of pytorch is broken otherwise
    """
    key_list = []
    for key, item in model_state_dict.items():
        key_list.append((key, item))


    new_model_state_dict = OrderedDict()
    counter = 0
    last = None
    stack_counter = 1
    for i in range(0, len(key_list)):
        pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\."
        key = key_list[i][0]       
        number_list = re.search(pattern, key)

        if number_list:
            incoming = int(key[31])
            current_stack = int(key[18])
            if current_stack > stack_counter:
                stack_counter = current_stack
                counter = 0
                last = None 
  
            if incoming == counter:
                new_model_state_dict[key] = model_state_dict[key]
            elif incoming - 1 == counter and last == None:
                new_model_state_dict[key] = model_state_dict[key]
                counter += 1
            elif incoming == 0:
                new_model_state_dict[key] = model_state_dict[key]
                last == None
                counter = 0
            elif last == None:
                counter += 1
                last = incoming
                new_key_name = key[:31] + str(counter) + key[32:]
                new_model_state_dict[new_key_name] = model_state_dict[key]
            elif last < incoming:
                counter +=1
                last = incoming
                new_key_name = key[:31] + str(counter) + key[32:]
                new_model_state_dict[new_key_name] = model_state_dict[key]
            elif last == incoming:
                new_key_name = key[:31] + str(counter) + key[32:]
                new_model_state_dict[new_key_name] = model_state_dict[key]
        else:
            new_model_state_dict[key] = model_state_dict[key]
    return new_model_state_dict

def fix_dimension_problems(model_state_dict):
    """
    After pruning residual blocks, it is possible that two layers with incorparable input and output layers have to work together
    this methods fixes these problems by cutting the weights of the actual layer if the previous has to less channels
    """

    last_dimension = None
    last_layer = None 

    for key, value in model_state_dict.items():
        if last_dimension == None:
            last_dimension = value.size()
            last_layer = key.split('.')[-1]
        else:
            current_dimension = value.size()
            current_layer = key.split('.')[-1]

            if value.dim() > 1:
                input_last = last_dimension[0]
                input_current = current_dimension[1]
                ouput = current_dimension[0]
                if input_last < input_current:
                    model_state_dict[key] = model_state_dict[key][:,:input_last,:,:]
            elif current_layer != "W" and current_layer != "num_batches_tracked":
                if last_dimension[0] < current_dimension[0]:
                    model_state_dict[key] = value[:last_dimension[0]]
            else:
                continue
            last_dimension = model_state_dict[key].size()
            last_layer = key.split('.')[-1]
            #print(key + ": " + str(np.shape(model_state_dict[key])))

    return model_state_dict    

def test_fix_dim_problems(new_state_dict, state_dict):
    for key, value in new_state_dict.items():
        shape = np.shape(state_dict[key])
        if value.dim() > 1:         
            new_state_dict[key] = value[:shape[0],:shape[1],:shape[2],:shape[3]]
        elif shape != torch.Size([]): 
            new_state_dict[key] = value[:shape[0]]
    return new_state_dict


def create_mask(weights, ratio):
    weights = np.array(weights.cpu())
    ordered_weights = []
    for i in range(0, len(weights)):
        ordered_weights.append([weights[i], i])
    #print(ordered_weights)
    ordered_weights = sorted(ordered_weights, key = lambda x: abs(x[0]))
    for i in range(0, int(len(ordered_weights) * ratio)):
        ordered_weights[i][0] = 0
    ordered_weights = sorted(ordered_weights, key = lambda x: x[1])
    mask = []
    for value, index in ordered_weights:
        if value != 0:
            mask.append(True)
        else:
            mask.append(False)
    return torch.tensor(mask)

def create_new_channel(conv_bn_pair, ratio):
    conv_weight, conv_bias = conv_bn_pair[0], conv_bn_pair[1]
    bn_weight, bn_bias, bn_mean, bn_var, bn_tracked = conv_bn_pair[2], conv_bn_pair[3], conv_bn_pair[4], conv_bn_pair[5], conv_bn_pair[6]

    mask = create_mask(bn_weight[1], ratio).cuda()
    conv_bias = (conv_bias[0], torch.masked_select(conv_bias[1], mask))
    bn_weight = (bn_weight[0], torch.masked_select(bn_weight[1], mask))
    bn_bias = (bn_bias[0], torch.masked_select(bn_bias[1], mask))
    bn_mean = (bn_mean[0], torch.masked_select(bn_mean[1], mask))
    bn_var = (bn_var[0], torch.masked_select(bn_var[1], mask))

    new_size = np.shape(bn_var[1])[0]
    i = 0
    buffer = torch.zeros([new_size, np.shape(conv_weight[1])[1], np.shape(conv_weight[1])[2], np.shape(conv_weight[1])[3]])
    for bool, net in zip(mask, conv_weight[1]):
        if bool:
            buffer[i] = net 
            i += 1
    conv_weight = (conv_weight[0], buffer)

    conv_bn_pair = [conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var]

    return conv_bn_pair, new_size


def prune_channels(model_state_dict, ratio, filters):
    conv_bn_pair = []
    filter_counter = 0
    for key, value in model_state_dict.items():
        pattern_scaling_factor = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.W"
        pattern_modules = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[2-9]"
        softmax = nn.Softmax(dim=0)

        scaling_factor = re.search(pattern_scaling_factor, key)
        number_list = re.search(pattern_modules, key)

        if scaling_factor:
            block_ratio = 1 - softmax(value)[0]
        
        if number_list:
            conv_bn_pair.append((key, value))
            if len(conv_bn_pair) == 7:
                conv_bn_pair, new_size = create_new_channel(conv_bn_pair, 0.4)

                for name, value in conv_bn_pair:
                    model_state_dict[name] = value

                counter = 0
                for i in range (1, len(filters)- 1):
                    for j in range(1, len(filters[i])):
                        for k in range(0, len(filters[i][j])):
                            if counter == filter_counter:
                                filters[i][j][k] = new_size
                                break
                            counter += 1
                conv_bn_pair = []
                filter_counter += 1
    return model_state_dict, filters

def prune_blocks(model_state_dict, filters, ratio):
    remove_index = []
    remove_list = []

    #Create list of scaling factors to remove the least important
    scaling_factors = []
    for i in range(1, len(filters) - 1):
        for j in range (1, len(filters[i])):
            name = f"module.classifier.{i}.classifier.{j}."
            scaling_factors.append((model_state_dict[name + "W"]))

    scaling_factors = sorted(scaling_factors, key=lambda x: x[1])

    threshold = scaling_factors[int((len(scaling_factors)-1) * ratio)][1]

    #Iterate over filters to build names of saved layers and find layers to drop
    for i in range(1, len(filters) - 1):
        for j in range (1, len(filters[i])):
            name = f"module.classifier.{i}.classifier.{j}."

            #If the condition to the custom weights is True drop the whole layer
            if (abs(model_state_dict[name + "W"][1]) < threshold):
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


def prune(load_path, ratio=0.2, lr=0.001, save_path=""):
    checkpoint = torch.load(load_path)
    model_state_dict = checkpoint['model_state_dict']

    filters = checkpoint['filters']
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
    birdnet.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    model_state_dict, filters = prune_blocks(model_state_dict, filters, ratio)
    
    model_state_dict, filters = prune_channels(model_state_dict, ratio, filters)


    #Build new pruned model
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 

    #Prepare weights after pruning for new model
    model_state_dict = rename_parameter_keys(model_state_dict)
    model_state_dict = test_fix_dim_problems(model_state_dict, birdnet.state_dict())

    #Load parameter to model
    birdnet.load_state_dict(model_state_dict)
    

    # for key, value in model_state_dict.items():
    #     print(key + ": " + str(np.shape(value)))
    print(birdnet)
    if save_path:
        retrain(birdnet, criterion, save_path, lr)


def retrain(birdnet, criterion, save_path, lr=0.001):
    analyze = AnalyzeBirdnet(birdnet)
    dataset = CallsDataset()
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=16, shuffle=True)
    #Start Training
    analyze = AnalyzeBirdnet(birdnet=birdnet, lr=lr, criterion=criterion, train_loader=train_loader, 
                                test_loader=test_loader, save_path=save_path, gamma=0.2)
    analyze.start_training(3)

if __name__ == '__main__':
    #prune(ratio=0.7, finetune=False)
    prune("models/crap/birdnet_final.pt", ratio=0.5, save_path="models/crap_split_dataset_pruned/") #
    #prune(ratio=0.9, finetune=False)
