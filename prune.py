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

module_mask_list2 = {}

module_mask_list = []

def apply_mask_to_tensor(mask, new_size, tensor):  
    buffer = torch.zeros([np.shape(tensor)[0], new_size, np.shape(tensor)[2], np.shape(tensor)[3]])
    for j in range(0, np.shape(buffer)[0]):
        i = 0
        for bool, net in zip(mask, tensor[j]):
            if bool:
                buffer[j][i] = net 
                i += 1
    return buffer

def rename_parameter_keys(new_state_dict, state_dict):
    model_state_dict = OrderedDict()
    for (_, value), (key, _) in zip(new_state_dict.items(), state_dict.items()):
        model_state_dict[key] = value

    return model_state_dict


def fix_dim_problems(new_state_dict, state_dict):
    prefix = "module.classifier.1.classifier.2"
    for key, value in new_state_dict.items():
        shape = np.shape(state_dict[key])
        if value.dim() > 1:   
            if np.shape(new_state_dict[key]) == np.shape(state_dict[key]):
                new_state_dict[key] = value 
            else:   
                name = key[:len("module.classifier.1.classifier.1")]
                #print(shape)
                #print(np.shape(value))
                if name in module_mask_list2:
                    number = int(key[-8])
                    if number == 6:
                        mask = module_mask_list2[name][1]
                    elif number == 2:
                        mask = module_mask_list2[name][0]
                    else:
                        print("ERROR")
                        exit()
                    new_state_dict[key] = apply_mask_to_tensor(mask, shape[1], value)
                else:             
                    new_state_dict[key] = apply_mask_to_tensor(module_mask_list[0][1], shape[1], value)

        elif shape != torch.Size([]): 
            if np.shape(new_state_dict[key]) == np.shape(state_dict[key]):
                new_state_dict[key] = value 
            else:
                if prefix in key or "module.classifier.5" in key:
                    new_state_dict[key] = torch.masked_select(value, module_mask_list[0][1])
                else: 
                    del module_mask_list[0]
                    new_state_dict[key] = torch.masked_select(value, module_mask_list[0][1])
                    prefix = key[:len(prefix)]
    return new_state_dict

def fix_dim_problems1(new_state_dict, state_dict):
    for key, value in new_state_dict.items():
        shape = np.shape(state_dict[key])
        if value.dim() > 1:         
            new_state_dict[key] = value[:shape[0],:shape[1],:shape[2],:shape[3]]
        elif shape != torch.Size([]): 
            new_state_dict[key] = value[:shape[0]]
    return new_state_dict

def create_mask(weight1, weight2, channel_ratio, evenly=False):
    """
    Creates a mask over two batchnorm layer for the values that are dropped and returns 
    the mask for both layers
    If evenly is false the channel_ratio percent lowest values from both layers together are pruned
    If it is true only the channel_ratio percent lowest values from each layer for themselve are pruned
    """
    weight1 = np.array(weight1.cpu())
    weight2 = np.array(weight2.cpu())

    ordered_weights = []
    for i in range(0, len(weight1)):
        ordered_weights.append([weight1[i], i, 1])
    for i in range(0, len(weight2)):
        ordered_weights.append([weight2[i], len(weight1) + i, 2])

    if evenly:
        ow1 = ordered_weights[:len(weight1)]
        ow2 = ordered_weights[len(weight1):]
        ow1 = sorted(ow1, key = lambda x: abs(x[0]))
        ow2 = sorted(ow2, key = lambda x: abs(x[0]))
        for i in range(0, int(len(ow1) * channel_ratio)):
            ow1[i][0] = 0
        for i in range(0, int(len(ow2) * channel_ratio)):
            ow2[i][0] = 0
        ordered_weights = ow1 + ow2
    else: 
        ordered_weights = sorted(ordered_weights, key = lambda x: abs(x[0]))   
        for i in range(0, int(len(ordered_weights) * channel_ratio)):
            ordered_weights[i][0] = 0
    
    ordered_weights = sorted(ordered_weights, key = lambda x: x[1])
    ordered_weights1 = list(filter(lambda x: x[2] == 1, ordered_weights))
    ordered_weights2 = list(filter(lambda x: x[2] == 2, ordered_weights))
    mask1 = []
    for value, _, _ in ordered_weights1:
        if value != 0:
            mask1.append(True)
        else:
            mask1.append(False)
    mask2 = []
    for value, _, _ in ordered_weights2:
        if value != 0:
            mask2.append(True)
        else:
            mask2.append(False)
    return torch.tensor(mask1), torch.tensor(mask2)

def apply_mask_to_conv_bn_block(mask, conv_bn_pair):
    conv_weight, conv_bias = conv_bn_pair[0], conv_bn_pair[1]
    bn_weight, bn_bias, bn_mean, bn_var, _ = conv_bn_pair[2], conv_bn_pair[3], conv_bn_pair[4], conv_bn_pair[5], conv_bn_pair[6]

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


def create_new_channel(conv_bn_pair, channel_ratio, evenly, module_name):
    bn_weight1 = conv_bn_pair[2]
    bn_weight2 = conv_bn_pair[9]

    mask1, mask2 = create_mask(bn_weight1[1], bn_weight2[1], channel_ratio, evenly)
    mask1 = mask1.cuda()
    mask2 = mask2.cuda()

    module_mask_list.append((module_name, mask2))
    module_mask_list2[module_name] = (mask1, mask2)

    conv_bn_pair1, new_size1 = apply_mask_to_conv_bn_block(mask1, conv_bn_pair[:7])
    conv_bn_pair2, new_size2 = apply_mask_to_conv_bn_block(mask2, conv_bn_pair[7:])

    return (conv_bn_pair1, new_size1), (conv_bn_pair2, new_size2)


def update_filter_size(filters, new_size, filter_counter):
    counter = 0
    for i in range (1, len(filters)- 1):
        for j in range(1, len(filters[i])):
            for k in range(0, len(filters[i][j])):
                if counter == filter_counter:
                    filters[i][j][k] = new_size
                    break
                counter += 1
    return filters

def prune_channels(model_state_dict, ratio, filters, evenly, channel_ratio):
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
            if len(conv_bn_pair) == 14:
                (conv_bn_pair1, new_size1), (conv_bn_pair2, new_size2) = create_new_channel(conv_bn_pair, channel_ratio, evenly, number_list[0][:-13])

                for name, value in conv_bn_pair1:
                    model_state_dict[name] = value
                
                for name, value in conv_bn_pair2:
                    model_state_dict[name] = value

                filters = update_filter_size(filters, new_size1, filter_counter)
                filters = update_filter_size(filters, new_size2, filter_counter + 1)
                conv_bn_pair = []
                filter_counter += 2
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

def retrain(birdnet, criterion, save_path, lr=0.001):
    train_dataset = CallsDataset("1dataset/1data/calls/train/")
    test_dataset = CallsDataset("1dataset/1data/calls/test/")
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=16, shuffle=True)

    #Start Training
    analyze = AnalyzeBirdnet(birdnet=birdnet, lr=lr, criterion=criterion, train_loader=train_loader, 
                                test_loader=test_loader, save_path=save_path, gamma=0.2)
    analyze.start_training(10)

def prune(load_path, ratio, lr=0.001, save_path="", evenly=False, channel_ratio=0.5):
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
    model_state_dict, filters = prune_channels(model_state_dict, ratio, filters, evenly, channel_ratio)


    #Build new pruned model
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 

    #Prepare weights after pruning for new model
    model_state_dict = rename_parameter_keys(model_state_dict, birdnet.state_dict())
    print("Start fix dim problems")
    model_state_dict = fix_dim_problems(model_state_dict, birdnet.state_dict())
    print("Stop fix dim problems")

    #Load parameter to model
    birdnet.load_state_dict(model_state_dict)
    #print(birdnet.state_dict()["module.classifier.1.classifier.4.weight"])

    if save_path:
        retrain(birdnet, criterion, save_path, lr)


if __name__ == '__main__':
    checkpoint = torch.load("models/birdnet_v1/birdnet_final.pt")
    model_state_dict = checkpoint['model_state_dict']

    #print(model_state_dict['module.classifier.1.classifier.4.weight'])
    prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.03, save_path="models/pruned/channels_01/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.05, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_05/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.1, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_10/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.15, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_15/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.2, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_20/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.25, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_25/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.3, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_30/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.4, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_40/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.5, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_50/") 

    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.05, save_path="models/pruned/channels_05/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.1, save_path="models/pruned/channels_10/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.15, save_path="models/pruned/channels_15/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.2, save_path="models/pruned/channels_20/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.3, save_path="models/pruned/channels_30/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.4, save_path="models/pruned/channels_40/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.5, save_path="models/pruned/channels_50/") 

    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.1, evenly=True, channel_ratio=0.1, save_path="models/pruned/block_10_channels_10/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.2, evenly=True, channel_ratio=0.2, save_path="models/pruned/block_20_channels_20/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.3, evenly=True, channel_ratio=0.3, save_path="models/pruned/block_30_channels_30/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.4, evenly=True, channel_ratio=0.4, save_path="models/pruned/block_40_channels_40/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.5, evenly=True, channel_ratio=0.5, save_path="models/pruned/block_50_channels_50/") 

    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.5, evenly=True, channel_ratio=0.1, save_path="models/pruned/block_50_channels_10/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.4, evenly=True, channel_ratio=0.2, save_path="models/pruned/block_40_channels_20/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.3, evenly=True, channel_ratio=0.3, save_path="models/pruned/block_30_channels_30/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.2, evenly=True, channel_ratio=0.4, save_path="models/pruned/block_20_channels_40/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.1, evenly=True, channel_ratio=0.5, save_path="models/pruned/block_10_channels_50/") 
