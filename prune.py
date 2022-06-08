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


# filters = [[[32]], 
# [[16, 16, 32], [64, 64], [64, 64], [64, 64]], 
# [[32, 32, 64], [128, 128], [128, 128], [128, 128]], 
# [[64, 64, 128], [256, 256], [256, 256], [256, 256]], 
# [[128, 128, 256], [512, 512], [512, 512], [512, 512]],
# [512, 512, 83]]


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


def fix_dim_problems(new_state_dict, state_dict):
    for key, value in new_state_dict.items():
        shape = np.shape(state_dict[key])
        if value.dim() > 1:         
            new_state_dict[key] = value[:shape[0],:shape[1],:shape[2],:shape[3]]
        elif shape != torch.Size([]): 
            new_state_dict[key] = value[:shape[0]]
    return new_state_dict


def create_mask(weight1, weight2, channel_ratio, evenly = False):
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

def create_new_channel(conv_bn_pair, channel_ratio, evenly):
    conv_weight1, conv_bias1 = conv_bn_pair[0], conv_bn_pair[1]
    bn_weight1, bn_bias1, bn_mean1, bn_var1, _ = conv_bn_pair[2], conv_bn_pair[3], conv_bn_pair[4], conv_bn_pair[5], conv_bn_pair[6]
    conv_weight2, conv_bias2 = conv_bn_pair[7], conv_bn_pair[8]
    bn_weight2, bn_bias2, bn_mean2, bn_var2, _ = conv_bn_pair[9], conv_bn_pair[10], conv_bn_pair[11], conv_bn_pair[12], conv_bn_pair[13]

    mask1, mask2 = create_mask(bn_weight1[1], bn_weight2[1], channel_ratio, evenly)
    mask1 = mask1.cuda()
    mask2 = mask2.cuda()
    conv_bias1 = (conv_bias1[0], torch.masked_select(conv_bias1[1], mask1))
    bn_weight1 = (bn_weight1[0], torch.masked_select(bn_weight1[1], mask1))
    bn_bias1 = (bn_bias1[0], torch.masked_select(bn_bias1[1], mask1))
    bn_mean1 = (bn_mean1[0], torch.masked_select(bn_mean1[1], mask1))
    bn_var1 = (bn_var1[0], torch.masked_select(bn_var1[1], mask1))

    conv_bias2 = (conv_bias2[0], torch.masked_select(conv_bias2[1], mask2))
    bn_weight2 = (bn_weight2[0], torch.masked_select(bn_weight2[1], mask2))
    bn_bias2 = (bn_bias2[0], torch.masked_select(bn_bias2[1], mask2))
    bn_mean2 = (bn_mean2[0], torch.masked_select(bn_mean2[1], mask2))
    bn_var2 = (bn_var2[0], torch.masked_select(bn_var2[1], mask2))

    new_size1 = np.shape(bn_var1[1])[0]
    i = 0
    buffer = torch.zeros([new_size1, np.shape(conv_weight1[1])[1], np.shape(conv_weight1[1])[2], np.shape(conv_weight1[1])[3]])
    for bool, net in zip(mask1, conv_weight1[1]):
        if bool:
            buffer[i] = net 
            i += 1
    conv_weight1 = (conv_weight1[0], buffer)
    conv_bn_pair1 = [conv_weight1, conv_bias1, bn_weight1, bn_bias1, bn_mean1, bn_var1]


    new_size2 = np.shape(bn_var2[1])[0]
    i = 0
    buffer = torch.zeros([new_size2, np.shape(conv_weight2[1])[1], np.shape(conv_weight2[1])[2], np.shape(conv_weight2[1])[3]])
    for bool, net in zip(mask2, conv_weight2[1]):
        if bool:
            buffer[i] = net 
            i += 1
    conv_weight2 = (conv_weight2[0], buffer)
    conv_bn_pair2 = [conv_weight2, conv_bias2, bn_weight2, bn_bias2, bn_mean2, bn_var2]

    return (conv_bn_pair1, new_size1), (conv_bn_pair2, new_size2)


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
                (conv_bn_pair1, new_size1), (conv_bn_pair2, new_size2) = create_new_channel(conv_bn_pair, block_ratio * channel_ratio, evenly)

                for name, value in conv_bn_pair1:
                    model_state_dict[name] = value
                
                for name, value in conv_bn_pair2:
                    model_state_dict[name] = value

                counter = 0
                for i in range (1, len(filters)- 1):
                    for j in range(1, len(filters[i])):
                        for k in range(0, len(filters[i][j])):
                            if counter == filter_counter:
                                filters[i][j][k] = new_size1
                                break
                            counter += 1

                counter = 0
                for i in range (1, len(filters)- 1):
                    for j in range(1, len(filters[i])):
                        for k in range(0, len(filters[i][j])):
                            if counter == filter_counter + 1:
                                filters[i][j][k] = new_size2
                                break
                            counter += 1
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


def prune(load_path, ratio, lr=0.001, save_path="", evenly=False, channel_ratio=0.9):
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
    model_state_dict = rename_parameter_keys(model_state_dict)
    model_state_dict = fix_dim_problems(model_state_dict, birdnet.state_dict())

    #Load parameter to model
    birdnet.load_state_dict(model_state_dict)
    
    #print(birdnet)
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
    analyze.start_training(30)

if __name__ == '__main__':
    prune("models/birdnet/birdnet_final.pt", ratio=0.6, save_path="models/pruned3/") 
