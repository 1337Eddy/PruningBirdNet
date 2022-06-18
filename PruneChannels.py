from collections import OrderedDict
import model
import re

import numpy as np
import torch
from torch import nn
from prune import Channel_Pruning_Mode


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


def get_mask_to_key(key):
    key_names = list(module_mask_list2)
    pattern_ds_block = "module\.classifier\.[0-9]+\.classifier\.[0-9]\.(classifierPath|skipPath)"
    pattern_resblock = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[0-9]+\."
    pattern_resstack_appendix = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\."
    resstack_index = 18
    resblock_index = 31
    resblock_layer_index = 44

    ds_block = re.search(pattern_ds_block, key)
    resblock = re.search(pattern_resblock, key)
    resstack_appendix = re.search(pattern_resstack_appendix, key)

    for i, name in zip(range(0, len(key_names)), key_names):
        if ds_block:
            if key[resstack_index] == name[resstack_index]:
                index_key = key_names[i-1]
                mask = module_mask_list2[index_key][1]
                return mask
            else:
                continue
        elif resblock:
            if key[resstack_index] == name[resstack_index] and key[resblock_index] == name[resblock_index]:
                resblock_layer_number = int(key[resblock_layer_index])
                if resblock_layer_number < 4:
                    index_key = key_names[i-1]
                    mask = module_mask_list2[index_key][1]
                else: 
                    mask = module_mask_list2[name][0]
                return mask
            else: 
                continue
        elif resstack_appendix:
            if int(key[resstack_index]) + 1 == int(name[resstack_index]):
                index_key = key_names[i-1]
                mask = module_mask_list2[index_key][1]
                return mask
            else:
                continue
    index_key = key_names[-1]

    mask = module_mask_list2[index_key][1]
    return mask 

def fix_dim_problems(new_state_dict, state_dict):
    prefix = "module.classifier.1.classifier.2"
    for key, value in new_state_dict.items():
        shape = np.shape(state_dict[key])

        if value.dim() > 1:   
            if np.shape(new_state_dict[key]) == np.shape(state_dict[key]):
                new_state_dict[key] = value 
            else:   
                mask = get_mask_to_key(key)
                new_state_dict[key] = apply_mask_to_tensor(mask, shape[1], value)

        elif shape != torch.Size([]): 
            if np.shape(new_state_dict[key]) == np.shape(state_dict[key]):
                new_state_dict[key] = value 
            else:
                mask = get_mask_to_key(key)
                new_state_dict[key] = torch.masked_select(value, mask)

        assert np.shape(new_state_dict[key]) == shape
    return new_state_dict


def isZero(values):
    for item in values:
        if item != 0:
            return False 
    return True

def create_mask(weight1, weight2, channel_ratio, mode=Channel_Pruning_Mode.MIN):
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

    if mode == Channel_Pruning_Mode.EVENLY:
        ow1 = ordered_weights[:len(weight1)]
        ow2 = ordered_weights[len(weight1):]
        ow1 = sorted(ow1, key = lambda x: abs(x[0]))
        ow2 = sorted(ow2, key = lambda x: abs(x[0]))
        for i in range(0, int(len(ow1) * channel_ratio)):
            ow1[i][0] = 0
        for i in range(0, int(len(ow2) * channel_ratio)):
            ow2[i][0] = 0
        ordered_weights = ow1 + ow2
    elif mode == Channel_Pruning_Mode.NO_PADD:
        ow1 = ordered_weights[:len(weight1)]
        ow2 = ordered_weights[len(weight1):]
        ow1 = sorted(ow1, key = lambda x: abs(x[0]))
        ow2 = sorted(ow2, key = lambda x: abs(x[0]))
        for i in range(0, int(len(ow1) * channel_ratio)):
            ow1[i][0] = 0
        ordered_weights = ow1 + ow2
    else: 
        ordered_weights = sorted(ordered_weights, key = lambda x: abs(x[0]))   
        for i in range(0, int(len(ordered_weights) * channel_ratio)):
            ordered_weights[i][0] = 0
    
    ordered_weights = sorted(ordered_weights, key = lambda x: x[1])
    ordered_weights1 = list(filter(lambda x: x[2] == 1, ordered_weights))
    ordered_weights2 = list(filter(lambda x: x[2] == 2, ordered_weights))

    if isZero([item[0] for item in ordered_weights1]):
        index = np.argmax(weight1)
        ordered_weights1[index][0] = weight1[index]
    
    if isZero([item[0] for item in ordered_weights2]):
        index = np.argmax(weight2)
        ordered_weights2[index][0] = weight2[index]

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


def create_new_channel(conv_bn_pair, channel_ratio, mode, module_name):
    bn_weight1 = conv_bn_pair[2]
    bn_weight2 = conv_bn_pair[9]

    mask1, mask2 = create_mask(bn_weight1[1], bn_weight2[1], channel_ratio, mode)
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

def prune_channels(model_state_dict, ratio, filters, mode, channel_ratio):
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
                (conv_bn_pair1, new_size1), (conv_bn_pair2, new_size2) = create_new_channel(conv_bn_pair, channel_ratio, mode, number_list[0][:-13])

                for name, value in conv_bn_pair1:
                    model_state_dict[name] = value
                
                for name, value in conv_bn_pair2:
                    model_state_dict[name] = value

                filters = update_filter_size(filters, new_size1, filter_counter)
                filters = update_filter_size(filters, new_size2, filter_counter + 1)
                conv_bn_pair = []
                filter_counter += 2
    return model_state_dict, filters


def prune(model_state_dict, ratio, filters, mode, channel_ratio):
    print("prune channels")
    model_state_dict, filters = prune_channels(model_state_dict, ratio, filters, mode, channel_ratio)
    #Build new pruned model
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()

    model_state_dict = fix_dim_problems(model_state_dict, birdnet.state_dict())

    return model_state_dict, filters