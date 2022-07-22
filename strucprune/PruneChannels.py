from collections import OrderedDict
from math import tanh
import model
import re
import numpy as np
import torch
from torch import nn, tensor
import torch.nn.functional as F
from strucprune.MaskSelection import SelectMask
from strucprune.MaskSelectionCURL import SelectMaskCURL
from strucprune.MaskSelectionEVENLY import SelectMaskEvenly
from strucprune.MaskSelectionMIN import SelectMaskMin
from strucprune.MaskSelectionNOPADD import SelectMaskNoPadd 


module_mask_list = {}


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
    key_names = list(module_mask_list)
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
            if key[resstack_index] <= name[resstack_index]:
                index_key = key_names[i-1]
                mask = module_mask_list[index_key][1]
                return mask
            else: 
                continue
        elif resblock:
            if key[resstack_index] == name[resstack_index] and key[resblock_index] == name[resblock_index]:
                resblock_layer_number = int(key[resblock_layer_index])
                if resblock_layer_number < 4:
                    index_key = key_names[i-1]
                    mask = module_mask_list[index_key][1]
                else: 
                    mask = module_mask_list[name][0]
                return mask
            else: 
                continue
        elif resstack_appendix:
            if int(key[resstack_index]) + 1 <= int(name[resstack_index]):
                index_key = key_names[i-1]
                mask = module_mask_list[index_key][1]
                return mask
            else: 
                continue
    index_key = key_names[-1]

    mask = module_mask_list[index_key][1]
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



def create_mask(weight1, weight2, channel_ratio, mode):
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

    if mode.value == 0: #EVENLY
        ow1 = ordered_weights[:len(weight1)]
        ow2 = ordered_weights[len(weight1):]
        ow1 = sorted(ow1, key = lambda x: abs(x[0]))
        ow2 = sorted(ow2, key = lambda x: abs(x[0]))
        for i in range(0, int(len(ow1) * channel_ratio)):
            ow1[i][0] = 0
        for i in range(0, int(len(ow2) * channel_ratio)):
            ow2[i][0] = 0
        ordered_weights = ow1 + ow2
    elif mode.value == 1: #NO_PADD
        ow1 = ordered_weights[:len(weight1)]
        ow2 = ordered_weights[len(weight1):]
        ow1 = sorted(ow1, key = lambda x: abs(x[0]))
        ow2 = sorted(ow2, key = lambda x: abs(x[0]))
        for i in range(0, int(len(ow1) * channel_ratio)):
            ow1[i][0] = 0
        ordered_weights = ow1 + ow2
    elif mode.value == 2: #MIN
        ordered_weights = sorted(ordered_weights, key = lambda x: abs(x[0]))   
        for i in range(0, int(len(ordered_weights) * channel_ratio) - 1):
            ordered_weights[i][0] = 0
    else: 
        raise RuntimeError(f'mode {mode} doesnt exist')
    
    ordered_weights = sorted(ordered_weights, key = lambda x: x[1])
    ordered_weights1 = list(filter(lambda x: x[2] == 1, ordered_weights))
    ordered_weights2 = list(filter(lambda x: x[2] == 2, ordered_weights))

    mask1 = []
    for value, _, _ in ordered_weights1:
        isNotZero = value != 0
        mask1.append(isNotZero)

    mask2 = []
    for value, _, _ in ordered_weights2:
        isNotZero = value != 0
        mask2.append(isNotZero)

    mask1, mask2 = torch.tensor(mask1), torch.tensor(mask2)

    #Prevent that all values of mask are False. This is important because otherwise the layer has no channels
    if torch.all(~mask1):
        mask1[np.argmax(weight1)] = True 
    if torch.all(~mask2):
        mask2[np.argmax(weight2)] = True 
    
    return mask1, mask2

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


def create_new_resblock(conv_bn_pair, channel_ratio, mode, module_name):
    bn_weight1 = conv_bn_pair[2]
    bn_weight2 = conv_bn_pair[9]

    mask1, mask2 = create_mask(bn_weight1[1], bn_weight2[1], channel_ratio, mode)
    mask1 = mask1.cuda()
    mask2 = mask2.cuda()

    module_mask_list[module_name] = (mask1, mask2)

    conv_bn_pair1, new_size1 = apply_mask_to_conv_bn_block(mask1, conv_bn_pair[:7])
    conv_bn_pair2, new_size2 = apply_mask_to_conv_bn_block(mask2, conv_bn_pair[7:])

    return (conv_bn_pair1, new_size1), (conv_bn_pair2, new_size2)


def update_filter_size(filters, new_size, filter_counter):
    counter = 0
    for stack in range (1, len(filters)- 1):
        for block in range(1, len(filters[stack])):
            for conv_layer in range(0, len(filters[stack][block])):
                if counter == filter_counter:
                    filters[stack][block][conv_layer] = new_size
                    break
                counter += 1
    return filters

def calc_mean_of_block_ratios(model_state_dict, pattern_scaling_factor, softmax):
    block_ratios = np.array([])
    for key, value in model_state_dict.items():
        scaling_factor = re.search(pattern_scaling_factor, key)
        if scaling_factor:
            value = softmax(value)[0]
            value = np.array(value.cpu())
            block_ratios = np.append(block_ratios, value)
    mean = np.mean(block_ratios)

    return torch.tensor([mean]).cuda()

def get_block_ratios(model_state_dict, pattern_scaling_factor):
    block_ratios = torch.tensor([]).cuda()
    softmax = nn.Softmax(dim=0)
    for key, value in model_state_dict.items():
        scaling_factor = re.search(pattern_scaling_factor, key)
        if scaling_factor:
            print(softmax(value))
            value = softmax(value)[0]
            value = value.unsqueeze(dim=0)
            block_ratios = torch.cat((block_ratios, value), 0)
    return block_ratios

def prune_channels(model_state_dict, filters, mode, channel_ratio, block_temperature):
    conv_bn_pair = []
    filter_counter = 0
    pattern_scaling_factor = "module\.classifier\.[0-9]+\.classifier\.[1-9]+\.W"
    pattern_modules = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[2-9]"
    softmax = nn.Softmax(dim=0)
    block_ratio_mean = calc_mean_of_block_ratios(model_state_dict, pattern_scaling_factor, softmax)
    # block_ratios = get_block_ratios(model_state_dict=model_state_dict, pattern_scaling_factor=pattern_scaling_factor)
    # block_ratios *= len(block_ratios)
    # print(block_ratios)
    # print(softmax(block_ratios)*len(block_ratios))
    # exit()

    for key, value in model_state_dict.items():
        scaling_factor = re.search(pattern_scaling_factor, key)
        number_list = re.search(pattern_modules, key)

        if scaling_factor:
            block_ratio = 1 - softmax(value)[0]
        
        if number_list:
            conv_bn_pair.append((key, value))   #Collect convolutional and batchnorm layer of a Residual Block
            if len(conv_bn_pair) == 14:
                if block_temperature:
                    channel_ratio_with_block_temperature = tanh(channel_ratio * (block_ratio_mean+block_ratio))
                    (conv_bn_pair1, new_size1), (conv_bn_pair2, new_size2) = create_new_resblock(conv_bn_pair, channel_ratio_with_block_temperature, mode, number_list[0][:-13])
                else: 
                    (conv_bn_pair1, new_size1), (conv_bn_pair2, new_size2) = create_new_resblock(conv_bn_pair, channel_ratio, mode, number_list[0][:-13])

                for name, value in conv_bn_pair1:
                    model_state_dict[name] = value
                
                for name, value in conv_bn_pair2:
                    model_state_dict[name] = value

                filters = update_filter_size(filters, new_size1, filter_counter)
                filters = update_filter_size(filters, new_size2, filter_counter + 1)
                conv_bn_pair = []
                filter_counter += 2
    return model_state_dict, filters

def apply_mask_to_layer(tensor, mask):
    if tensor.dim() == 1:
        return torch.masked_select(tensor, mask.cuda())
    else: 
        new_size = np.shape(tensor)[0]
        i = 0
        buffer = torch.zeros([new_size, np.shape(tensor)[1], np.shape(tensor)[2], np.shape(tensor)[3]])
        for bool, net in zip(mask, tensor[1]):
            if bool:
                buffer[i] = net 
                i += 1
        return buffer


def apply_masks(model_state_dict, masks):
    fst = ['2.weight', '2.bias', '3.weight', '3.bias', '3.running_mean', '3.running_var']
    snd = ['6.weight', '6.bias', '7.weight', '7.bias', '7.running_mean', '7.running_var']

    for key, mask in masks.items():
        layer_suffix_name_list = fst if key[44] == '3' else snd
        for layer in layer_suffix_name_list:
            layer_name = key[:44] + layer
            model_state_dict[layer_name] = apply_mask_to_layer(model_state_dict[layer_name], mask)
            
    return model_state_dict


def update_filter(filters, keys_grouped_in_stacks, model_state_dict):
    buffer = []
    new_filters = []
    for stack in keys_grouped_in_stacks:
        filters_per_stack = []
        for key in stack:
            print(key)
            layer_size = len(model_state_dict[key])
            if len(buffer) < 2:
                buffer.append(layer_size)
            else: 
                filters_per_stack.append(buffer)
                buffer = [layer_size]
        filters_per_stack.append(buffer)
        buffer = []
        new_filters.append(filters_per_stack)

        filters_per_stack = []

    for i in range(0, len(new_filters)):
        filters[i+1][1:] = new_filters[i]
    print(filters)

def new_prune_channels(model_state_dict, mode= "MIN", channel_ratio=0.4, filters=None, block_temperature=None):
    select_mask = SelectMask()
    if mode == "MIN":
        select_mask = SelectMaskMin()
    elif mode == "EVENLY":
        select_mask = SelectMaskEvenly()
    elif mode == "NO_PADD":
        select_mask = SelectMaskNoPadd()
    elif mode == "CURL":
        select_mask = SelectMaskCURL()
    masks = select_mask.get_masks(model_state_dict, channel_ratio, None)
    model_state_dict = apply_masks(model_state_dict, masks)
    

    keys_grouped_in_stacks = select_mask.group_key_name_list_in_stacks(list(masks.keys()))
    filters = update_filter(filters, keys_grouped_in_stacks, model_state_dict)
    return model_state_dict, filters
        

    

def prune(model_state_dict, ratio, filters, mode, channel_ratio, block_momentum=True):
    #print("prune channels")
    #PruneCURL.prune(model_state_dict, filters, channel_ratio)
    #model_state_dict, filters = new_prune_channels(model_state_dict, filters=filters)
    #exit()
    model_state_dict, filters = prune_channels(model_state_dict, filters, mode, channel_ratio, block_momentum)



    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()

    model_state_dict = fix_dim_problems(model_state_dict, birdnet.state_dict())
    return model_state_dict, filters