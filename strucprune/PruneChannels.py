from collections import OrderedDict
import copy
from curses import KEY_A1
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
    pattern_ds_block_classifier_path = "module\.classifier\.[0-9]+\.classifier\.[0-9]\.classifierPath"
    pattern_ds_block_skip_path = "module\.classifier\.[0-9]+\.classifier\.[0-9]\.skipPath"
    pattern_resblock = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[0-9]+\."
    pattern_resstack_appendix = "module\.classifier\.[0-9]+\.classifier\.[1-9]+\."
    resstack_index = 18
    resblock_index = 31
    resblock_layer_index = 44
    classifierPath_index = 48
    skipPath_index = 42
    ds_last_bn__layer_index = 44

    ds_block_classifier_path = re.search(pattern_ds_block_classifier_path, key)
    ds_block_skip_path = re.search(pattern_ds_block_skip_path, key)
    resblock = re.search(pattern_resblock, key)
    resstack_appendix = re.search(pattern_resstack_appendix, key)
    for i, name in zip(range(0, len(key_names)), key_names):
        if ds_block_classifier_path:
            if key[resstack_index] == name[resstack_index] and key[resblock_index] == name[resblock_index]:
                ds_cp_number = int(key[classifierPath_index])
                if ds_cp_number < 2:
                    index_key = key_names[i-1]
                    mask = module_mask_list[index_key][-1] 
                elif ds_cp_number < 4:
                    mask = module_mask_list[name][0]
                else: 
                    mask = module_mask_list[name][1]
                return mask
            else: 
                continue
        elif ds_block_skip_path:
            if key[resstack_index] == name[resstack_index] and key[resblock_index] == name[resblock_index]:
                ds_sp_number = int(key[skipPath_index])
                if ds_sp_number < 2:
                    index_key = key_names[i-1]
                    mask = module_mask_list[index_key][-1] 
                return mask
            else: 
                continue
        elif resblock:
            if key[resstack_index] == name[resstack_index] and key[resblock_index] == name[resblock_index]:
                resblock_layer_number = int(key[resblock_layer_index])
                if resblock_layer_number < 4:
                    index_key = key_names[i-1]
                    if index_key[-1] == '0':
                        mask = module_mask_list[index_key][2] 
                    else: 
                        mask = module_mask_list[index_key][1]
                else: 
                    mask = module_mask_list[name][0]
                return mask
            else: 
                continue
        elif resstack_appendix:
            if int(key[resstack_index]) + 1 <= int(name[resstack_index]):
                index_key = key_names[i-1]
                mask = module_mask_list[index_key][-1]
                return mask
            else: 
                continue
    index_key = key_names[-1]

    mask = module_mask_list[index_key][1]
    return mask 

def fix_dim_problems(new_state_dict, state_dict):
    
    for key, value in new_state_dict.items():
        #print(key)
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
                new_state_dict[key] = torch.masked_select(value, mask.cuda())

        assert np.shape(new_state_dict[key]) == shape
    return new_state_dict


def apply_mask_to_layer(tensor, mask):
    if tensor.dim() == 1:
        return torch.masked_select(tensor, mask.cuda())
    else: 
        new_size = torch.numel(mask[mask==True])
        i = 0
        buffer = torch.zeros([new_size, np.shape(tensor)[1], np.shape(tensor)[2], np.shape(tensor)[3]])
        for bool, net in zip(mask, tensor):
            if bool:
                buffer[i] = net 
                i += 1
        return buffer


def select_suffix_list(key):
    fst_bn_layer = ['2.weight', '2.bias', '3.weight', '3.bias', '3.running_mean', '3.running_var']
    snd_bn_layer = ['6.weight', '6.bias', '7.weight', '7.bias', '7.running_mean', '7.running_var']
    fst_ds_layer = ['0.weight', '0.bias', '1.weight', '1.bias', '1.running_mean', '1.running_var']
    snd_ds_layer = ['3.weight', '3.bias', '4.weight', '4.bias', '4.running_mean', '4.running_var']
    last_ds_layer = ['classifierPath.8.weight', 'classifierPath.8.bias', 'skipPath.1.weight', 'skipPath.1.bias', 'batchnorm.0.weight', 'batchnorm.0.bias', 'batchnorm.0.running_mean', 'batchnorm.0.running_var']
    
    if key[44] == '3':
        return fst_bn_layer, 44
    elif key[44] == '7':
        return snd_bn_layer, 44
    elif key[43] == '0':
        return last_ds_layer, 33
    elif key[48] == '1':
        return fst_ds_layer, 48 
    elif key[48] == '4':
        return snd_ds_layer, 48 

def apply_masks(model_state_dict, masks):
    
    for key, mask in masks.items():
         
        layer_suffix_name_list, index = select_suffix_list(key)

        for layer in layer_suffix_name_list:
            layer_name = key[:index] + layer
            model_state_dict[layer_name] = apply_mask_to_layer(model_state_dict[layer_name], mask) 
    return model_state_dict


def update_filter_all(filters, keys_grouped_in_stacks, model_state_dict):
    buffer = []
    new_filters = []

    for stack in keys_grouped_in_stacks:
        filters_per_stack = []
        is_ds_block_appended = False
        for struc in stack:
            if is_ds_block_appended:
                layer_size = len(model_state_dict[struc])
                if len(buffer) < 2:
                    buffer.append(layer_size)
                else: 
                    filters_per_stack.append(buffer)
                    buffer = [layer_size]
            else: 
                layer_size = len(model_state_dict[struc])
                if len(buffer) < 3:
                    buffer.append(layer_size)
                else: 
                    filters_per_stack.append(buffer)
                    buffer = [layer_size]
                    is_ds_block_appended = True
        filters_per_stack.append(buffer)
        buffer = []
        new_filters.append(filters_per_stack)

        filters_per_stack = []

    for i in range(0, len(new_filters)):
        filters[i+1] = new_filters[i]
    return filters

def create_module_mask_list(masks):
    last_key = None
    for key, item in masks.items():
        name = key[:32]
        if name == last_key:
            module_mask_list[name] += [item]
        else: 
            module_mask_list[name] = [item]
            last_key = name

def prune_channels(model_state_dict, mode, channel_ratio, filters, block_temperature, prune_structure):
    select_mask = SelectMask(model_state_dict)
    if mode.value == 2:
        select_mask = SelectMaskMin(model_state_dict)
    elif mode.value == 0:
        select_mask = SelectMaskEvenly(model_state_dict)
    elif mode.value == 1:
        select_mask = SelectMaskNoPadd(model_state_dict)
    elif mode.value == 3:
        select_mask = SelectMaskCURL(model_state_dict)
    masks = select_mask.get_masks(model_state_dict, channel_ratio, block_temperature, part=prune_structure)
    
    model_state_dict = apply_masks(model_state_dict, masks)

    create_module_mask_list(masks)


    keys_grouped_in_stacks = select_mask.group_key_name_list_in_stacks(list(masks.keys()))

    filters = update_filter_all(filters, keys_grouped_in_stacks, model_state_dict)


    return model_state_dict, filters
        


def prune(model_state_dict, ratio, filters, mode, channel_ratio, prune_structure, block_temperature):
    model_state_dict, filters = prune_channels(copy.copy(model_state_dict), mode=mode, filters=copy.copy(filters), 
                                                channel_ratio=channel_ratio, block_temperature=block_temperature, prune_structure=prune_structure)
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    model_state_dict = fix_dim_problems(model_state_dict, birdnet.state_dict())
    return model_state_dict, filters