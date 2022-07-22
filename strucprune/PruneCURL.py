import torch 
import re 



def select_layers(model_state_dict, pattern):
    sub_dict = {}
    for key, value in model_state_dict.items():
        element = re.search(pattern, key)
        if element:
            sub_dict[key] = value 
    return sub_dict

def create_masks_evenly(layers, channel_ratio):
    masks = []
    for layer in layers:
        layer_size = len(layer)
        indices = torch.topk(layer, int(channel_ratio * layer_size))[1]
        mask = torch.zeros(layer_size, dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    return masks

def apply_masks_to_model_state_dict(model_state_dict, masks):
    
    return

def prune(model_state_dict, filters, channel_ratio):
    pattern_last_bn_layer_of_resblock = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[7].weight"
    last_bn_layer_of_resblock = select_layers(model_state_dict, pattern_last_bn_layer_of_resblock)
    sum_layers_per_resstack = []
    index = None
    for key, item in last_bn_layer_of_resblock.items():
        if index == key[18]:
            sum_layers_per_resstack[-1][0] += torch.abs(item)
            sum_layers_per_resstack[-1][1] += 1
        else: 
            sum_layers_per_resstack.append([torch.abs(item), 1])
            index = key[18]
    
    masks = create_masks_evenly([x[0] for x in sum_layers_per_resstack], channel_ratio)

    
    exit()
    return None