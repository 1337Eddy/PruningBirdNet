import torch 
import numpy as np

from strucprune.MaskSelection import SelectMask


class SelectMaskCURL(SelectMask):

    def create_mask(self, tensor, ratio):
        layer_size = len(tensor)
        indices = torch.topk(tensor, int((1-ratio) * layer_size))[1]
        mask = torch.zeros(layer_size, dtype=torch.bool)
        mask[indices] = True
        return mask 

    def sum_layers_in_stack(self, stacks, model_state_dict):
        summation_dict = {}
        for stack in stacks:
            sum = None 
            for block in stack:
                if sum != None: 
                    sum += torch.abs(model_state_dict[block]) 
                else:
                    sum = torch.abs(model_state_dict[block])
            for block in stack:
                summation_dict[block] = sum 
        return summation_dict
    
    def get_masks(self, model_state_dict, ratio, block_temperature):
        masks = {}
        
        fst_layers = self.select_layers(model_state_dict, self.fst_bn_layer_in_resblock_pattern) 
        snd_layers = self.select_layers(model_state_dict, self.snd_bn_layer_in_resblock_pattern) 

        stacks = self.group_key_name_list_in_stacks(list(snd_layers.keys()))
        
        sum_of_snd_bn_layer_per_resstack = self.sum_layers_in_stack(stacks, model_state_dict)

        layers_temp = {**fst_layers, **sum_of_snd_bn_layer_per_resstack}
        keys = sorted(layers_temp)
        layers = {}
        for key in keys:
            layers[key] = layers_temp[key]

        for key in list(layers): 
            mask = self.create_mask(layers[key], ratio)
            masks[key] = mask
        return masks