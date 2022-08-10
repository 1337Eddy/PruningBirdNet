import re
import torch 
import numpy as np
from prune import Pruning_Structure

from strucprune.MaskSelection import SelectMask


class SelectMaskMin(SelectMask):

    def create_mask_resblock(self, tensor1, tensor2, ratio):
        tensor = torch.cat((tensor1, tensor2))
        layer_size = len(tensor)
        indices = torch.topk(tensor, int((1-ratio) * layer_size))[1]
        mask = torch.zeros(layer_size, dtype=torch.bool)
        mask[indices] = True

        mask1, mask2 = torch.split(mask, len(tensor1))

        #Prevent that all values of mask are False. This is important because otherwise the layer has no channels
        if torch.all(~mask1):
            mask1[torch.argmax(tensor1)] = True 
        if torch.all(~mask2):
            mask2[torch.argmax(tensor2)] = True 

        return mask1, mask2 

    def create_mask_dsblock(self, tensor1, tensor2, tensor3, ratio):
        tensor = torch.cat((tensor1, tensor2, tensor3))
        layer_size = len(tensor)
        indices = torch.topk(tensor, int((1-ratio) * layer_size))[1]
        mask = torch.zeros(layer_size, dtype=torch.bool)
        mask[indices] = True
        mask1, mask2, mask3 = mask[0:len(tensor1)], mask[len(tensor1): len(tensor1)+len(tensor2)], mask[len(tensor1)+len(tensor2):]

        #Prevent that all values of mask are False. This is important because otherwise the layer has no channels
        if torch.all(~mask1):
            mask1[torch.argmax(tensor1)] = True 
        if torch.all(~mask2):
            mask2[torch.argmax(tensor2)] = True 
        if torch.all(~mask3):
            mask3[torch.argmax(tensor3)] = True 

        return mask1, mask2, mask3


    def get_masks(self, model_state_dict, ratio, block_temperature, part="ALL"):
        masks = {}
        
        layers = self.select_layers(model_state_dict, self.bn_layer_in_resblock_pattern) 
        

        
        layer_list = list(layers)
        for i in range(0, len(layer_list), 2):
            key1 = layer_list[i]
            key2 = layer_list[i+1] 
            mask1, mask2 = self.create_mask_resblock(layers[key1], layers[key2], ratio)
            masks[key1] = mask1
            masks[key2] = mask2
        
        ds_layers= self.select_layers(model_state_dict, self.bn_layer_in_dsblock_pattern)
        last_ds_layer = self.select_layers(model_state_dict, self.last_bn_layer_of_dsblock_pattern)
        layers_temp = {**ds_layers, **last_ds_layer}

        ds_layers = self.sort_dict_by_key(layers_temp)
        ds_layer_list = list(ds_layers)

        for i in range(0, len(ds_layer_list), 3):
            key1 = ds_layer_list[i]
            key2 = ds_layer_list[i+1] 
            key3 = ds_layer_list[i+2] 
            new_ratio = self.get_temperature_ratio(key1, ratio, block_temperature)
            if part == Pruning_Structure.RESBLOCK:
                if re.search(self.bn_layer_in_dsblock_pattern, key1) or re.search(self.last_bn_layer_of_dsblock_pattern, key1):
                    mask1, mask2, mask3 = self.create_mask_dsblock(ds_layers[key1], ds_layers[key2], ds_layers[key3], 0)
                    masks[key1] = mask1
                    masks[key2] = mask2
                    masks[key3] = mask3
                    continue
            mask1, mask2, mask3 = self.create_mask_dsblock(ds_layers[key1], ds_layers[key2], ds_layers[key3], new_ratio)
            masks[key1] = mask1
            masks[key2] = mask2
            masks[key3] = mask3
            
        masks = self.sort_dict_by_key(masks)


        return masks