import torch 
import numpy as np

from strucprune.MaskSelection import SelectMask


class SelectMaskMin(SelectMask):

    def create_mask(self, tensor1, tensor2, ratio):
        tensor = torch.cat((tensor1, tensor2))
        layer_size = len(tensor)
        indices = torch.topk(tensor, int(ratio * layer_size))[1]
        mask = torch.ones(layer_size, dtype=torch.bool)
        mask[indices] = False

        mask1, mask2 = torch.split(mask, len(tensor1))

        #Prevent that all values of mask are False. This is important because otherwise the layer has no channels
        if torch.all(~mask1):
            mask1[torch.argmax(tensor1)] = True 
        if torch.all(~mask2):
            mask2[torch.argmax(tensor2)] = True 

        return mask1, mask2 

    def get_masks(self, model_state_dict, ratio, block_temperature):
        masks = {}
        
        layers = self.select_layers(model_state_dict, self.bn_layer_in_resblock_pattern) 

        
        layer_list = list(layers)
        for i in range(0, len(layer_list), 2):
            key1 = layer_list[i]
            key2 = layer_list[i+1] 
            mask1, mask2 = self.create_mask(layers[key1], layers[key2], ratio)
            masks[key1] = mask1
            masks[key2] = mask2
        return masks