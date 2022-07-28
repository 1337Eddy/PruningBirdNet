import re
import torch
from prune import Pruning_Structure 

from strucprune.MaskSelection import SelectMask


class SelectMaskEvenly(SelectMask):

    def create_mask(self, tensor, ratio):
        layer_size = len(tensor)
        indices = torch.topk(tensor, int((1-ratio) * layer_size))[1]
        mask = torch.zeros(layer_size, dtype=torch.bool)
        mask[indices] = True
        return mask 

    def get_masks(self, model_state_dict, ratio, block_temperature, part="ALL"):
        masks = {}     
        layers = self.select_layers(model_state_dict, [self.bn_layer_in_resblock_pattern, self.bn_layer_in_dsblock_pattern, self.last_bn_layer_of_dsblock_pattern]) 


        for key in list(layers): 
            if part == Pruning_Structure.RESBLOCK:
                if re.search(self.bn_layer_in_dsblock_pattern, key) or re.search(self.last_bn_layer_of_dsblock_pattern, key):
                    mask = self.create_mask(layers[key], 0)
                    masks[key] = mask.cuda()
                    continue
            mask = self.create_mask(layers[key], ratio)
            masks[key] = mask.cuda()
        return masks