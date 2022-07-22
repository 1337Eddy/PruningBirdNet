import torch 

from strucprune.MaskSelection import SelectMask


class SelectMaskEvenly(SelectMask):

    def create_mask(self, tensor, ratio):
        layer_size = len(tensor)
        indices = torch.topk(tensor, int(ratio * layer_size))[1]
        mask = torch.ones(layer_size, dtype=torch.bool)
        mask[indices] = False
        return mask 

    def get_masks(self, model_state_dict, ratio, block_temperature):
        masks = {}
        
        layers = self.select_layers(model_state_dict, self.fst_bn_layer_in_resblock_pattern) 

        for key in list(layers): 
            mask = self.create_mask(layers[key], ratio)
            masks[key] = mask
        return masks