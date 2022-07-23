from numpy import sort
import torch 

from strucprune.MaskSelection import SelectMask


class SelectMaskNoPadd(SelectMask):

    def create_mask(self, tensor, ratio):
        layer_size = len(tensor)
        indices = torch.topk(tensor, int((1-ratio) * layer_size))[1]
        mask = torch.zeros(layer_size, dtype=torch.bool)
        mask[indices] = True
        return mask 

    def get_masks(self, model_state_dict, ratio, block_temperature):
        masks = {}
        
        fst_layers = self.select_layers(model_state_dict, self.fst_bn_layer_in_resblock_pattern) 
        snd_layers = self.select_layers(model_state_dict, self.snd_bn_layer_in_resblock_pattern) 

        for key in list(fst_layers): 
            mask = self.create_mask(fst_layers[key], ratio)
            masks[key] = mask
        
        for key in list(snd_layers):
            mask = torch.ones(len(snd_layers[key]), dtype=torch.bool)
            masks[key] = mask

        masks = self.sort_dict_by_key(masks)

        return masks