from abc import abstractclassmethod
import re

import numpy as np
import torch

class SelectMask():

    def __init__(self, model_state_dict):
        self.resstack_pattern = "module\.classifier\.[0-9]+\.classifier\."
        self.resblock_pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\."
        self.fst_bn_layer_in_resblock_pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[3].weight"
        self.snd_bn_layer_in_resblock_pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[7].weight"
        self.bn_layer_in_resblock_pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]+\.classifier\.[3|7].weight"
        self.bn_layer_in_dsblock_pattern = "module\.classifier\.[1-9].classifier\.0\.classifierPath\.[1|4]\.weight"
        self.last_bn_layer_of_dsblock_pattern = "module\.classifier\.[1-9]\.classifier\.0\.batchnorm\.0\.weight"
        self.bn_layer_in_resstack_pattern = f"[{self.last_bn_layer_of_dsblock_pattern}|{self.bn_layer_in_dsblock_pattern}|{self.bn_layer_in_resblock_pattern}]" #
        self.scaling_block_pattern = "module\.classifier\.[0-9]+\.classifier\.[1-9]+\.W"
        self.len_resstack = len("module.classifier.1")
        self.len_resblock = len("module.classifier.1.classifier.2")
        self.model_state_dict = model_state_dict
        self.scaling_factors = self.get_scaling_factors()
        self.min, self.max = self.get_min_max_sacling_factors() 

    
    def scale(self, x, out_range=(-1, 1)):
        y = (x - (self.max + self.min) / 2) / (self.max - self.min)
        return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2



    def get_temperature_ratio(self, key, ratio, temperature):
        elem_of_resblock = re.search(self.resstack_pattern, key)
        if elem_of_resblock:
            key = key[:32]
            value = self.scaling_factors[key][0]
            spread = min(ratio, 1-ratio)
            scale = self.scale(value, (-spread, spread))*temperature
            print(scale)
            return ratio + scale
        else: 
            return ratio 

    def get_min_max_sacling_factors(self):
        min = 1
        max = 0
        for _, item in self.scaling_factors.items():
            if item[0] < min:
                min = item[0]
            if item[0] > max:
                max = item[0]
        return min, max

    def get_scaling_factors(self):
        pattern = "module\.classifier\.[0-9]+\.classifier\.[0-9]\.W"
        softmax = torch.nn.Softmax(dim=0)
        scaling_factors = {}
        for key, item in self.model_state_dict.items():
            result = re.search(pattern, key)
            if result:
                factors = softmax(item)
                scaling_factors[key[:-2]] = factors
        return scaling_factors



    def group_key_name_list_in_stacks(self, layer_name_list):
        stacks = []
        temp = []
        name = layer_name_list[0]
        for key in layer_name_list:
            if key[:self.len_resstack] == name[:self.len_resstack]:
                temp.append(key)
            else: 
                stacks.append(temp)
                temp = [] 
                name = key[:self.len_resstack]
                temp.append(key)
        stacks.append(temp)
        return stacks 

    def sort_dict_by_key(self, dict):
        buffer = {}
        keys = dict.keys()
        keys = sorted(keys)
        for key in keys:
            buffer[key] = dict[key] 
        return buffer
        
    def calc_mean_of_block_ratios(self, model_state_dict):
        pattern_scaling_factor = ""
        softmax = torch.nn.Softmax(dim=0)
        block_ratios = np.array([])
        for key, value in model_state_dict.items():
            scaling_factor = re.search(pattern_scaling_factor, key)
            if scaling_factor:
                value = 1 - softmax(value)[0]
                value = np.array(value.cpu())
                block_ratios = np.append(block_ratios, value)
        mean = np.mean(block_ratios)

        return torch.tensor([mean]).cuda()

    def select_layers(self, model_state_dict, pattern):
        sub_dict = {}
        for key, value in model_state_dict.items():
            if isinstance(pattern, list):
                for elem in pattern:
                    element = re.search(elem, key)
                    if element:
                        sub_dict[key] = value 
            elif isinstance(pattern, str):
                element = re.search(pattern, key)
                if element:
                    sub_dict[key] = value 
        return sub_dict

    @classmethod
    @abstractclassmethod
    def get_masks(self, model_state_dict, ratio, block_temperature, part="ALL"):
        pass 


