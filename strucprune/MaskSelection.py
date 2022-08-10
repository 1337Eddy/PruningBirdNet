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
        #for key, item in self.scaling_factors.items():
        #    print(f"key: {key} item: {item}")

    def softmax(self, tensor, alpha):
        new = []
        for elem in tensor:
            zaehler = np.exp(elem) ** alpha
            nenner = 0
            for i in tensor:
                nenner += np.exp(i) ** alpha 
            new.append(zaehler/nenner)
        return new 

    def get_temperature_ratio(self, key, ratio, temperature):
        factors = []
        for _, item in self.scaling_factors.items():
            factors.append(item[0].cpu())

        scaling_factors = torch.tensor(self.softmax(factors, temperature))*len(self.scaling_factors)
        elem_of_resblock = re.search(self.resstack_pattern, key)
        if elem_of_resblock:
            key = key[:32]
            index = list(self.scaling_factors.keys()).index(key)
            factors = ratio + (ratio - scaling_factors*ratio)
            new_factor = torch.relu(torch.tanh(1.3 * factors))[index]

            return new_factor
        else: 
            return ratio 

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

    def sort_keys(self, layers_temp):
        keys = sorted(layers_temp)

        skip_counter = 0
        for i in range(0, len(keys)):  
            if skip_counter > 0:
                skip_counter -= 1
                continue    
            if "batchnorm" in keys[i]:
                tmp = keys[i]
                keys[i] = keys[i+1]
                keys[i+1] = keys[i+2]
                keys[i+2] = tmp 
                skip_counter = 2
        return keys

    def sort_dict_by_key(self, dict):
        buffer = {}
        keys = dict.keys()
        keys = self.sort_keys(keys)
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


